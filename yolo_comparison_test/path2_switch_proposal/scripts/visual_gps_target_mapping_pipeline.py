#!/usr/bin/env python3
"""Lightweight visual-GPS target mapping pipeline plus sim benchmark harness.

Implements a Jetson-friendly pipeline:
  1) YOLO/HSV detections -> bottom-center pixels
  2) Horizon line analysis (pitch/roll)
  3) Heading resolver from GPS jitter (COG) with optical-flow fallback
  4) Monocular inverse projection to local meters
  5) Local->global GPS conversion
  6) Spatial registry with deduplication / EMA smoothing / confirmation threshold

This file also includes a benchmark mode that compares the new pipeline against the
current reprojection baseline using simulated drone positions as pseudo-GPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
GZ_SCRIPTS = REPO_ROOT / "gazebo" / "scripts"
if str(GZ_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(GZ_SCRIPTS))

from load_camera_intrinsics import default_calibration_json, resolve_pinhole_intrinsics  # noqa: E402
from nadir_projection import project_pixel_to_ground_ned  # noqa: E402
from reproject_detection_map import parse_buoys_from_world_sdf  # noqa: E402


EARTH_R_M = 6378137.0


@dataclass
class DetectionObs:
    u: float
    v: float
    x1: float
    y1: float
    x2: float
    y2: float
    color: str
    matched_gt_color: str
    image_name: str
    clip: str
    box_conf: float
    color_conf: float


def meters_to_latlon(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    d_lat = (north_m / EARTH_R_M) * (180.0 / math.pi)
    cos_lat = max(1e-6, abs(float(math.cos(math.radians(lat_deg)))))
    d_lon = (east_m / (EARTH_R_M * cos_lat)) * (180.0 / math.pi)
    return lat_deg + d_lat, lon_deg + d_lon


def latlon_to_local_ne(origin_lat: float, origin_lon: float, lat_deg: float, lon_deg: float) -> tuple[float, float]:
    dn = math.radians(lat_deg - origin_lat) * EARTH_R_M
    de = math.radians(lon_deg - origin_lon) * EARTH_R_M * math.cos(math.radians(origin_lat))
    return dn, de


def flat_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dn, de = latlon_to_local_ne(lat1, lon1, lat2, lon2)
    return float(math.hypot(dn, de))


def clip_from_symlink(dataset_test_dir: Path, image_name: str) -> str | None:
    p = dataset_test_dir / image_name
    if not p.is_symlink():
        return None
    try:
        raw = os.readlink(str(p))
    except OSError:
        return None
    parts = Path(raw).parts
    if "gazebo_uav_batch" in parts:
        i = parts.index("gazebo_uav_batch")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def source_image_name(test_image_name: str) -> str:
    if "__" in test_image_name:
        return test_image_name.split("__", 1)[1]
    return test_image_name


def load_pose_by_image(pose_csv: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not pose_csv.is_file():
        return out
    with pose_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            img = (r.get("image_name") or "").strip()
            if not img:
                continue
            try:
                out[img] = {
                    "x": float(r.get("drone_x_m") or 0.0),
                    "y": float(r.get("drone_y_m") or 0.0),
                    "z": float(r.get("drone_z_m") or 0.0),
                    "yaw_deg": float(r.get("drone_yaw_deg") or 0.0),
                    "lat_deg": float(r.get("drone_lat_deg") or 0.0) if (r.get("drone_lat_deg") or "").strip() else float("nan"),
                    "lon_deg": float(r.get("drone_lon_deg") or 0.0) if (r.get("drone_lon_deg") or "").strip() else float("nan"),
                }
            except ValueError:
                continue
    return out


def world_sdf_for_clip(worlds_dir: Path, clip_name: str) -> Path | None:
    exact = worlds_dir / f"{clip_name}.sdf"
    if exact.is_file():
        return exact
    m = re.search(r"robotx_dr_(\d{3})", clip_name)
    if not m:
        return None
    cands = sorted(worlds_dir.glob(f"robotx_dr_{m.group(1)}_*.sdf"))
    return cands[0] if cands else None


def nearest_same_color_error_m(est_n: float, est_e: float, color: str, gt_pts: list[tuple[float, float, str]]) -> float | None:
    best = float("inf")
    for gn, ge, gc in gt_pts:
        if gc != color:
            continue
        d = float(math.hypot(est_n - gn, est_e - ge))
        if d < best:
            best = d
    return None if best == float("inf") else best


class TargetLocalizationPipeline:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_height: float,
        *,
        origin_lat_deg: float,
        origin_lon_deg: float,
    ) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.H_cam = float(camera_height)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        self.origin_lat_deg = float(origin_lat_deg)
        self.origin_lon_deg = float(origin_lon_deg)

        self.current_heading = 0.0
        self.last_drone_lat: float | None = None
        self.last_drone_lon: float | None = None
        self.prev_horizon_features: np.ndarray | None = None
        self.prev_gray_frame: np.ndarray | None = None
        self.last_pitch_deg = 0.0
        self.last_roll_deg = 0.0

        self.tracked_targets: dict[int, dict[str, float | int | str | bool]] = {}
        self.target_id_counter = 0
        self.frame_counter = 0
        self.transmissions: list[dict[str, float | int | str]] = []

    def resolve_heading(self, current_lat: float, current_lon: float, gray_frame: np.ndarray) -> float:
        """Resolve heading from GPS COG, fall back to optical flow when stationary."""
        if self.last_drone_lat is not None and self.last_drone_lon is not None:
            dist_moved = flat_distance_m(self.last_drone_lat, self.last_drone_lon, current_lat, current_lon)
            if dist_moved > 0.5:
                lat1, lon1, lat2, lon2 = map(math.radians, [self.last_drone_lat, self.last_drone_lon, current_lat, current_lon])
                d_lon_rad = lon2 - lon1
                x = math.sin(d_lon_rad) * math.cos(lat2)
                y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon_rad)
                self.current_heading = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
                self.last_drone_lat = current_lat
                self.last_drone_lon = current_lon
                self.prev_gray_frame = gray_frame.copy()
                return self.current_heading

        self.last_drone_lat = current_lat
        self.last_drone_lon = current_lon

        if self.prev_gray_frame is not None and self.prev_horizon_features is not None and len(self.prev_horizon_features) > 0:
            next_features, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray_frame,
                gray_frame,
                self.prev_horizon_features,
                None,
            )
            if next_features is not None and status is not None and int(np.sum(status)) > 3:
                good_prev = self.prev_horizon_features[status.flatten() == 1].reshape(-1, 2)
                good_next = next_features[status.flatten() == 1].reshape(-1, 2)
                if len(good_prev) > 0:
                    avg_dx = float(np.mean(good_next[:, 0] - good_prev[:, 0]))
                    delta_yaw = math.degrees(math.atan2(avg_dx, self.fx))
                    self.current_heading = (self.current_heading - delta_yaw + 360.0) % 360.0

        self.prev_gray_frame = gray_frame.copy()
        return self.current_heading

    def estimate_horizon_angles(self, frame: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Estimate pitch/roll from dominant horizon line via Hough lines."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=100, maxLineGap=12)

        if lines is None or len(lines) == 0:
            self.prev_horizon_features = None
            return self.last_pitch_deg, self.last_roll_deg, gray

        best_line = None
        best_score = -1.0
        for entry in lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(float, entry)
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 20.0:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx)))
            angle = min(angle, 180.0 - angle)
            horizon_score = max(0.0, 35.0 - angle) / 35.0
            score = length * (0.3 + horizon_score)
            if score > best_score:
                best_score = score
                best_line = (x1, y1, x2, y2)

        if best_line is None:
            self.prev_horizon_features = None
            return self.last_pitch_deg, self.last_roll_deg, gray

        x1, y1, x2, y2 = best_line
        roll = math.degrees(math.atan2(y2 - y1, x2 - x1))
        mid_x = self.cx
        mid_y = y1 if abs(x2 - x1) < 1e-6 else y1 + (mid_x - x1) * (y2 - y1) / (x2 - x1)
        pitch = math.degrees(math.atan2(mid_y - self.cy, self.fy))

        features = []
        if abs(x2 - x1) < 1e-6:
            xs = np.full(12, x1, dtype=np.float32)
            ys = np.linspace(y1, y2, 12, dtype=np.float32)
        else:
            xs = np.linspace(x1, x2, 12, dtype=np.float32)
            ys = y1 + (xs - x1) * ((y2 - y1) / (x2 - x1))
        for xx, yy in zip(xs, ys):
            features.append([[np.float32(xx), np.float32(yy)]])
        self.prev_horizon_features = np.array(features, dtype=np.float32)
        self.last_pitch_deg = pitch
        self.last_roll_deg = roll
        return pitch, roll, gray

    def inverse_camera_plane_intersector(
        self,
        u: float,
        v: float,
        pitch_deg: float,
        roll_deg: float,
        *,
        height_m: float | None = None,
    ) -> tuple[float, float]:
        """Project pixel to camera-relative local plane (east, north) in meters.

        For efficiency and stability on our downward-looking simulated camera, this uses a small-angle,
        nadir-compatible approximation and injects horizon-derived roll/pitch as offsets.
        """
        H = float(height_m if height_m is not None else self.H_cam)
        x_c = (float(u) - self.cx) / self.fx
        y_c = (float(v) - self.cy) / self.fy
        east_local = x_c * H
        north_local = -y_c * H
        east_local += H * math.tan(math.radians(roll_deg))
        north_local += H * math.tan(math.radians(pitch_deg))
        return east_local, north_local

    def pixel_to_global_gps(
        self,
        u: float,
        v: float,
        pitch_deg: float,
        roll_deg: float,
        drone_lat: float,
        drone_lon: float,
        *,
        height_m: float | None = None,
    ) -> tuple[float, float, float, float]:
        east_local, north_local = self.inverse_camera_plane_intersector(u, v, pitch_deg, roll_deg, height_m=height_m)
        psi = math.radians(self.current_heading)
        dn = north_local * math.cos(psi) - east_local * math.sin(psi)
        de = north_local * math.sin(psi) + east_local * math.cos(psi)
        lat, lon = meters_to_latlon(drone_lat, drone_lon, dn, de)
        return lat, lon, dn, de

    def update_spatial_registry(self, new_lat: float, new_lon: float, color: str) -> None:
        alpha = 0.1
        threshold_m = 3.0
        matched_id = None
        for obj_id, data in self.tracked_targets.items():
            if str(data["color"]) != color:
                continue
            dist = flat_distance_m(float(data["lat"]), float(data["lon"]), new_lat, new_lon)
            if dist < threshold_m:
                matched_id = obj_id
                break

        if matched_id is not None:
            data = self.tracked_targets[matched_id]
            smoothed_lat = (1.0 - alpha) * float(data["lat"]) + alpha * new_lat
            smoothed_lon = (1.0 - alpha) * float(data["lon"]) + alpha * new_lon
            hits = int(data["hits"]) + 1
            data.update({"lat": smoothed_lat, "lon": smoothed_lon, "hits": hits, "last_seen_frame": self.frame_counter})
            if hits == 15 and not bool(data.get("transmitted", False)):
                data["transmitted"] = True
                self.transmit_to_receiver(matched_id, color, smoothed_lat, smoothed_lon)
        else:
            self.target_id_counter += 1
            self.tracked_targets[self.target_id_counter] = {
                "lat": new_lat,
                "lon": new_lon,
                "hits": 1,
                "last_seen_frame": self.frame_counter,
                "color": color,
                "transmitted": False,
            }

    def transmit_to_receiver(self, target_id: int, color: str, lat: float, lon: float) -> None:
        payload = {
            "target_id": target_id,
            "color": color,
            "lat": lat,
            "lon": lon,
            "frame": self.frame_counter,
        }
        self.transmissions.append(payload)
        print(f"[TRANSMIT] {json.dumps(payload)}")

    def process_frame(
        self,
        frame: np.ndarray,
        drone_lat: float,
        drone_lon: float,
        detections: list[DetectionObs],
        *,
        camera_height_m: float | None = None,
    ) -> tuple[list[dict[str, float | str]], dict[str, float]]:
        self.frame_counter += 1
        pitch_deg, roll_deg, gray = self.estimate_horizon_angles(frame)
        heading_deg = self.resolve_heading(drone_lat, drone_lon, gray)

        outputs: list[dict[str, float | str]] = []
        for det in detections:
            lat, lon, dn, de = self.pixel_to_global_gps(
                det.u,
                det.v,
                pitch_deg,
                roll_deg,
                drone_lat,
                drone_lon,
                height_m=camera_height_m,
            )
            self.update_spatial_registry(lat, lon, det.color)
            outputs.append(
                {
                    "image": det.image_name,
                    "clip": det.clip,
                    "color": det.color,
                    "matched_gt_color": det.matched_gt_color,
                    "lat": lat,
                    "lon": lon,
                    "dn_m": dn,
                    "de_m": de,
                    "heading_deg": heading_deg,
                    "pitch_deg": pitch_deg,
                    "roll_deg": roll_deg,
                }
            )
        return outputs, {"heading_deg": heading_deg, "pitch_deg": pitch_deg, "roll_deg": roll_deg}


def make_detection_rows(
    csv_path: Path,
    dataset_test_dir: Path,
    *,
    matched_only: bool,
    min_box_conf: float,
    min_color_conf: float,
    clips_allow: set[str] | None,
) -> dict[str, list[DetectionObs]]:
    by_image: dict[str, list[DetectionObs]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            clip = row.get("clip", "").strip() or (clip_from_symlink(dataset_test_dir, row.get("image", "")) or "")
            if clips_allow is not None and clip not in clips_allow:
                continue
            gt_color = (row.get("matched_gt_color") or "unmatched").strip().lower()
            box_conf = float(row.get("box_conf") or 0.0)
            color_conf = float(row.get("color_conf") or 0.0)
            if matched_only and gt_color in ("", "unmatched"):
                continue
            if box_conf < min_box_conf or color_conf < min_color_conf:
                continue
            try:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except (KeyError, ValueError):
                continue
            obs = DetectionObs(
                u=0.5 * (x1 + x2),
                v=y2,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                color=(row.get("pred_color_hsv") or "unknown").strip().lower(),
                matched_gt_color=gt_color,
                image_name=row["image"],
                clip=clip,
                box_conf=box_conf,
                color_conf=color_conf,
            )
            by_image.setdefault(obs.image_name, []).append(obs)
    return by_image


def resolved_frame_path(dataset_test_dir: Path, captures_root: Path, clip: str, image_name: str) -> Path:
    symlink = dataset_test_dir / image_name
    if symlink.is_symlink():
        return Path(os.path.realpath(symlink))
    return captures_root / clip / source_image_name(image_name)


def pose_to_pseudo_gps(pose: dict[str, float], origin_lat: float, origin_lon: float) -> tuple[float, float]:
    lat = pose.get("lat_deg", float("nan"))
    lon = pose.get("lon_deg", float("nan"))
    if not math.isnan(lat) and not math.isnan(lon):
        return float(lat), float(lon)
    return meters_to_latlon(origin_lat, origin_lon, float(pose["y"]), float(pose["x"]))


def summarize_errors(errs: list[float]) -> dict[str, float | int]:
    if not errs:
        return {"count": 0}
    arr = np.array(sorted(errs), dtype=np.float64)
    p95_idx = min(len(arr) - 1, int(round(0.95 * (len(arr) - 1))))
    return {
        "count": int(len(arr)),
        "mean_m": float(np.mean(arr)),
        "median_m": float(np.median(arr)),
        "p95_m": float(arr[p95_idx]),
        "rmse_m": float(np.sqrt(np.mean(arr * arr))),
    }


def evaluate_current_baseline(
    detections_by_image: dict[str, list[DetectionObs]],
    pose_by_image: dict[str, dict[str, float]],
    gt_pts: list[tuple[float, float, str]],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    proj_extra_yaw_deg: float,
    proj_yaw_sign: float,
    proj_mirror_east_cam: bool,
    proj_mirror_north_cam: bool,
) -> tuple[dict[str, float | int], list[float]]:
    errs: list[float] = []
    for image_name, dets in detections_by_image.items():
        pose = pose_by_image.get(source_image_name(image_name))
        if pose is None:
            continue
        for det in dets:
            n, e = project_pixel_to_ground_ned(
                u=det.u,
                v=0.5 * (det.y1 + det.y2),
                altitude_m=max(0.01, float(pose["z"])),
                fx_px=fx,
                fy_px=fy,
                cx_px=cx,
                cy_px=cy,
                heading_deg=float(proj_yaw_sign) * float(pose["yaw_deg"]),
                extra_yaw_deg=proj_extra_yaw_deg,
                mirror_east_cam=proj_mirror_east_cam,
                mirror_north_cam=proj_mirror_north_cam,
            )
            wn = float(pose["y"]) + n
            we = float(pose["x"]) + e
            dd = nearest_same_color_error_m(wn, we, det.matched_gt_color, gt_pts)
            if dd is not None:
                errs.append(dd)
    return summarize_errors(errs), errs


def evaluate_visual_gps_pipeline(
    detections_by_image: dict[str, list[DetectionObs]],
    pose_by_image: dict[str, dict[str, float]],
    gt_pts: list[tuple[float, float, str]],
    *,
    ordered_frames: list[tuple[str, Path]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    origin_lat: float,
    origin_lon: float,
) -> tuple[dict[str, object], list[float]]:
    pipe = TargetLocalizationPipeline(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_height=10.0,
        origin_lat_deg=origin_lat,
        origin_lon_deg=origin_lon,
    )
    errs: list[float] = []
    frame_states: list[dict[str, float | str]] = []
    for image_name, fp in ordered_frames:
        pose = pose_by_image.get(source_image_name(image_name))
        if pose is None:
            continue
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        drone_lat, drone_lon = pose_to_pseudo_gps(pose, origin_lat, origin_lon)
        outputs, state = pipe.process_frame(
            frame,
            drone_lat,
            drone_lon,
            detections_by_image.get(image_name, []),
            camera_height_m=max(0.01, float(pose["z"])),
        )
        frame_states.append({"image": image_name, **state})
        for out in outputs:
            est_n, est_e = latlon_to_local_ne(origin_lat, origin_lon, float(out["lat"]), float(out["lon"]))
            dd = nearest_same_color_error_m(est_n, est_e, str(out["matched_gt_color"]), gt_pts)
            if dd is not None:
                errs.append(dd)
    return (
        {
            **summarize_errors(errs),
            "transmissions": pipe.transmissions,
            "confirmed_targets": len(pipe.transmissions),
            "final_heading_deg": float(pipe.current_heading),
            "frame_states": frame_states[:10],
        },
        errs,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    dataset_test_dir = args.dataset_test_dir.resolve()
    captures_root = args.captures_root.resolve()
    worlds_dir = args.worlds_dir.resolve()

    clips_allow = {c.strip() for c in args.clips.split(",")} if args.clips else None
    detections_by_image = make_detection_rows(
        args.csv,
        dataset_test_dir,
        matched_only=args.matched_only,
        min_box_conf=args.min_box_conf,
        min_color_conf=args.min_color_conf,
        clips_allow=clips_allow,
    )
    if not detections_by_image:
        raise RuntimeError("No usable detections found for benchmark.")

    clip_to_images: dict[str, list[str]] = {}
    for image_name, dets in detections_by_image.items():
        clip = dets[0].clip
        clip_to_images.setdefault(clip, []).append(image_name)

    summary: dict[str, object] = {"clips": {}, "overall": {}}
    overall_current: list[float] = []
    overall_visual: list[float] = []
    for clip, image_names in sorted(clip_to_images.items()):
        pose_csv = captures_root / clip / "uav_pose_log.csv"
        pose_by_image = load_pose_by_image(pose_csv)
        world_sdf = world_sdf_for_clip(worlds_dir, clip)
        gt_pts = parse_buoys_from_world_sdf(world_sdf) if world_sdf and world_sdf.is_file() else []
        if not gt_pts:
            continue

        first_path = resolved_frame_path(dataset_test_dir, captures_root, clip, image_names[0])
        first = cv2.imread(str(first_path))
        if first is None:
            continue
        h, w = first.shape[:2]
        fx, fy, cx, cy = resolve_pinhole_intrinsics(
            intrinsics_path=args.intrinsics if args.intrinsics.is_file() else None,
            fx_override=None,
            fy_override=None,
            cx_override=None,
            cy_override=None,
            image_w=w,
            image_h=h,
        )

        ordered_frames = [(name, resolved_frame_path(dataset_test_dir, captures_root, clip, name)) for name in sorted(image_names)]
        dets_for_clip = {name: detections_by_image[name] for name in image_names}
        current, current_errs = evaluate_current_baseline(
            dets_for_clip,
            pose_by_image,
            gt_pts,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            proj_extra_yaw_deg=args.current_proj_extra_yaw_deg,
            proj_yaw_sign=args.current_proj_yaw_sign,
            proj_mirror_east_cam=args.current_proj_mirror_east_cam,
            proj_mirror_north_cam=args.current_proj_mirror_north_cam,
        )
        visual, visual_errs = evaluate_visual_gps_pipeline(
            dets_for_clip,
            pose_by_image,
            gt_pts,
            ordered_frames=ordered_frames,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            origin_lat=args.origin_lat,
            origin_lon=args.origin_lon,
        )

        clip_summary = {"current_pipeline": current, "visual_gps_pipeline": visual, "world_sdf": str(world_sdf)}
        summary["clips"][clip] = clip_summary
        overall_current.extend(current_errs)
        overall_visual.extend(visual_errs)

    summary["overall"] = {
        "current_pipeline": summarize_errors(overall_current),
        "visual_gps_pipeline": summarize_errors(overall_visual),
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "yolo_comparison_test/path2_switch_proposal/scripts/roi_hsv_eval_center_circle_cv_rerun/test_eval/roi_hsv_test_results.csv",
    )
    p.add_argument(
        "--dataset-test-dir",
        type=Path,
        default=REPO_ROOT / "yolo_comparison_test/path2_switch_proposal/scripts/roi_hsv_eval_center_circle_cv_rerun/dataset_roi/images/test",
    )
    p.add_argument(
        "--captures-root",
        type=Path,
        default=REPO_ROOT / "captures/gazebo_uav_batch",
    )
    p.add_argument(
        "--worlds-dir",
        type=Path,
        default=REPO_ROOT / "gazebo/worlds/generated",
    )
    p.add_argument(
        "--intrinsics",
        type=Path,
        default=default_calibration_json(REPO_ROOT),
    )
    p.add_argument("--clips", type=str, default="", help="Comma-separated clip names to benchmark; empty = all available.")
    p.add_argument("--matched-only", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--min-box-conf", type=float, default=0.35)
    p.add_argument("--min-color-conf", type=float, default=0.35)
    p.add_argument("--origin-lat", type=float, default=32.88010)
    p.add_argument("--origin-lon", type=float, default=-117.23420)
    p.add_argument("--current-proj-extra-yaw-deg", type=float, default=0.0)
    p.add_argument("--current-proj-yaw-sign", type=float, default=-1.0)
    p.add_argument("--current-proj-mirror-east-cam", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--current-proj-mirror-north-cam", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "yolo_comparison_test/path2_switch_proposal/scripts/visual_gps_mapping_benchmark_summary.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_benchmark(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote benchmark summary to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

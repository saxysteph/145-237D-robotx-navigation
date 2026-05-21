#!/usr/bin/env python3
"""Create per-clip test videos and ray-projection map overlays from camera pose logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
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

COLOR_BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 120, 0),
    "unknown": (220, 220, 220),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="roi_hsv_test_results.csv")
    p.add_argument("--frames-dir", type=Path, required=True, help="annotated_frames dir")
    p.add_argument("--dataset-test-dir", type=Path, required=True, help="dataset_roi/images/test symlink dir")
    p.add_argument("--captures-root", type=Path, default=REPO_ROOT / "captures" / "gazebo_uav_batch")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, default=default_calibration_json(REPO_ROOT))
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--max-clips", type=int, default=0)
    p.add_argument(
        "--proj-extra-yaw-deg",
        type=float,
        default=90.0,
        help="Extra yaw applied in reprojection to match camera frame convention (default 90).",
    )
    p.add_argument(
        "--auto-proj-yaw",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Auto-select per-clip projection yaw offset from {-90,0,90,180} using matched GT fit.",
    )
    p.add_argument(
        "--proj-yaw-sign",
        type=float,
        default=1.0,
        help="Manual yaw sign (1 or -1) used when --no-auto-proj-yaw.",
    )
    p.add_argument(
        "--proj-swap-xy",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Swap camera origin mapping x/y when --no-auto-proj-yaw.",
    )
    p.add_argument(
        "--proj-mirror-east-cam",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Mirror camera east axis before world rotation (fixes left/right flip).",
    )
    p.add_argument(
        "--proj-mirror-north-cam",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Mirror camera north axis before world rotation.",
    )
    p.add_argument("--min-box-conf", type=float, default=0.35,
                   help="Minimum box_conf to include in reprojection map (default 0.35).")
    p.add_argument("--min-color-conf", type=float, default=0.35,
                   help="Minimum color_conf to include in reprojection map (default 0.35).")
    p.add_argument(
        "--matched-only",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include only matched GT detections in reprojection map (default true).",
    )
    return p.parse_args()


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


def draw_map_panel(
    w: int,
    h: int,
    pts_bounds: list[tuple[float, float, str]],
    pts_cur: list[tuple[float, float, str]],
) -> np.ndarray:
    panel = np.full((h, w, 3), 25, dtype=np.uint8)
    # No detections for this frame: keep panel clean (no historical backdrop points).
    if not pts_cur:
        cv2.putText(panel, "No projected points", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        return panel

    # Keep a stable map scale using clip-wide filtered points when available,
    # but only draw current-frame points to avoid noisy backdrops.
    src = pts_bounds if pts_bounds else pts_cur
    ns = np.array([p[0] for p in src], dtype=np.float32)
    es = np.array([p[1] for p in src], dtype=np.float32)
    nmin, nmax = float(ns.min()), float(ns.max())
    emin, emax = float(es.min()), float(es.max())
    pad_n = max(1.0, 0.1 * (nmax - nmin + 1e-6))
    pad_e = max(1.0, 0.1 * (emax - emin + 1e-6))
    nmin -= pad_n
    nmax += pad_n
    emin -= pad_e
    emax += pad_e

    def to_px(n: float, e: float) -> tuple[int, int]:
        x = int((e - emin) / max(1e-6, (emax - emin)) * (w - 1))
        y = int((1.0 - (n - nmin) / max(1e-6, (nmax - nmin))) * (h - 1))
        return x, y

    cv2.putText(
        panel,
        "Ray-projected buoy coordinates (current frame, world N/E)",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        2,
    )
    for n, e, c in pts_cur:
        x, y = to_px(n, e)
        col = COLOR_BGR.get(c, COLOR_BGR["unknown"])
        cv2.circle(panel, (x, y), 8, col, -1, cv2.LINE_AA)
        cv2.circle(panel, (x, y), 10, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


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
                }
            except ValueError:
                continue
    return out


def source_image_name(test_image_name: str) -> str:
    if "__" in test_image_name:
        return test_image_name.split("__", 1)[1]
    return test_image_name


def world_sdf_for_clip(worlds_dir: Path, clip_name: str) -> Path | None:
    exact = worlds_dir / f"{clip_name}.sdf"
    if exact.is_file():
        return exact
    stem = clip_name.split("__", 1)[0]
    cands = sorted(worlds_dir.glob(f"{stem}_*.sdf"))
    return cands[0] if cands else None


def nearest_same_color_error_m(wn: float, we: float, color: str, gt_pts: list[tuple[float, float, str]]) -> float | None:
    best = float("inf")
    for gn, ge, gc in gt_pts:
        if gc != color:
            continue
        d = float(np.hypot(wn - gn, we - ge))
        if d < best:
            best = d
    return None if best == float("inf") else best


def world_xy_from_pose(pose: dict[str, float], *, swap_xy: bool) -> tuple[float, float]:
    """Return (world_north, world_east) camera origin from pose CSV mapping."""
    if swap_xy:
        return float(pose["x"]), float(pose["y"])
    return float(pose["y"]), float(pose["x"])


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_image[row["image"]].append(row)

    frame_paths = sorted([p for p in args.frames_dir.glob("*.jpg") if p.name in rows_by_image])
    clip_to_frames: dict[str, list[Path]] = defaultdict(list)
    for fp in frame_paths:
        clip = clip_from_symlink(args.dataset_test_dir, fp.name)
        if clip:
            clip_to_frames[clip].append(fp)

    clip_names = sorted(clip_to_frames.keys())
    if args.max_clips > 0:
        clip_names = clip_names[: args.max_clips]

    summary: dict[str, object] = {}
    worlds_dir = REPO_ROOT / "gazebo" / "worlds" / "generated"
    for clip in clip_names:
        frames = sorted(clip_to_frames[clip])
        if not frames:
            continue
        first = cv2.imread(str(frames[0]))
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

        pose_map = load_pose_by_image(args.captures_root / clip / "uav_pose_log.csv")
        clip_dir = args.out_dir / clip
        clip_dir.mkdir(parents=True, exist_ok=True)
        frames_mp4 = clip_dir / "test_frames.mp4"
        map_mp4 = clip_dir / "ray_projection_map.mp4"
        vw_frames = cv2.VideoWriter(str(frames_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        vw_map = cv2.VideoWriter(str(map_mp4), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w * 2, h))
        frames_path_used = frames_mp4
        map_path_used = map_mp4
        if not vw_frames.isOpened():
            frames_avi = clip_dir / "test_frames.avi"
            vw_frames = cv2.VideoWriter(str(frames_avi), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w, h))
            frames_path_used = frames_avi
        if not vw_map.isOpened():
            map_avi = clip_dir / "ray_projection_map.avi"
            vw_map = cv2.VideoWriter(str(map_avi), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w * 2, h))
            map_path_used = map_avi
        if not vw_frames.isOpened() or not vw_map.isOpened():
            continue

        clip_proj_extra_yaw = float(args.proj_extra_yaw_deg)
        clip_yaw_sign = float(args.proj_yaw_sign)
        clip_swap_xy = bool(args.proj_swap_xy)
        clip_mirror_east = bool(args.proj_mirror_east_cam)
        clip_mirror_north = bool(args.proj_mirror_north_cam)
        if args.auto_proj_yaw and worlds_dir.is_dir():
            ws = world_sdf_for_clip(worlds_dir, clip)
            gt_pts = parse_buoys_from_world_sdf(ws) if ws and ws.is_file() else []
            if gt_pts:
                yaw_cands = (-90.0, 0.0, 90.0, 180.0)
                cfg_errs: dict[tuple[float, float, bool, bool, bool], list[float]] = {}
                for yy in yaw_cands:
                    for yaw_sign in (1.0, -1.0):
                        for swap_xy in (False, True):
                            for mirror_east in (False, True):
                                cfg_errs[(yy, yaw_sign, swap_xy, mirror_east, False)] = []
                for fp in frames:
                    pose = pose_map.get(source_image_name(fp.name))
                    if pose is None:
                        continue
                    for row in rows_by_image[fp.name]:
                        gt_color = (row.get("matched_gt_color") or "unmatched").strip().lower()
                        if gt_color in ("", "unmatched"):
                            continue
                        try:
                            x1 = float(row["x1"])
                            y1 = float(row["y1"])
                            x2 = float(row["x2"])
                            y2 = float(row["y2"])
                        except (KeyError, ValueError):
                            continue
                        u = 0.5 * (x1 + x2)
                        v = 0.5 * (y1 + y2)
                        for yy in yaw_cands:
                            for yaw_sign in (1.0, -1.0):
                                for swap_xy in (False, True):
                                    for mirror_east in (False, True):
                                        n, e = project_pixel_to_ground_ned(
                                            u=u,
                                            v=v,
                                            altitude_m=max(0.01, pose["z"]),
                                            fx_px=fx,
                                            fy_px=fy,
                                            cx_px=cx,
                                            cy_px=cy,
                                            heading_deg=float(yaw_sign) * pose["yaw_deg"],
                                            extra_yaw_deg=yy,
                                            mirror_east_cam=mirror_east,
                                            mirror_north_cam=False,
                                        )
                                        cam_n, cam_e = world_xy_from_pose(pose, swap_xy=swap_xy)
                                        wn = cam_n + n
                                        we = cam_e + e
                                        dd = nearest_same_color_error_m(wn, we, gt_color, gt_pts)
                                        if dd is not None:
                                            cfg_errs[(yy, yaw_sign, swap_xy, mirror_east, False)].append(dd)
                scored: list[tuple[float, float, float, bool, bool, bool]] = []
                for (yy, yaw_sign, swap_xy, mirror_east, mirror_north), vals in cfg_errs.items():
                    if not vals:
                        continue
                    scored.append((float(np.median(vals)), yy, yaw_sign, swap_xy, mirror_east, mirror_north))
                if scored:
                    scored.sort()
                    _, clip_proj_extra_yaw, clip_yaw_sign, clip_swap_xy, clip_mirror_east, clip_mirror_north = scored[0]

        all_points: list[tuple[float, float, str]] = []
        per_frame_points: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        records: list[dict[str, object]] = []
        for fp in frames:
            pose = pose_map.get(source_image_name(fp.name))
            if pose is None:
                continue
            for row in rows_by_image[fp.name]:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                u = 0.5 * (x1 + x2)
                v = 0.5 * (y1 + y2)
                n, e = project_pixel_to_ground_ned(
                    u=u,
                    v=v,
                    altitude_m=max(0.01, pose["z"]),
                    fx_px=fx,
                    fy_px=fy,
                    cx_px=cx,
                    cy_px=cy,
                    heading_deg=float(clip_yaw_sign) * pose["yaw_deg"],
                    extra_yaw_deg=clip_proj_extra_yaw,
                    mirror_east_cam=clip_mirror_east,
                    mirror_north_cam=clip_mirror_north,
                )
                cam_n, cam_e = world_xy_from_pose(pose, swap_xy=clip_swap_xy)
                wn = cam_n + n
                we = cam_e + e
                c = (row.get("pred_color_hsv") or "unknown").strip().lower()
                box_conf = float(row.get("box_conf") or 0.0)
                color_conf = float(row.get("color_conf") or 0.0)
                gt_color = (row.get("matched_gt_color") or "unmatched").strip().lower()
                # Apply confidence and match filters for reprojection map
                passes_filter = (
                    box_conf >= args.min_box_conf
                    and color_conf >= args.min_color_conf
                    and (not args.matched_only or gt_color not in ("unmatched", ""))
                )
                if passes_filter:
                    all_points.append((wn, we, c))
                    per_frame_points[fp.name].append((wn, we, c))
                records.append(
                    {
                        "image": fp.name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "pred_color_hsv": c,
                        "box_conf": box_conf,
                        "color_conf": color_conf,
                        "matched_gt_color": gt_color,
                        "passes_map_filter": passes_filter,
                        "camera_x_m": pose["x"],
                        "camera_y_m": pose["y"],
                        "camera_z_m": pose["z"],
                        "camera_yaw_deg": pose["yaw_deg"],
                        "buoy_world_north_m": wn,
                        "buoy_world_east_m": we,
                    }
                )

        for fp in frames:
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            vw_frames.write(frame)
            panel = draw_map_panel(w, h, all_points, per_frame_points.get(fp.name, []))
            combo = np.hstack([frame, panel])
            vw_map.write(combo)

        vw_frames.release()
        vw_map.release()
        (clip_dir / "reconstructed_buoys.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        summary[clip] = {
            "frames": len(frames),
            "detections_reconstructed": len(records),
            "frames_video": str(frames_path_used),
            "map_video": str(map_path_used),
            "proj_extra_yaw_deg": float(clip_proj_extra_yaw),
            "proj_yaw_sign": float(clip_yaw_sign),
            "proj_swap_xy": bool(clip_swap_xy),
            "proj_mirror_east_cam": bool(clip_mirror_east),
            "proj_mirror_north_cam": bool(clip_mirror_north),
        }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote per-clip videos and reconstructions to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


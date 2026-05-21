#!/usr/bin/env python3
"""
Reproject detection centroids onto a local map and render overlays.

Input: ROI HSV test CSV (x1,y1,x2,y2 + predicted color) and annotated frame images.
Output:
  - side-by-side annotated video with top-down projected points
  - sample image snapshot
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GZ_SCRIPTS = _REPO_ROOT / "gazebo" / "scripts"
if str(_GZ_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_GZ_SCRIPTS))
from load_camera_intrinsics import default_calibration_json, resolve_pinhole_intrinsics  # noqa: E402
from nadir_projection import project_pixel_to_ground_ned  # noqa: E402


COLOR_BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 120, 0),
    "unknown": (220, 220, 220),
}


def meters_to_latlon(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    earth_r = 6378137.0
    d_lat = (north_m / earth_r) * (180.0 / np.pi)
    cos_lat = max(1e-6, abs(float(np.cos(np.deg2rad(lat_deg)))))
    d_lon = (east_m / (earth_r * cos_lat)) * (180.0 / np.pi)
    return lat_deg + d_lat, lon_deg + d_lon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--intrinsics",
        type=Path,
        default=default_calibration_json(_REPO_ROOT),
        help="camera_intrinsics_latest.json; used when --fx-px/--fy-px not set.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("roi_hsv_pipeline_clip_split_4ep/test_eval/roi_hsv_test_results.csv"),
    )
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("roi_hsv_pipeline_clip_split_4ep/test_eval/annotated_frames"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("roi_hsv_pipeline_clip_split_4ep/test_eval/reprojection"),
    )
    p.add_argument("--altitude-m", type=float, default=10.0)
    p.add_argument("--fx-px", type=float, default=None, help="Override horizontal focal length (pixels).")
    p.add_argument("--fy-px", type=float, default=None)
    p.add_argument("--cx-px", type=float, default=None, help="Override principal point; <=0 uses intrinsics or image center.")
    p.add_argument("--cy-px", type=float, default=None)
    p.add_argument("--heading-deg", type=float, default=0.0)
    p.add_argument("--drone-lat", type=float, default=32.88010)
    p.add_argument("--drone-lon", type=float, default=-117.23420)
    p.add_argument(
        "--demo-label",
        type=str,
        default="",
        help="Optional caption drawn on both panels (presentation).",
    )
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument(
        "--world-sdf",
        type=Path,
        default=None,
        help="Optional Gazebo world SDF to overlay ground-truth buoy positions.",
    )
    p.add_argument("--drone-x-m", type=float, default=0.0, help="Drone world X (east) in the SDF frame.")
    p.add_argument("--drone-y-m", type=float, default=0.0, help="Drone world Y (north) in the SDF frame.")
    return p.parse_args()


def parse_buoys_from_world_sdf(world_sdf: Path) -> list[tuple[float, float, str]]:
    """
    Return [(north_m, east_m, color)] from world SDF.
    Uses model name hints (red/green/blue) and model pose x/y.
    """
    if world_sdf is None or not world_sdf.is_file():
        return []
    tree = ET.parse(world_sdf)
    root = tree.getroot()
    out: list[tuple[float, float, str]] = []
    # Pattern A: explicit <model ...> elements.
    for m in root.findall(".//model"):
        name = (m.get("name") or "").lower()
        if "buoy" not in name:
            continue
        color = "unknown"
        if "red" in name:
            color = "red"
        elif "green" in name:
            color = "green"
        elif "blue" in name:
            color = "blue"
        pose = m.find("pose")
        if pose is None or not (pose.text or "").strip():
            continue
        vals = (pose.text or "").strip().split()
        if len(vals) < 2:
            continue
        try:
            x_east = float(vals[0])
            y_north = float(vals[1])
        except ValueError:
            continue
        out.append((y_north, x_east, color))

    # Pattern B: <include><uri>model://robotx_buoy_* ... + <pose>
    for inc in root.findall(".//include"):
        uri = (inc.findtext("uri") or "").lower()
        if "buoy" not in uri:
            continue
        color = "unknown"
        if "red" in uri:
            color = "red"
        elif "green" in uri:
            color = "green"
        elif "blue" in uri:
            color = "blue"
        pose_txt = (inc.findtext("pose") or "").strip()
        if not pose_txt:
            continue
        vals = pose_txt.split()
        if len(vals) < 2:
            continue
        try:
            x_east = float(vals[0])
            y_north = float(vals[1])
        except ValueError:
            continue
        out.append((y_north, x_east, color))
    return out


def draw_map_panel(
    w: int,
    h: int,
    points_all: list[tuple[float, float, str]],
    points_cur: list[tuple[float, float, str]],
    points_gt: list[tuple[float, float, str]] | None = None,
) -> np.ndarray:
    panel = np.full((h, w, 3), 28, dtype=np.uint8)
    points_gt = points_gt or []
    bounds_pts = list(points_all) + list(points_gt)
    if not bounds_pts:
        return panel

    ns = np.array([p[0] for p in bounds_pts], dtype=np.float32)
    es = np.array([p[1] for p in bounds_pts], dtype=np.float32)
    nmin, nmax = float(ns.min()), float(ns.max())
    emin, emax = float(es.min()), float(es.max())
    pad_n = max(1.0, 0.1 * (nmax - nmin + 1e-6))
    pad_e = max(1.0, 0.1 * (emax - emin + 1e-6))
    nmin -= pad_n
    nmax += pad_n
    emin -= pad_e
    emax += pad_e

    def to_px(north_m: float, east_m: float) -> tuple[int, int]:
        x = int((east_m - emin) / max(1e-6, (emax - emin)) * (w - 1))
        y = int((1.0 - (north_m - nmin) / max(1e-6, (nmax - nmin))) * (h - 1))
        return x, y

    cv2.putText(panel, "Top-down reprojection (N/E meters)", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    cv2.putText(panel, f"E range: {emin:.1f} .. {emax:.1f}", (18, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(panel, f"N range: {nmin:.1f} .. {nmax:.1f}", (18, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Ground-truth buoy positions (if provided): hollow circles.
    for n, e, color in points_gt:
        x, y = to_px(n, e)
        c = COLOR_BGR.get(color, COLOR_BGR["unknown"])
        cv2.circle(panel, (x, y), 12, c, 2, cv2.LINE_AA)
        cv2.circle(panel, (x, y), 3, c, -1, cv2.LINE_AA)

    # Current-frame detections: filled circles.
    for n, e, color in points_cur:
        x, y = to_px(n, e)
        cv2.circle(panel, (x, y), 7, COLOR_BGR.get(color, COLOR_BGR["unknown"]), -1, cv2.LINE_AA)
        cv2.circle(panel, (x, y), 10, (255, 255, 255), 1, cv2.LINE_AA)

    if points_gt:
        cv2.putText(
            panel,
            "GT buoys: hollow circles  |  Reprojected detections: filled",
            (18, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return panel


def main() -> int:
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV missing: {args.csv}")
    if not args.frames_dir.is_dir():
        raise NotADirectoryError(f"Frames dir missing: {args.frames_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_image[row["image"]].append(row)

    frame_paths = sorted([p for p in args.frames_dir.glob("*.jpg") if p.name in rows_by_image])
    if args.max_frames > 0:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError("No frame images match CSV rows.")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    intr_src = args.intrinsics if args.intrinsics.is_file() else None
    fx_px, fy_px, cx_px, cy_px = resolve_pinhole_intrinsics(
        intrinsics_path=intr_src,
        fx_override=args.fx_px,
        fy_override=args.fy_px,
        cx_override=args.cx_px,
        cy_override=args.cy_px,
        image_w=w,
        image_h=h,
    )

    # Precompute projected points for map bounds.
    points_all: list[tuple[float, float, str]] = []
    proj_by_image: dict[str, list[tuple[float, float, str, tuple[float, float]]]] = defaultdict(list)
    for p in frame_paths:
        for row in rows_by_image[p.name]:
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            color = row.get("pred_color_hsv", "unknown")
            north_m, east_m = project_pixel_to_ground_ned(
                u=cx,
                v=cy,
                altitude_m=args.altitude_m,
                fx_px=fx_px,
                fy_px=fy_px,
                cx_px=cx_px,
                cy_px=cy_px,
                heading_deg=args.heading_deg,
            )
            est_lat, est_lon = meters_to_latlon(args.drone_lat, args.drone_lon, north_m, east_m)
            world_n = args.drone_y_m + north_m
            world_e = args.drone_x_m + east_m
            points_all.append((world_n, world_e, color))
            proj_by_image[p.name].append((north_m, east_m, color, (est_lat, est_lon)))

    points_gt_world = parse_buoys_from_world_sdf(args.world_sdf) if args.world_sdf else []

    out_video = args.out_dir / "reprojection_overlay.mp4"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w * 2, h),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to create output video writer.")

    sample_img_written = False
    for fp in frame_paths:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        points_cur = [(args.drone_y_m + n, args.drone_x_m + e, c) for (n, e, c, _) in proj_by_image[fp.name]]
        panel = draw_map_panel(w, h, points_all, points_cur, points_gt_world)

        if args.demo_label:
            cap = args.demo_label
            cv2.putText(frame, cap, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(frame, cap, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, cap, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2, cv2.LINE_AA)

        # Add per-detection text to left frame.
        ytxt = 26
        for i, (n, e, c, (lat, lon)) in enumerate(proj_by_image[fp.name][:8]):
            txt = f"{c:<5} N={n:+6.2f}m E={e:+6.2f}m  lat={lat:.7f} lon={lon:.7f}"
            cv2.putText(frame, txt, (16, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BGR.get(c, (200, 200, 200)), 1, cv2.LINE_AA)
            ytxt += 20
        combo = np.hstack([frame, panel])
        writer.write(combo)

        if not sample_img_written:
            sample_path = args.out_dir / "reprojection_sample.jpg"
            cv2.imwrite(str(sample_path), combo)
            sample_img_written = True

    writer.release()
    print(f"Wrote video: {out_video}")
    print(f"Wrote sample image: {args.out_dir / 'reprojection_sample.jpg'}")
    print(f"Frames used: {len(frame_paths)}")
    print(f"Drone ref GPS: lat={args.drone_lat:.7f}, lon={args.drone_lon:.7f}")
    if args.world_sdf:
        print(f"World SDF: {args.world_sdf}")
        print(f"GT buoy points parsed: {len(points_gt_world)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


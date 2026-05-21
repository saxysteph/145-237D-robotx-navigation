#!/usr/bin/env python3
"""
Project simulator buoy positions into normalized YOLO boxes (single class 0 = buoy ROI).

Uses ``nadir_projection.ground_delta_en_to_pixel`` (inverse of path2 ``project_pixel_to_ground_ned``).

For Gazebo captures, use --projection sim_sdf (default): intrinsics come from the world's
horizontal_fov + image size, and camera pose from ``uav_pose_log.csv`` when ``drone_*`` is
filled from gz-transport Pose_V. Per-frame buoy XYZ comes from ``buoy_poses_json`` (same
recording path) when present; otherwise buoy centers use static ``<include><pose>`` in
``world.sdf``. Altitude for weak-perspective scale uses ``drone_z_m - buoy_z`` per buoy.
Real checkerboard calibration JSON does **not** match synthetic renders and will misplace boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from load_camera_intrinsics import (  # noqa: E402
    default_calibration_json,
    load_intrinsics_matrix,
    sim_nadir_camera_from_world_sdf,
)
from nadir_projection import ground_delta_en_to_pixel  # noqa: E402
from render_gt_in_image_from_pose import parse_buoy_instances_world_sdf  # noqa: E402


def parse_uav_pose_log_enriched(
    pose_csv: Path,
    default_xyz_yaw: tuple[float, float, float, float],
) -> tuple[
    dict[str, tuple[float, float, float, float]],
    dict[str, dict[str, list[float]] | None],
]:
    poses: dict[str, tuple[float, float, float, float]] = {}
    buoy_snapshots: dict[str, dict[str, list[float]] | None] = {}
    dx, dy, dz, dyaw = default_xyz_yaw

    with pose_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        has_buoy_json = reader.fieldnames and ("buoy_poses_json" in reader.fieldnames)
        for row in reader:
            name = (row.get("image_name") or "").strip()
            if not name:
                continue

            def _num(key: str, fallback: float) -> float:
                s = (row.get(key) or "").strip()
                if s == "":
                    return fallback
                try:
                    return float(s)
                except ValueError:
                    return fallback

            x = _num("drone_x_m", dx)
            y = _num("drone_y_m", dy)
            z = _num("drone_z_m", dz)
            yaw = _num("drone_yaw_deg", dyaw)
            poses[name] = (x, y, z, yaw)

            buoy_snapshots[name] = None
            if has_buoy_json:
                rawj = (row.get("buoy_poses_json") or "").strip()
                if rawj:
                    try:
                        parsed = json.loads(rawj)
                        if isinstance(parsed, dict):
                            buoy_snapshots[name] = {str(k): v for k, v in parsed.items()}  # type: ignore[assignment]
                        else:
                            buoy_snapshots[name] = None
                    except json.JSONDecodeError:
                        buoy_snapshots[name] = None

    return poses, buoy_snapshots


def corners_square_enu(bx: float, by: float, half_m: float) -> list[tuple[float, float]]:
    r = half_m
    return [(bx - r, by - r), (bx + r, by - r), (bx - r, by + r), (bx + r, by + r)]


def box_from_projections(
    uv_list: list[tuple[float, float]],
    w: int,
    h: int,
    min_side_px: float,
) -> tuple[float, float, float, float] | None:
    xs = [u for u, _ in uv_list]
    ys = [v for _, v in uv_list]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    x1 = max(0.0, min(float(w - 1), x1))
    x2 = max(0.0, min(float(w - 1), x2))
    y1 = max(0.0, min(float(h - 1), y1))
    y2 = max(0.0, min(float(h - 1), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw < min_side_px or bh < min_side_px:
        return None
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return (cx / float(w), cy / float(h), bw / float(w), bh / float(h))


def class_id_from_color(color: str) -> int:
    """
    Canonical class mapping for projected labels.
    0=red, 1=green, 2=blue, 3=unknown
    """
    c = (color or "").strip().lower()
    if c == "red":
        return 0
    if c == "green":
        return 1
    if c == "blue":
        return 2
    return 3


def process_clip(
    clip_dir: Path,
    world_sdf: Path,
    intrinsics: Path,
    labels_subdir: str,
    ground_half_m: float,
    default_pose: tuple[float, float, float, float],
    min_side_px: float,
    pose_csv_name: str,
    projection: str = "sim_sdf",
    *,
    extra_yaw_deg: float = 0.0,
    mirror_east_camera: bool = False,
    mirror_north_camera: bool = False,
    transpose_en_pixel: bool = False,
) -> int:
    pose_csv = clip_dir / pose_csv_name
    if not pose_csv.is_file():
        print(f"Missing pose CSV: {pose_csv}", file=sys.stderr)
        return 1

    buoy_specs = parse_buoy_instances_world_sdf(world_sdf)
    if not buoy_specs:
        print(f"No buoys parsed from {world_sdf}", file=sys.stderr)
        return 1

    frames = sorted(clip_dir.glob("uav_*.jpg"))
    if not frames:
        print(f"No uav_*.jpg under {clip_dir}", file=sys.stderr)
        return 1

    import cv2

    first = cv2.imread(str(frames[0]))
    if first is None:
        print(f"Cannot read {frames[0]}", file=sys.stderr)
        return 1
    h, w = first.shape[:2]

    if projection == "sim_sdf":
        fx, fy, cx, cy, sdx, sdy, sdz, syaw = sim_nadir_camera_from_world_sdf(world_sdf, w, h)
        pose_fallback: tuple[float, float, float, float] = (sdx, sdy, sdz, syaw)
    else:
        fx, fy, tcx, tcy, jw, jh = load_intrinsics_matrix(intrinsics)
        cx, cy = tcx, tcy
        if jw > 0 and jh > 0 and (jw != w or jh != h):
            sx = w / float(jw)
            sy = h / float(jh)
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy
        pose_fallback = default_pose

    poses, buoy_by_image = parse_uav_pose_log_enriched(pose_csv, pose_fallback)

    any_csv_pose = False
    has_buoy_col = False
    any_live_buoy_xyz = False
    try:
        with pose_csv.open("r", encoding="utf-8", newline="") as pcf:
            rdr = csv.DictReader(pcf)
            has_buoy_col = bool(rdr.fieldnames and ("buoy_poses_json" in rdr.fieldnames))
            for row in rdr:
                if (row.get("drone_x_m") or "").strip():
                    any_csv_pose = True
                if has_buoy_col:
                    raw = (row.get("buoy_poses_json") or "").strip()
                    if raw not in ("", "{}", "null"):
                        any_live_buoy_xyz = True
    except OSError:
        pass
    if not any_csv_pose:
        print(
            f"{clip_dir.name}: uav_pose_log.csv has empty drone_x/y/z/yaw — using fixed fallback pose "
            f"({pose_fallback[0]:g}, {pose_fallback[1]:g}, {pose_fallback[2]:g}) m / {pose_fallback[3]:g}°. "
            "Re-record with gz-transport Pose_V for camera-relative labeling when the UAV moves.",
            file=sys.stderr,
        )

    if not has_buoy_col:
        print(
            f"{clip_dir.name}: CSV has no buoy_poses_json column — buoy XY/Z use static world.sdf only; "
            "re-record with an updated record_uav_dataset_gz_transport.py for per-frame buoy link poses.",
            file=sys.stderr,
        )
    elif not any_live_buoy_xyz and pose_fallback:
        print(
            f"{clip_dir.name}: buoy_poses_json empty on all rows — buoy centers fall back to SDF placers "
            "(check Pose_V + world.sdf instance names matching SceneBroadcaster).",
            file=sys.stderr,
        )

    uniq_pose = len({poses.get(fp.name, pose_fallback) for fp in frames})
    if any_csv_pose and uniq_pose <= 1 and len(frames) > 1:
        print(
            f"{clip_dir.name}: per-frame drone pose is constant across {len(frames)} frames "
            "(typical static `uav_dataset_camera`).",
            file=sys.stderr,
        )

    out_dir = clip_dir / labels_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for fp in frames:
        pose = poses.get(fp.name, pose_fallback)
        dx, dy, dz, yaw = pose
        live_buoys = buoy_by_image.get(fp.name)
        lines: list[str] = []
        for inst_name, bx0, by0, bz0, buoy_color in buoy_specs:
            bx, by, bz = bx0, by0, bz0
            src = live_buoys
            if src and inst_name in src:
                arr = src.get(inst_name)
                if isinstance(arr, (list, tuple)) and len(arr) >= 3:
                    try:
                        bx, by, bz = float(arr[0]), float(arr[1]), float(arr[2])
                    except (TypeError, ValueError):
                        pass
            alt_m = max(1e-3, dz - bz)
            uv_square = []
            for ex, ny in corners_square_enu(bx, by, ground_half_m):
                uv = ground_delta_en_to_pixel(
                    ex - dx,
                    ny - dy,
                    alt_m,
                    fx,
                    fy,
                    cx,
                    cy,
                    yaw,
                    extra_yaw_deg=extra_yaw_deg,
                    mirror_east_cam=mirror_east_camera,
                    mirror_north_cam=mirror_north_camera,
                    transpose_en_pixel=transpose_en_pixel,
                )
                if uv is None:
                    continue
                uv_square.append(uv)
            if len(uv_square) < 4:
                continue
            yolo = box_from_projections(uv_square, w, h, min_side_px)
            if yolo is None:
                continue
            tcx, tcy, tw, th = yolo
            cls_id = class_id_from_color(buoy_color)
            lines.append(f"{cls_id} {tcx:.6f} {tcy:.6f} {tw:.6f} {th:.6f}")

        label_path = out_dir / f"{fp.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    print(f"{clip_dir.name}: wrote {written} label files → {out_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip-dir", type=Path, help="Single clip folder with uav_*.jpg and pose CSV.")
    p.add_argument(
        "--captures-root",
        type=Path,
        help="Root containing clip subfolders (each with uav_*.jpg); needs --world-sdf per clip or world.sdf in clip.",
    )
    p.add_argument("--world-sdf", type=Path, help="World file for a single --clip-dir run.")
    p.add_argument(
        "--intrinsics",
        type=Path,
        default=default_calibration_json(Path(__file__).resolve().parents[2]),
        help="With --projection calibration: camera_intrinsics_latest.json. Ignored for sim_sdf.",
    )
    p.add_argument(
        "--projection",
        choices=("sim_sdf", "calibration"),
        default="sim_sdf",
        help="sim_sdf: K + camera pose from Gazebo world.sdf (recommended for sim). "
        "calibration: physical JSON (for real camera / non-Gazebo).",
    )
    p.add_argument("--labels-subdir", type=str, default="labels_proj")
    p.add_argument("--pose-csv-name", type=str, default="uav_pose_log.csv")
    p.add_argument("--ground-half-m", type=float, default=0.35, help="Half-edge of square footprint on water plane.")
    p.add_argument("--min-side-px", type=float, default=4.0)
    p.add_argument(
        "--default-drone-x-m",
        type=float,
        default=0.0,
        help="Used when pose CSV has empty pose columns (static sim camera).",
    )
    p.add_argument("--default-drone-y-m", type=float, default=0.0)
    p.add_argument("--default-drone-z-m", type=float, default=26.0)
    p.add_argument("--default-drone-yaw-deg", type=float, default=0.0)
    p.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="With --captures-root, skip clips missing world.sdf, pose CSV, or uav_*.jpg instead of failing.",
    )
    p.add_argument(
        "--extra-yaw-deg",
        type=float,
        default=0.0,
        help="Added to drone yaw before EN→camera-plane rotation (debug sim/image azimuth mismatch).",
    )
    p.add_argument(
        "--mirror-east-camera",
        action="store_true",
        help="Flip east offset in the camera plane (mirrors boxes left↔right if flat model sign is wrong).",
    )
    p.add_argument(
        "--mirror-north-camera",
        action="store_true",
        help="Flip north offset in the camera plane (mirrors vertical placement vs flat model).",
    )
    p.add_argument(
        "--transpose-en-pixel",
        action="store_true",
        help="Map north→horizontal pixel offset and east→vertical (try if boxes look rotated 90° vs buoys).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    default_pose = (
        args.default_drone_x_m,
        args.default_drone_y_m,
        args.default_drone_z_m,
        args.default_drone_yaw_deg,
    )

    if args.clip_dir:
        world = args.world_sdf
        if world is None:
            cand = args.clip_dir / "world.sdf"
            if not cand.is_file():
                print("Provide --world-sdf or place world.sdf in the clip directory.", file=sys.stderr)
                return 1
            world = cand
        return process_clip(
            args.clip_dir,
            world,
            args.intrinsics,
            args.labels_subdir,
            args.ground_half_m,
            default_pose,
            args.min_side_px,
            args.pose_csv_name,
            projection=args.projection,
            extra_yaw_deg=args.extra_yaw_deg,
            mirror_east_camera=args.mirror_east_camera,
            mirror_north_camera=args.mirror_north_camera,
            transpose_en_pixel=args.transpose_en_pixel,
        )

    if args.captures_root:
        if not args.captures_root.is_dir():
            print(f"Not a directory: {args.captures_root}", file=sys.stderr)
            return 1
        rc = 0
        for clip_dir in sorted(p for p in args.captures_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            world = clip_dir / "world.sdf"
            if not world.is_file():
                print(f"Skip {clip_dir.name}: no world.sdf", file=sys.stderr)
                if not args.skip_incomplete:
                    rc = 1
                continue
            pose_csv = clip_dir / args.pose_csv_name
            if args.skip_incomplete:
                if not pose_csv.is_file():
                    print(f"Skip {clip_dir.name}: missing {args.pose_csv_name}", file=sys.stderr)
                    continue
                if not list(clip_dir.glob("uav_*.jpg")):
                    print(f"Skip {clip_dir.name}: no uav_*.jpg", file=sys.stderr)
                    continue
            r = process_clip(
                clip_dir,
                world,
                args.intrinsics,
                args.labels_subdir,
                args.ground_half_m,
                default_pose,
                args.min_side_px,
                args.pose_csv_name,
                projection=args.projection,
                extra_yaw_deg=args.extra_yaw_deg,
                mirror_east_camera=args.mirror_east_camera,
                mirror_north_camera=args.mirror_north_camera,
                transpose_en_pixel=args.transpose_en_pixel,
            )
            rc = rc or r
        return rc

    print("Specify --clip-dir or --captures-root.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

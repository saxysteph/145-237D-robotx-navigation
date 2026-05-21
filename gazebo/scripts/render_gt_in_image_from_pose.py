#!/usr/bin/env python3
"""
Render frame-by-frame moving GT buoy projections into camera images.

Inputs:
  - world SDF (buoy XYZ)
  - frame directory (uav_<stamp>.jpg)
  - pose CSV from record_uav_dataset.py (uav_pose_log.csv)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from load_camera_intrinsics import default_calibration_json, resolve_pinhole_intrinsics
from nadir_projection import ground_delta_en_to_pixel


COLOR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 120, 0), "unknown": (220, 220, 220)}


def parse_buoys(world_sdf: Path) -> list[tuple[float, float, float, str]]:
    """Nominal buoy centroids (+ color hint) from static ``<include><pose>`` in world SDF."""
    return [(bx, by, bz, c) for (_nm, bx, by, bz, c) in parse_buoy_instances_world_sdf(world_sdf)]


def parse_buoy_instances_world_sdf(world_sdf: Path) -> list[tuple[str, float, float, float, str]]:
    """
    Per-buoy world instance ``<include><name>`` plus nominal pose (+ color hint from URI).

    Used to match SceneBroadcaster ``Pose_V`` names (scoped as ``instance::body_link``, etc.).
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(world_sdf)
    root = tree.getroot()
    out: list[tuple[str, float, float, float, str]] = []
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
        nm = (inc.findtext("name") or "").strip()
        pose = (inc.findtext("pose") or "").strip().split()
        if len(pose) < 3:
            continue
        try:
            x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
        except ValueError:
            continue
        if nm:
            out.append((nm, x, y, z, color))
        else:
            out.append((f"buoy_{len(out)}", x, y, z, color))
    return out


def parse_pose_csv(pose_csv: Path) -> dict[str, tuple[float, float, float, float]]:
    by_image: dict[str, tuple[float, float, float, float]] = {}
    with pose_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("image_name", "")
            if not name:
                continue
            try:
                x = float(row["drone_x_m"])
                y = float(row["drone_y_m"])
                z = float(row["drone_z_m"])
                yaw_deg = float(row["drone_yaw_deg"])
            except (KeyError, ValueError):
                continue
            by_image[name] = (x, y, z, yaw_deg)
    return by_image


def world_to_pixel(
    buoy_x: float,
    buoy_y: float,
    drone_x: float,
    drone_y: float,
    drone_z: float,
    yaw_deg: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float] | None:
    """Delegates to :func:`ground_delta_en_to_pixel` (inverse of path2 ``project_pixel_to_ground_ned``)."""
    u, v = ground_delta_en_to_pixel(
        buoy_x - drone_x,
        buoy_y - drone_y,
        drone_z,
        fx,
        fy,
        cx,
        cy,
        yaw_deg,
    )
    return u, v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--pose-csv", type=Path, required=True)
    p.add_argument("--world-sdf", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--intrinsics",
        type=Path,
        default=default_calibration_json(repo),
        help="camera_intrinsics_latest.json (default: repo calibration/). Used unless overridden below.",
    )
    p.add_argument("--fx-px", type=float, default=None, help="Override intrinsics fx (pixels).")
    p.add_argument("--fy-px", type=float, default=None, help="Override intrinsics fy (pixels).")
    p.add_argument("--cx-px", type=float, default=None, help="Override principal point; omit to use intrinsics or image center.")
    p.add_argument("--cy-px", type=float, default=None)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    buoys = parse_buoys(args.world_sdf)
    poses = parse_pose_csv(args.pose_csv)
    frames = sorted(args.frames_dir.glob("uav_*.jpg"))
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError("No frames found.")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot read {frames[0]}")
    h, w = first.shape[:2]
    intr_src = args.intrinsics if args.intrinsics.is_file() else None
    fx, fy, cx, cy = resolve_pinhole_intrinsics(
        intrinsics_path=intr_src,
        fx_override=args.fx_px,
        fy_override=args.fy_px,
        cx_override=args.cx_px,
        cy_override=args.cy_px,
        image_w=w,
        image_h=h,
    )

    out_video = args.out_dir / "gt_projected_in_image.mp4"
    vw = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter open failed.")

    sample_written = False
    drawn = 0
    for fp in frames:
        fr = cv2.imread(str(fp))
        if fr is None:
            continue
        pose = poses.get(fp.name)
        if pose is None:
            vw.write(fr)
            continue
        dx, dy, dz, yaw = pose
        for bx, by, _bz, color in buoys:
            uv = world_to_pixel(
                buoy_x=bx,
                buoy_y=by,
                drone_x=dx,
                drone_y=dy,
                drone_z=dz,
                yaw_deg=yaw,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
            if uv is None:
                continue
            u, v = uv
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                c = COLOR.get(color, COLOR["unknown"])
                cv2.circle(fr, (ui, vi), 10, c, 2, cv2.LINE_AA)
                cv2.circle(fr, (ui, vi), 3, c, -1, cv2.LINE_AA)
                cv2.putText(fr, f"GT-{color}", (ui + 8, max(18, vi - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
                drawn += 1
        cv2.putText(fr, f"pose x={dx:.2f} y={dy:.2f} z={dz:.2f} yaw={yaw:.1f}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
        vw.write(fr)
        if not sample_written:
            cv2.imwrite(str(args.out_dir / "gt_projected_sample.jpg"), fr)
            sample_written = True
    vw.release()
    print(f"Wrote video: {out_video}")
    print(f"Wrote sample: {args.out_dir / 'gt_projected_sample.jpg'}")
    print(f"Frames: {len(frames)}  GT buoys: {len(buoys)}  projected points drawn: {drawn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


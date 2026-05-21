#!/usr/bin/env python3
"""
Per-clip world-matched reprojection renderer.

For each clip in a test split:
  - find matching world SDF (robotx_dr_XXX_*.sdf),
  - project detections to local N/E,
  - overlay simulator GT buoy positions from that world,
  - write one video + sample image per clip.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
import json
import math

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GZ_SCRIPTS = _REPO_ROOT / "gazebo" / "scripts"
if str(_GZ_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_GZ_SCRIPTS))
from load_camera_intrinsics import default_calibration_json, resolve_pinhole_intrinsics  # noqa: E402

from reproject_detection_map import (
    COLOR_BGR,
    draw_map_panel,
    meters_to_latlon,
    parse_buoys_from_world_sdf,
    project_pixel_to_ground_ned,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
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
        "--dataset-test-dir",
        type=Path,
        default=Path("roi_hsv_pipeline_clip_split_4ep/dataset_roi/images/test"),
        help="Used to recover original clip from symlink targets.",
    )
    p.add_argument(
        "--worlds-dir",
        type=Path,
        default=Path("../../../gazebo/worlds/generated"),
        help="Directory containing robotx_dr_XXX_*.sdf files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("roi_hsv_pipeline_clip_split_4ep/test_eval/reprojection_per_clip"),
    )
    p.add_argument(
        "--intrinsics",
        type=Path,
        default=default_calibration_json(_REPO_ROOT),
        help="camera_intrinsics_latest.json; used when focal lengths not overridden.",
    )
    p.add_argument("--altitude-m", type=float, default=10.0)
    p.add_argument("--fx-px", type=float, default=None)
    p.add_argument("--fy-px", type=float, default=None)
    p.add_argument("--cx-px", type=float, default=None)
    p.add_argument("--cy-px", type=float, default=None)
    p.add_argument("--heading-deg", type=float, default=0.0)
    p.add_argument("--drone-lat", type=float, default=32.88010)
    p.add_argument("--drone-lon", type=float, default=-117.23420)
    p.add_argument("--drone-x-m", type=float, default=0.0)
    p.add_argument("--drone-y-m", type=float, default=0.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max-clips", type=int, default=0, help="0=all clips")
    p.add_argument("--hide-gt", action="store_true", help="Do not draw GT buoy markers in output videos.")
    return p.parse_args()


def _clip_from_symlink(dataset_test_dir: Path, image_name: str) -> str | None:
    p = dataset_test_dir / image_name
    if not p.is_symlink():
        return None
    try:
        raw = os.readlink(str(p))
    except OSError:
        return None
    # target expected like .../captures/gazebo_uav_batch/<clip>/uav_....jpg
    parts = Path(raw).parts
    if "gazebo_uav_batch" in parts:
        i = parts.index("gazebo_uav_batch")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _world_for_clip(worlds_dir: Path, clip_name: str) -> Path | None:
    # Prefer exact scenario token match:
    # clip: 008_robotx_dr_008_clear_blue_mild_regular_wide_lane_s01
    # world: robotx_dr_008_clear_blue_mild_regular_wide_lane.sdf
    m_scn = re.search(r"robotx_dr_\d{3}_(.+)_s\d+$", clip_name)
    if m_scn:
        scenario = m_scn.group(1)
        exact = sorted(worlds_dir.glob(f"robotx_dr_*_{scenario}.sdf"))
        if exact:
            # If multiple, prefer same dr id if present in clip.
            m_id = re.search(r"robotx_dr_(\d{3})", clip_name)
            if m_id:
                same_id = [p for p in exact if f"robotx_dr_{m_id.group(1)}_" in p.name]
                if same_id:
                    return same_id[0]
            return exact[0]

    # Fallback to ID-only match.
    m = re.search(r"robotx_dr_(\d{3})", clip_name)
    if not m:
        return None
    idx = m.group(1)
    cands = sorted(worlds_dir.glob(f"robotx_dr_{idx}_*.sdf"))
    return cands[0] if cands else None


def _nearest_same_color_gt(
    north_m: float,
    east_m: float,
    color: str,
    gt_pts: list[tuple[float, float, str]],
) -> tuple[float, float, str, float] | None:
    best = None
    best_d = float("inf")
    for gn, ge, gc in gt_pts:
        if gc != color:
            continue
        d = math.hypot(north_m - gn, east_m - ge)
        if d < best_d:
            best_d = d
            best = (gn, ge, gc, d)
    if best is None:
        return None
    return best


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir = args.worlds_dir.resolve()

    rows_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_image[row["image"]].append(row)

    frame_paths = sorted([p for p in args.frames_dir.glob("*.jpg") if p.name in rows_by_image])
    if not frame_paths:
        raise RuntimeError("No matching frames for CSV.")

    clip_to_frames: dict[str, list[Path]] = defaultdict(list)
    for fp in frame_paths:
        clip = _clip_from_symlink(args.dataset_test_dir, fp.name)
        if clip is None:
            continue
        clip_to_frames[clip].append(fp)

    clip_names = sorted(clip_to_frames.keys())
    if args.max_clips > 0:
        clip_names = clip_names[: args.max_clips]
    if not clip_names:
        raise RuntimeError("No clip mappings found from dataset symlinks.")

    made = 0
    skipped = 0
    summary_rows: list[list[object]] = []
    for clip in clip_names:
        world_sdf = _world_for_clip(worlds_dir, clip)
        if world_sdf is None:
            skipped += 1
            continue

        gt_pts = parse_buoys_from_world_sdf(world_sdf)
        frames = clip_to_frames[clip]
        first = cv2.imread(str(frames[0]))
        if first is None:
            skipped += 1
            continue
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

        # Precompute projected points for this clip.
        points_all: list[tuple[float, float, str]] = []
        proj_by_image: dict[str, list[tuple[float, float, str, tuple[float, float]]]] = defaultdict(list)
        for fp in frames:
            for row in rows_by_image[fp.name]:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                u = 0.5 * (x1 + x2)
                v = 0.5 * (y1 + y2)
                color = row.get("pred_color_hsv", "unknown")
                north_m, east_m = project_pixel_to_ground_ned(
                    u=u,
                    v=v,
                    altitude_m=args.altitude_m,
                    fx_px=args.fx_px,
                    fy_px=args.fy_px,
                    cx_px=cx_px,
                    cy_px=cy_px,
                    heading_deg=args.heading_deg,
                )
                world_n = args.drone_y_m + north_m
                world_e = args.drone_x_m + east_m
                points_all.append((world_n, world_e, color))
                est_lat, est_lon = meters_to_latlon(args.drone_lat, args.drone_lon, north_m, east_m)
                proj_by_image[fp.name].append((north_m, east_m, color, (est_lat, est_lon)))

        out_sub = args.out_dir / clip
        out_sub.mkdir(parents=True, exist_ok=True)
        out_video = out_sub / "reprojection_overlay.mp4"
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w * 2, h))
        if not writer.isOpened():
            skipped += 1
            continue

        sample_written = False
        eval_rows: list[list[object]] = []
        per_color_err: dict[str, list[float]] = {"red": [], "green": [], "blue": [], "unknown": []}
        for fp in frames:
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            cur = [
                (args.drone_y_m + n, args.drone_x_m + e, c)
                for (n, e, c, _) in proj_by_image[fp.name]
            ]
            panel = draw_map_panel(w, h, points_all, cur, [] if args.hide_gt else gt_pts)
            ytxt = 26
            for (n, e, c, (lat, lon)) in proj_by_image[fp.name]:
                wn = args.drone_y_m + n
                we = args.drone_x_m + e
                nearest = _nearest_same_color_gt(wn, we, c, gt_pts)
                if nearest is not None:
                    gn, ge, gc, err_m = nearest
                    per_color_err.setdefault(c, []).append(err_m)
                    eval_rows.append([fp.name, c, wn, we, gn, ge, err_m, lat, lon])
                    # Draw match line in map panel.
                    # Reuse same projection transform by plotting in a temporary panel pass is expensive;
                    # instead draw summary text on image and let map markers speak visually.
                    txt = f"{c:<5} err={err_m:5.2f}m N={n:+6.2f} E={e:+6.2f}"
                else:
                    txt = f"{c:<5} err= n/a  N={n:+6.2f} E={e:+6.2f}"
                    eval_rows.append([fp.name, c, wn, we, "", "", "", lat, lon])
                cv2.putText(
                    frame,
                    txt,
                    (16, ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLOR_BGR.get(c, COLOR_BGR["unknown"]),
                    1,
                    cv2.LINE_AA,
                )
                ytxt += 20
                if ytxt > h - 20:
                    break
            combo = np.hstack([frame, panel])
            writer.write(combo)
            if not sample_written:
                cv2.imwrite(str(out_sub / "reprojection_sample.jpg"), combo)
                sample_written = True
        writer.release()

        # Per-clip metric outputs.
        eval_csv = out_sub / "reprojection_eval.csv"
        with eval_csv.open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(
                [
                    "image",
                    "color",
                    "est_world_north_m",
                    "est_world_east_m",
                    "gt_world_north_m",
                    "gt_world_east_m",
                    "error_m",
                    "est_lat",
                    "est_lon",
                ]
            )
            wcsv.writerows(eval_rows)

        clip_summary = {
            "clip": clip,
            "world_sdf": str(world_sdf),
            "frames": len(frames),
            "gt_buoys": len(gt_pts),
            "detections_total": len(eval_rows),
            "matched_total": int(sum(1 for r in eval_rows if r[6] != "")),
            "error_mean_m": None,
            "error_median_m": None,
            "error_p95_m": None,
            "per_color": {},
        }
        all_err = [float(r[6]) for r in eval_rows if r[6] != ""]
        if all_err:
            arr = np.array(all_err, dtype=np.float32)
            clip_summary["error_mean_m"] = float(np.mean(arr))
            clip_summary["error_median_m"] = float(np.median(arr))
            clip_summary["error_p95_m"] = float(np.percentile(arr, 95))
        for color in ("red", "green", "blue"):
            ce = per_color_err.get(color, [])
            if ce:
                arr = np.array(ce, dtype=np.float32)
                clip_summary["per_color"][color] = {
                    "count": int(len(arr)),
                    "mean_m": float(np.mean(arr)),
                    "median_m": float(np.median(arr)),
                    "p95_m": float(np.percentile(arr, 95)),
                }
            else:
                clip_summary["per_color"][color] = {"count": 0}

        with (out_sub / "reprojection_eval_summary.json").open("w", encoding="utf-8") as f:
            json.dump(clip_summary, f, indent=2)

        with (out_sub / "meta.txt").open("w", encoding="utf-8") as f:
            f.write(f"clip={clip}\n")
            f.write(f"world_sdf={world_sdf}\n")
            f.write(f"frames={len(frames)}\n")
            f.write(f"gt_buoys={len(gt_pts)}\n")
            f.write(f"detections_total={clip_summary['detections_total']}\n")
            f.write(f"matched_total={clip_summary['matched_total']}\n")
            f.write(f"error_mean_m={clip_summary['error_mean_m']}\n")
            f.write(f"error_median_m={clip_summary['error_median_m']}\n")
            f.write(f"error_p95_m={clip_summary['error_p95_m']}\n")

        summary_rows.append(
            [
                clip,
                str(world_sdf),
                len(frames),
                len(gt_pts),
                clip_summary["detections_total"],
                clip_summary["matched_total"],
                clip_summary["error_mean_m"],
                clip_summary["error_median_m"],
                clip_summary["error_p95_m"],
            ]
        )
        made += 1
        print(f"[{made}] {clip}: {len(frames)} frames, gt={len(gt_pts)} -> {out_video}")

    # Global roll-up across clips.
    if summary_rows:
        with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(
                [
                    "clip",
                    "world_sdf",
                    "frames",
                    "gt_buoys",
                    "detections_total",
                    "matched_total",
                    "error_mean_m",
                    "error_median_m",
                    "error_p95_m",
                ]
            )
            wcsv.writerows(summary_rows)

    print(f"Done. rendered={made}, skipped={skipped}, out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


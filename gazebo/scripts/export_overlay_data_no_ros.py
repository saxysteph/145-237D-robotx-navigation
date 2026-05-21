#!/usr/bin/env python3
"""
Export presentation-friendly overlay/eval data without ROS2/rclpy.

This consumes already-generated per-clip reprojection outputs from:
  yolo_comparison_test/path2_switch_proposal/scripts/.../reprojection_per_clip_eval_v2/

and writes:
  - one flattened CSV across all clips
  - one compact JSON summary for slides
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reproj-root",
        type=Path,
        default=Path(
            "yolo_comparison_test/path2_switch_proposal/scripts/"
            "roi_hsv_pipeline_clip_split_4ep/test_eval/reprojection_per_clip_eval_v2"
        ),
    )
    p.add_argument("--out-dir", type=Path, default=Path("captures/overlay_data_export"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.reproj_root.is_dir():
        raise NotADirectoryError(f"Missing reproj root: {args.reproj_root}")

    flat_rows: list[list[object]] = []
    clip_summaries: list[dict] = []
    for clip_dir in sorted([p for p in args.reproj_root.iterdir() if p.is_dir()]):
        eval_csv = clip_dir / "reprojection_eval.csv"
        eval_json = clip_dir / "reprojection_eval_summary.json"
        if not eval_csv.is_file() or not eval_json.is_file():
            continue
        js = json.loads(eval_json.read_text(encoding="utf-8"))
        clip_summaries.append(js)
        with eval_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                flat_rows.append(
                    [
                        clip_dir.name,
                        row["image"],
                        row["color"],
                        row["est_world_north_m"],
                        row["est_world_east_m"],
                        row["gt_world_north_m"],
                        row["gt_world_east_m"],
                        row["error_m"],
                        row.get("est_lat", ""),
                        row.get("est_lon", ""),
                    ]
                )

    flat_csv = args.out_dir / "overlay_eval_all_detections.csv"
    with flat_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "clip",
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
        w.writerows(flat_rows)

    all_err = []
    for row in flat_rows:
        try:
            all_err.append(float(row[7]))
        except Exception:
            pass
    summary = {
        "source_reprojection_root": str(args.reproj_root),
        "clips_included": len(clip_summaries),
        "detections_included": len(flat_rows),
        "error_mean_m": float(np.mean(all_err)) if all_err else None,
        "error_median_m": float(np.median(all_err)) if all_err else None,
        "error_p95_m": float(np.percentile(all_err, 95)) if all_err else None,
        "per_clip": clip_summaries,
    }
    summary_json = args.out_dir / "overlay_eval_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {flat_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Clips: {summary['clips_included']}, detections: {summary['detections_included']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


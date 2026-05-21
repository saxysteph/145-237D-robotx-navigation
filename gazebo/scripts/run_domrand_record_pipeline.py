#!/usr/bin/env python3
"""
Optional end-to-end glue around the same pieces documented in gazebo/README.md (generate worlds,
ros2 launch + record, project_buoys_to_yolo_labels, train_roi_then_hsv). Prefer those manual steps
when debugging; use this script for batch timing.

Domain-randomized worlds → timed ROS/Gazebo clips → reprojection labels → (optional) train_roi_then_hsv.

Example:
  source /opt/ros/jazzy/setup.bash
  python3 gazebo/scripts/run_domrand_record_pipeline.py \\
    --manifest gazebo/worlds/generated/manifest.json \\
    --out-batch captures/gazebo_uav_batch \\
    --seconds 10 --max-clips 5

Camera pan: the stock world uses a static downward camera; frames move only with waves/light.
True horizontal pan requires rotating the camera model in Gazebo (not implemented here).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).resolve().parent
    gen_worlds = script_dir / "generate_domain_randomized_worlds.py"
    record_clip = script_dir / "record_uav_clip_timed.py"
    label_proj = script_dir / "project_buoys_to_yolo_labels.py"
    train_script = repo_root / "yolo_comparison_test" / "path2_switch_proposal" / "scripts" / "train_roi_then_hsv.py"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generate-only", action="store_true", help="Run world generator and exit.")
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--worlds-dir", type=Path, default=repo_root / "gazebo" / "worlds" / "generated")
    p.add_argument("--max-scenarios", type=int, default=0, help="Passed to generate_domain_randomized_worlds.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", type=Path, default=None, help="Defaults to --worlds-dir/manifest.json")
    p.add_argument("--out-batch", type=Path, default=repo_root / "captures" / "gazebo_uav_batch")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--max-recordings", type=int, default=0, help="Cap manifest entries to record (0=all).")
    p.add_argument("--skip-record", action="store_true", help="Only run projection labeling on existing clips.")
    p.add_argument("--skip-annotate", action="store_true")
    p.add_argument("--train", action="store_true", help="Run train_roi_then_hsv after labeling.")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--train-max-clips", type=int, default=0, help="Forward --max-clips to train_roi_then_hsv.")
    p.add_argument("--train-subset-samples", type=int, default=0)
    args = p.parse_args()

    manifest_path = args.manifest or (args.worlds_dir / "manifest.json")

    if not args.skip_generate:
        cmd = [sys.executable, str(gen_worlds), "--output-dir", str(args.worlds_dir), "--seed", str(args.seed)]
        if args.max_scenarios and args.max_scenarios > 0:
            cmd.extend(["--max-scenarios", str(args.max_scenarios)])
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if args.generate_only:
        return 0

    if not manifest_path.is_file():
        print(f"No manifest at {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.max_recordings and args.max_recordings > 0:
        manifest = manifest[: args.max_recordings]

    args.out_batch.mkdir(parents=True, exist_ok=True)

    if not args.skip_record:
        for item in manifest:
            world_path = Path(item["world_path"])
            name = item["name"]
            clip_dir = args.out_batch / name
            if (clip_dir / "uav_pose_log.csv").is_file() and any(clip_dir.glob("uav_*.jpg")):
                print(f"Skip existing clip: {clip_dir}")
                continue
            cmd = [
                sys.executable,
                str(record_clip),
                str(world_path),
                str(clip_dir),
                "--seconds",
                str(args.seconds),
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=False)

    if not args.skip_annotate:
        cmd = [
            sys.executable,
            str(label_proj),
            "--captures-root",
            str(args.out_batch),
            "--intrinsics",
            str(repo_root / "calibration" / "camera_intrinsics_latest.json"),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if args.train:
        if not train_script.is_file():
            print(f"Missing {train_script}", file=sys.stderr)
            return 1
        cmd = [
            sys.executable,
            str(train_script),
            "--captures-root",
            str(args.out_batch),
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
        ]
        if args.train_max_clips and args.train_max_clips > 0:
            cmd.extend(["--max-clips", str(args.train_max_clips)])
        if args.train_subset_samples and args.train_subset_samples > 0:
            cmd.extend(["--subset-max-samples", str(args.train_subset_samples)])
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

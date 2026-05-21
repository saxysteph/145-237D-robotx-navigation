#!/usr/bin/env python3
"""
Record multiple domain-randomized clips using gz sim + record_uav_dataset_gz_transport (no ROS).

Prereqs:
  - Run from bash AFTER:  source gazebo/scripts/gz_env.sh   (so GZ_SIM_* + macOS DYLD for gz-transport)
  - uv (or create .venv_gz manually) + protobuf, numpy, opencv-python
  - Optional repo .venv_train (py3.12) with ultralytics+torch+pi-heif for YOLO; else falls back to python3
  - gz on PATH

Example:
  cd repo && source gazebo/scripts/gz_env.sh && python3 gazebo/scripts/regen_clips_gz_transport.py --max-clips 50 --worldgen-seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import importlib.util
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=repo / "gazebo" / "worlds" / "generated" / "manifest.json")
    p.add_argument("--out-batch", type=Path, default=repo / "captures" / "gazebo_uav_batch")
    p.add_argument("--max-clips", type=int, default=50, help="Clip count (matches --max-scenarios for worldgen).")
    p.add_argument("--seconds", type=float, default=10.0, help="Recording duration per world (after startup sleep)")
    p.add_argument("--gz-startup-s", type=float, default=5.0, help="Wait after gz sim before subscribing")
    p.add_argument("--worldgen-seed", type=int, default=42, help="Seed for generate_domain_randomized_worlds (glare×wave×layout shuffle).")
    p.add_argument(
        "--clutter-count",
        type=int,
        default=18,
        help="Forwarded to generate_domain_randomized_worlds: static non-buoy objects (0 disables).",
    )
    p.add_argument(
        "--no-camera-motion",
        action="store_true",
        help="Forwarded to worldgen: keep nadir camera fixed (no overhead orbit plugin).",
    )
    p.add_argument(
        "--annotate-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After labeling, write colored GT boxes under each clip's annotated_gt/.",
    )
    p.add_argument(
        "--annot-max-per-clip",
        type=int,
        default=0,
        help="Cap GT viz frames per clip (0 = all labeled frames).",
    )
    p.add_argument(
        "--proj-extra-yaw-deg",
        type=float,
        default=90.0,
        help="Forwarded to project_buoys_to_yolo_labels --extra-yaw-deg (camera/image frame alignment).",
    )
    p.add_argument("--skip-worldgen", action="store_true")
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Do not run train_roi_then_hsv (only worldgen/record/labels/GT viz).",
    )
    p.add_argument("--venv-python", type=Path, default=repo / ".venv_gz" / "bin" / "python")
    args = p.parse_args()

    if not os.environ.get("GZ_SIM_RESOURCE_PATH"):
        print(
            "GZ_SIM_RESOURCE_PATH is not set. Run:\n"
            f"  export REPO_ROOT={repo}\n"
            f"  source {repo}/gazebo/scripts/gz_env.sh",
            file=sys.stderr,
        )
        return 1

    if not args.venv_python.is_file():
        print(
            f"Missing venv Python at {args.venv_python}\n"
            "  Run:  cd repo && uv venv .venv_gz --python 3.12 && . .venv_gz/bin/activate && uv pip install protobuf numpy opencv-python",
            file=sys.stderr,
        )
        return 1

    record_script = here / "record_uav_dataset_gz_transport.py"
    if not record_script.is_file():
        print(f"Missing {record_script}", file=sys.stderr)
        return 1

    gz = shutil.which("gz")
    if not gz:
        print("gz not on PATH", file=sys.stderr)
        return 1

    if not args.skip_worldgen:
        wg_cmd = [
            sys.executable,
            str(here / "generate_domain_randomized_worlds.py"),
            "--seed",
            str(args.worldgen_seed),
            "--max-scenarios",
            str(args.max_clips),
            "--clutter-count",
            str(max(0, args.clutter_count)),
        ]
        if args.no_camera_motion:
            wg_cmd.append("--no-camera-motion")
        subprocess.run(wg_cmd, cwd=str(repo), check=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))[: args.max_clips]
    if not manifest:
        print("Manifest empty.", file=sys.stderr)
        return 1

    args.out_batch.mkdir(parents=True, exist_ok=True)
    # Match record_uav_dataset_gz_transport so subprocess can load libgz-transport*.dylib.
    _spec = importlib.util.spec_from_file_location("_gzrec", here / "record_uav_dataset_gz_transport.py")
    if _spec is None or _spec.loader is None:
        print("Cannot load record_uav_dataset_gz_transport module spec.", file=sys.stderr)
        return 1
    rec_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rec_mod)
    rec_mod._ensure_dyld_library_path_macos()
    env_base = os.environ.copy()

    # Ultralytics stacks often assume Python 3.10+; /usr/bin/python3 is often 3.9 on macOS.
    _train_venv = repo / ".venv_train" / "bin" / "python"
    label_train_py = (
        _train_venv
        if _train_venv.is_file()
        else (Path("/usr/bin/python3") if Path("/usr/bin/python3").is_file() else Path(sys.executable))
    )

    for i, item in enumerate(manifest):
        world = Path(item["world_path"])
        name = item["name"]
        if not world.is_file():
            print(f"Skip missing world: {world}", file=sys.stderr)
            continue
        clip = args.out_batch / name
        shutil.rmtree(clip, ignore_errors=True)
        clip.mkdir(parents=True)
        shutil.copy(world, clip / "world.sdf")

        print(f"\n=== [{i + 1}/{len(manifest)}] {name} ===")
        gz_proc = subprocess.Popen(
            [gz, "sim", "-s", "-r", str(world.resolve())],
            cwd=str(repo),
            env=env_base,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            time.sleep(args.gz_startup_s)
            rc = subprocess.run(
                [
                    str(args.venv_python),
                    str(record_script),
                    "--topic",
                    "/robotx/uav/camera/image_raw",
                    "--out-dir",
                    str(clip),
                    "--duration",
                    str(args.seconds),
                ],
                cwd=str(repo),
                env=env_base,
                check=False,
            )
            if rc.returncode != 0:
                print(f"Recorder exit {rc.returncode}", file=sys.stderr)
        finally:
            gz_proc.send_signal(signal.SIGTERM)
            try:
                gz_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                gz_proc.kill()
            time.sleep(1.0)

    print("\n=== project_buoys_to_yolo_labels ===")
    subprocess.run(
        [
            str(label_train_py),
            str(here / "project_buoys_to_yolo_labels.py"),
            "--captures-root",
            str(args.out_batch),
            "--skip-incomplete",
            "--projection",
            "sim_sdf",
            "--extra-yaw-deg",
            str(args.proj_extra_yaw_deg),
            "--intrinsics",
            str(repo / "calibration" / "camera_intrinsics_latest.json"),
        ],
        cwd=str(repo),
        check=True,
    )

    if args.annotate_gt:
        print("\n=== draw_yolo_labels_on_captures (GT visualization) ===")
        vis_cmd = [
            str(label_train_py),
            str(here / "draw_yolo_labels_on_captures.py"),
            "--captures-root",
            str(args.out_batch),
            "--labels-subdir",
            "labels_proj",
            "--out-subdir",
            "annotated_gt",
        ]
        if args.annot_max_per_clip and args.annot_max_per_clip > 0:
            vis_cmd.extend(["--max-per-clip", str(args.annot_max_per_clip)])
        vis_cmd.append("--hsv-mean-label")
        subprocess.run(vis_cmd, cwd=str(repo), check=False)

    if not args.skip_train:
        print("\n=== train_roi_then_hsv ===")
        train = repo / "yolo_comparison_test" / "path2_switch_proposal" / "scripts" / "train_roi_then_hsv.py"
        subprocess.run(
            [
                str(label_train_py),
                str(train),
                "--captures-root",
                str(args.out_batch),
                "--epochs",
                "8",
                "--device",
                "cpu",
                "--workers",
                "0",
                "--batch",
                "8",
                "--imgsz",
                "640",
                "--max-test-images",
                "0",
            ],
            cwd=str(repo),
            check=False,
        )
    else:
        print("\n=== train_roi_then_hsv (skipped via --skip-train) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

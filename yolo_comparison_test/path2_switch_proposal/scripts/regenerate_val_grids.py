#!/usr/bin/env python3
"""Re-run model.val() for Path 2 (red/green/blue balloons) with readable 4-tile val_batch plots."""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS = os.path.join(_SCRIPT_DIR, "training", "balloon_proper", "weights", "best.pt")
_DEFAULT_DATA = os.path.join(_SCRIPT_DIR, "dataset", "dataset_abs.yaml")
_DEFAULT_PROJECT = os.path.join(_SCRIPT_DIR, "balloon_ultralytics_runs")

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from ultralytics import YOLO  # noqa: E402

from ultralytics_plot_patch import apply_plot_max_subplots  # noqa: E402


def _reject_non_balloon_data(data_path: str) -> str | None:
    """Return error message if data path looks like COCO / public demo, else None."""
    norm = os.path.normpath(data_path).lower()
    parts = norm.replace("\\", "/").split("/")
    if "coco" in norm or "coco8" in parts or norm.endswith("coco.yaml") or "datasets/coco" in norm.replace("\\", "/"):
        return (
            "Refusing this --data path: regenerate_val_grids is for the Path 2 balloon dataset only "
            "(scripts/dataset/dataset_abs.yaml after validation_step1). "
            "Do not pass COCO or coco8."
        )
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weights",
        default=None,
        help=f"Path to .pt weights (default: {_DEFAULT_WEIGHTS})",
    )
    p.add_argument(
        "--data",
        default=None,
        help=f"Path to dataset YAML (default: {_DEFAULT_DATA})",
    )
    p.add_argument(
        "--project",
        default=None,
        help=(
            "Absolute or relative Ultralytics project parent directory "
            f"(default: {_DEFAULT_PROJECT}). "
            "Plots are saved under <project>/<name>/."
        ),
    )
    p.add_argument("--name", default="val", help="Run folder name under project (default: val)")
    p.add_argument("--batch", type=int, default=4, help="Val dataloader batch size (matches mosaic cap)")
    p.add_argument(
        "--i-know-this-is-not-balloon-data",
        action="store_true",
        help="Skip COCO/demo guard on --data (not recommended).",
    )
    args = p.parse_args()

    weights = args.weights or _DEFAULT_WEIGHTS
    data = args.data or _DEFAULT_DATA
    project = args.project if args.project is not None else _DEFAULT_PROJECT
    project = os.path.abspath(project)

    if not args.i_know_this_is_not_balloon_data:
        msg = _reject_non_balloon_data(data)
        if msg:
            print(msg, file=sys.stderr)
            return 2

    if not os.path.isfile(weights):
        print(
            f"Weights not found: {weights}\n"
            "Run validation_step2_retrain.py first (after step1 split), or pass --weights explicitly.",
            file=sys.stderr,
        )
        return 1
    if not os.path.isfile(data):
        print(
            f"Dataset YAML not found: {data}\n"
            "Run validation_step1_proper_split.py and ensure dataset_abs.yaml exists, or pass --data explicitly.",
            file=sys.stderr,
        )
        return 1

    apply_plot_max_subplots(4)
    model = YOLO(weights)
    model.val(
        data=data,
        project=project,
        name=args.name,
        batch=args.batch,
        plots=True,
        verbose=True,
        exist_ok=True,
    )
    print(f"Balloon val plots (val_batch*.jpg) are under: {os.path.join(project, args.name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

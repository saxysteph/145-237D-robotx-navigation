#!/usr/bin/env python3
"""Retrain YOLOv11n on proper train split and report held-out mAP50."""

from __future__ import annotations

import os

from ultralytics import YOLO


def extract_map50(results_csv: str) -> float:
    import csv

    if not os.path.isfile(results_csv):
        return 0.0
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        best = 0.0
        for row in reader:
            try:
                v = float(row.get("metrics/mAP50(B)", "0") or 0.0)
            except ValueError:
                v = 0.0
            if v > best:
                best = v
    return best


def main() -> int:
    root = os.path.dirname(__file__)
    dataset_dir = os.path.join(root, "dataset")
    dataset_yaml_abs = os.path.join(dataset_dir, "dataset_abs.yaml")

    with open(dataset_yaml_abs, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_dir.replace('\\', '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 3\n")
        f.write("names: ['red', 'green', 'blue']\n")

    project = os.path.join(root, "training")
    name = "balloon_proper"
    os.makedirs(project, exist_ok=True)

    model = YOLO("yolo11n.pt")
    model.train(
        data=dataset_yaml_abs,
        epochs=50,
        imgsz=640,
        batch=8,
        project=project,
        name=name,
        patience=15,
        exist_ok=True,
    )

    results_csv = os.path.join(project, name, "results.csv")
    best_map50 = extract_map50(results_csv)

    print(f"Best mAP50 on HELD-OUT val set: {best_map50:.3f}")
    print("This is the honest accuracy number")

    out_txt = os.path.join(root, "honest_map50.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"{best_map50:.3f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

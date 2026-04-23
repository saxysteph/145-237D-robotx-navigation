#!/usr/bin/env python3
"""Summarize HSV detection CSV and generate a simple metrics chart."""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np


def _bar_chart(counts: dict[str, int], out_path: str) -> None:
    width, height = 900, 360
    img = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(img, "Detections per class", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)

    classes = ["red", "green", "blue"]
    colors = {"red": (0, 0, 255), "green": (0, 180, 0), "blue": (255, 100, 0)}
    max_count = max(1, max(counts[c] for c in classes))

    bar_left = 180
    bar_right = width - 50
    bar_max_w = bar_right - bar_left
    y0 = 85
    step = 85
    bar_h = 46

    for i, cls in enumerate(classes):
        y = y0 + i * step
        w = int((counts[cls] / max_count) * bar_max_w)
        cv2.putText(img, cls, (45, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2)
        cv2.rectangle(img, (bar_left, y), (bar_left + w, y + bar_h), colors[cls], -1)
        cv2.rectangle(img, (bar_left, y), (bar_left + bar_max_w, y + bar_h), (100, 100, 100), 1)
        cv2.putText(img, str(counts[cls]), (bar_left + w + 10, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    cv2.imwrite(out_path, img)


def main() -> int:
    csv_path = os.path.join("captures", "hsv_results", "detections.csv")
    if not os.path.isfile(csv_path):
        print(f"Detections CSV not found: {csv_path}")
        return 1

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    classes = ["red", "green", "blue"]
    by_class: dict[str, list[float]] = {c: [] for c in classes}
    images: dict[str, int] = {}
    low_conf = 0

    for row in rows:
        image = row.get("image", "")
        color = row.get("color", "").lower()
        conf_str = row.get("confidence", "0")
        conf = float(conf_str) if conf_str else 0.0
        if image:
            images[image] = images.get(image, 0) + 1
        if color in by_class:
            by_class[color].append(conf)
        if conf < 0.25:
            low_conf += 1

    captures_dir = "captures"
    image_files = [
        name
        for name in sorted(os.listdir(captures_dir))
        if os.path.isfile(os.path.join(captures_dir, name)) and name.lower().endswith(".jpg")
    ]
    zero_images = [name for name in image_files if images.get(name, 0) == 0]

    total = len(rows)
    image_count = len(image_files)
    counts = {c: len(by_class[c]) for c in classes}

    avg_conf_all = float(np.mean([float(r.get("confidence", "0") or 0.0) for r in rows])) if rows else 0.0

    print("=== Detection Metrics ===")
    print(f"Images processed: {image_count}")
    print(f"Total detections: {total}")
    print("")
    print("Per-class breakdown:")
    for cls in classes:
        vals = by_class[cls]
        if vals:
            print(
                f"  {cls}: count={len(vals)}  avg_conf={np.mean(vals):.3f}  "
                f"min={np.min(vals):.3f}  max={np.max(vals):.3f}"
            )
        else:
            print(f"  {cls}: count=0  avg_conf=0.000  min=0.000  max=0.000")
    print("")
    print(f"Potential false positives (conf < 0.25): {low_conf}")
    if len(zero_images) <= 5:
        print(f"Images with zero detections: {len(zero_images)}  {zero_images}")
    else:
        print(f"Images with zero detections: {len(zero_images)}")
    print("")
    print("=== Presentation Summary Line ===")
    print(
        "Stage-A HSV Pipeline | "
        f"{image_count} frames | {total} detections | "
        f"red:{counts['red']} green:{counts['green']} blue:{counts['blue']} | "
        f"avg_conf: {avg_conf_all:.3f}"
    )

    out_chart = os.path.join("captures", "hsv_results", "metrics.png")
    _bar_chart(counts, out_chart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

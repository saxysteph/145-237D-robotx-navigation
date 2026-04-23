#!/usr/bin/env python3
"""Plot train vs val loss from Ultralytics results.csv using OpenCV."""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np


def main() -> int:
    root = os.path.dirname(__file__)
    results_csv = os.path.join(root, "training", "balloon_proper", "results.csv")
    if not os.path.isfile(results_csv):
        print(f"Missing training curve file: {results_csv}")
        return 1

    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []

    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = int(float(row.get("epoch", "0")))
            tr = (
                float(row.get("train/box_loss", "0") or 0.0)
                + float(row.get("train/cls_loss", "0") or 0.0)
                + float(row.get("train/dfl_loss", "0") or 0.0)
            )
            va = (
                float(row.get("val/box_loss", "0") or 0.0)
                + float(row.get("val/cls_loss", "0") or 0.0)
                + float(row.get("val/dfl_loss", "0") or 0.0)
            )
            epochs.append(e)
            train_loss.append(tr)
            val_loss.append(va)

    if not epochs:
        print("No epoch data found in results.csv")
        return 1

    best_idx = int(np.argmin(np.array(val_loss)))
    best_epoch = epochs[best_idx]
    tr_best = train_loss[best_idx]
    va_best = val_loss[best_idx]
    gap = va_best - tr_best

    w, h = 1200, 700
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 90, 30, 60, 80
    x0, y0 = margin_l, h - margin_b
    x1, y1 = w - margin_r, margin_t
    cv2.rectangle(img, (x0, y1), (x1, y0), (230, 230, 230), 1)
    cv2.putText(img, "Train vs Val Loss", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(img, "Epoch", (w // 2 - 25, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(img, "Loss", (15, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)

    min_y = min(min(train_loss), min(val_loss))
    max_y = max(max(train_loss), max(val_loss))
    if max_y <= min_y:
        max_y = min_y + 1.0

    def to_pt(i: int, y: float) -> tuple[int, int]:
        px = int(x0 + (i / max(1, len(epochs) - 1)) * (x1 - x0))
        py = int(y0 - ((y - min_y) / (max_y - min_y)) * (y0 - y1))
        return px, py

    for i in range(1, len(epochs)):
        cv2.line(img, to_pt(i - 1, train_loss[i - 1]), to_pt(i, train_loss[i]), (200, 80, 40), 2)
        cv2.line(img, to_pt(i - 1, val_loss[i - 1]), to_pt(i, val_loss[i]), (40, 40, 220), 2)

    cv2.putText(img, "Train loss", (x1 - 220, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 80, 40), 2, cv2.LINE_AA)
    cv2.putText(img, "Val loss", (x1 - 220, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 220), 2, cv2.LINE_AA)

    out_png = os.path.join(root, "loss_curves.png")
    cv2.imwrite(out_png, img)

    if gap < 0.05:
        verdict = "HEALTHY"
    elif gap <= 0.15:
        verdict = "MILD OVERFIT"
    else:
        verdict = "SEVERE OVERFIT"

    print(f"Best epoch: {best_epoch}")
    print(f"Train loss at best epoch: {tr_best:.3f}")
    print(f"Val loss at best epoch:   {va_best:.3f}")
    print(f"Gap (val - train): {gap:.3f}")
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

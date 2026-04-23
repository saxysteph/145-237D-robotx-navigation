#!/usr/bin/env python3
"""Run honest inference on held-out val set and compute detection metrics."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
from ultralytics import YOLO


@dataclass
class Box:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f


def read_gt_labels(lbl_path: str, img_w: int, img_h: int) -> list[Box]:
    boxes: list[Box] = []
    if not os.path.isfile(lbl_path):
        return boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx_n, cy_n, w_n, h_n = map(float, parts[1:])
            bw = w_n * img_w
            bh = h_n * img_h
            cx = cx_n * img_w
            cy = cy_n * img_h
            boxes.append(
                Box(
                    cls=cls,
                    conf=1.0,
                    x1=cx - bw / 2.0,
                    y1=cy - bh / 2.0,
                    x2=cx + bw / 2.0,
                    y2=cy + bh / 2.0,
                )
            )
    return boxes


def greedy_match(preds: list[Box], gts: list[Box], cls_filter: int | None = None) -> tuple[int, int, int]:
    pred_ids = [i for i, p in enumerate(preds) if cls_filter is None or p.cls == cls_filter]
    gt_ids = [i for i, g in enumerate(gts) if cls_filter is None or g.cls == cls_filter]
    matches: list[tuple[float, int, int]] = []
    for pi in pred_ids:
        for gi in gt_ids:
            ov = iou(preds[pi], gts[gi])
            if ov > 0.5:
                matches.append((ov, pi, gi))
    matches.sort(reverse=True, key=lambda t: t[0])
    used_p: set[int] = set()
    used_g: set[int] = set()
    tp = 0
    for _, pi, gi in matches:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        tp += 1
    fp = len(pred_ids) - len(used_p)
    fn = len(gt_ids) - len(used_g)
    return tp, fp, fn


def main() -> int:
    root = os.path.dirname(__file__)
    val_img_dir = os.path.join(root, "dataset", "images", "val")
    val_lbl_dir = os.path.join(root, "dataset", "labels", "val")
    ann_dir = os.path.join(root, "val_annotated")
    os.makedirs(ann_dir, exist_ok=True)

    weights = os.path.join(root, "training", "balloon_proper", "weights", "best.pt")
    dataset_yaml_abs = os.path.join(root, "dataset", "dataset_abs.yaml")
    if not os.path.isfile(dataset_yaml_abs):
        with open(dataset_yaml_abs, "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.join(root, 'dataset').replace('\\', '/')}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("nc: 3\n")
            f.write("names: ['red', 'green', 'blue']\n")

    model = YOLO(weights)

    images = sorted([n for n in os.listdir(val_img_dir) if n.lower().endswith(".jpg")])
    total_tp = total_fp = total_fn = 0
    cls_stats = {0: [0, 0, 0], 1: [0, 0, 0], 2: [0, 0, 0]}  # tp, fp, fn

    for name in images:
        path = os.path.join(val_img_dir, name)
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_gt_labels(os.path.join(val_lbl_dir, os.path.splitext(name)[0] + ".txt"), w, h)

        result = model(img, verbose=False)[0]
        preds: list[Box] = []
        if result.boxes is not None:
            for i in range(len(result.boxes)):
                cls = int(result.boxes.cls[i].item())
                conf = float(result.boxes.conf[i].item())
                if conf < 0.25:
                    continue
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                preds.append(Box(cls, conf, x1, y1, x2, y2))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
                cv2.putText(
                    img,
                    f"{cls}:{conf:.2f}",
                    (int(x1), max(20, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        tp, fp, fn = greedy_match(preds, gt, None)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for cls in (0, 1, 2):
            ctp, cfp, cfn = greedy_match(preds, gt, cls)
            cls_stats[cls][0] += ctp
            cls_stats[cls][1] += cfp
            cls_stats[cls][2] += cfn

        cv2.imwrite(os.path.join(ann_dir, name), img)

    p, r, f1 = prf(total_tp, total_fp, total_fn)
    map_metrics = model.val(data=dataset_yaml_abs, split="val", verbose=False)
    map50 = float(map_metrics.box.map50)

    by_name = {0: "red", 1: "green", 2: "blue"}
    lines = [
        "=== HONEST VALIDATION RESULTS ===",
        f"Val set size: {len(images)} images (never seen during training)",
        f"True Positives:  {total_tp}",
        f"False Positives: {total_fp}  (things detected that aren't balloons)",
        f"False Negatives: {total_fn}  (balloons missed)",
        f"Precision: {p:.3f}",
        f"Recall:    {r:.3f}",
        f"F1 Score:  {f1:.3f}",
        f"mAP50 (ultralytics): {map50:.3f}",
        "",
        "Per class:",
    ]
    for cls in (0, 1, 2):
        cp, cr, cf = prf(cls_stats[cls][0], cls_stats[cls][1], cls_stats[cls][2])
        lines.append(f"  {by_name[cls]}: precision={cp:.3f} recall={cr:.3f} F1={cf:.3f}")

    train_map = 0.980
    overfit_drop = train_map - map50
    lines += [
        "",
        f"VERDICT: {'LEGITIMATE' if map50 > 0.80 else 'NOT LEGITIMATE'} if mAP50 > 0.80 on val set",
        (
            f"VERDICT: {'OVERFIT' if overfit_drop > 0.15 else 'NOT OVERFIT'} "
            "if mAP50 drops more than 0.15 vs training mAP50"
        ),
    ]

    text = "\n".join(lines) + "\n"
    print(text, end="")
    out_txt = os.path.join(root, "honest_results.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

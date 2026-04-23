#!/usr/bin/env python3
"""Stress test trained model on held-out val images with UAV-like noise."""

from __future__ import annotations

import os
import random

import cv2
import numpy as np
from ultralytics import YOLO


def _motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    base = np.zeros((size, size), dtype=np.float32)
    cv2.line(base, (0, size // 2), (size - 1, size // 2), 1.0, 1)
    rot = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle_deg, 1.0)
    kernel = cv2.warpAffine(base, rot, (size, size))
    s = float(kernel.sum())
    if s > 0:
        kernel /= s
    return kernel


def apply_uav_noise(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    out = cv2.GaussianBlur(out, (0, 0), 2.5)
    out = cv2.filter2D(out, -1, _motion_blur_kernel(13, random.uniform(0, 180)))

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    v += random.randint(-20, 20)

    hh, ww = v.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    for _ in range(3):
        gx = random.randint(0, ww - 1)
        gy = random.randint(0, hh - 1)
        radius = random.randint(25, 55)
        dist2 = (xx - gx) ** 2 + (yy - gy) ** 2
        sigma = max(1.0, radius / 2.0)
        v += 65.0 * np.exp(-dist2 / (2.0 * sigma * sigma))

    hsv[:, :, 2] = np.clip(v, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    noise = np.random.normal(0, 8, out.shape).astype(np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def infer_count_conf(model: YOLO, img: np.ndarray) -> tuple[int, float, np.ndarray]:
    result = model(img, verbose=False)[0]
    boxes = result.boxes
    out = img.copy()
    cnt = 0
    confs: list[float] = []
    if boxes is not None:
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if conf < 0.25:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
            cv2.putText(
                out,
                f"{conf:.2f}",
                (int(x1), max(20, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cnt += 1
            confs.append(conf)
    avg = float(np.mean(confs)) if confs else 0.0
    return cnt, avg, out


def main() -> int:
    random.seed(42)
    np.random.seed(42)

    root = os.path.dirname(__file__)
    val_img_dir = os.path.join(root, "dataset", "images", "val")
    weights = os.path.join(root, "training", "balloon_proper", "weights", "best.pt")
    model = YOLO(weights)

    names = sorted([n for n in os.listdir(val_img_dir) if n.lower().endswith(".jpg")])
    clean_det = 0
    aug_det = 0
    clean_avg_list: list[float] = []
    aug_avg_list: list[float] = []
    clean_missed = 0
    aug_missed = 0

    strip_triplets: list[np.ndarray] = []
    pick = set(random.sample(range(len(names)), k=min(3, len(names))))

    for i, name in enumerate(names):
        img = cv2.imread(os.path.join(val_img_dir, name))
        if img is None:
            continue
        aug = apply_uav_noise(img)
        c_cnt, c_avg, _ = infer_count_conf(model, img)
        a_cnt, a_avg, a_ann = infer_count_conf(model, aug)

        clean_det += c_cnt
        aug_det += a_cnt
        clean_avg_list.append(c_avg)
        aug_avg_list.append(a_avg)
        if c_cnt == 0:
            clean_missed += 1
        if a_cnt == 0:
            aug_missed += 1

        if i in pick:
            h = 280
            clean_r = cv2.resize(img, (420, h))
            aug_r = cv2.resize(aug, (420, h))
            ann_r = cv2.resize(a_ann, (420, h))
            strip_triplets.append(np.hstack([clean_r, aug_r, ann_r]))

    clean_avg = float(np.mean(clean_avg_list)) if clean_avg_list else 0.0
    aug_avg = float(np.mean(aug_avg_list)) if aug_avg_list else 0.0
    retention = (aug_det / clean_det * 100.0) if clean_det > 0 else 0.0

    lines = [
        "=== STRESS TEST (UAV noise simulation) ===",
        f"Clean val:     detections={clean_det}  avg_conf={clean_avg:.3f}  missed={clean_missed}",
        f"Augmented val: detections={aug_det}  avg_conf={aug_avg:.3f}  missed={aug_missed}",
        f"Retention rate: {retention:.1f}%",
        f"Confidence drop: {clean_avg:.3f} → {aug_avg:.3f}",
    ]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    with open(os.path.join(root, "stress_test_results.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    if strip_triplets:
        strip = np.vstack(strip_triplets)
        cv2.imwrite(os.path.join(root, "stress_strip.png"), strip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

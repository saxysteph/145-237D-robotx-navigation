#!/usr/bin/env python3
"""Run HSV detector on clean and UAV-noised image variants."""

from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np

from color_utils import load_color_ranges
from hsv_batch_detect import Detection, detect_in_image, draw_detections


def _first_capture_jpg(captures_dir: str) -> str | None:
    for name in sorted(os.listdir(captures_dir)):
        if name.lower().endswith(".jpg"):
            path = os.path.join(captures_dir, name)
            if os.path.isfile(path):
                return path
    return None


def _motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    base = np.zeros((size, size), dtype=np.float32)
    cv2.line(base, (0, size // 2), (size - 1, size // 2), 1.0, 1)
    center = (size / 2.0, size / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(base, rot, (size, size))
    s = float(kernel.sum())
    if s > 0:
        kernel /= s
    return kernel


def apply_uav_noise(img: np.ndarray) -> np.ndarray:
    """Apply realistic UAV flight noise. Returns augmented copy."""
    out = img.copy()
    out = cv2.GaussianBlur(out, (0, 0), 2.5)

    angle = random.uniform(0.0, 180.0)
    kernel = _motion_blur_kernel(13, angle)
    out = cv2.filter2D(out, -1, kernel)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    v += random.randint(-35, 35)

    h_img, w_img = v.shape
    yy, xx = np.mgrid[0:h_img, 0:w_img]
    for _ in range(3):
        gx = random.randint(0, w_img - 1)
        gy = random.randint(0, h_img - 1)
        radius = random.randint(25, 55)
        dist2 = (xx - gx) ** 2 + (yy - gy) ** 2
        sigma = max(1.0, radius / 2.0)
        falloff = np.exp(-dist2 / (2.0 * sigma * sigma))
        v += 65.0 * falloff

    hsv[:, :, 2] = np.clip(v, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    noise = np.random.normal(0, 8, out.shape).astype(np.int16)
    out_i16 = out.astype(np.int16) + noise
    return np.clip(out_i16, 0, 255).astype(np.uint8)


def _counts(detections: list[Detection]) -> dict[str, int]:
    out = {"red": 0, "green": 0, "blue": 0}
    for det in detections:
        if det.color in out:
            out[det.color] += 1
    return out


def _line(label: str, detections: list[Detection]) -> str:
    c = _counts(detections)
    return (
        f"{label}:  {len(detections)} detections "
        f"(red:{c['red']}  green:{c['green']}  blue:{c['blue']})"
    )


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (8, 8), (520, 44), (0, 0, 0), -1)
    cv2.putText(out, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return out


def main() -> int:
    captures_dir = "captures"
    out_dir = os.path.join(captures_dir, "hsv_results")
    os.makedirs(out_dir, exist_ok=True)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        if not os.path.isdir(captures_dir):
            print(f"Captures directory not found: {captures_dir}")
            return 1
        image_path = _first_capture_jpg(captures_dir)
        if image_path is None:
            print("No .jpg files found in captures/.")
            return 1

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return 1

    ranges_map = load_color_ranges(classes_dir="captures/classes")
    clean_det = detect_in_image(img, ranges_map, min_area=1500.0, kernel_size=5)

    aug_images = [apply_uav_noise(img) for _ in range(3)]
    aug_dets = [detect_in_image(a, ranges_map, min_area=1500.0, kernel_size=5) for a in aug_images]

    clean_draw = draw_detections(img, clean_det)
    aug_draws = [draw_detections(a, d) for a, d in zip(aug_images, aug_dets)]

    p0 = _label_panel(clean_draw, f"clean ({len(clean_det)} detections)")
    p1 = _label_panel(aug_draws[0], f"uav_noise_1 blur+motion+glare ({len(aug_dets[0])})")
    p2 = _label_panel(aug_draws[1], f"uav_noise_2 blur+motion+glare ({len(aug_dets[1])})")
    p3 = _label_panel(aug_draws[2], f"uav_noise_3 blur+motion+glare ({len(aug_dets[2])})")

    top = np.hstack([p0, p1])
    bottom = np.hstack([p2, p3])
    grid = np.vstack([top, bottom])

    out_path = os.path.join(out_dir, "augmentation_test.jpg")
    cv2.imwrite(out_path, grid)

    clean_n = len(clean_det)
    if clean_n > 0:
        retention = (sum(len(d) / clean_n for d in aug_dets) / 3.0) * 100.0
    else:
        retention = 0.0

    print("=== Augmentation Test ===")
    print(_line("Clean", clean_det))
    print(_line("Aug 1", aug_dets[0]))
    print(_line("Aug 2", aug_dets[1]))
    print(_line("Aug 3", aug_dets[2]))
    print(f"Avg retention rate: {retention:.1f}%")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

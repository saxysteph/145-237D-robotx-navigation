#!/usr/bin/env python3
"""Batch HSV detection on capture images using class reference swatches."""

import argparse
import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np
from color_utils import HSVRange, build_mask, derive_class_hsv_ranges


@dataclass
class Detection:
    color: str
    confidence: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HSV detections on images in captures/ using class references.")
    parser.add_argument("--captures-dir", type=str, default="captures")
    parser.add_argument("--classes-dir", type=str, default="captures/classes")
    parser.add_argument("--out-dir", type=str, default="captures/hsv_results")
    parser.add_argument("--hue-margin", type=int, default=12)
    parser.add_argument("--sat-min-floor", type=int, default=50)
    parser.add_argument("--val-min-floor", type=int, default=45)
    parser.add_argument("--min-area", type=float, default=1500.0)
    parser.add_argument("--kernel-size", type=int, default=5)
    return parser.parse_args()


def list_images(folder: str) -> list[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(exts):
            paths.append(path)
    return paths


def detect_in_image(img: np.ndarray, ranges_map: dict[str, list[HSVRange]], min_area: float, kernel_size: int) -> list[Detection]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    detections: list[Detection] = []
    img_h, img_w = img.shape[:2]
    border_x = int(0.12 * img_w)
    border_y = int(0.12 * img_h)

    for color, ranges in ranges_map.items():
        mask = build_mask(hsv, ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue
            aspect = float(w) / float(h)
            if aspect < 0.3 or aspect > 3.3:
                continue
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = (area / hull_area) if hull_area > 0 else 0.0
            if solidity < 0.55:
                continue
            cx_val = x + w / 2.0
            cy_val = y + h / 2.0
            if cx_val < border_x or cx_val > img_w - border_x:
                continue
            if cy_val < border_y or cy_val > img_h - border_y:
                continue

            pad_x = int(0.15 * w)
            pad_y = int(0.15 * h)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.shape[1], x + w + pad_x)
            y2 = min(img.shape[0], y + h + pad_y)
            roi_crop = hsv[y1:y2, x1:x2]
            if roi_crop.size == 0:
                continue

            roi_mask = build_mask(roi_crop, ranges)
            denom = float(roi_crop.shape[0] * roi_crop.shape[1])
            ratio = float(np.count_nonzero(roi_mask)) / denom if denom > 0 else 0.0
            m = cv2.moments(roi_mask)
            if m["m00"] > 1e-6:
                cx = float(x1 + (m["m10"] / m["m00"]))
                cy = float(y1 + (m["m01"] / m["m00"]))
            else:
                cx, cy = x1 + (x2 - x1) / 2.0, y1 + (y2 - y1) / 2.0
            solidity_score = solidity
            size_score = min(1.0, area / 4000.0)
            conf = max(0.0, min(1.0, 0.50 * ratio + 0.30 * solidity_score + 0.20 * size_score))
            detections.append(
                Detection(
                    color=color,
                    confidence=conf,
                    bbox=(x, y, w, h),
                    centroid=(cx, cy),
                    area=area,
                )
            )
    return detections


def draw_detections(img: np.ndarray, detections: list[Detection]) -> np.ndarray:
    color_draw = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
    }
    out = img.copy()
    for det in detections:
        x, y, w, h = det.bbox
        draw = color_draw.get(det.color, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), draw, 2)
        cv2.circle(out, (int(det.centroid[0]), int(det.centroid[1])), 3, draw, -1)
        label = f"{det.color} {det.confidence:.2f}"
        cv2.putText(out, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw, 2)
    return out


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    annotated_dir = os.path.join(args.out_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    if not os.path.isdir(args.captures_dir):
        print(f"Captures directory not found: {args.captures_dir}")
        return 1
    if not os.path.isdir(args.classes_dir):
        print(f"Classes directory not found: {args.classes_dir}")
        return 1

    ranges_map = derive_class_hsv_ranges(
        args.classes_dir, args.hue_margin, args.sat_min_floor, args.val_min_floor
    )
    if not ranges_map:
        print("No class reference images found.")
        return 1

    print("Derived HSV ranges:")
    for color, ranges in sorted(ranges_map.items()):
        for i, r in enumerate(ranges):
            print(f"  {color}[{i}] low={r.low} high={r.high}")

    image_paths = list_images(args.captures_dir)
    image_paths = [p for p in image_paths if os.path.dirname(p) == args.captures_dir]
    if not image_paths:
        print("No capture images found.")
        return 1

    csv_path = os.path.join(args.out_dir, "detections.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "color", "confidence", "cx", "cy", "x", "y", "w", "h", "area"])
        total = 0
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            detections = detect_in_image(img, ranges_map, args.min_area, args.kernel_size)
            total += len(detections)

            annotated = draw_detections(img, detections)
            out_name = os.path.basename(path)
            out_path = os.path.join(annotated_dir, out_name)
            cv2.imwrite(out_path, annotated)

            for det in detections:
                x, y, w, h = det.bbox
                writer.writerow(
                    [
                        os.path.basename(path),
                        det.color,
                        f"{det.confidence:.4f}",
                        f"{det.centroid[0]:.2f}",
                        f"{det.centroid[1]:.2f}",
                        x,
                        y,
                        w,
                        h,
                        f"{det.area:.2f}",
                    ]
                )
    print(f"Processed {len(image_paths)} images, found {total} detections.")
    print(f"CSV: {csv_path}")
    print(f"Annotated images: {annotated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

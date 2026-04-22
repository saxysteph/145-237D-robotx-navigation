#!/usr/bin/env python3
"""Batch HSV detection on capture images using class reference swatches."""

import argparse
import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HSVRange:
    low: tuple[int, int, int]
    high: tuple[int, int, int]


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
    parser.add_argument("--min-area", type=float, default=120.0)
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


def circular_hue_mean(hues: np.ndarray) -> int:
    angles = (hues.astype(np.float32) / 180.0) * 2.0 * np.pi
    s = np.sin(angles).mean()
    c = np.cos(angles).mean()
    mean_angle = np.arctan2(s, c)
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return int(round((mean_angle / (2.0 * np.pi)) * 180.0)) % 180


def make_ranges_for_hue(hue_center: int, hue_margin: int, s_min: int, v_min: int) -> list[HSVRange]:
    h_low = hue_center - hue_margin
    h_high = hue_center + hue_margin
    if h_low < 0:
        return [
            HSVRange((0, s_min, v_min), (h_high, 255, 255)),
            HSVRange((180 + h_low, s_min, v_min), (179, 255, 255)),
        ]
    if h_high > 179:
        return [
            HSVRange((h_low, s_min, v_min), (179, 255, 255)),
            HSVRange((0, s_min, v_min), (h_high - 180, 255, 255)),
        ]
    return [HSVRange((h_low, s_min, v_min), (h_high, 255, 255))]


def derive_class_hsv_ranges(classes_dir: str, hue_margin: int, sat_floor: int, val_floor: int) -> dict[str, list[HSVRange]]:
    out: dict[str, list[HSVRange]] = {}
    for path in list_images(classes_dir):
        color = os.path.splitext(os.path.basename(path))[0].lower()
        img = cv2.imread(path)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        valid = (s > 25) & (v > 25)
        if int(valid.sum()) < 10:
            valid = np.ones_like(h, dtype=bool)

        hue_center = circular_hue_mean(h[valid])
        s_min = max(sat_floor, int(np.percentile(s[valid], 15)))
        v_min = max(val_floor, int(np.percentile(v[valid], 15)))
        out[color] = make_ranges_for_hue(hue_center, hue_margin, s_min, v_min)
    return out


def build_mask(hsv: np.ndarray, ranges: list[HSVRange]) -> np.ndarray:
    acc = None
    for r in ranges:
        m = cv2.inRange(
            hsv,
            np.array(r.low, dtype=np.uint8),
            np.array(r.high, dtype=np.uint8),
        )
        acc = m if acc is None else cv2.bitwise_or(acc, m)
    if acc is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    return acc


def detect_in_image(img: np.ndarray, ranges_map: dict[str, list[HSVRange]], min_area: float, kernel_size: int) -> list[Detection]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    detections: list[Detection] = []

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
            roi = mask[y : y + h, x : x + w]
            ratio = float(np.count_nonzero(roi)) / float(w * h)
            m = cv2.moments(cnt)
            if m["m00"] > 1e-6:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
            conf = max(0.0, min(1.0, 0.6 * ratio + 0.4 * min(1.0, area / 2500.0)))
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
        writer.writerow(
            ["image", "color", "confidence", "cx", "cy", "x", "y", "w", "h", "area", "annotated_image"]
        )
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
                        os.path.relpath(out_path, start=args.out_dir),
                    ]
                )
    print(f"Processed {len(image_paths)} images, found {total} detections.")
    print(f"CSV: {csv_path}")
    print(f"Annotated images: {annotated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

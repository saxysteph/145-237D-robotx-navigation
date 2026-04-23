#!/usr/bin/env python3
"""Path 2A: auto-label images with isolated HSV logic."""

from __future__ import annotations

import os
import shutil
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
    bbox: tuple[int, int, int, int]


def list_images(folder: str, exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> list[str]:
    out: list[str] = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(exts):
            out.append(path)
    return out


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
        if color not in {"red", "green", "blue"}:
            continue
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
        m = cv2.inRange(hsv, np.array(r.low, dtype=np.uint8), np.array(r.high, dtype=np.uint8))
        acc = m if acc is None else cv2.bitwise_or(acc, m)
    if acc is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    return acc


def detect_in_image(img: np.ndarray, ranges_map: dict[str, list[HSVRange]], min_area: float = 1500.0, kernel_size: int = 5) -> list[Detection]:
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
            detections.append(Detection(color=color, bbox=(x, y, w, h)))
    return detections


def main() -> int:
    root = os.path.dirname(__file__)
    repo_root = os.path.join(root, "..")
    captures_dir = os.path.join(repo_root, "captures")
    classes_dir = os.path.join(captures_dir, "classes")

    dataset_dir = os.path.join(root, "path2_dataset")
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    ranges_map = derive_class_hsv_ranges(classes_dir, hue_margin=12, sat_floor=50, val_floor=45)
    if not ranges_map:
        print("Failed to derive class HSV ranges from captures/classes.")
        return 1

    class_to_id = {"red": 0, "green": 1, "blue": 2}
    images = [p for p in list_images(captures_dir, exts=(".jpg",)) if os.path.dirname(p) == captures_dir]

    total_boxes = 0
    per_class = {"red": 0, "green": 0, "blue": 0}
    labeled_images = 0

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        detections = detect_in_image(img, ranges_map)

        dst_img = os.path.join(images_dir, os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)

        label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        lines: list[str] = []
        for det in detections:
            if det.color not in class_to_id:
                continue
            x, y, bw, bh = det.bbox
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            cx_n = cx / w
            cy_n = cy / h
            bw_n = bw / w
            bh_n = bh / h
            lines.append(f"{class_to_id[det.color]} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")
            total_boxes += 1
            per_class[det.color] += 1

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        labeled_images += 1

    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: ./path2_dataset\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("nc: 3\n")
        f.write("names: ['red', 'green', 'blue']\n")

    lines = [
        f"Auto-labeled {labeled_images} images, {total_boxes} total bounding boxes",
        f"Per class: red={per_class['red']}, green={per_class['green']}, blue={per_class['blue']}",
    ]
    for line in lines:
        print(line)

    summary_path = os.path.join(dataset_dir, "autolabel_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

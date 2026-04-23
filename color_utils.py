#!/usr/bin/env python3
"""Shared color range helpers for Stage-A HSV detection pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HSVRange:
    low: tuple[int, int, int]
    high: tuple[int, int, int]


FALLBACK_COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 100, 70), (10, 255, 255)), ((170, 100, 70), (179, 255, 255))],
    "green": [((75, 60, 50), (105, 255, 255)),
    ],
    "blue": [((100, 80, 60), (130, 255, 255))],
}


def list_images(folder: str) -> list[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths: list[str] = []
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
    if not os.path.isdir(classes_dir):
        return out

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


def build_mask(
    hsv: np.ndarray,
    ranges: list[HSVRange] | list[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> np.ndarray:
    acc = None
    for r in ranges:
        if isinstance(r, HSVRange):
            low, high = r.low, r.high
        else:
            low, high = r
        m = cv2.inRange(
            hsv,
            np.array(low, dtype=np.uint8),
            np.array(high, dtype=np.uint8),
        )
        acc = m if acc is None else cv2.bitwise_or(acc, m)
    if acc is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    return acc


def _ranges_to_tuples(ranges_map: dict[str, list[HSVRange]]) -> dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    out: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {}
    for color, ranges in ranges_map.items():
        out[color] = [(r.low, r.high) for r in ranges]
    return out


def _print_tuple_ranges(title: str, ranges_map: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]]) -> None:
    print(title)
    for color in ("red", "green", "blue"):
        for i, (low, high) in enumerate(ranges_map.get(color, [])):
            print(f"  {color}[{i}] low={low} high={high}")


def load_color_ranges(
    classes_dir: str = "captures/classes",
    hue_margin: int = 12,
    sat_floor: int = 50,
    val_floor: int = 45,
) -> dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    derived = derive_class_hsv_ranges(classes_dir, hue_margin, sat_floor, val_floor)
    if derived:
        tuple_ranges = _ranges_to_tuples(derived)
        _print_tuple_ranges("Derived HSV ranges:", tuple_ranges)
        return tuple_ranges

    print(f"Classes directory missing or empty: {classes_dir}")
    print("Using fallback HSV ranges:")
    _print_tuple_ranges("Fallback HSV ranges:", FALLBACK_COLOR_RANGES)
    return {color: list(ranges) for color, ranges in FALLBACK_COLOR_RANGES.items()}

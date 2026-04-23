#!/usr/bin/env python3
"""Batch object-proposal-first + HSV ROI color classification."""

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
    parser = argparse.ArgumentParser(description="Run object-first ROI color classification on captures/.")
    parser.add_argument("--captures-dir", type=str, default="captures")
    parser.add_argument("--classes-dir", type=str, default="captures/classes")
    parser.add_argument("--out-dir", type=str, default="captures/hsv_results")
    parser.add_argument("--hue-margin", type=int, default=12)
    parser.add_argument("--sat-min-floor", type=int, default=50)
    parser.add_argument("--val-min-floor", type=int, default=45)
    parser.add_argument("--min-proposal-area", type=float, default=900.0)
    parser.add_argument("--max-proposal-area-ratio", type=float, default=0.28)
    parser.add_argument("--roi-margin", type=float, default=0.10)
    parser.add_argument("--proposal-padding", type=float, default=0.15)
    parser.add_argument("--min-circularity", type=float, default=0.08)
    parser.add_argument("--min-solidity", type=float, default=0.50)
    parser.add_argument("--min-color-ratio", type=float, default=0.07)
    parser.add_argument("--nms-iou", type=float, default=0.45)
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


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _nms_candidates(candidates: list[tuple[np.ndarray, float]], iou_th: float) -> list[np.ndarray]:
    ordered = sorted(candidates, key=lambda c: c[1], reverse=True)
    kept: list[np.ndarray] = []
    kept_boxes: list[tuple[int, int, int, int]] = []
    for cnt, score in ordered:
        _ = score
        x, y, w, h = cv2.boundingRect(cnt)
        box = (x, y, w, h)
        suppress = False
        for k in kept_boxes:
            if _bbox_iou(box, k) > iou_th:
                suppress = True
                break
        if not suppress:
            kept.append(cnt)
            kept_boxes.append(box)
    return kept


def _proposal_contours(
    img: np.ndarray,
    ranges_map: dict[str, list[HSVRange]],
    kernel_size: int,
    min_area: float,
    max_area_ratio: float,
    roi_margin: float,
    min_circularity: float,
    min_solidity: float,
    nms_iou: float,
) -> list[np.ndarray]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sat_blur = cv2.GaussianBlur(sat, (5, 5), 0)
    _, sat_bin = cv2.threshold(sat_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proposal_seed = cv2.bitwise_or(edges, sat_bin)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    proposal_seed = cv2.morphologyEx(proposal_seed, cv2.MORPH_CLOSE, kernel)
    proposal_seed = cv2.morphologyEx(proposal_seed, cv2.MORPH_OPEN, kernel)

    margin_x = int(roi_margin * w)
    margin_y = int(roi_margin * h)
    roi_mask = np.zeros_like(proposal_seed)
    roi_mask[margin_y : h - margin_y, margin_x : w - margin_x] = proposal_seed[
        margin_y : h - margin_y, margin_x : w - margin_x
    ]

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max_area_ratio * float(w * h)
    candidates: list[tuple[np.ndarray, float]] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue
        aspect = float(bw) / float(bh)
        if aspect < 0.25 or aspect > 4.0:
            continue

        per = cv2.arcLength(cnt, True)
        if per <= 1e-6:
            continue
        circularity = float((4.0 * np.pi * area) / (per * per))
        if circularity < min_circularity:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = (area / hull_area) if hull_area > 0 else 0.0
        if solidity < min_solidity:
            continue

        score = 0.50 * min(1.0, area / 5000.0) + 0.25 * solidity + 0.25 * min(1.0, circularity)
        candidates.append((cnt, score))

    # Hybrid proposals: include color-mask contours as additional proposals.
    for _color, ranges in ranges_map.items():
        c_mask = build_mask(hsv, ranges)
        c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel)
        c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, kernel)
        c_mask = cv2.bitwise_and(c_mask, roi_mask)
        color_contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in color_contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            aspect = float(bw) / float(bh)
            if aspect < 0.25 or aspect > 4.0:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = (area / hull_area) if hull_area > 0 else 0.0
            if solidity < (min_solidity * 0.9):
                continue

            per = cv2.arcLength(cnt, True)
            circularity = float((4.0 * np.pi * area) / (per * per)) if per > 1e-6 else 0.0
            ratio = float(np.count_nonzero(c_mask[y : y + bh, x : x + bw])) / float(bw * bh)
            score = 0.45 * ratio + 0.35 * min(1.0, area / 5000.0) + 0.20 * max(0.0, min(1.0, circularity))
            candidates.append((cnt, score))

    return _nms_candidates(candidates, nms_iou)


def detect_in_image(
    img: np.ndarray,
    ranges_map: dict[str, list[HSVRange]],
    min_proposal_area: float,
    max_proposal_area_ratio: float,
    roi_margin: float,
    proposal_padding: float,
    min_circularity: float,
    min_solidity: float,
    min_color_ratio: float,
    nms_iou: float,
    kernel_size: int,
) -> list[Detection]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    detections: list[Detection] = []

    proposals = _proposal_contours(
        img=img,
        ranges_map=ranges_map,
        kernel_size=kernel_size,
        min_area=min_proposal_area,
        max_area_ratio=max_proposal_area_ratio,
        roi_margin=roi_margin,
        min_circularity=min_circularity,
        min_solidity=min_solidity,
        nms_iou=nms_iou,
    )

    for cnt in proposals:
        area = float(cv2.contourArea(cnt))
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue

        pad_x = int(proposal_padding * w)
        pad_y = int(proposal_padding * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h + pad_y)
        roi_hsv = hsv[y1:y2, x1:x2]
        if roi_hsv.size == 0:
            continue
        roi_area = float(roi_hsv.shape[0] * roi_hsv.shape[1])
        if roi_area <= 0:
            continue

        best_color = "unknown"
        best_ratio = 0.0
        best_moments = None
        for color, ranges in ranges_map.items():
            mask = build_mask(roi_hsv, ranges)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            ratio = float(np.count_nonzero(mask)) / roi_area
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color
                best_moments = cv2.moments(mask)

        if best_ratio < min_color_ratio:
            continue

        if best_moments is not None and best_moments["m00"] > 1e-6:
            cx = float(x1 + (best_moments["m10"] / best_moments["m00"]))
            cy = float(y1 + (best_moments["m01"] / best_moments["m00"]))
        else:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = (area / hull_area) if hull_area > 0 else 0.0
        size_score = min(1.0, area / 5000.0)
        conf = max(0.0, min(1.0, 0.55 * best_ratio + 0.25 * solidity + 0.20 * size_score))

        detections.append(
            Detection(
                color=best_color,
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
            detections = detect_in_image(
                img=img,
                ranges_map=ranges_map,
                min_proposal_area=args.min_proposal_area,
                max_proposal_area_ratio=args.max_proposal_area_ratio,
                roi_margin=args.roi_margin,
                proposal_padding=args.proposal_padding,
                min_circularity=args.min_circularity,
                min_solidity=args.min_solidity,
                min_color_ratio=args.min_color_ratio,
                nms_iou=args.nms_iou,
                kernel_size=args.kernel_size,
            )
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

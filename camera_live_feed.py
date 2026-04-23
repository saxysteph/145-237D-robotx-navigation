#!/usr/bin/env python3
"""Stage-A RGB color detection pipeline for USB camera feeds."""

import argparse
import csv
import os
import platform
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from color_utils import build_mask, load_color_ranges


@dataclass
class Detection:
    color: str
    confidence: float
    cx_full: float
    cy_full: float
    radius_det: float
    bbox_full: tuple[int, int, int, int]


@dataclass
class Track:
    track_id: int
    color: str
    kf: cv2.KalmanFilter
    missed: int


COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {}
COLOR_DRAW = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 100, 0),
    "unknown": (255, 255, 255),
}


def make_kalman(init_x: float, init_y: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[init_x], [init_y], [0], [0]], np.float32)
    return kf


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    system = platform.system().lower()
    if system == "darwin":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def find_working_camera(max_index: int, width: int, height: int) -> int | None:
    for index in range(max_index + 1):
        cap = open_camera(index, width, height)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        cap.release()
        if ok:
            return index
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-A object-first + HSV color classification.")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--max-index", type=int, default=10)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--det-width", type=int, default=960)
    parser.add_argument("--det-height", type=int, default=540)
    parser.add_argument("--altitude-m", type=float, default=10.0)
    parser.add_argument("--fx-px", type=float, default=1500.0)
    parser.add_argument("--target-diameter-m", type=float, default=0.32)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--roi-margin", type=float, default=0.10)
    parser.add_argument("--min-circularity", type=float, default=0.35)
    parser.add_argument("--min-color-ratio", type=float, default=0.12)
    parser.add_argument("--track-gate-px", type=float, default=70.0)
    parser.add_argument("--max-track-missed", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="detection_logs")
    parser.add_argument("--calib-color", type=str, default="red", choices=["red", "green", "blue"])
    return parser.parse_args()


def find_detections(
    frame_full: np.ndarray,
    frame_det: np.ndarray,
    hsv_det: np.ndarray,
    hsv_full: np.ndarray,
    roi: tuple[int, int, int, int],
    args: argparse.Namespace,
) -> list[Detection]:
    detections: list[Detection] = []
    h_det, w_det = frame_det.shape[:2]
    x0, y0, x1, y1 = roi
    expected_d = args.fx_px * args.target_diameter_m / max(args.altitude_m, 0.1)
    min_d = 0.5 * expected_d
    max_d = 2.0 * expected_d
    kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
    scale_x = frame_full.shape[1] / float(w_det)
    scale_y = frame_full.shape[0] / float(h_det)

    # Stage 1: object detection (color-agnostic).
    gray = cv2.cvtColor(frame_det, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    mask_roi = np.zeros_like(edges)
    mask_roi[y0:y1, x0:x1] = edges[y0:y1, x0:x1]
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 1e-6:
            continue
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < args.min_circularity:
            continue

        (cx_det, cy_det), radius = cv2.minEnclosingCircle(cnt)
        diameter = 2.0 * radius
        if diameter < min_d or diameter > max_d:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue

        x_f = int(max(0, x * scale_x))
        y_f = int(max(0, y * scale_y))
        w_f = int(min(frame_full.shape[1] - x_f, w * scale_x))
        h_f = int(min(frame_full.shape[0] - y_f, h * scale_y))
        if w_f <= 0 or h_f <= 0:
            continue

        hsv_roi = hsv_full[y_f : y_f + h_f, x_f : x_f + w_f]
        roi_area = float(hsv_roi.shape[0] * hsv_roi.shape[1])
        if roi_area <= 0:
            continue

        # Stage 2: color thresholding/classification inside object ROI only.
        best_color = "unknown"
        best_ratio = 0.0
        best_moments = None
        for color, ranges in COLOR_RANGES.items():
            c_mask = build_mask(hsv_roi, ranges)
            c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel)
            c_mask = cv2.morphologyEx(c_mask, cv2.MORPH_CLOSE, kernel)
            ratio = float(np.count_nonzero(c_mask)) / roi_area
            if ratio > best_ratio:
                best_ratio = ratio
                best_color = color
                best_moments = cv2.moments(c_mask)

        if best_ratio < args.min_color_ratio:
            continue

        if best_moments is not None and best_moments["m00"] > 0:
            cx_full = x_f + (best_moments["m10"] / best_moments["m00"])
            cy_full = y_f + (best_moments["m01"] / best_moments["m00"])
        else:
            cx_full = cx_det * scale_x
            cy_full = cy_det * scale_y

        size_term = max(0.0, 1.0 - abs(diameter - expected_d) / max(expected_d, 1e-6))
        conf = float(0.45 * size_term + 0.55 * best_ratio)
        detections.append(
            Detection(
                color=best_color,
                confidence=max(0.0, min(1.0, conf)),
                cx_full=float(cx_full),
                cy_full=float(cy_full),
                radius_det=radius,
                bbox_full=(x_f, y_f, w_f, h_f),
            )
        )
    return detections


def update_tracks(
    tracks: list[Track], detections: list[Detection], gate_px: float, max_missed: int, next_track_id: int
) -> tuple[list[Track], list[tuple[Detection, int, tuple[float, float]]], int]:
    assigned = []
    used_dets = set()

    for track in tracks:
        pred = track.kf.predict()
        px, py = float(pred[0, 0]), float(pred[1, 0])
        best_idx = None
        best_dist = float("inf")
        for i, det in enumerate(detections):
            if i in used_dets or det.color != track.color:
                continue
            dist = ((det.cx_full - px) ** 2 + (det.cy_full - py) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist <= gate_px:
            det = detections[best_idx]
            meas = np.array([[det.cx_full], [det.cy_full]], np.float32)
            corr = track.kf.correct(meas)
            track.missed = 0
            used_dets.add(best_idx)
            assigned.append((det, track.track_id, (float(corr[0, 0]), float(corr[1, 0]))))
        else:
            track.missed += 1

    tracks = [t for t in tracks if t.missed <= max_missed]

    for i, det in enumerate(detections):
        if i in used_dets:
            continue
        kf = make_kalman(det.cx_full, det.cy_full)
        track = Track(track_id=next_track_id, color=det.color, kf=kf, missed=0)
        tracks.append(track)
        assigned.append((det, track.track_id, (det.cx_full, det.cy_full)))
        next_track_id += 1
    return tracks, assigned, next_track_id


def apply_clahe_to_v(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def calibrate_sv_threshold(frame_det_bgr: np.ndarray, calib_color: str) -> None:
    hsv = cv2.cvtColor(frame_det_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    patch = hsv[h // 2 - 20 : h // 2 + 20, w // 2 - 20 : w // 2 + 20]
    if patch.size == 0:
        return
    s_med = int(np.median(patch[:, :, 1]))
    v_med = int(np.median(patch[:, :, 2]))
    updated = []
    for low, high in COLOR_RANGES[calib_color]:
        new_low = (low[0], max(0, min(low[1], s_med - 20)), max(0, min(low[2], v_med - 20)))
        updated.append((new_low, high))
    COLOR_RANGES[calib_color] = updated
    print(f"Calibrated {calib_color}: S_min={updated[0][0][1]}, V_min={updated[0][0][2]}")


def main() -> int:
    global COLOR_RANGES
    args = parse_args()
    COLOR_RANGES = load_color_ranges(classes_dir="captures/classes")
    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, "detections.csv")
    csv_exists = os.path.exists(csv_path)

    if args.camera_index is None:
        print(f"Probing camera indices 0..{args.max_index}...")
        camera_index = find_working_camera(args.max_index, args.width, args.height)
        if camera_index is None:
            print("No working camera stream found.")
            print("On macOS, grant camera permission to your terminal/IDE and retry.")
            return 1
        print(f"Using camera index: {camera_index}")
    else:
        camera_index = args.camera_index

    cap = open_camera(camera_index, args.width, args.height)
    if not cap.isOpened():
        print(f"Failed to open camera index {camera_index}.")
        return 1

    tracks = []
    next_track_id = 1
    window_name = f"Stage-A RGB Detection (camera {camera_index})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(
                ["timestamp", "image_path", "track_id", "color", "confidence", "cx", "cy", "x", "y", "w", "h"]
            )

        while True:
            ok, frame_full = cap.read()
            if not ok:
                print("Frame read failed. Camera may have disconnected.")
                break

            frame_full = apply_clahe_to_v(frame_full)
            frame_det = cv2.resize(frame_full, (args.det_width, args.det_height), interpolation=cv2.INTER_AREA)
            hsv_det = cv2.cvtColor(frame_det, cv2.COLOR_BGR2HSV)
            hsv_full = cv2.cvtColor(frame_full, cv2.COLOR_BGR2HSV)

            margin_x = int(args.roi_margin * args.det_width)
            margin_y = int(args.roi_margin * args.det_height)
            roi = (margin_x, margin_y, args.det_width - margin_x, args.det_height - margin_y)

            detections = find_detections(frame_full, frame_det, hsv_det, hsv_full, roi, args)
            tracks, assigned, next_track_id = update_tracks(
                tracks, detections, args.track_gate_px, args.max_track_missed, next_track_id
            )

            frame_out = frame_full.copy()
            image_path = ""
            if assigned:
                ts = time.time()
                image_path = os.path.join(args.log_dir, f"frame_{int(ts * 1000)}.jpg")
                cv2.imwrite(image_path, frame_full)

            for det, track_id, (sx, sy) in assigned:
                x, y, w, h = det.bbox_full
                color_bgr = COLOR_DRAW[det.color]
                cv2.rectangle(frame_out, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.circle(frame_out, (int(sx), int(sy)), 4, color_bgr, -1)
                label = f"{det.color} t{track_id} conf={det.confidence:.2f}"
                cv2.putText(frame_out, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
                writer.writerow(
                    [
                        f"{time.time():.3f}",
                        image_path,
                        track_id,
                        det.color,
                        f"{det.confidence:.4f}",
                        f"{sx:.2f}",
                        f"{sy:.2f}",
                        x,
                        y,
                        w,
                        h,
                    ]
                )
            f.flush()

            cv2.imshow(window_name, frame_out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calibrate_sv_threshold(frame_det, args.calib_color)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

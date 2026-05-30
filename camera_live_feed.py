#!/usr/bin/env python3
"""Stage-A RGB color detection pipeline for USB camera feeds."""
from __future__ import annotations

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


def meters_to_latlon(lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Local tangent-plane EN offsets -> WGS84 lat/lon approximation."""
    earth_r = 6378137.0
    d_lat = (north_m / earth_r) * (180.0 / np.pi)
    cos_lat = float(np.cos(np.deg2rad(lat_deg)))
    cos_lat = max(1e-6, abs(cos_lat))
    d_lon = (east_m / (earth_r * cos_lat)) * (180.0 / np.pi)
    return lat_deg + d_lat, lon_deg + d_lon


def project_pixel_to_ground_ned(
    u: float,
    v: float,
    width: int,
    height: int,
    altitude_m: float,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    heading_deg: float,
) -> tuple[float, float]:
    """
    Project image pixel to local N/E offsets (meters) under nadir-camera assumption.
    Assumptions:
      - Camera optical axis points straight down (nadir).
      - Ground/water surface is a flat plane at known altitude_m below camera.
      - heading_deg is yaw clockwise from North.
    """
    if fx_px <= 1e-6 or fy_px <= 1e-6:
        return 0.0, 0.0
    if altitude_m <= 1e-6:
        return 0.0, 0.0

    # Camera-local ground offsets before heading rotation.
    # Image center is (cx, cy), +x right, +y down.
    east_cam = ((u - cx_px) / fx_px) * altitude_m
    north_cam = ((cy_px - v) / fy_px) * altitude_m

    yaw = np.deg2rad(heading_deg)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    north = c * north_cam - s * east_cam
    east = s * north_cam + c * east_cam
    return north, east


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
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        # On Linux/Jetson let the driver negotiate format — don't force MJPG
        # or a specific resolution, as H264 cameras will reject it.
        cap = cv2.VideoCapture(camera_index)
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
    parser.add_argument("--fy-px", type=float, default=0.0, help="If <= 0, fallback to --fx-px.")
    parser.add_argument("--cx-px", type=float, default=0.0, help="If <= 0, fallback to image center.")
    parser.add_argument("--cy-px", type=float, default=0.0, help="If <= 0, fallback to image center.")
    parser.add_argument("--target-diameter-m", type=float, default=0.32)
    parser.add_argument("--drone-lat", type=float, default=None, help="Drone camera GPS latitude in degrees.")
    parser.add_argument("--drone-lon", type=float, default=None, help="Drone camera GPS longitude in degrees.")
    parser.add_argument("--heading-deg", type=float, default=0.0, help="Drone yaw clockwise from North.")
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--roi-margin", type=float, default=0.10)
    parser.add_argument("--min-circularity", type=float, default=0.35)
    parser.add_argument("--min-color-ratio", type=float, default=0.12)
    parser.add_argument("--track-gate-px", type=float, default=70.0)
    parser.add_argument("--max-track-missed", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="detection_logs")
    parser.add_argument("--calib-color", type=str, default="red", choices=["red", "green", "blue"])
    # Jetson / headless operation
    parser.add_argument("--headless", action="store_true", help="Disable all cv2.imshow windows (for Jetson/SSH).")
    parser.add_argument("--save-video", action="store_true", help="Save raw camera feed to a timestamped .avi in --log-dir.")
    parser.add_argument("--gcs-ip", type=str, default=None, help="Ground station IP for MAVLink UDP (e.g. 192.168.55.100). Enables live MAVLink transmission.")
    parser.add_argument("--gcs-port", type=int, default=14555, help="Ground station UDP port (default: 14555).")
    # YOLO
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="Path to YOLO .pt model file.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--no-yolo", action="store_true", help="Fall back to edge-detection proposals (no YOLO).")
    return parser.parse_args()


def _get_proposals_yolo(
    frame_det: np.ndarray,
    frame_full: np.ndarray,
    yolo_model,
    conf_thresh: float,
) -> list[tuple[int, int, int, int, float, str]]:
    """Run YOLO on full-res frame, return boxes in full-res pixel coords.
    Returns list of (x, y, w, h, yolo_conf, yolo_class).
    """
    results = yolo_model(frame_full, conf=conf_thresh, verbose=False)
    CLASS_NAMES = {0: "red", 1: "green", 2: "blue"}
    proposals = []
    h_f, w_f_frame = frame_full.shape[:2]
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            yolo_conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            yolo_class = CLASS_NAMES.get(cls_idx, "unknown")
            x = int(max(0, x1))
            y = int(max(0, y1))
            w = int(min(w_f_frame - x, x2 - x1))
            h = int(min(h_f - y, y2 - y1))
            if w > 4 and h > 4:
                proposals.append((x, y, w, h, yolo_conf, yolo_class))
    return proposals


def _get_proposals_edges(
    frame_det: np.ndarray,
    frame_full: np.ndarray,
    roi: tuple[int, int, int, int],
    args: argparse.Namespace,
) -> list[tuple[int, int, int, int, float]]:
    """Fallback edge+contour proposals. Returns (x, y, w, h, pseudo_conf)."""
    h_det, w_det = frame_det.shape[:2]
    x0, y0, x1, y1 = roi
    expected_d = args.fx_px * args.target_diameter_m / max(args.altitude_m, 0.1)
    min_d = 0.5 * expected_d
    max_d = 2.0 * expected_d
    kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
    scale_x = frame_full.shape[1] / float(w_det)
    scale_y = frame_full.shape[0] / float(h_det)

    gray = cv2.cvtColor(frame_det, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    mask_roi = np.zeros_like(edges)
    mask_roi[y0:y1, x0:x1] = edges[y0:y1, x0:x1]
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals = []
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
        (_, _), radius = cv2.minEnclosingCircle(cnt)
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
        if w_f > 0 and h_f > 0:
            proposals.append((x_f, y_f, w_f, h_f, 0.5))
    return proposals


def find_detections(
    frame_full: np.ndarray,
    frame_det: np.ndarray,
    hsv_det: np.ndarray,
    hsv_full: np.ndarray,
    roi: tuple[int, int, int, int],
    args: argparse.Namespace,
    yolo_model=None,
) -> list[Detection]:
    detections: list[Detection] = []
    kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
    expected_d = args.fx_px * args.target_diameter_m / max(args.altitude_m, 0.1)

    # Stage 1: object proposals — YOLO finds buoy ROIs, edges as fallback
    if yolo_model is not None:
        raw_proposals = _get_proposals_yolo(frame_det, frame_full, yolo_model, args.yolo_conf)
        # Strip YOLO class label — HSV does all color classification
        proposals = [(x, y, w, h, conf) for x, y, w, h, conf, _ in raw_proposals]
    else:
        proposals = _get_proposals_edges(frame_det, frame_full, roi, args)

    # Stage 2: HSV color classification inside each YOLO ROI
    for x_f, y_f, w_f, h_f, proposal_conf in proposals:
        hsv_roi = hsv_full[y_f : y_f + h_f, x_f : x_f + w_f]
        roi_area = float(hsv_roi.shape[0] * hsv_roi.shape[1])
        if roi_area <= 0:
            continue

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
            cx_full = x_f + w_f * 0.5
            cy_full = y_f + h_f * 0.5

        diameter = (w_f + h_f) * 0.5
        size_term = max(0.0, 1.0 - abs(diameter - expected_d) / max(expected_d, 1e-6))
        conf = float(0.3 * proposal_conf + 0.3 * size_term + 0.4 * best_ratio)
        detections.append(
            Detection(
                color=best_color,
                confidence=max(0.0, min(1.0, conf)),
                cx_full=float(cx_full),
                cy_full=float(cy_full),
                radius_det=diameter * 0.5,
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
    # Sanity-check derived ranges: if all three hue centers are in the same
    # yellow-green band (H 37-73), the reference images are bad — use fallback.
    from color_utils import FALLBACK_COLOR_RANGES
    hue_centers = []
    for color in ("red", "green", "blue"):
        ranges = COLOR_RANGES.get(color, [])
        if ranges:
            hue_centers.append((ranges[0][0][0] + ranges[0][1][0]) / 2)
    if hue_centers and max(hue_centers) - min(hue_centers) < 20:
        print("[WARN] Derived HSV ranges look degenerate (all similar hue) — using fallback ranges.")
        COLOR_RANGES = {color: list(ranges) for color, ranges in FALLBACK_COLOR_RANGES.items()}
        from color_utils import _print_tuple_ranges
        _print_tuple_ranges("Fallback HSV ranges:", COLOR_RANGES)
    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, "detections.csv")
    csv_exists = os.path.exists(csv_path)

    # YOLO model
    yolo_model = None
    if not args.no_yolo:
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(args.yolo_model)
            # Warm-up inference so first frame isn't slow
            dummy = np.zeros((args.det_height, args.det_width, 3), dtype=np.uint8)
            yolo_model(dummy, conf=args.yolo_conf, verbose=False)
            print(f"YOLO loaded: {args.yolo_model}")
        except Exception as e:
            print(f"[WARN] YOLO unavailable ({e}); falling back to edge detection.")
            yolo_model = None

    # MAVLink transmitter (only when --gcs-ip is provided)
    mavlink_tx = None
    if args.gcs_ip:
        try:
            import sys as _sys
            _repo_root = os.path.dirname(os.path.abspath(__file__))
            if _repo_root not in _sys.path:
                _sys.path.insert(0, _repo_root)
            from mavlink_comms.transmitter import BuoyMavlinkTransmitter
            connection = f"udpout:{args.gcs_ip}:{args.gcs_port}"
            mavlink_tx = BuoyMavlinkTransmitter(connection=connection)
            print(f"MAVLink transmitter → {connection}")
        except Exception as e:
            print(f"[WARN] MAVLink init failed ({e}); continuing without telemetry.")

    if args.camera_index is None:
        print(f"Probing camera indices 0..{args.max_index}...")
        camera_index = find_working_camera(args.max_index, args.width, args.height)
        if camera_index is None:
            print("No working camera stream found.")
            return 1
        print(f"Using camera index: {camera_index}")
    else:
        camera_index = args.camera_index

    cap = open_camera(camera_index, args.width, args.height)
    if not cap.isOpened():
        print(f"Failed to open camera index {camera_index}.")
        return 1

    # Video recorder
    video_writer = None
    if args.save_video:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_path = os.path.join(args.log_dir, f"recording_{int(time.time())}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(vid_path, fourcc, actual_fps, (actual_w, actual_h))
        print(f"Recording to {vid_path} ({actual_w}x{actual_h} @ {actual_fps:.0f}fps)")

    tracks = []
    next_track_id = 1
    window_name = f"Stage-A RGB Detection (camera {camera_index})"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(
                [
                    "timestamp",
                    "image_path",
                    "track_id",
                    "color",
                    "confidence",
                    "cx",
                    "cy",
                    "x",
                    "y",
                    "w",
                    "h",
                    "north_m",
                    "east_m",
                    "est_lat",
                    "est_lon",
                ]
            )

        frame_counter = 0
        while True:
            ok, frame_full = cap.read()
            if not ok:
                print("Frame read failed. Camera may have disconnected.")
                break

            frame_counter += 1
            if video_writer is not None:
                video_writer.write(frame_full)
            frame_full = apply_clahe_to_v(frame_full)
            frame_det = cv2.resize(frame_full, (args.det_width, args.det_height), interpolation=cv2.INTER_AREA)
            hsv_det = cv2.cvtColor(frame_det, cv2.COLOR_BGR2HSV)
            hsv_full = cv2.cvtColor(frame_full, cv2.COLOR_BGR2HSV)

            margin_x = int(args.roi_margin * args.det_width)
            margin_y = int(args.roi_margin * args.det_height)
            roi = (margin_x, margin_y, args.det_width - margin_x, args.det_height - margin_y)

            detections = find_detections(frame_full, frame_det, hsv_det, hsv_full, roi, args, yolo_model)
            tracks, assigned, next_track_id = update_tracks(
                tracks, detections, args.track_gate_px, args.max_track_missed, next_track_id
            )

            frame_out = frame_full.copy() if not args.headless else None
            image_path = ""
            if assigned:
                ts = time.time()
                image_path = os.path.join(args.log_dir, f"frame_{int(ts * 1000)}.jpg")
                cv2.imwrite(image_path, frame_full)

            for det, track_id, (sx, sy) in assigned:
                x, y, w, h = det.bbox_full
                color_bgr = COLOR_DRAW[det.color]
                fy_px = args.fy_px if args.fy_px > 0 else args.fx_px
                cx_px = args.cx_px if args.cx_px > 0 else (frame_full.shape[1] * 0.5)
                cy_px = args.cy_px if args.cy_px > 0 else (frame_full.shape[0] * 0.5)
                north_m, east_m = project_pixel_to_ground_ned(
                    u=float(sx),
                    v=float(sy),
                    width=frame_full.shape[1],
                    height=frame_full.shape[0],
                    altitude_m=args.altitude_m,
                    fx_px=args.fx_px,
                    fy_px=fy_px,
                    cx_px=cx_px,
                    cy_px=cy_px,
                    heading_deg=args.heading_deg,
                )
                est_lat = ""
                est_lon = ""
                if args.drone_lat is not None and args.drone_lon is not None:
                    est_lat_v, est_lon_v = meters_to_latlon(args.drone_lat, args.drone_lon, north_m, east_m)
                    est_lat = f"{est_lat_v:.8f}"
                    est_lon = f"{est_lon_v:.8f}"

                if not args.headless and frame_out is not None:
                    cv2.rectangle(frame_out, (x, y), (x + w, y + h), color_bgr, 2)
                    cv2.circle(frame_out, (int(sx), int(sy)), 4, color_bgr, -1)
                    label = f"{det.color} t{track_id} conf={det.confidence:.2f}"
                    cv2.putText(frame_out, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 2)
                    if est_lat and est_lon:
                        cv2.putText(
                            frame_out,
                            f"lat={est_lat} lon={est_lon}",
                            (x, min(frame_out.shape[0] - 10, y + h + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color_bgr,
                            1,
                            cv2.LINE_AA,
                        )

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
                        f"{north_m:.3f}",
                        f"{east_m:.3f}",
                        est_lat,
                        est_lon,
                    ]
                )

                # Transmit over MAVLink if GCS is configured and GPS is available
                if mavlink_tx and est_lat and est_lon:
                    try:
                        mavlink_tx.transmit(
                            target_id=track_id,
                            color=det.color,
                            lat=float(est_lat),
                            lon=float(est_lon),
                            frame=frame_counter,
                        )
                        print(f"[TX] t{track_id} {det.color} lat={est_lat} lon={est_lon}", flush=True)
                    except Exception as e:
                        print(f"[TX ERROR] {e}", flush=True)
                else:
                    print(
                        f"[TRANSMIT] {{'target_id': {track_id}, 'color': '{det.color}', "
                        f"'north_m': {north_m:.3f}, 'east_m': {east_m:.3f}, "
                        f"'est_lat': '{est_lat}', 'est_lon': '{est_lon}'}}",
                        flush=True,
                    )

            f.flush()

            if not args.headless and frame_out is not None:
                cv2.imshow(window_name, frame_out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    calibrate_sv_threshold(frame_det, args.calib_color)

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to {vid_path}")
    if not args.headless:
        cv2.destroyAllWindows()
    if mavlink_tx:
        mavlink_tx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

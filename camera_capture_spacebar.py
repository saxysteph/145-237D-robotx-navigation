#!/usr/bin/env python3
"""Capture photos from a USB camera when spacebar is pressed."""

import argparse
import os
import platform
import sys
import time

import cv2


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open camera with a backend suitable for host OS."""
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
    """Probe camera indices and return first index producing valid frame."""
    for idx in range(max_index + 1):
        cap = open_camera(idx, width, height)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        cap.release()
        if ok:
            return idx
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Press spacebar to save photos from camera.")
    parser.add_argument("--camera-index", type=int, default=None, help="Camera index to open.")
    parser.add_argument("--max-index", type=int, default=10, help="Max index to probe if auto mode.")
    parser.add_argument("--width", type=int, default=1920, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=1080, help="Requested capture height.")
    parser.add_argument("--output-dir", type=str, default="captures", help="Directory for saved photos.")
    parser.add_argument("--prefix", type=str, default="capture", help="Filename prefix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.camera_index is None:
        print(f"Probing camera indices 0..{args.max_index}...")
        camera_index = find_working_camera(args.max_index, args.width, args.height)
        if camera_index is None:
            print("No working camera stream found.")
            print("On macOS, grant Camera permission to your terminal/IDE and retry.")
            return 1
        print(f"Using camera index: {camera_index}")
    else:
        camera_index = args.camera_index

    cap = open_camera(camera_index, args.width, args.height)
    if not cap.isOpened():
        print(f"Failed to open camera index {camera_index}.")
        return 1

    window_name = f"Camera Capture (index {camera_index})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Spacebar: save photo | q: quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed. Camera may have disconnected.")
            break

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:  # spacebar
            timestamp_ms = int(time.time() * 1000)
            filename = f"{args.prefix}_{timestamp_ms}.jpg"
            path = os.path.join(args.output_dir, filename)
            if cv2.imwrite(path, frame):
                print(f"Saved: {path}")
            else:
                print(f"Failed to save: {path}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

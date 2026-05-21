#!/usr/bin/env python3
"""Load fx, fy, cx, cy (and image size) from calibration/camera_intrinsics_latest.json — shared by sim overlay + YOLO labels + eval."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def load_intrinsics_matrix(path: Path) -> tuple[float, float, float, float, int, int]:
    """Return fx, fy, cx, cy, image_width, image_height."""
    data = json.loads(path.read_text(encoding="utf-8"))
    k = data.get("K")
    if isinstance(k, list) and len(k) >= 2:
        fx = float(k[0][0])
        fy = float(k[1][1])
        cx = float(k[0][2])
        cy = float(k[1][2])
    else:
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data["cx"])
        cy = float(data["cy"])
    w = int(data.get("image_width", 1920))
    h = int(data.get("image_height", 1080))
    return fx, fy, cx, cy, w, h


def default_calibration_json(repo_root: Path) -> Path:
    return repo_root / "calibration" / "camera_intrinsics_latest.json"


def pinhole_intrinsics_from_horizontal_fov(
    hfov_rad: float, image_w: int, image_h: int
) -> tuple[float, float, float, float]:
    """
    Standard pinhole from horizontal FOV + resolution (matches typical Gazebo camera math).

    Vertical focal length fy is derived from the aspect ratio so vfov is consistent with hfov.
    Principal point is the image center.
    """
    fx = (0.5 * image_w) / math.tan(0.5 * hfov_rad)
    vfov = 2.0 * math.atan(math.tan(0.5 * hfov_rad) * (image_h / image_w))
    fy = (0.5 * image_h) / math.tan(0.5 * vfov)
    cx = 0.5 * image_w
    cy = 0.5 * image_h
    return fx, fy, cx, cy


def sim_nadir_camera_from_world_sdf(
    world_sdf: Path, image_w: int, image_h: int
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Parse Gazebo world SDF for `uav_dataset_camera` and first camera sensor.

    Returns:
        fx, fy, cx, cy — pinhole intrinsics consistent with sim render size /
            scaled to actual JPEG dimensions if they differ from SDF.
        drone_x_m, drone_y_m, drone_z_m, yaw_deg — static camera pose for nadir projection
        when pose CSV rows omit drone pose (gz-transport recorder).

    Physical checkerboard calibration (`camera_intrinsics_latest.json`) does **not** match
    synthetic Gazebo images; use this for sim labeling / overlay.
    """
    hfov = 1.05
    sdf_w, sdf_h = image_w, image_h

    try:
        root = ET.parse(world_sdf).getroot()
    except (ET.ParseError, OSError):
        fx, fy, cx, cy = pinhole_intrinsics_from_horizontal_fov(hfov, image_w, image_h)
        return fx, fy, cx, cy, 0.0, 0.0, 26.0, 0.0

    for cam in root.findall(".//camera"):
        hfov_el = cam.find("horizontal_fov")
        if hfov_el is not None and (hfov_el.text or "").strip():
            try:
                hfov = float(hfov_el.text.strip())
            except ValueError:
                pass
        im = cam.find("image")
        if im is not None:
            w_el = im.find("width")
            h_el = im.find("height")
            if w_el is not None and (w_el.text or "").strip():
                try:
                    sdf_w = int(float(w_el.text.strip()))
                except ValueError:
                    pass
            if h_el is not None and (h_el.text or "").strip():
                try:
                    sdf_h = int(float(h_el.text.strip()))
                except ValueError:
                    pass
        break

    if sdf_w <= 0:
        sdf_w = image_w
    if sdf_h <= 0:
        sdf_h = image_h

    fx, fy, cx, cy = pinhole_intrinsics_from_horizontal_fov(hfov, sdf_w, sdf_h)
    if sdf_w != image_w or sdf_h != image_h:
        sx = image_w / float(sdf_w)
        sy = image_h / float(sdf_h)
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy

    dx, dy, dz, dyaw = 0.0, 0.0, 26.0, 0.0
    for model in root.findall(".//model"):
        name = model.get("name") or ""
        if "uav_dataset_camera" not in name:
            continue
        pose_text = (model.findtext("pose") or "").strip()
        if pose_text:
            parts = pose_text.split()
            if len(parts) >= 3:
                try:
                    dx, dy, dz = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    pass
            if len(parts) >= 6:
                try:
                    dyaw = math.degrees(float(parts[5]))
                except ValueError:
                    pass
        break

    return fx, fy, cx, cy, dx, dy, dz, dyaw


def resolve_pinhole_intrinsics(
    *,
    intrinsics_path: Path | None,
    fx_override: float | None,
    fy_override: float | None,
    cx_override: float | None,
    cy_override: float | None,
    image_w: int,
    image_h: int,
) -> tuple[float, float, float, float]:
    """
    Pinhole intrinsics for nadir projection scripts.

    Order: calibration JSON if present, then per-parameter overrides (non-None),
    else legacy defaults (1500 / image center).
    cx/cy overrides apply only when > 0 (matches older scripts).
    """
    fx = fy = 1500.0
    cx = 0.5 * float(image_w)
    cy = 0.5 * float(image_h)
    if intrinsics_path is not None and intrinsics_path.is_file():
        fx, fy, tcx, tcy, _, _ = load_intrinsics_matrix(intrinsics_path)
        cx, cy = tcx, tcy
    if fx_override is not None:
        fx = fx_override
    if fy_override is not None:
        fy = fy_override
    if cx_override is not None:
        cx = cx_override if cx_override > 0 else (0.5 * float(image_w))
    if cy_override is not None:
        cy = cy_override if cy_override > 0 else (0.5 * float(image_h))
    return fx, fy, cx, cy

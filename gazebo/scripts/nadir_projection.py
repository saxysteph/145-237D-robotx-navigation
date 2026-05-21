#!/usr/bin/env python3
"""
Flat-earth nadir camera: east/north ground deltas ↔ image pixels.

This is the single projection convention shared by:
  - ``reproject_detection_map.project_pixel_to_ground_ned`` (inverse path)
  - ``render_gt_in_image_from_pose`` / ``project_buoys_to_yolo_labels`` (forward path)

Normalized YOLO boxes drawn by ``train_roi_then_hsv.yolo_line_to_box`` multiply raw image
width/height — consistent with pixel coordinates produced here.

World frame (RobotX Gazebo worlds): +X ≈ east, +Y ≈ north, +Z up. Ground plane Z≈0;
``altitude_m`` is camera height above that plane (positive).

Heading/yaw: degrees; identical rotation semantics as the path2 reprojection demos.

Rendering mismatch (esp. Gazebo Harmonic / ogre2): sensors attach an ``Ogre2Camera`` that applies a
fixed yaw(-90°) then roll(-90°) to match classic +X-forward convention—your labels can look **mirrored**
or **rotated 90°** vs this naive flat model. Use ``mirror_*``, ``transpose_en_pixel``, or
``extra_yaw_deg`` on ``project_buoys_to_yolo_labels.py`` to compensate when reviewing ``annotated_gt``.
"""

from __future__ import annotations

import numpy as np


def project_pixel_to_ground_ned(
    u: float,
    v: float,
    altitude_m: float,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    heading_deg: float,
    *,
    extra_yaw_deg: float = 0.0,
    mirror_east_cam: bool = False,
    mirror_north_cam: bool = False,
    transpose_en_pixel: bool = False,
) -> tuple[float, float]:
    """Pixel (u,v) → delta north / delta east (meters) on the ground plane."""
    if fx_px <= 1e-6 or fy_px <= 1e-6 or altitude_m <= 1e-6:
        return 0.0, 0.0
    if transpose_en_pixel:
        north_cam = ((u - cx_px) / fx_px) * altitude_m
        east_cam = ((cy_px - v) / fy_px) * altitude_m
    else:
        east_cam = ((u - cx_px) / fx_px) * altitude_m
        north_cam = ((cy_px - v) / fy_px) * altitude_m
    if mirror_east_cam:
        east_cam = -east_cam
    if mirror_north_cam:
        north_cam = -north_cam
    yaw = np.deg2rad(heading_deg + extra_yaw_deg)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    north = c * north_cam - s * east_cam
    east = s * north_cam + c * east_cam
    return north, east


def ground_delta_en_to_pixel(
    d_east_m: float,
    d_north_m: float,
    altitude_m: float,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    heading_deg: float,
    *,
    extra_yaw_deg: float = 0.0,
    mirror_east_cam: bool = False,
    mirror_north_cam: bool = False,
    transpose_en_pixel: bool = False,
) -> tuple[float, float]:
    """Delta east / delta north (meters) from camera ground reference → pixel (u, v)."""
    h = max(1e-6, altitude_m)
    yaw = np.deg2rad(heading_deg + extra_yaw_deg)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    north_cam = c * d_north_m + s * d_east_m
    east_cam = -s * d_north_m + c * d_east_m
    if mirror_east_cam:
        east_cam = -east_cam
    if mirror_north_cam:
        north_cam = -north_cam
    if transpose_en_pixel:
        u = cx_px + fx_px * (north_cam / h)
        v = cy_px - fy_px * (east_cam / h)
    else:
        u = cx_px + fx_px * (east_cam / h)
        v = cy_px - fy_px * (north_cam / h)
    return u, v

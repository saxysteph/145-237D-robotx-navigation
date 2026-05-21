#!/usr/bin/env python3
"""
Generate RobotX gazebo world variants for domain randomization.

Vary:
- sun / glare settings
- wave model choice
- buoy field: stochastic red/green pair count (layout-biased), 0–2 blue markers per world, corridor
  angles, per-buoy roll/pitch/yaw on includes, plus global rotate/translate/scale and FOV clamp
- non-buoy clutter: static inline boxes/cylinders spaced away from buoy XY (``--clutter-count``)
- overhead UAV camera: non-static model + ``robotx_overhead_motion_plugin`` elliptical path (``--no-camera-motion`` to disable)

Optional interactive walkthrough prints each scenario and waits for Enter
so you can launch one-by-one during dataset collection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple

# Nadir camera z≈26 m, hfov≈1.05 → ~15 m horizontal half-span; keep buoys inside ~14 m
# so domain-randomized translations do not push gate markers off-frame (empty YOLO boxes).
_MAX_XY_NORM_M = 13.8


class SurfPose(NamedTuple):
    """Ground-plane XY + roll/pitch/yaw (rad) for SDF poses (z handled separately)."""

    x: float
    y: float
    roll: float
    pitch: float
    yaw: float


GLARE_PROFILES = [
    {
        "name": "clear_blue_mild",
        "ambient": "0.20 0.20 0.20 1.0",
        "background": "0.58 0.70 0.88 1.0",
        "diffuse": "0.78 0.78 0.78 1.0",
        "specular": "0.45 0.45 0.45 1.0",
        "direction": "0.20 -0.08 -1.0",
    },
    {
        "name": "clear_blue_harsh",
        "ambient": "0.27 0.27 0.27 1.0",
        "background": "0.62 0.75 0.92 1.0",
        "diffuse": "1.00 0.98 0.94 1.0",
        "specular": "1.00 1.00 1.00 1.0",
        "direction": "0.26 -0.12 -1.0",
    },
    {
        "name": "sunset_glint",
        "ambient": "0.22 0.20 0.18 1.0",
        "background": "0.74 0.55 0.44 1.0",
        "diffuse": "0.95 0.74 0.56 1.0",
        "specular": "1.00 0.87 0.71 1.0",
        "direction": "-0.40 0.20 -0.88",
    },
    {
        # Greenish / murkier water appearance from reflected sky + reduced specular
        "name": "greenish_murky",
        "ambient": "0.18 0.22 0.18 1.0",
        "background": "0.35 0.55 0.40 1.0",
        "diffuse": "0.55 0.75 0.60 1.0",
        "specular": "0.28 0.45 0.32 1.0",
        "direction": "0.20 -0.06 -1.0",
    },
    {
        # Dim / low-visibility water: darker reflections, weaker highlights
        "name": "dark_overcast",
        "ambient": "0.14 0.14 0.14 1.0",
        "background": "0.32 0.32 0.38 1.0",
        "diffuse": "0.35 0.35 0.35 1.0",
        "specular": "0.12 0.12 0.12 1.0",
        "direction": "0.12 -0.05 -1.0",
    },
    {
        # "Silt haze" style: greyish sky, slightly greener highlight
        "name": "silt_haze",
        "ambient": "0.20 0.19 0.17 1.0",
        "background": "0.45 0.48 0.42 1.0",
        "diffuse": "0.60 0.58 0.52 1.0",
        "specular": "0.30 0.35 0.28 1.0",
        "direction": "0.22 -0.10 -1.0",
    },
]

WAVE_MODELS = [
    {"name": "ocean", "uri": "model://ocean_waves"},
    {"name": "regular", "uri": "model://regular_waves"},
    {"name": "trochoid", "uri": "model://trochoid_waves"},
]

def _rotate_xy(x: float, y: float, yaw_rad: float) -> Tuple[float, float]:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return (c * x - s * y, s * x + c * y)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _layout_placement_ranges(layout_name: str) -> dict:
    """Biases for procedural gate spawning (pair count, corridors, lateral spacing)."""
    if layout_name == "wide_lane":
        return {
            "n_pairs": (4, 8),
            "half_sep_m": (1.9, 4.6),
            "corridor_yaw_deg": (-78.0, 78.0),
            "spine_x": (-10.2, 10.2),
            "spine_y": (-7.2, 7.2),
            "min_center_sep_m": 3.5,
            "min_pair_internal_m": 1.45,
        }
    if layout_name == "offset":
        return {
            "n_pairs": (3, 8),
            "half_sep_m": (1.25, 4.0),
            "corridor_yaw_deg": (-85.0, 85.0),
            "spine_x": (-9.8, 9.8),
            "spine_y": (-6.8, 6.8),
            "min_center_sep_m": 2.45,
            "min_pair_internal_m": 1.15,
        }
    if layout_name == "baseline":
        return {
            "n_pairs": (3, 7),
            "half_sep_m": (1.3, 3.6),
            "corridor_yaw_deg": (-70.0, 70.0),
            "spine_x": (-9.2, 9.2),
            "spine_y": (-6.2, 6.2),
            "min_center_sep_m": 2.75,
            "min_pair_internal_m": 1.25,
        }
    raise ValueError(f"Unknown layout: {layout_name}")


def _buoy_rpy_sample(rng: random.Random) -> Tuple[float, float, float]:
    """Per-buoy roll/pitch/yaw (rad): strong orientation diversity."""
    roll = math.radians(rng.uniform(-18.0, 18.0))
    pitch = math.radians(rng.uniform(-16.0, 16.0))
    yaw = math.radians(rng.uniform(-180.0, 180.0))
    return roll, pitch, yaw


def _sample_raw_layout(
    rng: random.Random,
    layout_name: str,
) -> Tuple[List[Tuple[SurfPose, SurfPose]], List[SurfPose]]:
    """Variable RG gate count, 0–2 blues, corridor angles & per-buoy RPY."""
    pr = _layout_placement_ranges(layout_name)
    n_pairs = rng.randint(pr["n_pairs"][0], pr["n_pairs"][1])
    n_blue = rng.randint(0, 2)

    placed_xy: List[Tuple[float, float]] = []
    pairs: List[Tuple[SurfPose, SurfPose]] = []

    def min_dist_ok(q: Tuple[float, float], mind: float) -> bool:
        return all(_dist(q, p) >= mind for p in placed_xy)

    for _ in range(n_pairs):
        placed_attempt = False
        for _attempt in range(95):
            half_sep = rng.uniform(pr["half_sep_m"][0], pr["half_sep_m"][1])
            theta_deg = rng.uniform(pr["corridor_yaw_deg"][0], pr["corridor_yaw_deg"][1])
            theta = math.radians(theta_deg)
            mx = rng.uniform(pr["spine_x"][0], pr["spine_x"][1])
            my = rng.uniform(pr["spine_y"][0], pr["spine_y"][1])
            tx, ty = math.cos(theta), math.sin(theta)
            nx, ny = -ty, tx
            rx, ry = mx - half_sep * nx, my - half_sep * ny
            gx, gy = mx + half_sep * nx, my + half_sep * ny
            if _dist((rx, ry), (gx, gy)) < pr["min_pair_internal_m"] * 2.0 - 1e-3:
                continue
            mind = float(pr["min_center_sep_m"])
            if not min_dist_ok((rx, ry), mind):
                continue
            if not min_dist_ok((gx, gy), mind):
                continue
            rr, rp, ryaw = _buoy_rpy_sample(rng)
            gr, gp, gyaw = _buoy_rpy_sample(rng)
            pairs.append((SurfPose(rx, ry, rr, rp, ryaw), SurfPose(gx, gy, gr, gp, gyaw)))
            placed_xy.append((rx, ry))
            placed_xy.append((gx, gy))
            placed_attempt = True
            break
        if not placed_attempt:
            break

    if len(pairs) < 2:
        pairs = [
            (
                SurfPose(-6.0, 2.0, 0.0, 0.0, 0.0),
                SurfPose(-6.0, -2.0, 0.0, 0.0, math.radians(rng.uniform(-40, 40))),
            ),
            (
                SurfPose(6.0, 3.2, 0.0, 0.0, math.radians(rng.uniform(-120, 120))),
                SurfPose(6.0, -3.2, 0.0, 0.0, math.radians(rng.uniform(-120, 120))),
            ),
        ]
        placed_xy = [(p.x, p.y) for pr, pg in pairs for p in (pr, pg)]

    blues: List[SurfPose] = []
    for _bi in range(n_blue):
        ok = False
        for _attempt in range(140):
            bx = rng.uniform(-11.5, 11.5)
            by = rng.uniform(-7.5, 7.5)
            if not min_dist_ok((bx, by), 2.55):
                continue
            if blues and _dist((bx, by), (blues[0].x, blues[0].y)) < 3.2:
                continue
            br, bp, byaw = _buoy_rpy_sample(rng)
            blues.append(SurfPose(bx, by, br, bp, byaw))
            placed_xy.append((bx, by))
            ok = True
            break
        if not ok:
            break

    return pairs, blues


def _clamp_surf_field(
    pairs: List[Tuple[SurfPose, SurfPose]],
    blues: List[SurfPose],
    max_norm: float,
) -> Tuple[List[Tuple[SurfPose, SurfPose]], List[SurfPose]]:
    """Scale XY only toward origin when outside max radius."""
    pts: List[Tuple[float, float]] = []
    for pr, pg in pairs:
        pts.append((pr.x, pr.y))
        pts.append((pg.x, pg.y))
    for b in blues:
        pts.append((b.x, b.y))
    if not pts:
        return pairs, blues
    peak = max(math.hypot(x, y) for x, y in pts)
    if peak <= max_norm or peak < 1e-6:
        return pairs, blues
    scale = (max_norm / peak) * 0.995

    def sc_sp(sp: SurfPose) -> SurfPose:
        return SurfPose(sp.x * scale, sp.y * scale, sp.roll, sp.pitch, sp.yaw)

    return ([(sc_sp(r), sc_sp(g)) for r, g in pairs], [sc_sp(bl) for bl in blues])


def _apply_global_field_perturb(
    rng: random.Random,
    pairs_in: List[Tuple[SurfPose, SurfPose]],
    blues_in: List[SurfPose],
    *,
    max_translate_m: float,
    max_yaw_deg: float,
    max_scale_jitter: float,
) -> Tuple[List[Tuple[SurfPose, SurfPose]], List[SurfPose]]:
    """Rotate + scale + translate XY; add global planar yaw into each buoy body yaw."""
    dx = rng.uniform(-max_translate_m, max_translate_m)
    dy = rng.uniform(-max_translate_m, max_translate_m)
    yaw_g = math.radians(rng.uniform(-max_yaw_deg, max_yaw_deg))
    sc = 1.0 + rng.uniform(-max_scale_jitter, max_scale_jitter)

    def map_xy(x: float, y: float) -> Tuple[float, float]:
        xr, yr = _rotate_xy(x * sc, y * sc, yaw_g)
        return xr + dx, yr + dy

    out_pairs: List[Tuple[SurfPose, SurfPose]] = []
    for pr, pg in pairs_in:
        rx, ry = map_xy(pr.x, pr.y)
        gx, gy = map_xy(pg.x, pg.y)
        out_pairs.append(
            (
                SurfPose(rx, ry, pr.roll, pr.pitch, pr.yaw + yaw_g),
                SurfPose(gx, gy, pg.roll, pg.pitch, pg.yaw + yaw_g),
            )
        )
    out_blues: List[SurfPose] = []
    for b in blues_in:
        bx, by = map_xy(b.x, b.y)
        out_blues.append(SurfPose(bx, by, b.roll, b.pitch, b.yaw + yaw_g))
    return _clamp_surf_field(out_pairs, out_blues, _MAX_XY_NORM_M)


def _micro_jitter_offset_lane(
    rng: random.Random,
    pairs: List[Tuple[SurfPose, SurfPose]],
    blues: List[SurfPose],
) -> Tuple[List[Tuple[SurfPose, SurfPose]], List[SurfPose]]:
    """XY + RPY wobble for ``offset`` layouts."""
    j_pairs: List[Tuple[SurfPose, SurfPose]] = []
    for pr, pg in pairs:
        dx = rng.uniform(-2.95, 2.95)
        dy = rng.uniform(-2.15, 2.15)
        gfx = rng.uniform(-1.35, 1.35)
        gfy = rng.uniform(-1.45, 1.45)
        j_pairs.append(
            (
                SurfPose(
                    pr.x + dx,
                    pr.y + dy,
                    pr.roll + math.radians(rng.uniform(-6, 6)),
                    pr.pitch + math.radians(rng.uniform(-6, 6)),
                    pr.yaw + math.radians(rng.uniform(-14, 14)),
                ),
                SurfPose(
                    pg.x + dx + gfx,
                    pg.y + dy + gfy,
                    pg.roll + math.radians(rng.uniform(-6, 6)),
                    pg.pitch + math.radians(rng.uniform(-6, 6)),
                    pg.yaw + math.radians(rng.uniform(-14, 14)),
                ),
            )
        )
    j_blues = [
        SurfPose(
            b.x + rng.uniform(-2.1, 2.1),
            b.y + rng.uniform(-1.9, 1.9),
            b.roll + math.radians(rng.uniform(-7, 7)),
            b.pitch + math.radians(rng.uniform(-7, 7)),
            b.yaw + math.radians(rng.uniform(-18, 18)),
        )
        for b in blues
    ]
    return _clamp_surf_field(j_pairs, j_blues, _MAX_XY_NORM_M)


def buoy_positions_for_layout(
    seed: int,
    layout_name: str,
    *,
    max_translate_m: float,
    max_yaw_deg: float,
    max_scale_jitter: float,
) -> Tuple[
    List[Tuple[SurfPose, SurfPose]],
    List[SurfPose],
    List[Tuple[float, float]],
]:
    """RGB layout: procedural gates + blues; XY list feeds clutter avoidance."""
    rng = random.Random(seed)
    raw_pairs, raw_blues = _sample_raw_layout(rng, layout_name)
    pairs, blues = _apply_global_field_perturb(
        rng,
        raw_pairs,
        raw_blues,
        max_translate_m=max_translate_m,
        max_yaw_deg=max_yaw_deg,
        max_scale_jitter=max_scale_jitter,
    )
    if layout_name == "offset":
        jitter_seed = rng.randint(1, 1_000_000_537)
        pairs, blues = _micro_jitter_offset_lane(random.Random(jitter_seed), pairs, blues)
    buoy_xy: List[Tuple[float, float]] = []
    for pr, pg in pairs:
        buoy_xy.append((pr.x, pr.y))
        buoy_xy.append((pg.x, pg.y))
    for b in blues:
        buoy_xy.append((b.x, b.y))
    return pairs, blues, buoy_xy


def _sample_clutter_placements(
    rng: random.Random,
    buoy_xy: Sequence[Tuple[float, float]],
    target_n: int,
    *,
    min_dist_buoy_m: float,
    min_dist_clutter_m: float,
    xy_limit: float,
) -> List[Tuple[float, float, float, float]]:
    """(x, y, yaw_deg, approximate_footprint_radius_m) avoiding buoy disks."""
    placed: List[Tuple[float, float, float]] = []
    out: List[Tuple[float, float, float, float]] = []

    def ok(x: float, y: float, r_pad: float) -> bool:
        if abs(x) > xy_limit - r_pad or abs(y) > xy_limit - r_pad:
            return False
        for bx, by in buoy_xy:
            if math.hypot(x - bx, y - by) < min_dist_buoy_m + r_pad:
                return False
        for px, py, pr in placed:
            if math.hypot(x - px, y - py) < pr + min_dist_clutter_m + r_pad:
                return False
        return True

    trials = 0
    max_trials = max(400, target_n * 350)
    while len(out) < target_n and trials < max_trials:
        trials += 1
        r_foot = rng.uniform(0.55, 1.42)
        x = rng.uniform(-xy_limit, xy_limit)
        y = rng.uniform(-xy_limit, xy_limit)
        if ok(x, y, r_foot):
            yaw_deg = rng.uniform(-180.0, 180.0)
            placed.append((x, y, r_foot))
            out.append((x, y, yaw_deg, r_foot))
    return out


_MATERIAL_PALETTE = (
    ("0.45 0.38 0.32 1.0", "0.62 0.52 0.38 1.0"),  # wood-ish
    ("0.18 0.22 0.28 1.0", "0.28 0.34 0.42 1.0"),  # slate
    ("0.35 0.16 0.12 1.0", "0.55 0.28 0.18 1.0"),  # rust
    ("0.12 0.30 0.14 1.0", "0.22 0.48 0.24 1.0"),  # tarp green
    ("0.40 0.35 0.50 1.0", "0.55 0.50 0.68 1.0"),  # painted metal
)


def clutter_includes(seed: int, buoy_xy: Sequence[Tuple[float, float]], n_objects: int) -> str:
    """Static inline primitives (boxes/cylinders) — URIs deliberately avoid ``buoy`` so labels stay buoy-only."""
    rng = random.Random(seed + 901_273)
    if n_objects <= 0:
        return ""
    placements = _sample_clutter_placements(
        rng,
        buoy_xy,
        n_objects,
        min_dist_buoy_m=2.7,
        min_dist_clutter_m=1.2,
        xy_limit=min(_MAX_XY_NORM_M, 13.6),
    )

    lines: List[str] = []
    lines.append("")
    lines.append("    <!-- Domain clutter (not buoys): boxes/cylinders; static, no buoy URI substring -->")

    prim_roll = rng.random()
    for i, (x, y, yaw_deg, _rf) in enumerate(placements):
        ambient, diffuse = _MATERIAL_PALETTE[i % len(_MATERIAL_PALETTE)]
        yaw_rad = math.radians(yaw_deg)

        use_cylinder = prim_roll > 0.58 and rng.random() < 0.32
        prim_roll += 0.137

        if use_cylinder:
            rad = rng.uniform(0.32, 0.85)
            length = rng.uniform(0.85, 2.4)
            zpose = 0.5 * length + 0.05
            lines.extend(
                [
                    f'    <model name="scene_clutter_cyl_{i:03d}">',
                    "      <static>true</static>",
                    f"      <pose>{x:.3f} {y:.3f} {zpose:.3f} 0 0 {yaw_rad:.6f}</pose>",
                    '      <link name="link">',
                    '        <visual name="v">',
                    "          <geometry>",
                    "            <cylinder>",
                    f"              <radius>{rad:.3f}</radius>",
                    f"              <length>{length:.3f}</length>",
                    "            </cylinder>",
                    "          </geometry>",
                    "          <material>",
                    f"            <ambient>{ambient}</ambient>",
                    f"            <diffuse>{diffuse}</diffuse>",
                    "          </material>",
                    "        </visual>",
                    "      </link>",
                    "    </model>",
                ]
            )
        else:
            w = rng.uniform(0.85, 2.85)
            d = rng.uniform(0.85, 2.65)
            h = rng.uniform(0.45, 1.95)
            zpose = max(0.15, h * 0.5 + 0.02)
            lines.extend(
                [
                    f'    <model name="scene_clutter_box_{i:03d}">',
                    "      <static>true</static>",
                    f"      <pose>{x:.3f} {y:.3f} {zpose:.3f} 0 0 {yaw_rad:.6f}</pose>",
                    '      <link name="link">',
                    '        <visual name="v">',
                    "          <geometry>",
                    "            <box>",
                    f"              <size>{w:.3f} {d:.3f} {h:.3f}</size>",
                    "            </box>",
                    "          </geometry>",
                    "          <material>",
                    f"            <ambient>{ambient}</ambient>",
                    f"            <diffuse>{diffuse}</diffuse>",
                    "          </material>",
                    "        </visual>",
                    "      </link>",
                    "    </model>",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def buoy_includes(
    red_green_pairs: List[Tuple[SurfPose, SurfPose]],
    blue_markers: List[SurfPose],
) -> str:
    """Emit ``<include>`` buoys with full 6-DOF pose (XY + roll/pitch/yaw in radians); z≈waterline."""
    lines: List[str] = []
    idx = 1
    for pr, pg in red_green_pairs:
        lines.extend(
            [
                "    <include>",
                "      <uri>model://robotx_buoy_red_led</uri>",
                f"      <name>safe_passage_red_{idx}</name>",
                f"      <pose>{pr.x:.4f} {pr.y:.4f} 0.0 "
                f"{pr.roll:.5f} {pr.pitch:.5f} {pr.yaw:.5f}</pose>",
                "    </include>",
                "    <include>",
                "      <uri>model://robotx_buoy_green_led</uri>",
                f"      <name>safe_passage_green_{idx}</name>",
                f"      <pose>{pg.x:.4f} {pg.y:.4f} 0.0 "
                f"{pg.roll:.5f} {pg.pitch:.5f} {pg.yaw:.5f}</pose>",
                "    </include>",
            ]
        )
        idx += 1

    for i, b in enumerate(blue_markers, start=1):
        marker_name = f"marker_blue_{i}"
        lines.extend(
            [
                "    <include>",
                "      <uri>model://robotx_buoy_blue_led</uri>",
                f"      <name>{marker_name}</name>",
                f"      <pose>{b.x:.4f} {b.y:.4f} 0.0 "
                f"{b.roll:.5f} {b.pitch:.5f} {b.yaw:.5f}</pose>",
                "    </include>",
            ]
        )

    return "\n".join(lines)


def _camera_plugin_filename() -> str:
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    return f"librobotx_overhead_motion_plugin{ext}"


def camera_model_sdf(
    *,
    radius_x_m: float,
    radius_y_m: float,
    angular_speed_rad_s: float,
    phase_rad: float,
    z_wobble_m: float,
    z_wobble_hz: float,
    z_wobble_phase_rad: float,
    enable_motion: bool,
) -> str:
    """UAV rig: downward RGB camera; optional overhead orbit via WorldPoseCmd plugin."""
    plugin_fn = _camera_plugin_filename()
    motion_xml = ""
    if enable_motion:
        motion_xml = f"""
      <plugin filename="{plugin_fn}" name="robotx::OverheadMotionPlugin">
        <radius_x_m>{radius_x_m:.5f}</radius_x_m>
        <radius_y_m>{radius_y_m:.5f}</radius_y_m>
        <angular_speed_rad_s>{angular_speed_rad_s:.6f}</angular_speed_rad_s>
        <phase_rad>{phase_rad:.6f}</phase_rad>
        <z_wobble_m>{z_wobble_m:.5f}</z_wobble_m>
        <z_wobble_hz>{z_wobble_hz:.5f}</z_wobble_hz>
        <z_wobble_phase_rad>{z_wobble_phase_rad:.6f}</z_wobble_phase_rad>
      </plugin>"""
    static_tag = "false" if enable_motion else "true"
    grav_line = "\n        <gravity>false</gravity>" if enable_motion else ""
    return f"""    <model name="uav_dataset_camera">
      <static>{static_tag}</static>
      <pose>0 0 26 0 0 0</pose>{motion_xml}
      <link name="camera_link">{grav_line}
        <sensor name="top_down_rgb" type="camera">
          <always_on>true</always_on>
          <update_rate>20</update_rate>
          <pose>0 0 0 0 1.57079632679 0</pose>
          <camera>
            <horizontal_fov>1.05</horizontal_fov>
            <image>
              <width>1920</width>
              <height>1080</height>
              <format>R8G8B8</format>
            </image>
            <clip>
              <near>0.2</near>
              <far>600</far>
            </clip>
          </camera>
          <topic>/robotx/uav/camera/image_raw</topic>
          <visualize>true</visualize>
        </sensor>
      </link>
    </model>"""


def world_sdf(
    glare: Dict[str, str],
    wave: Dict[str, str],
    buoy_block: str,
    clutter_block: str,
    camera_block: str,
) -> str:
    return f"""<?xml version="1.0"?>
<sdf version="1.10">
  <world name="robotx_task1_uav_view">
    <gravity>0 0 -9.81</gravity>

    <physics name="physics" type="ode">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <scene>
      <ambient>{glare["ambient"]}</ambient>
      <background>{glare["background"]}</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>

    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 300 0 0 0</pose>
      <diffuse>{glare["diffuse"]}</diffuse>
      <specular>{glare["specular"]}</specular>
      <direction>{glare["direction"]}</direction>
      <attenuation>
        <range>2500</range>
        <constant>0.85</constant>
        <linear>0.001</linear>
        <quadratic>0.00005</quadratic>
      </attenuation>
    </light>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <include>
      <uri>{wave["uri"]}</uri>
      <name>robotx_ocean</name>
      <pose>0 0 0 0 0 0</pose>
    </include>

{buoy_block}
{clutter_block}

{camera_block}
  </world>
</sdf>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RobotX world variants for data collection.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "worlds" / "generated",
        help="Directory where generated .sdf files and manifest are written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic layouts.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After generation, walk through scenarios and wait for Enter between each.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=0,
        help="If >0, shuffle glare×wave×layout combinations with --seed and emit only this many worlds.",
    )
    parser.add_argument(
        "--clutter-count",
        type=int,
        default=18,
        help="Inline static boxes/cylinders scattered in the nominal FOV while keeping clearance from buoy X/Y.",
    )
    parser.add_argument(
        "--buoy-translate-max-m",
        type=float,
        default=2.9,
        help="Half-range (m) of random global shift applied to entire gate pattern for every scenario.",
    )
    parser.add_argument(
        "--buoy-yaw-max-deg",
        type=float,
        default=31.0,
        help="Max |yaw| (deg) of global gate rotation applied about world +Z.",
    )
    parser.add_argument(
        "--buoy-scale-jitter",
        type=float,
        default=0.11,
        help="Symmetric scale perturbation ± this value around the gate (before clamping inside FOV).",
    )
    parser.add_argument(
        "--no-camera-motion",
        action="store_true",
        help="Keep the nadir camera static (no overhead orbit plugin).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    layouts = ["baseline", "offset", "wide_lane"]
    scenarios: List[Tuple[Dict[str, str], Dict[str, str], str]] = []
    for glare in GLARE_PROFILES:
        for wave in WAVE_MODELS:
            for layout in layouts:
                scenarios.append((glare, wave, layout))

    rng = random.Random(args.seed)
    if args.max_scenarios and args.max_scenarios > 0 and args.max_scenarios < len(scenarios):
        rng.shuffle(scenarios)
        scenarios = scenarios[: args.max_scenarios]

    manifest = []
    scenario_id = 0

    for glare, wave, layout in scenarios:
        scenario_id += 1
        seed = args.seed + scenario_id * 13
        pairs, blues, buoy_xy_clear = buoy_positions_for_layout(
            seed,
            layout,
            max_translate_m=args.buoy_translate_max_m,
            max_yaw_deg=args.buoy_yaw_max_deg,
            max_scale_jitter=args.buoy_scale_jitter,
        )
        buoy_block = buoy_includes(pairs, blues)
        clutter_block = clutter_includes(seed, buoy_xy_clear, max(0, int(args.clutter_count)))
        mrng = random.Random(seed + 50_003)
        cam_on = not args.no_camera_motion
        # Extra-strong cinematic panning: even wider ellipse, modestly faster angular speed.
        cam_rx = mrng.uniform(8.0, 14.0)
        cam_ry = mrng.uniform(6.5, 11.5)
        cam_omega = mrng.uniform(0.035, 0.085)
        cam_phase = mrng.uniform(0.0, 2.0 * math.pi)
        cam_zw = mrng.uniform(0.02, 0.12)
        cam_zh = mrng.uniform(0.03, 0.08)
        cam_zp = mrng.uniform(0.0, 2.0 * math.pi)
        camera_block = camera_model_sdf(
            radius_x_m=cam_rx,
            radius_y_m=cam_ry,
            angular_speed_rad_s=cam_omega,
            phase_rad=cam_phase,
            z_wobble_m=cam_zw,
            z_wobble_hz=cam_zh,
            z_wobble_phase_rad=cam_zp,
            enable_motion=cam_on,
        )
        sdf = world_sdf(
            glare=glare,
            wave=wave,
            buoy_block=buoy_block,
            clutter_block=clutter_block,
            camera_block=camera_block,
        )
        stem = f"robotx_dr_{scenario_id:03d}_{glare['name']}_{wave['name']}_{layout}"
        sdf_path = output_dir / f"{stem}.sdf"
        sdf_path.write_text(sdf)
        manifest.append(
            {
                "id": scenario_id,
                "name": stem,
                "glare": glare["name"],
                "wave": wave["name"],
                "layout": layout,
                "seed": seed,
                "world_path": str(sdf_path),
                "clutter_instances": max(0, int(args.clutter_count)),
                "buoy_translate_max_m": args.buoy_translate_max_m,
                "buoy_yaw_max_deg": args.buoy_yaw_max_deg,
                "buoy_scale_jitter": args.buoy_scale_jitter,
                "red_green_gate_count": len(pairs),
                "blue_marker_count": len(blues),
                "camera_overhead_motion": cam_on,
                "camera_orbit_radius_x_m": round(cam_rx, 4) if cam_on else None,
                "camera_orbit_radius_y_m": round(cam_ry, 4) if cam_on else None,
                "camera_orbit_omega_rad_s": round(cam_omega, 5) if cam_on else None,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Generated {len(manifest)} scenarios in {output_dir}")
    print(f"Manifest: {manifest_path}")

    if not args.interactive:
        print("\nExample launch:")
        print(f"  gz sim -s -r {manifest[0]['world_path']}")
        print("  # in second terminal: gz sim -g")
        return

    print("\nInteractive walkthrough (press Enter for next scenario, 'q' to quit):")
    for item in manifest:
        print(
            f"\n[{item['id']:03d}] {item['name']}\n"
            f"  glare={item['glare']} wave={item['wave']} layout={item['layout']}\n"
            f"  launch: gz sim -s -r {item['world_path']}"
        )
        reply = input("Enter=next, q=quit > ").strip().lower()
        if reply == "q":
            break


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate RobotX gazebo world variants for domain randomization.

Vary:
- sun / glare settings
- wave model choice
- buoy arrangement

Optional interactive walkthrough prints each scenario and waits for Enter
so you can launch one-by-one during dataset collection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


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

BASE_RED_GREEN_PAIRS: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    ((-8.0, 2.0), (-8.0, -2.0)),
    ((8.0, 3.5), (8.0, -3.5)),
]

BASE_BLUE_MARKERS: List[Tuple[float, float]] = [(-14.0, 0.0), (14.0, 0.0)]


def buoy_includes(seed: int, layout_name: str) -> str:
    rng = random.Random(seed)

    if layout_name == "baseline":
        red_green_pairs = BASE_RED_GREEN_PAIRS
        blue_markers = BASE_BLUE_MARKERS
    elif layout_name == "offset":
        red_green_pairs = []
        for red_xy, green_xy in BASE_RED_GREEN_PAIRS:
            dx = rng.uniform(-1.2, 1.2)
            dy = rng.uniform(-0.8, 0.8)
            red_green_pairs.append(
                ((red_xy[0] + dx, red_xy[1] + dy), (green_xy[0] + dx, green_xy[1] - dy))
            )
        blue_markers = [
            (p[0] + rng.uniform(-0.8, 0.8), p[1] + rng.uniform(-0.6, 0.6))
            for p in BASE_BLUE_MARKERS
        ]
    elif layout_name == "wide_lane":
        red_green_pairs = [((-10.0, 3.4), (-10.0, -3.4)), ((10.0, 4.6), (10.0, -4.6))]
        blue_markers = [(-16.0, 0.0), (16.0, 0.0)]
    else:
        raise ValueError(f"Unknown layout: {layout_name}")

    lines: List[str] = []
    idx = 1
    for red_xy, green_xy in red_green_pairs:
        lines.extend(
            [
                "    <include>",
                "      <uri>model://robotx_buoy_red_led</uri>",
                f"      <name>safe_passage_red_{idx}</name>",
                f"      <pose>{red_xy[0]:.3f} {red_xy[1]:.3f} 0.0 0 0 0</pose>",
                "    </include>",
                "    <include>",
                "      <uri>model://robotx_buoy_green_led</uri>",
                f"      <name>safe_passage_green_{idx}</name>",
                f"      <pose>{green_xy[0]:.3f} {green_xy[1]:.3f} 0.0 0 0 0</pose>",
                "    </include>",
            ]
        )
        idx += 1

    for i, p in enumerate(blue_markers, start=1):
        marker_name = "start_marker_blue" if i == 1 else "end_marker_blue"
        lines.extend(
            [
                "    <include>",
                "      <uri>model://robotx_buoy_blue_led</uri>",
                f"      <name>{marker_name}</name>",
                f"      <pose>{p[0]:.3f} {p[1]:.3f} 0.0 0 0 0</pose>",
                "    </include>",
            ]
        )

    return "\n".join(lines)


def world_sdf(glare: Dict[str, str], wave: Dict[str, str], buoy_block: str) -> str:
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

    <model name="uav_dataset_camera">
      <static>true</static>
      <pose>0 0 26 0 0 0</pose>
      <link name="camera_link">
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
    </model>
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
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    layouts = ["baseline", "offset", "wide_lane"]
    manifest = []
    scenario_id = 0

    for glare in GLARE_PROFILES:
        for wave in WAVE_MODELS:
            for layout in layouts:
                scenario_id += 1
                seed = args.seed + scenario_id * 13
                buoy_block = buoy_includes(seed=seed, layout_name=layout)
                sdf = world_sdf(glare=glare, wave=wave, buoy_block=buoy_block)
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

#!/usr/bin/env python3
"""
Dot-map visualizer for RobotX buoy detections.

Usage:
    # Post-run (static):
    python fulldemo/visualize_detections.py fulldemo/detections.jsonl

    # Live (polls file while ground station is running):
    python fulldemo/visualize_detections.py fulldemo/detections.jsonl --live
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COLOR_MAP = {
    "red":   "#e74c3c",
    "green": "#2ecc71",
    "blue":  "#3498db",
}
MARKER_SIZE = 120

def load_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return records


def plot(records: list[dict], ax: plt.Axes) -> None:
    ax.cla()

    by_color: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for r in records:
        color = r.get("color", "unknown")
        lat = r.get("lat")
        lon = r.get("lon")
        tid = r.get("target_id", 0)
        if lat is not None and lon is not None:
            by_color[color].append((lon, lat, tid))

    if not any(by_color.values()):
        ax.text(0.5, 0.5, "No detections yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title("RobotX Buoy Detection Map")
        return

    for color, points in by_color.items():
        lons = [p[0] for p in points]
        lats = [p[1] for p in points]
        tids = [p[2] for p in points]
        hex_color = COLOR_MAP.get(color, "#aaaaaa")
        ax.scatter(lons, lats, c=hex_color, s=MARKER_SIZE,
                   edgecolors="white", linewidths=0.8,
                   label=f"{color} ({len(points)})", zorder=3)
        # Label each dot with track ID
        for lon, lat, tid in zip(lons, lats, tids):
            ax.annotate(f"t{tid}", (lon, lat),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color=hex_color)

    # Unique track IDs — draw convex hull or just connect first occurrences
    all_lons = [r["lon"] for r in records if "lon" in r]
    all_lats = [r["lat"] for r in records if "lat" in r]
    if all_lons:
        pad_lon = (max(all_lons) - min(all_lons)) * 0.2 + 1e-5
        pad_lat = (max(all_lats) - min(all_lats)) * 0.2 + 1e-5
        ax.set_xlim(min(all_lons) - pad_lon, max(all_lons) + pad_lon)
        ax.set_ylim(min(all_lats) - pad_lat, max(all_lats) + pad_lat)

    total = sum(len(v) for v in by_color.values())
    ax.set_title(f"RobotX Buoy Detection Map  —  {total} detections", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="Path to detections.jsonl")
    parser.add_argument("--live", action="store_true",
                        help="Poll the file and refresh every 2 seconds")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Refresh interval in seconds (live mode)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    if args.live:
        plt.ion()
        print(f"Live mode — watching {args.jsonl} (Ctrl+C to stop)")
        last_count = -1
        while True:
            records = load_jsonl(args.jsonl)
            if len(records) != last_count:
                plot(records, ax)
                last_count = len(records)
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
            time.sleep(args.interval)
    else:
        records = load_jsonl(args.jsonl)
        if not records:
            print(f"No records found in {args.jsonl}")
        plot(records, ax)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

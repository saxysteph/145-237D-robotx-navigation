#!/usr/bin/env python3
"""Mock onboard sender: emit sample buoy reports for local MAVLink testing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mavlink_comms.buoy_report import BuoyReport
from mavlink_comms.transmitter import DEFAULT_ONBOARD_CONNECTION, BuoyMavlinkTransmitter


def sample_reports() -> list[BuoyReport]:
    # ARC field-ish coordinates from mavcore SITL README
    return [
        BuoyReport(1, "red", 33.6429321, -117.8262882, frame=100),
        BuoyReport(2, "green", 33.6431000, -117.8260000, frame=101),
        BuoyReport(3, "blue", 33.6428000, -117.8265000, frame=102),
    ]


def load_json_reports(path: Path) -> list[BuoyReport]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON file must be a list of buoy objects")
    out: list[BuoyReport] = []
    for i, row in enumerate(data, start=1):
        lat = row.get("lat") or row.get("est_lat") or row.get("buoy_lat")
        lon = row.get("lon") or row.get("est_lon") or row.get("buoy_lon")
        if lat is None or lon is None:
            continue
        out.append(
            BuoyReport(
                target_id=int(row.get("target_id", i)),
                color=str(row.get("color") or row.get("pred_color_hsv", "red")),
                lat=float(lat),
                lon=float(lon),
                frame=int(row.get("frame", 0)),
            )
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Mock RobotX buoy MAVLink sender")
    p.add_argument(
        "--connection",
        default=DEFAULT_ONBOARD_CONNECTION,
        help=f"pymavlink connection string (default: {DEFAULT_ONBOARD_CONNECTION})",
    )
    p.add_argument("--interval-s", type=float, default=1.0, help="Seconds between sends")
    p.add_argument("--count", type=int, default=0, help="Max reports to send (0 = all samples once)")
    p.add_argument("--from-json", type=Path, default=None, help="JSON list of buoy dicts")
    args = p.parse_args()

    reports = load_json_reports(args.from_json) if args.from_json else sample_reports()
    if not reports:
        print("No reports to send", file=sys.stderr)
        return 1

    tx = BuoyMavlinkTransmitter(args.connection)
    print(f"Sending {len(reports)} report(s) to {args.connection}", flush=True)

    sent = 0
    try:
        for report in reports:
            tx.transmit_report(report)
            print(f"[ONBOARD] {report}", flush=True)
            sent += 1
            if args.count > 0 and sent >= args.count:
                break
            time.sleep(max(0.0, args.interval_s))
    finally:
        tx.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

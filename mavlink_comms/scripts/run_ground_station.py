#!/usr/bin/env python3
"""Ground station: listen for RobotX buoy STATUSTEXT over MAVLink UDP."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mavlink_comms._mavcore_path  # noqa: F401

from mavcore import MAVDevice
from mavcore.protocols.heartbeat_protocol import HeartbeatProtocol

from mavlink_comms.messages.robotx_buoy_msg import RobotXBuoyStatusText
from mavlink_comms.transmitter import DEFAULT_GCS_CONNECTION


def main() -> int:
    p = argparse.ArgumentParser(description="RobotX buoy MAVLink ground station receiver")
    p.add_argument(
        "--connection",
        default=DEFAULT_GCS_CONNECTION,
        help=f"pymavlink connection string (default: {DEFAULT_GCS_CONNECTION})",
    )
    p.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Append each received buoy as one JSON line",
    )
    p.add_argument("--duration-s", type=float, default=0.0, help="Exit after N seconds (0 = run forever)")
    args = p.parse_args()

    device = MAVDevice(
        args.connection,
        source_system=255,
        source_component=0,
        attempt_reconnect=False,
    )
    device.run_protocol(HeartbeatProtocol())

    received: list[dict] = []

    def on_buoy(msg: RobotXBuoyStatusText) -> None:
        r = msg.report
        row = {
            "target_id": r.target_id,
            "color": r.color,
            "lat": r.lat,
            "lon": r.lon,
            "frame": r.frame,
            "timestamp_ms": msg.timestamp,
        }
        received.append(row)
        print(f"[GCS] {json.dumps(row)}", flush=True)
        if args.output_jsonl:
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    buoy_listener = RobotXBuoyStatusText(callback_func=on_buoy)
    device.add_listener(buoy_listener)

    print(f"Listening on {args.connection} for RXB| buoy reports (Ctrl+C to stop)", flush=True)
    try:
        if args.duration_s > 0:
            time.sleep(args.duration_s)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        device.stop_reading()

    print(f"Received {len(received)} buoy report(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

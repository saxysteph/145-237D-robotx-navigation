#!/usr/bin/env python3
"""Single-process loopback test: GCS listener + mock sender on UDP 14555."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mavlink_comms._mavcore_path  # noqa: F401

from mavlink_comms.buoy_report import BuoyReport
from mavlink_comms.messages.robotx_buoy_msg import RobotXBuoyStatusText
from mavlink_comms.transmitter import (
    DEFAULT_GCS_CONNECTION,
    DEFAULT_ONBOARD_CONNECTION,
    BuoyMavlinkTransmitter,
)
from mavcore import MAVDevice


def main() -> int:
    received: list[BuoyReport] = []
    done = threading.Event()

    def on_buoy(msg: RobotXBuoyStatusText) -> None:
        received.append(msg.report)

    device = MAVDevice(
        DEFAULT_GCS_CONNECTION,
        source_system=255,
        source_component=0,
        attempt_reconnect=False,
    )
    device.add_listener(RobotXBuoyStatusText(callback_func=on_buoy))

    # Allow listener thread to start
    time.sleep(1.5)

    tx = BuoyMavlinkTransmitter(DEFAULT_ONBOARD_CONNECTION)
    expected = [
        BuoyReport(10, "red", 33.64, -117.82, 1),
        BuoyReport(11, "green", 33.65, -117.83, 2),
    ]
    for report in expected:
        tx.transmit_report(report)
        time.sleep(0.3)

    tx.close()
    time.sleep(1.0)
    device.stop_reading()

    if len(received) < len(expected):
        print(f"FAIL: received {len(received)} / {len(expected)}", file=sys.stderr)
        return 1

    for exp in expected:
        match = [
            r
            for r in received
            if r.target_id == exp.target_id
            and r.color == exp.color
            and abs(r.lat - exp.lat) < 1e-5
            and abs(r.lon - exp.lon) < 1e-5
        ]
        if not match:
            print(f"FAIL: missing report for id={exp.target_id}", file=sys.stderr)
            return 1

    print(f"PASS: loopback received {len(received)} report(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

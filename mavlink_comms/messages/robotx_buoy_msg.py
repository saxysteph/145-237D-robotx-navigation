"""MAVCore STATUSTEXT wrapper for RobotX buoy reports."""

from __future__ import annotations

from typing import Callable

import mavlink_comms._mavcore_path  # noqa: F401

from mavcore.mav_message import MAVMessage, thread_safe
from mavcore.messages.status_text_msg import MAVSeverity, StatusText

from mavlink_comms.buoy_report import BuoyReport, decode_statustext, encode_statustext


class RobotXBuoyStatusText(StatusText):
    """
    Encodes BuoyReport as STATUSTEXT with prefix RXB|...

    Receivers should register this listener and read .report after each callback.
    """

    def __init__(
        self,
        report: BuoyReport | None = None,
        callback_func: Callable[["RobotXBuoyStatusText"], None] = lambda _msg: None,
    ):
        report = report or BuoyReport(0, "red", 0.0, 0.0, 0)
        text = encode_statustext(report)
        super().__init__(text=text, severity=MAVSeverity.INFO, cb=callback_func)
        self.report = report

    def set_report(self, report: BuoyReport) -> None:
        self.report = report
        self.text = encode_statustext(report)

    def encode(self, system_id, component_id):
        self.text = encode_statustext(self.report)
        return super().encode(system_id, component_id)

    def decode(self, msg) -> None:
        super().decode(msg)
        parsed = decode_statustext(self.text)
        if parsed is not None:
            self.report = parsed

    @thread_safe
    def __repr__(self) -> str:
        r = self.report
        return (
            f"(ROBOTX_BUOY) id={r.target_id} color={r.color} "
            f"lat={r.lat:.7f} lon={r.lon:.7f} frame={r.frame}"
        )

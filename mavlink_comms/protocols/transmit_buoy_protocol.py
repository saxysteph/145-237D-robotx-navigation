"""Send one RobotX buoy report over MAVLink."""

from __future__ import annotations

from mavlink_comms import _mavcore_path  # noqa: F401

from mavcore.mav_protocol import MAVProtocol
from mavcore.mav_receiver import Receiver
from mavcore.mav_sender import Sender

from mavlink_comms.buoy_report import BuoyReport
from mavlink_comms.messages.robotx_buoy_msg import RobotXBuoyStatusText


class TransmitBuoyProtocol(MAVProtocol):
    """Single-shot protocol: encode and send one buoy STATUSTEXT."""

    def __init__(self, report: BuoyReport):
        super().__init__()
        self.report = report
        self.msg = RobotXBuoyStatusText(self.report)

    def run(self, sender: Sender, receiver: Receiver) -> None:
        self.msg.set_report(self.report)
        sender.send_msg(self.msg)

    def __repr__(self) -> str:
        return f"TransmitBuoyProtocol({self.report})"

"""Buoy detection report encoding for MAVLink STATUSTEXT (50-byte limit)."""

from __future__ import annotations

from dataclasses import dataclass
import re

PREFIX = "RXB|"
_COLOR_TO_CODE = {"red": "r", "green": "g", "blue": "b"}
_CODE_TO_COLOR = {v: k for k, v in _COLOR_TO_CODE.items()}

# RXB|{id}|{c}|{lat_e7}|{lon_e7}|{frame}
_PATTERN = re.compile(
    r"^RXB\|(?P<id>\d+)\|(?P<color>[rgb])\|(?P<lat>-?\d+)\|(?P<lon>-?\d+)\|(?P<frame>\d+)$"
)


@dataclass(frozen=True)
class BuoyReport:
    target_id: int
    color: str
    lat: float
    lon: float
    frame: int = 0

    def __post_init__(self) -> None:
        color = str(self.color).strip().lower()
        if color not in _COLOR_TO_CODE:
            raise ValueError(f"color must be one of {sorted(_COLOR_TO_CODE)}; got {self.color!r}")
        object.__setattr__(self, "color", color)


def encode_statustext(report: BuoyReport) -> str:
    """Compact wire form for STATUSTEXT (must stay within 50 bytes)."""
    lat_e7 = int(round(report.lat * 1e7))
    lon_e7 = int(round(report.lon * 1e7))
    text = (
        f"{PREFIX}{report.target_id}|{_COLOR_TO_CODE[report.color]}"
        f"|{lat_e7}|{lon_e7}|{report.frame}"
    )
    if len(text.encode("utf-8")) > 50:
        raise ValueError(f"encoded buoy report exceeds STATUSTEXT limit: {text!r}")
    return text


def decode_statustext(text: str) -> BuoyReport | None:
    """Parse STATUSTEXT payload; returns None if not a RobotX buoy report."""
    raw = str(text).strip("\x00").strip()
    if not raw.startswith(PREFIX):
        return None
    match = _PATTERN.match(raw)
    if not match:
        return None
    color = _CODE_TO_COLOR[match.group("color")]
    return BuoyReport(
        target_id=int(match.group("id")),
        color=color,
        lat=int(match.group("lat")) / 1e7,
        lon=int(match.group("lon")) / 1e7,
        frame=int(match.group("frame")),
    )


def report_from_pipeline_payload(payload: dict) -> BuoyReport:
    """Build from visual_gps_target_mapping_pipeline transmit dict."""
    return BuoyReport(
        target_id=int(payload["target_id"]),
        color=str(payload["color"]),
        lat=float(payload["lat"]),
        lon=float(payload["lon"]),
        frame=int(payload.get("frame", 0)),
    )

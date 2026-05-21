from . import _mavcore_path  # noqa: F401

from mavlink_comms.buoy_report import BuoyReport, decode_statustext, encode_statustext
from mavlink_comms.transmitter import (
    DEFAULT_GCS_CONNECTION,
    DEFAULT_ONBOARD_CONNECTION,
)

__all__ = [
    "BuoyReport",
    "DEFAULT_GCS_CONNECTION",
    "DEFAULT_ONBOARD_CONNECTION",
    "decode_statustext",
    "encode_statustext",
]


def __getattr__(name: str):
    if name == "BuoyMavlinkTransmitter":
        from mavlink_comms.transmitter import BuoyMavlinkTransmitter

        return BuoyMavlinkTransmitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

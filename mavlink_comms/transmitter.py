"""High-level MAVLink buoy transmitter for pipeline integration."""

from __future__ import annotations

from . import _mavcore_path  # noqa: F401

from mavcore import MAVDevice

from mavlink_comms.buoy_report import BuoyReport, report_from_pipeline_payload
from mavlink_comms.protocols.transmit_buoy_protocol import TransmitBuoyProtocol

# Local loopback defaults (no SITL): GCS listens, onboard sends.
DEFAULT_GCS_CONNECTION = "udpin:0.0.0.0:14555"
DEFAULT_ONBOARD_CONNECTION = "udpout:127.0.0.1:14555"


class BuoyMavlinkTransmitter:
    """
    Sends confirmed buoy targets to a ground station over UDP.

    Wire format: STATUSTEXT with compact RXB|... payload (see buoy_report.py).
    """

    def __init__(
        self,
        connection: str = DEFAULT_ONBOARD_CONNECTION,
        *,
        source_system: int = 1,
        source_component: int = 191,
    ):
        self.connection = connection
        self.source_system = source_system
        self.source_component = source_component
        self._device: MAVDevice | None = None

    def _ensure_device(self) -> MAVDevice:
        if self._device is None:
            self._device = MAVDevice(
                self.connection,
                source_system=self.source_system,
                source_component=self.source_component,
                attempt_reconnect=False,
            )
        return self._device

    def transmit_report(self, report: BuoyReport) -> None:
        device = self._ensure_device()
        device.run_protocol(TransmitBuoyProtocol(report))

    def transmit(
        self,
        target_id: int,
        color: str,
        lat: float,
        lon: float,
        *,
        frame: int = 0,
    ) -> None:
        self.transmit_report(
            BuoyReport(target_id=target_id, color=color, lat=lat, lon=lon, frame=frame)
        )

    def transmit_from_pipeline(self, payload: dict) -> None:
        """Accept dict from visual_gps_target_mapping_pipeline.transmit_to_receiver."""
        self.transmit_report(report_from_pipeline_payload(payload))

    def close(self) -> None:
        if self._device is not None:
            self._device.stop_reading()
            self._device = None

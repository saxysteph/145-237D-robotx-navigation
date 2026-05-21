"""Offline tests for buoy STATUSTEXT codec (no MAVLink I/O)."""

from mavlink_comms.buoy_report import BuoyReport, decode_statustext, encode_statustext


def test_roundtrip() -> None:
    report = BuoyReport(7, "green", 33.6429321, -117.8262882, frame=999)
    text = encode_statustext(report)
    assert len(text.encode("utf-8")) <= 50
    back = decode_statustext(text)
    assert back == report


def test_ignore_non_buoy() -> None:
    assert decode_statustext("Armed") is None
    assert decode_statustext("RXB|bad") is None


if __name__ == "__main__":
    test_roundtrip()
    test_ignore_non_buoy()
    print("ok")

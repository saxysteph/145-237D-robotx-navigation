# Successful flight — 2025-06-02

End-to-end buoy detection demo: Jetson → MAVLink UDP → Mac GCS.

| Item | Value |
|------|--------|
| Mac IP (WiFi) | 192.168.8.184 |
| Jetson | babydragon @ 192.168.8.136 |
| GCS | `fulldemo/run_gcs_mac.sh` (UDP 14555) |
| Detection | `fulldemo/run_detection_jetson.sh` / `camera_live_feed.py` |
| Raw AVI (Jetson) | `../recording_1780463969.avi` (parent dir; remux → MP4 below) |
| Demo video (MP4) | `recording_demo_flight_mac.mp4.part-*` → see `RESTORE.md` |
| Screen capture (MOV) | `recording_demo_flight_mac_screen.mov.part-*` → see `RESTORE.md` |

## Log files

- `jetson_tx.log` — Jetson terminal `[TX]` lines (+ session header / crash tail)
- `mac_gcs_rx.log` — Mac ground station `[GCS]` JSON lines
- `combined_tx_rx.log` — both streams in chronological paste order

## Video in git

Large binaries are split for GitHub (under 100 MB per file). Run `cat …part-* > …` per `RESTORE.md`, then verify with `shasum -a 256 -c SHA256SUMS`.

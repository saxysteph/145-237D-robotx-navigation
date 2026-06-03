# Restore split video files

GitHub limits single blobs to 100 MB. Large recordings are stored as `split(1)` parts (45 MB / 30 MB chunks).

## Reassemble

From this directory:

```bash
cat recording_demo_flight_mac.mp4.part-* > recording_demo_flight_mac.mp4
cat recording_demo_flight_mac_screen.mov.part-* > recording_demo_flight_mac_screen.mov
shasum -a 256 -c SHA256SUMS
```

`shasum -c` should report OK for both restored files.

## What is committed

| Artifact | Parts |
|----------|--------|
| Jetson camera MP4 (OpenCV remux) | `recording_demo_flight_mac.mp4.part-aa`, `.part-ab` |
| Mac screen recording MOV | `recording_demo_flight_mac_screen.mov.part-aa` … `.part-af` |
| TX/RX logs | `jetson_tx.log`, `mac_gcs_rx.log`, `combined_tx_rx.log` |

Unsplit `.mp4` / `.mov` originals are gitignored to avoid duplicate storage; restore from parts above.

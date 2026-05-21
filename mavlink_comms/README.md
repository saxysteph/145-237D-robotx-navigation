# MAVLink buoy telemetry (MAVCore)

Transmits confirmed buoy/balloon detections (GPS lat/lon + color) from the onboard pipeline to a ground station using [MAVCore](https://github.com/uci-uav-forge/mavcore).

## Wire format

Reports are sent as standard MAVLink `STATUSTEXT` messages with a compact payload (≤50 bytes):

```text
RXB|{target_id}|{r|g|b}|{lat_e7}|{lon_e7}|{frame}
```

Example: `RXB|3|r|336429321|-1178262881|42`

- `r` / `g` / `b` → `red` / `green` / `blue` (teal balloons map to `green`)
- lat/lon are degrees × 1e7 (same convention as `GLOBAL_POSITION_INT`)

This avoids a custom MAVLink dialect while staying compatible with MAVCore’s message/protocol pattern.

## Setup

From the repo root:

```bash
bash mavlink_comms/scripts/setup_mavlink_env.sh
source .venv-mavlink/bin/activate
```

This installs `pymavlink` and clones [MAVCore](https://github.com/uci-uav-forge/mavcore) into `vendor/mavcore` (not pip-installable upstream).

## Local test (two terminals)

Uses UDP port **14555** so it does not conflict with ArduPilot SITL on 14550.

**Terminal 1 — ground station (laptop):**

```bash
python mavlink_comms/scripts/run_ground_station.py
```

**Terminal 2 — mock onboard sender:**

```bash
python mavlink_comms/scripts/run_mock_sender.py
```

You should see `[GCS]` JSON lines for each buoy.

**Single-process smoke test:**

```bash
python mavlink_comms/scripts/run_loopback_test.py
```

## SITL / real vehicle later

When sim or FC is on `udp:127.0.0.1:14550`, point the onboard transmitter at that endpoint and run the GCS on a separate `udpin` port, or share the mavlink router. Example onboard connection:

```bash
python mavlink_comms/scripts/run_mock_sender.py --connection udp:127.0.0.1:14550
```

## Pipeline integration

`visual_gps_target_mapping_pipeline.py` currently prints `[TRANSMIT]` JSON. To send over MAVLink:

```python
from mavlink_comms import BuoyMavlinkTransmitter

self._mavlink_tx = BuoyMavlinkTransmitter()  # udpout:127.0.0.1:14555 by default

def transmit_to_receiver(self, target_id, color, lat, lon):
    payload = {"target_id": target_id, "color": color, "lat": lat, "lon": lon, "frame": self.frame_counter}
    self.transmissions.append(payload)
    self._mavlink_tx.transmit_from_pipeline(payload)
```

Optional: load prior detections from `reconstructed_buoys.json`:

```bash
python mavlink_comms/scripts/run_mock_sender.py \
  --from-json yolo_comparison_test/path2_switch_proposal/scripts/roi_hsv_eval_center_circle_cv_rerun/test_eval/reprojection_sunset_samples_flip_fixed/robotx_dr_026_clear_blue_mild_ocean_baseline/reconstructed_buoys.json
```

(Only entries with lat/lon fields are sent.)

# Full Demo — RobotX Buoy Detection Pipeline

End-to-end guide: Jetson Orin Nano runs YOLO + HSV detection, laptop receives GPS + color over MAVLink.

---

## Prerequisites

- **Network link (choose one):**
  - **Option A — USB-C (default):** Jetson `192.168.55.1` ↔ laptop `192.168.55.100`
  - **Option B — WiFi router (recommended for field):** Jetson and laptop both join the same router SSID (e.g. `GL-AXT1800-*`) and use their router LAN IPs (e.g. `192.168.8.x`)
- Jetson venv and all packages installed (see `jetson_setup.sh`)
- `buoy_best.pt` present at `~/robotx-navigation/buoy_best.pt` on the Jetson

---

## Step 0 (Option A) — USB-C network (laptop ↔ Jetson)

Assign the laptop’s USB ethernet IP (adjust interface name):

```bash
sudo ifconfig en10 192.168.55.100 netmask 255.255.255.0
ping 192.168.55.1
```

---

## Step 0 (Option B) — Router WiFi network (recommended)

Join the same router SSID on both devices.

- **On Jetson:** connect the WiFi dongle to the router SSID and note the Jetson IP (e.g. `192.168.8.136`).
- **On laptop:** join the same SSID and note the laptop IP:

```bash
ipconfig getifaddr en0
```

For the rest of this guide, set `--gcs-ip` to the **laptop’s router IP**.

---

## Step 1 — Start the Ground Station (laptop)

Open a terminal on your laptop from the repo root:

```bash
cd ~/Downloads/SP26/CSE237D/145-237D-robotx-navigation
bash fulldemo/run_gcs_mac.sh
```

The ground station listens on UDP port `14555`. You should see:

```
Listening on udpin:0.0.0.0:14555 for RXB| buoy reports (Ctrl+C to stop)
```

Leave this running. Each received detection prints as JSON:

```
[GCS] {"target_id": 1, "color": "red", "lat": 32.88012, "lon": -117.23418, "frame": 42, "timestamp_ms": ...}
```

To save detections to a file for post-processing:

```bash
python mavlink_comms/scripts/run_ground_station.py --output-jsonl fulldemo/detections.jsonl
```

---

## Step 2 — Start Detection on the Jetson

SSH into the Jetson:

```bash
ssh babydragon@192.168.55.1   # USB-C option
# or:
ssh babydragon@<JETSON_ROUTER_IP>  # router WiFi option (e.g. 192.168.8.136)
cd ~/robotx-navigation
```

Run the full pipeline (YOLO → HSV → MAVLink transmit):

```bash
GCS_IP=<LAPTOP_IP_ON_LINK> bash fulldemo/run_detection_jetson.sh
```

Replace the placeholders with actual values. Example for benchtop testing:

```bash
# USB-C option (laptop IP is 192.168.55.100)
GCS_IP=192.168.55.100 bash fulldemo/run_detection_jetson.sh

# Router option (example laptop IP 192.168.8.184)
GCS_IP=192.168.8.184 bash fulldemo/run_detection_jetson.sh
```

**What the pipeline does:**
1. YOLO proposes bounding boxes on each frame at det resolution (960×540)
2. HSV thresholding classifies each ROI as `red`, `green`, or `blue`
3. Pixel coordinates are projected to GPS lat/lon (flat-earth, nadir camera model)
4. Each confirmed detection is transmitted as a MAVLink `STATUSTEXT` to the laptop

**Console output on the Jetson:**
- `YOLO loaded: ...` — model ready
- `MAVLink transmitter → udpout:<LAPTOP_IP_ON_LINK>:14555` — link established
- `[TX] t1 red lat=32.88012 lon=-117.23418` — live transmissions

---

## Step 3 — What to Look For

**On the Jetson terminal:**
- `[TX]` lines confirm detections are being sent
- No `[TX]` lines means nothing is passing both YOLO confidence threshold and HSV color ratio — point camera at a colored buoy/balloon

**On the laptop ground station:**
- `[GCS]` JSON lines confirm packets are arriving over the network
- If you see `[TX]` on Jetson but no `[GCS]` on laptop, check the USB network: `ifconfig en10`

**Tuning flags if detections are missed:**
```bash
--yolo-conf 0.15        # lower YOLO threshold (more proposals, more false positives)
--min-color-ratio 0.08  # lower HSV color ratio gate
```

---

## Step 4 — Post-Processing: Get Video from the Jetson

The Jetson saves raw video to `~/robotx-navigation/detection_logs/recording_<unix_ts>.avi`.
The detection CSV is at `~/robotx-navigation/detection_logs/detections.csv`.

**Copy both to your laptop:**

```bash
rsync -av babydragon@192.168.55.1:~/robotx-navigation/detection_logs/ \
  ~/Downloads/SP26/CSE237D/145-237D-robotx-navigation/fulldemo/session_data/
```

**Cross-referencing detections with video:**

The video filename contains its start Unix timestamp (e.g. `recording_1748000000.avi`).
Each CSV row has a `timestamp` column (Unix seconds). To find a detection in the video:

```
video_offset_seconds = detection_timestamp - recording_start_timestamp
```

Use VLC or ffmpeg to seek to that offset:

```bash
ffmpeg -ss <offset> -i recording_<ts>.avi -frames:v 1 frame_at_detection.jpg
```

---

## Step 5 — Visualize Received Coordinates

Run the coordinate visualizer on your laptop against the saved JSONL file:

```bash
cd ~/Downloads/SP26/CSE237D/145-237D-robotx-navigation
python fulldemo/visualize_detections.py fulldemo/detections.jsonl
```

This opens an interactive dot map — each buoy detection is a colored dot at its estimated GPS position.

To visualize live while the ground station is running, pass the same `--output-jsonl` file:

```bash
# Terminal 1 — ground station writing to file
python mavlink_comms/scripts/run_ground_station.py --output-jsonl fulldemo/detections.jsonl

# Terminal 2 — live visualizer (polls the file)
python fulldemo/visualize_detections.py fulldemo/detections.jsonl --live
```

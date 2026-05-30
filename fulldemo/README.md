# Full Demo — RobotX Buoy Detection Pipeline

End-to-end guide: Jetson Orin Nano runs YOLO + HSV detection, laptop receives GPS + color over MAVLink.

---

## Prerequisites

- Jetson connected to laptop via USB-C (`192.168.55.1`)
- Laptop USB-ethernet interface assigned: `sudo ifconfig en10 192.168.55.100 netmask 255.255.255.0`
- Jetson venv and all packages installed (see `jetson_setup.sh`)
- `buoy_best.pt` present at `~/robotx-navigation/buoy_best.pt` on the Jetson

---

## Step 1 — Start the Ground Station (laptop)

Open a terminal on your laptop from the repo root:

```bash
cd ~/Downloads/SP26/CSE237D/145-237D-robotx-navigation
source .venv-mavlink/bin/activate
python mavlink_comms/scripts/run_ground_station.py
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
ssh babydragon@192.168.55.1
cd ~/robotx-navigation
```

Run the full pipeline (YOLO → HSV → MAVLink transmit):

```bash
python3 camera_live_feed.py \
  --headless \
  --save-video \
  --camera-index 0 \
  --yolo-model buoy_best.pt \
  --gcs-ip 192.168.55.100 \
  --drone-lat <DRONE_LAT> \
  --drone-lon <DRONE_LON> \
  --altitude-m <ALTITUDE_M> \
  --heading-deg <HEADING_DEG>
```

Replace the placeholders with actual values. Example for benchtop testing:

```bash
python3 camera_live_feed.py \
  --headless \
  --save-video \
  --camera-index 0 \
  --yolo-model buoy_best.pt \
  --gcs-ip 192.168.55.100 \
  --drone-lat 32.88010 \
  --drone-lon -117.23420 \
  --altitude-m 10 \
  --heading-deg 0
```

**What the pipeline does:**
1. YOLO (`buoy_best.pt`) proposes bounding boxes on each frame at det resolution (960×540)
2. HSV thresholding classifies each ROI as `red`, `green`, or `blue`
3. Pixel coordinates are projected to GPS lat/lon (flat-earth, nadir camera model)
4. Each confirmed detection is transmitted as a MAVLink `STATUSTEXT` to the laptop

**Console output on the Jetson:**
- `YOLO loaded: buoy_best.pt` — model ready
- `MAVLink transmitter → udpout:192.168.55.100:14555` — link established
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

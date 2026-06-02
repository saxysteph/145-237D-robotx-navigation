# Setup & Replication

This page covers how to replicate the full pipeline from scratch: setting up the Jetson Orin Nano, installing dependencies, and running the end-to-end buoy detection demo.

For a quick-start guide focused on running (not setting up), see [fulldemo/PARTNER_INSTRUCTIONS.md](../blob/main/fulldemo/PARTNER_INSTRUCTIONS.md).

---

## Hardware Requirements

| Component | Spec |
|---|---|
| Edge computer | NVIDIA Jetson Orin Nano (JetPack 5 / R35.x) |
| Camera | USB H264 camera |
| Host laptop | macOS or Linux with Python 3.9+ |
| Link | USB-C cable (ethernet gadget mode) or USB WiFi dongle |

---

## Jetson Setup

### 1. Connect via USB-C

After plugging in the Jetson:

```bash
# On the laptop — assign USB ethernet interface
sudo ifconfig en10 192.168.55.100 netmask 255.255.255.0
# (replace en10 with your actual interface — check with: networksetup -listallhardwareports)

# Verify
ping 192.168.55.1

# SSH in
ssh babydragon@192.168.55.1   # [password: ask repo owners]
```

### 2. Clone the Repo (Jetson)

```bash
cd ~
git clone --no-local --depth=1 --filter=blob:none \
  https://github.com/saxysteph/145-237D-robotx-navigation.git robotx-navigation
cd robotx-navigation
```

### 3. Install Python Dependencies (Jetson)

```bash
# System packages
sudo apt-get install -y python3-pip python3-opencv \
    libxml2-dev libxslt-dev libopenblas-dev \
    libcudnn8 cuda-toolkit-11-4

# PyTorch for JetPack 5 (from NVIDIA wheel index)
pip3 install --no-cache \
  https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# pymavlink (no lxml build — install separately)
pip3 install --break-system-packages future pymavlink

# Ultralytics (no-deps to avoid polars/numpy conflicts)
pip3 install --break-system-packages ultralytics --no-deps
pip3 install --break-system-packages onnxruntime opencv-python-headless \
    numpy pillow pyyaml tqdm psutil py-cpuinfo
```

> **Note:** torchvision must be compiled from source for Python 3.8 / aarch64. See `jetson_setup.sh` for the full sequence.

### 4. Patch Ultralytics for numpy 1.x

The Jetson is limited to numpy ≤ 1.24.4. Ultralytics ships checks that reject it:

```bash
# Patch out numpy version check
sed -i 's/.*requires numpy>=1.26.1.*/#&/' \
  ~/.local/lib/python3.8/site-packages/ultralytics/nn/tasks.py

# Patch onnxruntime provider selection (avoids GPU retry spam)
sed -i "s/'onnxruntime-gpu'/'onnxruntime'/" \
  ~/.local/lib/python3.8/site-packages/ultralytics/engine/exporter.py
```

### 5. Copy the Model

```bash
# From laptop
scp buoy_best.onnx babydragon@192.168.55.1:~/robotx-navigation/buoy_best.onnx
```

---

## Laptop Setup

```bash
cd ~/Downloads/SP26/CSE237D/145-237D-robotx-navigation
python3 -m venv .venv-mavlink
source .venv-mavlink/bin/activate
pip install pymavlink future matplotlib
```

---

## Running the Pipeline

### Step 1 — Start Ground Station (laptop)

```bash
cd ~/Downloads/SP26/CSE237D/145-237D-robotx-navigation
source .venv-mavlink/bin/activate

# Print detections to terminal
python mavlink_comms/scripts/run_ground_station.py

# Or save to file for visualization
python mavlink_comms/scripts/run_ground_station.py --output-jsonl fulldemo/detections.jsonl
```

Expected output:
```
Listening on udpin:0.0.0.0:14555 for RXB| buoy reports (Ctrl+C to stop)
[GCS] {"target_id": 1, "color": "red", "lat": 32.88012, "lon": -117.23418, "frame": 42}
```

### Step 2 — Start Detection (Jetson, via SSH)

```bash
ssh babydragon@192.168.55.1
cd ~/robotx-navigation

python3 camera_live_feed.py \
  --headless \
  --save-video \
  --camera-index 0 \
  --yolo-model buoy_best.onnx \
  --gcs-ip 192.168.55.100 \
  --drone-lat 32.88010 \
  --drone-lon -117.23420 \
  --altitude-m 10 \
  --heading-deg 0
```

### Step 3 — Visualize Detections (laptop)

```bash
# Static (post-run)
python fulldemo/visualize_detections.py fulldemo/detections.jsonl

# Live (while session is running)
python fulldemo/visualize_detections.py fulldemo/detections.jsonl --live
```

---

## Retrieving Video from Jetson

```bash
rsync -av babydragon@192.168.55.1:~/robotx-navigation/detection_logs/ \
  ~/Downloads/SP26/CSE237D/145-237D-robotx-navigation/fulldemo/session_data/
```

Cross-reference a detection timestamp with video:
```bash
# video_offset = detection_timestamp - recording_start_timestamp
ffmpeg -ss <offset_seconds> -i recording_<ts>.avi -frames:v 1 frame_at_detection.jpg
```

---

## Tuning

| Flag | Default | Effect |
|---|---|---|
| `--yolo-conf` | 0.25 | Lower = more proposals, more false positives |
| `--min-color-ratio` | 0.10 | Lower = classify ROIs with weaker color signal |
| `--altitude-m` | required | Affects GPS projection scale |
| `--heading-deg` | required | Affects GPS projection direction |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ping 192.168.55.1` fails | Re-assign: `sudo ifconfig en10 192.168.55.100 netmask 255.255.255.0` |
| `[TX]` on Jetson but no `[GCS]` | USB network IP dropped; re-assign on laptop |
| Nothing detected | Lower `--yolo-conf` and `--min-color-ratio`; point camera at a brightly colored object |
| Camera not found | Try `--camera-index 1` or `2` |
| `YOLO loaded` not printed | Check `buoy_best.onnx` exists in `~/robotx-navigation/` |

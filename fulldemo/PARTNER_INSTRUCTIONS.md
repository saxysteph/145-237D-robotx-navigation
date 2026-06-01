# Partner Instructions — RobotX Buoy Detection Demo

End-to-end guide for running the buoy detection pipeline: Jetson Orin Nano detects buoys via YOLO + HSV, laptop receives GPS + color over MAVLink.

---

## Hardware Setup

- **Network link (choose one):**
  - **Option A — USB-C cable:** Jetson `192.168.55.1` ↔ laptop `192.168.55.100`
  - **Option B — Router WiFi (recommended):** Jetson and laptop both join the same router SSID (e.g. `GL-AXT1800-*`) and use their router LAN IPs (e.g. `192.168.8.x`)
- H264 camera connected to Jetson
- `buoy_best.onnx` present at `~/robotx-navigation/buoy_best.onnx` on the Jetson

---

## Step 1 — Assign USB Network on Laptop

After plugging in the USB-C cable, the Jetson appears as a USB ethernet device. You must assign it an IP:

```bash
sudo ifconfig en10 192.168.55.100 netmask 255.255.255.0
```

> **Note:** The interface may not be `en10`. Check with:
> ```bash
> networksetup -listallhardwareports | grep -A2 "Linux\|Tegra\|USB"
> ```
> Use whichever interface name appears next to the Jetson entry.

Verify the link is up:

```bash
ping 192.168.55.1
```

You should get replies. If not, try unplugging and replugging the USB-C cable, then reassign the IP.

### If ping fails after replugging

```bash
# List interfaces and find the Jetson one (named en9, en10, en11, etc.)
ifconfig | grep -E "^en|inet 192"

# Assign IP to whichever interface shows the Jetson
sudo ifconfig enXX 192.168.55.100 netmask 255.255.255.0
```

---

## Step 1B — Router WiFi Option (recommended)

If you have a portable router, connect both laptop and Jetson to the router SSID.

- **Laptop IP (router):**

```bash
ipconfig getifaddr en0
```

- **Jetson IP (router):** check on Jetson:

```bash
hostname -I
ip -4 addr show wlan0
```

Use the laptop’s router IP as `--gcs-ip` when running detection on the Jetson.

---

## Step 2 — SSH into the Jetson

```bash
ssh babydragon@192.168.55.1
# password: companion
```

---

## Step 3 — Start the Ground Station (laptop)

Open a **new terminal** on the laptop from the repo root:

```bash
cd ~/Downloads/SP26/CSE237D/145-237D-robotx-navigation
bash fulldemo/run_gcs_mac.sh
```

To save detections to a file for post-processing and visualization:

```bash
python mavlink_comms/scripts/run_ground_station.py --output-jsonl fulldemo/detections.jsonl
```

You should see:
```
Listening on udpin:0.0.0.0:14555 for RXB| buoy reports (Ctrl+C to stop)
```

Each received detection prints as:
```
[GCS] {"target_id": 1, "color": "red", "lat": 32.88012, "lon": -117.23418, "frame": 42, ...}
```

Leave this running.

---

## Step 4 — Start Detection on the Jetson

In your SSH session:

```bash
cd ~/robotx-navigation
source venv/bin/activate   # if venv exists, otherwise skip
```

Run the full pipeline:

```bash
GCS_IP=<LAPTOP_IP_ON_LINK> bash fulldemo/run_detection_jetson.sh
```

Example for benchtop testing (fake GPS):

```bash
# USB-C option
GCS_IP=192.168.55.100 bash fulldemo/run_detection_jetson.sh

# Router option (example laptop IP 192.168.8.184)
GCS_IP=192.168.8.184 bash fulldemo/run_detection_jetson.sh
```

### What to look for

**On the Jetson terminal:**
- `YOLO loaded: buoy_best.onnx` — model ready
- `MAVLink transmitter → udpout:192.168.55.100:14555` — link established
- `[TX] t1 red lat=32.88012 lon=-117.23418` — live transmissions

**On the laptop ground station:**
- `[GCS]` JSON lines confirm packets are arriving

**If `[TX]` appears on Jetson but no `[GCS]` on laptop:**
```bash
# On laptop — check USB network is still assigned
ifconfig en10
# If no inet address, reassign:
sudo ifconfig en10 192.168.55.100 netmask 255.255.255.0
```

**If nothing is detected:**
```bash
# Lower thresholds
python3 camera_live_feed.py ... --yolo-conf 0.15 --min-color-ratio 0.08
```

---

## Step 5 — Visualize Received Coordinates

```bash
# Static (after run)
python fulldemo/visualize_detections.py fulldemo/detections.jsonl

# Live (while ground station is running)
python fulldemo/visualize_detections.py fulldemo/detections.jsonl --live
```

---

## Step 6 — Retrieve Video from Jetson

The Jetson saves video to `~/robotx-navigation/detection_logs/recording_<unix_ts>.avi`.

Copy to laptop:

```bash
rsync -av babydragon@192.168.55.1:~/robotx-navigation/detection_logs/ \
  ~/Downloads/SP26/CSE237D/145-237D-robotx-navigation/fulldemo/session_data/
```

Cross-reference a detection with video:
```bash
# video_offset = detection_timestamp - recording_start_timestamp
ffmpeg -ss <offset_seconds> -i recording_<ts>.avi -frames:v 1 frame_at_detection.jpg
```

---

## Recovery Cheatsheet

| Problem | Fix |
|---|---|
| `ping 192.168.55.1` fails | `sudo ifconfig en10 192.168.55.100 netmask 255.255.255.0` (adjust interface name) |
| SSH refuses connection | Check ping first; Jetson may still be booting (~30s after power-on) |
| Ground station shows nothing | Check `[TX]` on Jetson; if present, USB network IP dropped — reassign |
| Camera not found | Try `--camera-index 1` or `--camera-index 2` |
| YOLO not loading | Confirm `buoy_best.onnx` exists: `ls ~/robotx-navigation/buoy_best.onnx` |
| SSH drops mid-run | Pipeline keeps running on Jetson; re-SSH and check `detection_logs/` for saved video |

---

## Wireless Option (future)

When a USB WiFi dongle is available on the Jetson, the pipeline works identically — just replace `192.168.55.100` with the laptop's IP on the shared WiFi network. No code changes needed.

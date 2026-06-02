# Repository Structure

```
145-237D-robotx-navigation/
│
├── camera_live_feed.py          # Main pipeline: YOLO → HSV → GPS projection → MAVLink TX
├── color_utils.py               # HSV color range definitions and fallback ranges
│
├── captures/
│   └── classes/                 # Reference images used to derive HSV color ranges
│       ├── red/
│       ├── green/
│       └── blue/
│
├── mavlink_comms/
│   ├── transmitter.py           # BuoyMavlinkTransmitter — wraps pymavlink UDP send
│   └── scripts/
│       └── run_ground_station.py  # Laptop GCS: listens on UDP 14555, logs detections
│
├── vendor/
│   └── mavcore/                 # Vendored pymavlink core (patched for Python 3.8)
│
├── fulldemo/
│   ├── README.md                # Full demo guide (GCS startup, Jetson startup, post-processing)
│   ├── PARTNER_INSTRUCTIONS.md  # Step-by-step guide for partner to run the pipeline
│   ├── visualize_detections.py  # Dot-map visualizer for received GPS detections
│   ├── bluetooth_comms_plan.md  # Planned BT PAN setup (not yet implemented)
│   └── session_data/            # (gitignored) Video and CSV logs from field sessions
│
├── wiki/                        # Source for GitHub Wiki pages
│   ├── Home.md
│   ├── Team.md
│   ├── Project-Overview.md
│   ├── Repository-Structure.md
│   └── Setup-and-Replication.md
│
├── jetson_setup.sh              # Dependency installation script for Jetson Orin Nano
├── buoy_best.onnx               # Trained YOLO model (ONNX opset 19, buoy/balloon detector)
└── detection_logs/              # (gitignored) Saved video and detection CSVs from Jetson
```

---

## Key Files

### `camera_live_feed.py`
The main entry point for the Jetson. Orchestrates the full pipeline:
- Opens the H264 USB camera
- Loads `buoy_best.onnx` via Ultralytics YOLO
- For each frame: runs YOLO for bounding boxes → crops ROIs → classifies color via HSV
- Projects pixel centroids to GPS lat/lon using drone altitude, heading, and camera FOV
- Transmits each confirmed detection as a MAVLink STATUSTEXT packet over UDP
- Optionally saves raw video to `detection_logs/`

Key flags:
```
--headless          Run without display (required on Jetson)
--save-video        Save raw video to detection_logs/
--yolo-model        Path to ONNX model (default: buoy_best.onnx)
--yolo-conf         YOLO confidence threshold (default: 0.25)
--gcs-ip            Ground station IP (default: 192.168.55.100)
--gcs-port          Ground station UDP port (default: 14555)
--drone-lat/lon     Drone GPS position (required for coordinate projection)
--altitude-m        Drone altitude in meters
--heading-deg       Drone heading in degrees (0=North)
--min-color-ratio   Minimum HSV color ratio to confirm a detection
```

### `mavlink_comms/scripts/run_ground_station.py`
Runs on the laptop. Listens on `udpin:0.0.0.0:14555` for incoming MAVLink packets, decodes the `RXB|` payload, and prints JSON. Optionally writes to a `.jsonl` file for post-processing and visualization.

### `fulldemo/visualize_detections.py`
Reads a `.jsonl` detection log and renders a color-coded GPS dot map using matplotlib. Supports `--live` mode to poll the file while a session is running.

### `color_utils.py`
Defines `FALLBACK_COLOR_RANGES` — the HSV hue/saturation/value bounds for red, green, and blue. Used when derived ranges from `captures/classes/` images are found to be degenerate.

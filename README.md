# 145-237D-robotx-navigation

Repository for CSE145/237D Spring 2026 course project with RobotX: drone CV buoy/balloon detection, color classification, and navigation support tooling.

## Environment Setup

Create and activate the conda environment:

```bash
conda create -y -n robotx python=3.10 opencv numpy pip
conda activate robotx
```

If you already created it earlier:

```bash
conda activate robotx
```

## Camera Permission (macOS)

If OpenCV cannot open the camera, grant camera access to the app launching Python:

- System Settings -> Privacy & Security -> Camera
- Enable access for Terminal / iTerm / Cursor (whichever you use)

Optional reset if needed:

```bash
tccutil reset Camera
```

## Scripts

### 1) Live feed + object/color pipeline

`camera_live_feed.py` runs:
- object-first detection (candidate blobs),
- then HSV color classification (`red`, `green`, `yellow`, `orange`),
- ROI cropping, CLAHE, morphology, size gating, and Kalman tracking,
- detection logging to `detection_logs/`.

Run:

```bash
python camera_live_feed.py --camera-index 0
```

Useful flags:

```bash
python camera_live_feed.py \
  --camera-index 0 \
  --altitude-m 10 \
  --fx-px 1500 \
  --target-diameter-m 0.32 \
  --det-width 960 --det-height 540
```

Hotkeys:
- `q`: quit
- `c`: calibrate S/V threshold floor for selected `--calib-color`

### 2) Capture images by spacebar

`camera_capture_spacebar.py` opens a preview and saves a JPG whenever spacebar is pressed.

Run:

```bash
python camera_capture_spacebar.py --camera-index 0
```

Example with output folder:

```bash
python camera_capture_spacebar.py --camera-index 0 --output-dir captures --prefix capture
```

Hotkeys:
- `Spacebar`: save frame
- `q`: quit

### 3) Batch HSV detection on saved captures

`hsv_batch_detect.py`:
- uses reference class images in `captures/classes` (e.g., `red.png`, `green.png`, `blue.png`),
- derives HSV ranges per class,
- detects on all images in `captures/`,
- saves outputs to `captures/hsv_results/`.

Run:

```bash
python hsv_batch_detect.py
```

Outputs:
- `captures/hsv_results/detections.csv`
- `captures/hsv_results/annotated/*.jpg`

## Typical Workflow

1. Capture dataset:

```bash
python camera_capture_spacebar.py --camera-index 0 --output-dir captures --prefix capture
```

2. Run batch HSV detection:

```bash
python hsv_batch_detect.py
```

3. Inspect annotated images and `detections.csv` for threshold tuning.

# 145-237D-robotx-navigation

Repository for CSE145/237D Spring 2026 course project with RobotX: drone CV buoy/balloon detection, color classification, and navigation support tooling.

Current Stage-A pipeline is standardized to exactly 3 classes:
- `red`
- `green` (teal/cyan physical balloon)
- `blue`

## Architecture Baselines (for comparison)

Baseline captured before replacing `hsv_batch_detect.py` with object-proposal-first ROI classification:

- Pipeline: color-first HSV contour detection + geometric filtering
- Dataset: `captures/` (110 images)
- Total detections: `868`
- Per class:
  - `red`: `225` (avg conf `0.702`)
  - `green`: `415` (avg conf `0.723`)
  - `blue`: `228` (avg conf `0.687`)
- Overall average confidence: `0.708`

First run after replacing with object-proposal-first ROI classification:

- Pipeline: color-agnostic proposal stage (edges + saturation saliency) -> one ROI per proposal -> all-color HSV scoring within ROI
- Dataset: `captures/` (110 images)
- Total detections: `765`
- Per class:
  - `red`: `177` (avg conf `0.583`)
  - `green`: `361` (avg conf `0.667`)
  - `blue`: `227` (avg conf `0.701`)
- Overall average confidence: `0.658`

Hybrid proposals run (object-agnostic + color-mask proposals merged with NMS):

- Pipeline: hybrid proposal stage (edges/saturation object proposals + legacy color-mask proposals) -> one ROI per proposal -> all-color HSV scoring within ROI
- Dataset: `captures/` (110 images)
- Total detections: `1067`
- Per class:
  - `red`: `308` (avg conf `0.589`)
  - `green`: `485` (avg conf `0.677`)
  - `blue`: `274` (avg conf `0.654`)
- Overall average confidence: `0.646`

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
- then HSV color classification (`red`, `green`, `blue`),
- HSV ranges loaded at startup from `captures/classes` via shared `color_utils.py`,
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
- derives HSV ranges per class using shared `color_utils.py`,
- detects on all images in `captures/`,
- applies additional false-positive filtering (higher min area, aspect ratio, solidity, border exclusion),
- saves outputs to `captures/hsv_results/`.

Run:

```bash
python hsv_batch_detect.py
```

Outputs:
- `captures/hsv_results/detections.csv`
- `captures/hsv_results/annotated/*.jpg`

### 4) Augmentation robustness demo (presentation)

`augment_test.py` runs detector on one clean capture plus 3 UAV-noise augmentations (blur, motion blur, brightness shift, glare, sensor noise), then builds a labeled 2x2 grid.

Run:

```bash
python augment_test.py
```

Optional input image:

```bash
python augment_test.py captures/myframe.jpg
```

Output:
- `captures/hsv_results/augmentation_test.jpg`

### 5) Metrics summary + chart (presentation)

`metrics_summary.py` reads `captures/hsv_results/detections.csv`, prints per-class metrics and summary line, and saves an OpenCV-only bar chart.

Run:

```bash
python metrics_summary.py
```

Output:
- `captures/hsv_results/metrics.png`

### 6) Presentation-ready diagram

`visualize_results.py` renders a single-slide style diagram (1400x900) with headline metrics and before/after comparison.

Run:

```bash
python visualize_results.py
```

Output:
- `captures/hsv_results/results_diagram.png`

## Typical Workflow

1. Capture dataset:

```bash
python camera_capture_spacebar.py --camera-index 0 --output-dir captures --prefix capture
```

2. Run batch HSV detection:

```bash
python hsv_batch_detect.py
```

3. Generate summary metrics:

```bash
python metrics_summary.py
```

4. Generate augmentation demo panel:

```bash
python augment_test.py
```

5. Generate presentation diagram:

```bash
python visualize_results.py
```

6. Inspect `captures/hsv_results/` outputs for tuning and presentation artifacts.

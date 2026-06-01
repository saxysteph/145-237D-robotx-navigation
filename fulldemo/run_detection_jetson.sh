#!/bin/bash
# Jetson detection pipeline — run ON Jetson (SSH babydragon@192.168.8.136)
# Usage: GCS_IP=192.168.8.XXX bash fulldemo/run_detection_jetson.sh
set -euo pipefail

GCS_IP="${GCS_IP:?Set GCS_IP to your Mac IP on GL-AXT1800 (ipconfig getifaddr en0 on Mac)}"
ROOT="${HOME}/robotx-navigation"
cd "$ROOT"

MODEL=""
for c in \
  "${ROOT}/buoy_best.onnx" \
  "${ROOT}/yolo_comparison_test/path2_switch_proposal/demo_preserved/weights/buoy_balloon_roboflow_best.onnx" \
  "${ROOT}/yolo_comparison_test/path2_switch_proposal/demo_preserved/weights/buoy_balloon_roboflow_best.pt"; do
  if [[ -f "$c" ]]; then
    MODEL="$c"
    break
  fi
done

if [[ -z "$MODEL" ]]; then
  echo "No YOLO model found under ${ROOT}. Copy buoy_best.onnx first."
  exit 1
fi

echo "Model: $MODEL"
echo "GCS IP: $GCS_IP"

python3 camera_live_feed.py \
  --headless \
  --save-video \
  --camera-index 0 \
  --yolo-model "$MODEL" \
  --yolo-conf 0.25 \
  --min-color-ratio 0.12 \
  --gcs-ip "$GCS_IP" \
  --drone-lat 32.88010 \
  --drone-lon -117.23420 \
  --altitude-m 10 \
  --heading-deg 0

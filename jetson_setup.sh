#!/usr/bin/env bash
# jetson_setup.sh — run once on the Jetson Orin Nano to prepare the detection pipeline.
# Usage: bash jetson_setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv-mavlink"

echo "=== [1/5] System packages ==="
sudo apt-get update -qq
# python3-opencv: Jetson-optimized build (avoids recompiling from pip)
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-opencv \
    git \
    v4l-utils

echo ""
echo "=== [2/5] Python venv ==="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo "Created venv at $VENV_DIR (with system-site-packages so python3-opencv is visible)"
else
    echo "Venv already exists at $VENV_DIR, skipping creation."
fi

source "$VENV_DIR/bin/activate"

echo ""
echo "=== [3/5] Python packages ==="
pip install --upgrade pip --quiet
pip install --quiet \
    numpy \
    pymavlink \
    future

echo ""
echo "=== [4/5] MAVCore (vendor clone) ==="
MAVCORE_DIR="$REPO_DIR/vendor/mavcore"
if [ ! -d "$MAVCORE_DIR/.git" ]; then
    mkdir -p "$REPO_DIR/vendor"
    git clone --depth 1 https://github.com/uci-uav-forge/mavcore.git "$MAVCORE_DIR"
    echo "Cloned MAVCore into $MAVCORE_DIR"
else
    echo "MAVCore already cloned, pulling latest..."
    git -C "$MAVCORE_DIR" pull --ff-only || echo "(pull failed — using cached version)"
fi

echo ""
echo "=== [5/5] Camera check ==="
echo "Available video devices:"
v4l2-ctl --list-devices 2>/dev/null || ls /dev/video* 2>/dev/null || echo "(no /dev/video* found yet)"

echo ""
echo "======================================================"
echo " Setup complete."
echo " Activate the venv with:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " Run the detector (headless, sending to laptop GCS):"
echo "   python3 camera_live_feed.py \\"
echo "     --headless \\"
echo "     --camera-index 0 \\"
echo "     --gcs-ip <YOUR_LAPTOP_IP> \\"
echo "     --drone-lat 32.88010 --drone-lon -117.23420"
echo ""
echo " Run the ground station on your laptop:"
echo "   source .venv-mavlink/bin/activate"
echo "   python mavlink_comms/scripts/run_ground_station.py"
echo "======================================================"

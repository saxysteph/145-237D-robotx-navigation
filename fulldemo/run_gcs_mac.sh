#!/bin/bash
# Mac ground station — run on GL-AXT1800-99a WiFi (same network as Jetson 192.168.8.136)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv-mavlink ]]; then
  echo "Creating .venv-mavlink..."
  bash mavlink_comms/scripts/setup_mavlink_env.sh
fi
source .venv-mavlink/bin/activate

MAC_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
echo "Mac IP on WiFi: ${MAC_IP:-unknown}"
echo "Jetson should use: --gcs-ip ${MAC_IP:-<mac-ip>}"
echo "Listening UDP 14555 for buoy reports (Ctrl+C to stop)"
echo

python mavlink_comms/scripts/run_ground_station.py --output-jsonl fulldemo/detections.jsonl

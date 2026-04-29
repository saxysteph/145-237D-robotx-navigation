#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gz_env.sh"

WORLD_PATH="${1:-/Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation/gazebo/worlds/robotx_task1_uav_view.sdf}"

exec gz sim -v4 -s -r "${WORLD_PATH}"

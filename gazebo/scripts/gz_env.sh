#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/xurui/Downloads/SP26/CSE237D/145-237D-robotx-navigation"
WAVES_ROOT="$HOME/robotx_sim_ws/src/asv_wave_sim"
VRX_ROOT="$HOME/robotx_sim_ws/src/vrx"

export GZ_SIM_RESOURCE_PATH="${WAVES_ROOT}/gz-waves-models/models:${WAVES_ROOT}/gz-waves-models/world_models:${WAVES_ROOT}/gz-waves-models/worlds:${REPO_ROOT}/gazebo/models:${VRX_ROOT}/vrx_gz/models"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${WAVES_ROOT}/gz-waves/build/lib:${REPO_ROOT}/gazebo/plugins/robotx_beacon_plugin/build"
export GZ_RENDERING_PLUGIN_PATH="${WAVES_ROOT}/gz-waves/build/lib"

echo "GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}"
echo "GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH}"
echo "GZ_RENDERING_PLUGIN_PATH=${GZ_RENDERING_PLUGIN_PATH}"

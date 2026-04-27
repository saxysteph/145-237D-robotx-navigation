#!/usr/bin/env bash
# No -u: ROS setup.bash references vars that may be unset (e.g. AMENT_TRACE_SETUP_FILES).
set -eo pipefail

# End-to-end setup + launch for Ubuntu 22.04, ROS 2 Humble, Gazebo Harmonic
# Run from repository root:
#   bash simulation/run_end_to_end.sh
#
# VRX: use branch "jazzy" (gz-sim8 / Harmonic). Branch "humble" is Garden (gz-sim7) and will not build against Harmonic.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VRX_WS="${HOME}/vrx_ws"

source /opt/ros/humble/setup.bash

mkdir -p "${VRX_WS}/src"
VRX_GZ_CMAKE="${VRX_WS}/src/vrx/vrx_gz/CMakeLists.txt"
if [[ -f "${VRX_GZ_CMAKE}" ]] && grep -q 'find_package(gz-sim7' "${VRX_GZ_CMAKE}"; then
  echo "Detected VRX humble (Gazebo Garden). Replacing with jazzy (Gazebo Harmonic)..."
  rm -rf "${VRX_WS}/src/vrx"
fi
if [[ ! -d "${VRX_WS}/src/vrx/.git" ]]; then
  git clone https://github.com/osrf/vrx -b jazzy "${VRX_WS}/src/vrx"
fi

cd "${VRX_WS}"
rosdep update
rosdep install --from-paths src --ignore-src -r -y
# No --symlink-install: on WSL, ament symlink-install can fail for large resource trees.
colcon build
source "${VRX_WS}/install/setup.bash"

mkdir -p "${VRX_WS}/src/vrx/vrx_gz/worlds"
mkdir -p "${VRX_WS}/src/vrx/vrx_gz/models/ucsd_drone"
mkdir -p "${VRX_WS}/src/vrx/vrx_gz/config"
mkdir -p "${VRX_WS}/src/vrx/vrx_gz/launch"

cp "${REPO_ROOT}/simulation/gazebo/worlds/ucsd_robotx_demo.world.sdf" "${VRX_WS}/src/vrx/vrx_gz/worlds/"
cp "${REPO_ROOT}/simulation/gazebo/models/ucsd_drone/model.sdf" "${VRX_WS}/src/vrx/vrx_gz/models/ucsd_drone/"
cp "${REPO_ROOT}/simulation/ros2_bridge/bridge.yaml" "${VRX_WS}/src/vrx/vrx_gz/config/"
cp "${REPO_ROOT}/simulation/gazebo/launch/ucsd_demo_launch.py" "${VRX_WS}/src/vrx/vrx_gz/launch/"

cd "${VRX_WS}"
colcon build
source "${VRX_WS}/install/setup.bash"

echo "=== Launching custom demo ==="
ros2 launch vrx_gz ucsd_demo_launch.py

#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source "${HOME}/vrx_ws/install/setup.bash"

echo "Checking required topics..."
ros2 topic list | grep -E "^/drone/camera$|^/clock$" || {
  echo "Required topics not found yet. Ensure demo launch is running."
  exit 1
}

echo "Sampling /clock..."
ros2 topic echo /clock --once

echo "Camera topic info..."
ros2 topic info /drone/camera

echo "Verification passed."

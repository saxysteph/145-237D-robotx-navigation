#!/usr/bin/env bash
# Source before: gz sim, ros2 launch, ros_gz_bridge, record_uav_dataset.py (rclpy).
#
# Typical (from repo README):
#   export REPO_ROOT="/path/to/145-237D-robotx-navigation"
#   source "$REPO_ROOT/gazebo/scripts/setup_robotx_sim_env.sh"
#
# Sets: GZ_SIM_* (gz_env.sh), ROS_DISTRO via /opt/ros/*/setup.bash, optional ~/robotx_sim_ws/install.
#
# Override ROS location (non-standard install, another machine’s mount, etc.):
#   export ROS_SETUP_BASH="/your/path/to/setup.bash"
#   source "$REPO_ROOT/gazebo/scripts/setup_robotx_sim_env.sh"

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${_SCRIPT_DIR}/../.." && pwd)}"

# Gazebo resource + wave plugins (see gazebo/README.md §4 / §8).
# shellcheck source=gz_env.sh
source "${_SCRIPT_DIR}/gz_env.sh"

# ROS 2 (Linux .deb install — not bundled with Homebrew Python).
ROS_SETUP=""
if [[ -n "${ROS_SETUP_BASH:-}" && -f "${ROS_SETUP_BASH}" ]]; then
  # shellcheck disable=SC1090
  source "${ROS_SETUP_BASH}"
  ROS_SETUP="${ROS_SETUP_BASH}"
  echo "Sourced ROS (ROS_SETUP_BASH): ${ROS_SETUP}"
else
  for _try in /opt/ros/jazzy/setup.bash /opt/ros/humble/setup.bash /opt/ros/iron/setup.bash; do
    if [[ -f "${_try}" ]]; then
      # shellcheck disable=SC1090
      source "${_try}"
      ROS_SETUP="${_try}"
      break
    fi
  done
fi
if [[ -z "${ROS_SETUP}" ]]; then
  echo "setup_robotx_sim_env.sh: ROS 2 setup.bash not found (tried ROS_SETUP_BASH, then /opt/ros/{jazzy,humble,iron})." >&2
  echo "  Ubuntu 24.04: sudo apt install ros-jazzy-ros-gz-bridge ros-jazzy-gz-sim-vendor  (see gazebo/README.md §6)." >&2
  echo "  Or set: export ROS_SETUP_BASH=/path/to/setup.bash" >&2
  echo "  Discover: bash \"$REPO_ROOT/gazebo/scripts/find_ros_env.sh\"" >&2
else
  if [[ -z "${ROS_SETUP_BASH:-}" ]]; then
    echo "Sourced ROS: ${ROS_SETUP}"
  fi
fi

if [[ -f "${HOME}/robotx_sim_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/robotx_sim_ws/install/setup.bash"
  echo "Sourced colcon workspace: ~/robotx_sim_ws/install"
fi

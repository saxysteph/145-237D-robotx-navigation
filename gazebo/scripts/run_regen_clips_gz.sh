#!/usr/bin/env bash
# Source Gazebo paths, then run regen_clips_gz_transport.py (no ROS).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export REPO_ROOT
# shellcheck source=gz_env.sh
source "${REPO_ROOT}/gazebo/scripts/gz_env.sh"
exec python3 "${REPO_ROOT}/gazebo/scripts/regen_clips_gz_transport.py" "$@"

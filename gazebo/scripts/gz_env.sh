#!/usr/bin/env bash
set -euo pipefail

# Bash sets BASH_SOURCE when sourced; zsh does not — use bash -lc 'source ...' from zsh.
if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  SCRIPT_DIR_GZ_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [[ -n "${REPO_ROOT:-}" ]]; then
  SCRIPT_DIR_GZ_ENV="${REPO_ROOT}/gazebo/scripts"
else
  echo "gz_env.sh: source with bash from the repo root, e.g.  bash -lc 'cd <repo> && source gazebo/scripts/gz_env.sh'" >&2
  echo "Or: export REPO_ROOT=<repo> then source this file (zsh-friendly)." >&2
  return 1 2>/dev/null || exit 1
fi
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR_GZ_ENV}/../.." && pwd)}"
WAVES_ROOT="$HOME/robotx_sim_ws/src/asv_wave_sim"
VRX_ROOT="$HOME/robotx_sim_ws/src/vrx"

export GZ_SIM_RESOURCE_PATH="${WAVES_ROOT}/gz-waves-models/models:${WAVES_ROOT}/gz-waves-models/world_models:${WAVES_ROOT}/gz-waves-models/worlds:${REPO_ROOT}/gazebo/models:${VRX_ROOT}/vrx_gz/models"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${WAVES_ROOT}/gz-waves/build/lib:${REPO_ROOT}/gazebo/plugins/robotx_beacon_plugin/build"
export GZ_RENDERING_PLUGIN_PATH="${WAVES_ROOT}/gz-waves/build/lib"

# Homebrew gz-transport Python bindings need libgz-transport on the loader path *before* the
# interpreter starts; setting DYLD_LIBRARY_PATH only inside Python is often too late on macOS.
if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
  _GZTP_LIB="$(brew --prefix gz-transport13 2>/dev/null)/lib"
  _HB_LIB="$(brew --prefix)/lib"
  if [[ -d "${_GZTP_LIB}" ]]; then
    export DYLD_LIBRARY_PATH="${_GZTP_LIB}:${_HB_LIB}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
  fi
fi

echo "GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}"
echo "GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH}"
echo "GZ_RENDERING_PLUGIN_PATH=${GZ_RENDERING_PLUGIN_PATH}"

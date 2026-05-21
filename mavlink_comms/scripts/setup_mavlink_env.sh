#!/usr/bin/env bash
# Install pymavlink + clone MAVCore into vendor/mavcore for local testing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 -m venv .venv-mavlink
# shellcheck disable=SC1091
source .venv-mavlink/bin/activate
pip install -q -r mavlink_comms/requirements.txt

if [[ ! -f vendor/mavcore/__init__.py ]]; then
  mkdir -p vendor
  git clone --depth 1 https://github.com/uci-uav-forge/mavcore.git vendor/mavcore
fi

echo "Ready. Activate with: source .venv-mavlink/bin/activate"
echo "MAVCore is on PYTHONPATH via mavlink_comms scripts (vendor/mavcore)."

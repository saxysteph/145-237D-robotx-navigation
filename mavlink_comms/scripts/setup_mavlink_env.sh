#!/usr/bin/env bash
# Install pymavlink + clone MAVCore into vendor/mavcore for local testing.
# Requires Python 3.10+ (MAVCore uses PEP 604 type unions).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    major="${ver%%.*}"
    minor="${ver#*.}"
    if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
      PY="$cand"
      break
    fi
  fi
done
if [[ -z "$PY" ]]; then
  echo "Need Python 3.10+ for MAVCore. Install or use: conda install python=3.13"
  exit 1
fi
echo "Using $PY ($($PY --version))"

"$PY" -m venv .venv-mavlink
# shellcheck disable=SC1091
source .venv-mavlink/bin/activate
pip install -q -r mavlink_comms/requirements.txt

if [[ ! -f vendor/mavcore/__init__.py ]]; then
  mkdir -p vendor
  git clone --depth 1 https://github.com/uci-uav-forge/mavcore.git vendor/mavcore
fi

echo "Ready. Activate with: source .venv-mavlink/bin/activate"
echo "MAVCore is on PYTHONPATH via mavlink_comms scripts (vendor/mavcore)."

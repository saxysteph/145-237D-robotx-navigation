#!/usr/bin/env bash
# Copy checkpoints / light demo assets into demo_preserved/ (binaries ignored by git; README is tracked).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROPOSAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEMO="$PROPOSAL_ROOT/demo_preserved"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

mkdir -p "$DEMO/weights" "$DEMO/videos" "$DEMO/representative_stills"

found=()
while IFS= read -r -d '' p; do
  found+=("$p")
done < <(
  find "$SCRIPT_DIR" "$REPO/runs/detect" -path "*/ultralytics_runs/*/weights/best.pt" -print0 2>/dev/null || true
)

if [[ ${#found[@]} -gt 0 ]]; then
  newest="${found[0]}"
  for p in "${found[@]}"; do
    [[ "$p" -nt "$newest" ]] && newest="$p"
  done
  cp -v "$newest" "$DEMO/weights/buoy_roi_best.pt"
  echo "Copied newest best.pt -> $DEMO/weights/buoy_roi_best.pt"
else
  echo "No best.pt under $SCRIPT_DIR/**/ultralytics_runs (retrain first or restore from backup / Time Machine)."
fi

ANN="$REPO/captures/hsv_results/annotated"
if [[ -d "$ANN" ]]; then
  count=0
  for img in "$ANN"/*.jpg; do
    [[ -f "$img" ]] || continue
    cp -v "$img" "$DEMO/representative_stills/"
    count=$((count + 1))
    [[ "$count" -ge 12 ]] && break
  done
  echo "Refreshed representative_stills (up to $count images from $ANN)."
fi

shopt -s nullglob
stills=( "$DEMO/representative_stills"/*.jpg )
shopt -u nullglob
if [[ ${#stills[@]} -gt 0 ]] && command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -hide_banner -loglevel error \
    -framerate 2 -pattern_type glob -i "$DEMO/representative_stills/*.jpg" \
    -c:v libx264 -pix_fmt yuv420p "$DEMO/videos/representative_annotated_slideshow.mp4" \
    && echo "Wrote $DEMO/videos/representative_annotated_slideshow.mp4"
elif [[ ${#stills[@]} -eq 0 ]]; then
  echo "No stills for slideshow."
else
  echo "ffmpeg not found; skipped slideshow mp4."
fi

echo "Done. See $DEMO/README.txt"

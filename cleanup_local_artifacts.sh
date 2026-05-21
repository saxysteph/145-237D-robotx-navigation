#!/usr/bin/env bash
# Safe disk helper (replacement for the old bulk-delete cleanup).
#
# The previous version ran `rm -rf` on roi_hsv_pipeline_* and large outputs — that deletion
# cannot be undone from git because those paths were intentionally gitignored.
#
# Usage: ./cleanup_local_artifacts.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$ROOT/yolo_comparison_test/path2_switch_proposal/scripts"

echo "Heavy directories under Path2 scripts (for manual review):"
du -sh \
  "${SCRIPT_DIR}"/roi_hsv_pipeline_* \
  "$SCRIPT_DIR/path2_video_results" \
  2>/dev/null || true

echo ""
echo "Before deleting anything:"
echo "  1. Copy checkpoints: see yolo_comparison_test/path2_switch_proposal/demo_preserved/README.txt"
echo "  2. Run: yolo_comparison_test/path2_switch_proposal/scripts/snapshot_demo_artifacts.sh"
echo ""
echo "To remove large dirs yourself after backup, delete only what you no longer need (Finder or rm)."
exit 0

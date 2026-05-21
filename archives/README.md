# Local archives (not in git)

Large zip bundles (Gazebo clip batches, ROI+HSV eval runs) live here on disk only.
They are gitignored to keep pushes small.

Restore a bundle by unzipping into the paths noted in each archive name, or re-run
the pipeline scripts under `gazebo/` and `yolo_comparison_test/path2_switch_proposal/scripts/`.

Tracked detector weights: `yolo_comparison_test/path2_switch_proposal/demo_preserved/weights/buoy_roi_best.pt`

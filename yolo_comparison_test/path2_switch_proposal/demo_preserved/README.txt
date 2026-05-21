Purpose
-------
Keep a SMALL, explicit copy of what you care about BEFORE deleting large pipeline folders.

Git does not restore `roi_hsv_pipeline_*`, `path2_video_results`, or `captures/gazebo_uav_batch`
if those were removed with rm -rf (they are/were gitignored).

What to save after each training run
-----------------------------------
1. ROI detector checkpoint (required for --hsv-reval-only):

   Ultralytics writes under your --out-root, for example:

   <out-root>/ultralytics_runs/roi_detector/weights/best.pt

   Copy or symlink here as:

     demo_preserved/weights/buoy_roi_best.pt

2. Optional: a few annotated MP4s / side-by-side reprojection clips into demo_preserved/videos/

3. Run from repo root:

   yolo_comparison_test/path2_switch_proposal/scripts/snapshot_demo_artifacts.sh

Representative stills / slideshow
---------------------------------
`snapshot_demo_artifacts.sh` also refreshes demo_preserved/representative_stills/ from
captures/hsv_results/annotated when present.

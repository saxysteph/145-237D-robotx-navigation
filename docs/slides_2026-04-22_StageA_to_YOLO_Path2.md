# RobotX 145/237D — Vision Pipeline Update (Week of 2026-04-22)
### Slide 1 — Title
- **CSE 145/237D — Drone / RobotX color marker detection**
- **Focus:** HSV + ROI gating, then YOLO fine-tune experiments (Path 2)
- **Codebase time window:** all cited commits on **2026-04-22**

### Slide 2 — What shipped last week (git commits, high-signal)
- `01e202c` — repo setup + runbook (`README.md`)
- `33213ab` — batch HSV script (`hsv_batch_detect.py`)
- `23c31bb` / `813600e` — unify to **3 classes** (red / green* / blue) + presentation tooling
- `233c58f` — **hybrid** proposal + ROI HSV re-score + refreshed `captures/hsv_results/`
- `90241ae` — **YOLO work packaged** under `yolo_comparison_test/path2_switch_proposal/`

*Green = teal/cyan physical balloon in the dataset, mapped to a single HSV class for MVP consistency.

### Slide 3 — Problem statement (one sentence each)
- **We need a detector that is cheap to iterate on, then a stronger learned detector.**
- **We need a color head that is stable enough for field lighting variation.**

### Slide 4 — “Approach” (with explicit industry references)
**What we are doing (our pipeline shape):** propose an object/region, then run color in a local ROI, then (optionally) track and log.

**References we are explicitly aligning to (read for philosophy + system framing):**
- **MIT (RobotX 2026) — technical design report (SLAM/vision system framing)**: [TDR_MITArcturus_RB2026.pdf](https://robonation.org/app/uploads/sites/3/2026/02/TDR_MITArcturus_RB2026.pdf)
- **Team Minion (RobotX 2024) — technical design / competition strategy (CV integration + UAV work package context)**: [TDR_Embry_Riddle_Aeronautical_University_RX2024-compressed.pdf](https://robonation.org/app/uploads/sites/2/2024/10/TDR_Embry_Riddle_Aeronautical_University_RX2024-compressed.pdf)

**How we use these references in this project (camera-only, honest):**
- We are **not** claiming the same multi-sensor fusion as Minion; we *borrow* the engineering pattern:
  - **(trusted geometry / trusted detector) → define ROI → confirm attributes (color) with lighter compute**
- In our case today: the “trusted geometry” is either **(A)** classical proposals + gating, or **(B)** YOLO once trained, and HSV becomes the **post-hoc** color head.

**Internal “Useful Links” to paste on the final deck (placeholders if not public):**
- **UAV Search & Report (team doc):** `[PASTE LINK HERE]`
- **Drone video (UAV / CV demo reel):** `[PASTE LINK HERE]`

### Slide 5 — HSV / ROI baseline (what we run today in-repo)
- **Class swatch calibration (RGB → HSV ranges):** `captures/classes/{red,green,blue}.png` → derived ranges in `color_utils.py`
- **Classical “proposal + geometry gate” (batch):** `hsv_batch_detect.py`
- **Color scoring:** run each class HSV mask **inside the proposal ROI** and pick the best in-range pixel ratio

**Important interpretation note for presentation honesty:**
- HSV “metrics” in `metrics_summary.py` are **pipeline-internal** (counts + self-reported “confidence”) — *not* ground-truth mAP.
- YOLO Path 2’s held-out file contains **TP/FP/FN** against labels — that is a different, stronger evaluation.

### Slide 6 — HSV / ROI tuning knobs (what we actually turn)
- **Range derivation knobs:** `--hue-margin`, `--sat-min-floor`, `--val-min-floor` (affects the swatch-derived HSV parallelepiped)
- **Proposal / ROI knobs:** `--roi-margin`, `--proposal-padding`, `--min-proposal-area`, `--max-proposal-area-ratio`, `--min-circularity`, `--min-solidity`, `--nms-iou`
- **Color acceptance knob:** `--min-color-ratio` (dominant failure mode: background color + weak saturation)
- **Morphology / noise:** `--kernel-size` (open/close on masks)
- **Live camera knobs (separate from batch):** resolution split (`--det-width/--det-height`), CLAHE on V, Kalman gating, altitude-based size gating in `camera_live_feed.py`

### Slide 7 — HSV “numbers we recorded” (apples: same dataset, same summary script)
Dataset: `captures/` = **110** JPGs (as counted by `metrics_summary.py`).

| Architecture (batch) | Total detections | Avg self-reported conf | What changed (one-liner) |
|---|---:|---:|---|
| Color-first (historical) | 868 | 0.708 | Per-color mask contours + geometric filters; higher recall, more HSV-naive FPs possible |
| Object-first (edges+sat) | 765 | 0.658 | Stricter “find object first” → fewer boxes; recall drops if proposals miss balloons |
| Hybrid (object + color proposals) | 1067 | 0.646 | More detections (often **more** candidate boxes) — **not automatically “better”** for precision |

**Presenter line:** *Higher HSV count ≠ higher accuracy* unless measured against labels.

**Artifacts to show on slide (pictures, not just numbers):**
- **Bar chart:** `captures/hsv_results/metrics.png`
- **2×2 augmentation panel (noise / glare demo):** `captures/hsv_results/augmentation_test.jpg`
- **Before/after storyboard:** `captures/hsv_results/results_diagram.png`

**Example labeled HSV visualizations (use 2–3 side-by-side on one slide):**
- `captures/hsv_results/annotated/capture_1776885031140.jpg` (visually “busy” scene — good stress example)
- `captures/hsv_results/annotated/capture_1776885201156.jpg` (nice multi-detection example)
- Raw vs annotated pair (if you want “evidence of labeling”):
  - raw: `captures/capture_1776885031140.jpg`
  - annotated: `captures/hsv_results/annotated/capture_1776885031140.jpg`

### Slide 8 — YOLO Path 2 (Stage B) — the “real accuracy story”
This is packaged under: `yolo_comparison_test/path2_switch_proposal/`

**Training/inference quick facts (from `results/path2_summary.txt`):**
- 30 epochs on **110** auto-labeled images
- **Training mAP50:** 0.980 (as reported in training run)
- Full inference pass (same 110 images): **892** detections, **avg conf 0.924**

**Held-out honest validation (from `results/honest_results.txt` + `results/honest_map50.txt`):**
- val size: **22** images
- mAP50 (Ultralytics val): **0.967** (file `honest_map50.txt` shows `0.968` — use **0.967** for consistency with the report text, or use **0.968** as the scalar export)
- TP/FP/FN: 169 / 10 / 4
- **Precision 0.944, Recall 0.977, F1 0.960**
- Per-class F1: red 1.000, green 0.994, blue 0.860

**UAV stress test (from `results/stress_test_results.txt`):**
- detections: 179 → 175 (retention **97.8%**)
- avg conf: 0.925 → 0.898

### Slide 9 — “Why we’re doing Path 2” (one slide, crisp)
- HSV is **fast** and a great bootstrap labeler, but is fundamentally **heuristic** under uncontrolled background + lighting.
- YOLO gives **data-driven** localization, and the held-out metrics show the jump from “self-scored HSV heuristics” to **precision/recall against labels**.

### Slide 10 — What’s next (project-shaped)
- Keep HSV for **(1)** data capture + auto-label, **(2)** sanity mode** on edge hardware.
- Promote YOLO to the **primary detector** once the dataset is large enough, keep HSV as **color head inside boxes** (or train color as separate class heads with more labels).

### Slide 11 — Q&A (common questions, prepared answers)
- **“Is hybrid HSV better?”** Not automatically — it increases candidate boxes. Compare only with **labeled** precision/recall, not count.
- **“Is training YOLO enough?”** Training solves localization; color still may need a dedicated head, especially under glare/white-balance.
- **“What’s the single biggest win last week?”** A reproducible Path 2 package + honest held-out metrics + stress test script chain.

### Appendix A — one-slide “knobs table” (copy to poster)
- **HSV:** hue margin, S/V floors, min color ratio, kernel size, ROI margin, proposal NMS, hybrid proposal on/off
- **YOLO:** model size, epochs, confidence threshold, autolabel quality filters, proper train/val split seed

### Appendix B — how to re-generate figures before presenting
```bash
conda activate robotx
python hsv_batch_detect.py
python metrics_summary.py
python augment_test.py
python visualize_results.py
```

# Export tips (Google Slides / Keynote)
- Use **16:9**, one main claim per slide, and paste **2 images max** on evidence slides.
- Put long URLs in **speaker notes** (this keeps slides readable in a 10–12 minute talk).

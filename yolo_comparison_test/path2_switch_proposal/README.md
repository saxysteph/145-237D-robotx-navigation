# Path 2 Switch Proposal (Stage B)

This folder is a concise review package for switching Stage B from HSV-only detection to a YOLO model trained from Stage A auto-labels.

## Why Switch

- Stage A HSV baseline was useful for bootstrapping labels and pipeline logic.
- Path 2 (fine-tuned YOLOv11n) outperformed zero-shot COCO ball detection in our data.
- Honest held-out validation remained high after a proper train/val split, indicating the result is not just memorization.

## Core Evidence

- `results/path2_summary.txt`
  - Full-dataset Path 2 run summary.
- `results/honest_map50.txt`
  - Held-out validation mAP50 after proper 80/20 split retraining.
- `results/honest_results.txt`
  - Manual TP/FP/FN, precision, recall, F1, per-class breakdown, verdict.
- `results/stress_test_results.txt`
  - UAV-noise stress test retention and confidence drop.

## Scripts Included

- `scripts/01_autolabel.py`
  - Auto-labels from HSV to YOLO format (red/green/blue classes).
- `scripts/02_finetune.py`
  - Fine-tunes YOLOv11n on auto-labeled data.
- `scripts/validation_step1_proper_split.py`
  - Builds reproducible 80/20 split (`seed=42`).
- `scripts/validation_step2_retrain.py`
  - Retrains YOLOv11n from base weights on train split only.
- `scripts/validation_step3_val_inference.py`
  - Held-out inference + TP/FP/FN metrics + per-class metrics.
- `scripts/validation_step4_overfit_check.py`
  - Train-vs-val loss curve analysis and gap verdict.
- `scripts/validation_step5_stress_test.py`
  - UAV-noise robustness test on held-out validation images.

## Recommendation

Proceed with Path 2 as Stage B:

- Use HSV auto-labeling only as weak-supervision bootstrap.
- Use YOLO fine-tune + held-out validation as the primary detector path.
- Keep HSV as fallback/sanity check in difficult lighting scenarios.

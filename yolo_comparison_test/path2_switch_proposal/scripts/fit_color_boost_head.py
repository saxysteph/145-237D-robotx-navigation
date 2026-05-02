#!/usr/bin/env python3
"""Train Histogram-Gradient-Boosting head on `stack_tree_features` for ROI color (see color_utils).

Trains on detector-matched crops when --detector-weights is set (recommended for deployment).
Saves `color_tree_head.joblib` for `train_roi_then_hsv.py`.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from color_utils import (  # noqa: E402
    LabColorModel,
    TREE_H_BINS,
    TREE_S_BINS,
    TREE_V_BINS,
    compute_color_evidence_bundle,
    crop_roi_center,
    derive_class_hsv_ranges,
    derive_lab_ab_models,
    geom_sampling_mask,
    stack_tree_features,
)

# Reuse helpers from fit_color_lr_head
from fit_color_lr_head import (  # noqa: E402
    CLASS_ID,
    iou_xyxy_pair,
    jitter_xyxy,
    yolo_line_to_box,
)


def _default_detector_weights() -> Path | None:
    """Ultralytics nests detect runs under runs/detect/<project>/ ; also allow flat out_root mirror."""
    cands = [
        SCRIPT_DIR
        / "runs"
        / "detect"
        / "roi_hsv_pipeline_mps"
        / "ultralytics_runs"
        / "roi_detector"
        / "weights"
        / "best.pt",
        SCRIPT_DIR / "roi_hsv_pipeline_mps" / "ultralytics_runs" / "roi_detector" / "weights" / "best.pt",
    ]
    for p in cands:
        if p.is_file():
            return p
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures-root", type=Path, default=REPO_ROOT / "captures" / "gazebo_uav_batch")
    p.add_argument("--classes-dir", type=Path, default=REPO_ROOT / "captures" / "classes")
    p.add_argument("--out-path", type=Path, default=SCRIPT_DIR / "color_tree_head.joblib")
    p.add_argument("--max-samples", type=int, default=40000)
    p.add_argument("--max-clips", type=int, default=0, help="0 = all clip folders under captures-root")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hsv-mask-shape",
        type=str,
        default="ellipse",
        choices=["ellipse", "circle", "inner_rect", "legacy_center_ellipse"],
    )
    p.add_argument("--ellipse-rx-frac", type=float, default=0.42)
    p.add_argument("--ellipse-ry-frac", type=float, default=0.42)
    p.add_argument("--circle-r-frac", type=float, default=0.38)
    p.add_argument("--inner-shrink-xy", type=float, default=0.62)
    p.add_argument("--cap-band-top-frac", type=float, default=0.22)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--aug-copies", type=int, default=2)
    p.add_argument("--aug-scale-lo", type=float, default=0.75)
    p.add_argument("--aug-scale-hi", type=float, default=1.25)
    p.add_argument("--aug-shift-frac", type=float, default=0.1)
    p.add_argument(
        "--detector-weights",
        type=Path,
        default=None,
        help="YOLO weights for IoU-matched crops. Default: roi_hsv_pipeline_mps best.pt when file exists.",
    )
    p.add_argument("--no-detector", action="store_true", help="Train on GT boxes only (ignore detector-weights).")
    p.add_argument("--det-conf", type=float, default=0.15, help="Match train_roi_then_hsv --conf-thres when possible.")
    p.add_argument("--match-iou", type=float, default=0.20, help="IoU GT–det threshold for adopting detector crop.")
    p.add_argument("--hgb-max-iter", type=int, default=320)
    p.add_argument("--hgb-depth", type=int, default=8)
    p.add_argument(
        "--roi-inner-crop",
        type=float,
        default=0.88,
        help="Match train_roi_then_hsv: crop each box to central fraction before features.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ranges_map = derive_class_hsv_ranges(str(args.classes_dir), hue_margin=12, sat_floor=50, val_floor=45)
    if not ranges_map:
        print(f"Missing HSV ranges from {args.classes_dir}")
        return 1
    lab_models: dict[str, LabColorModel] = derive_lab_ab_models(str(args.classes_dir))
    if not lab_models or len(lab_models) < 3:
        print("Need red/green/blue reference images in classes-dir for LAB models.")
        return 1

    det_model = None
    det_path = None if args.no_detector else args.detector_weights
    if det_path is None and not args.no_detector:
        resolved = _default_detector_weights()
        if resolved is not None:
            det_path = resolved
    if det_path is not None and Path(det_path).is_file():
        from ultralytics import YOLO

        det_model = YOLO(str(det_path))
        print(f"Detector crops from {det_path}")
    else:
        print("No detector weights; using GT boxes + jitter only")

    clip_dirs = sorted([p for p in args.captures_root.iterdir() if p.is_dir() and not p.name.startswith("_")])
    if args.max_clips > 0:
        clip_dirs = clip_dirs[: args.max_clips]
    print(f"Clips: {len(clip_dirs)}")

    pred_cache: dict[Path, object] = {}
    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    n_read = 0

    for clip_dir in clip_dirs:
        for label_path in sorted((clip_dir / "labels_proj").glob("*.txt")):
            img_path = clip_dir / f"{label_path.stem}.jpg"
            if not img_path.is_file():
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                b = yolo_line_to_box(parts, w, h)
                if b is None:
                    continue
                cls_id = int(float(parts[0]))
                x1, y1, x2, y2 = b
                gt_box = (x1, y1, x2, y2)
                boxes_try: list[tuple[int, int, int, int]] = []

                if det_model is not None:
                    if img_path not in pred_cache:
                        pred_cache[img_path] = det_model(frame, verbose=False, conf=args.det_conf)[0]
                    res = pred_cache[img_path]
                    boxes = res.boxes
                    best_iou = 0.0
                    best_det: tuple[int, int, int, int] | None = None
                    if boxes is not None:
                        for bi in range(len(boxes)):
                            x1f, y1f, x2f, y2f = boxes.xyxy[bi].tolist()
                            db = (
                                max(0, int(round(x1f))),
                                max(0, int(round(y1f))),
                                min(w - 1, int(round(x2f))),
                                min(h - 1, int(round(y2f))),
                            )
                            iv = iou_xyxy_pair(gt_box, db)
                            if iv > best_iou:
                                best_iou = iv
                                best_det = db
                    if best_det is not None and best_iou >= args.match_iou:
                        boxes_try.append(best_det)
                    boxes_try.append(gt_box)
                    for _ in range(max(0, args.aug_copies)):
                        boxes_try.append(
                            jitter_xyxy(
                                x1, y1, x2, y2, w, h, rng,
                                scale_lo=args.aug_scale_lo,
                                scale_hi=args.aug_scale_hi,
                                shift_frac=args.aug_shift_frac,
                            )
                        )
                else:
                    boxes_try = [(x1, y1, x2, y2)]
                    for _ in range(max(0, args.aug_copies)):
                        boxes_try.append(
                            jitter_xyxy(
                                x1, y1, x2, y2, w, h, rng,
                                scale_lo=args.aug_scale_lo,
                                scale_hi=args.aug_scale_hi,
                                shift_frac=args.aug_shift_frac,
                            )
                        )

                for bx1, by1, bx2, by2 in boxes_try:
                    crop_bgr = frame[by1:by2, bx1:bx2]
                    crop_hsv = hsv[by1:by2, bx1:bx2]
                    if args.roi_inner_crop < 0.999:
                        crop_bgr, crop_hsv = crop_roi_center(crop_bgr, crop_hsv, args.roi_inner_crop)
                    if crop_bgr.size < 80:
                        continue
                    rh, rw = crop_hsv.shape[:2]
                    gm = geom_sampling_mask(
                        rh,
                        rw,
                        shape=args.hsv_mask_shape,
                        ellipse_rx_frac=args.ellipse_rx_frac,
                        ellipse_ry_frac=args.ellipse_ry_frac,
                        circle_r_frac=args.circle_r_frac,
                        inner_shrink_xy=args.inner_shrink_xy,
                        cap_band_top_frac=args.cap_band_top_frac,
                    )
                    ev = compute_color_evidence_bundle(
                        crop_bgr,
                        crop_hsv,
                        ranges_map,
                        gm,
                        suppress_water=True,
                        water_sat_max=48,
                        water_val_min=30,
                        water_desat_blue_max=38,
                        lab_models=lab_models,
                    )
                    if ev is None:
                        continue
                    vec = stack_tree_features(
                        crop_hsv,
                        ev,
                        h_bins=TREE_H_BINS,
                        s_bins=TREE_S_BINS,
                        v_bins=TREE_V_BINS,
                    )
                    X_list.append(vec)
                    y_list.append(CLASS_ID[cls_id])
                    n_read += 1
                    if n_read >= args.max_samples:
                        break
                if n_read >= args.max_samples:
                    break
            if n_read >= args.max_samples:
                break
        if n_read >= args.max_samples:
            break

    if len(X_list) < 400:
        print(f"Too few samples: {len(X_list)}")
        return 1

    X = np.stack(X_list, axis=0)
    X = np.nan_to_num(np.clip(X, -1e3, 1e3), nan=0.0, posinf=1e3, neginf=-1e3)
    y = np.array(y_list)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_iter=args.hgb_max_iter,
                    max_depth=args.hgb_depth,
                    learning_rate=0.045,
                    l2_regularization=0.06,
                    random_state=args.seed,
                    class_weight="balanced",
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=25,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Hold-out accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, digits=4))

    hgb_step = clf.named_steps["hgb"]
    proba_classes = tuple(str(c) for c in hgb_step.classes_)
    blob = {
        "model": clf,
        "proba_classes": proba_classes,
        "h_bins": TREE_H_BINS,
        "s_bins": TREE_S_BINS,
        "v_bins": TREE_V_BINS,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(blob, args.out_path)
    print(f"Wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

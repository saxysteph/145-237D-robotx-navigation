#!/usr/bin/env python3
"""Train YOLOv11n as single-class buoy ROI detector, then classify ROI color with HSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil

try:
    import joblib
except ImportError:
    joblib = None
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from color_utils import (
    ColorLogisticHead,
    ColorTreeHead,
    LabColorModel,
    classify_buoy_roi_hsv,
    crop_roi_center,
    derive_bgr_centroids,
    derive_class_hsv_ranges,
    derive_lab_ab_models,
    geom_sampling_mask,
    infer_sklearn_predict_proba_class_order,
)
from ultralytics_plot_patch import apply_plot_max_subplots


@dataclass
class Sample:
    key: str
    image_path: Path
    label_path: Path


@dataclass
class GTBox:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float


CLASS_NAME = {0: "red", 1: "green", 2: "blue"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures-root", type=Path, default=REPO_ROOT / "captures" / "gazebo_uav_batch")
    p.add_argument("--labels-subdir", type=str, default="labels_proj")
    p.add_argument("--classes-dir", type=Path, default=REPO_ROOT / "captures" / "classes")
    p.add_argument("--out-root", type=Path, default=SCRIPT_DIR / "roi_hsv_pipeline")
    p.add_argument(
        "--hsv-reval-only",
        action="store_true",
        help="Skip dataset prep and training; re-run HSV test eval on existing out-root (needs dataset_roi + weights).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers for YOLO train (use 0–2 if unstable on MPS).")
    p.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Ultralytics training device (e.g., mps, cpu, 0).",
    )
    p.add_argument("--conf-thres", type=float, default=0.15)
    p.add_argument("--iou-match", type=float, default=0.20)
    p.add_argument(
        "--max-test-images",
        type=int,
        default=0,
        help="For --hsv-reval-only: cap number of test images (0 = all). Fast iteration.",
    )
    p.add_argument(
        "--hsv-mask-shape",
        choices=["ellipse", "circle", "inner_rect", "legacy_center_ellipse"],
        default="ellipse",
        help="Geometry for HSV sampling inside each detection box.",
    )
    p.add_argument("--ellipse-rx-frac", type=float, default=0.42)
    p.add_argument("--ellipse-ry-frac", type=float, default=0.42)
    p.add_argument("--circle-r-frac", type=float, default=0.38)
    p.add_argument("--inner-shrink-xy", type=float, default=0.62)
    p.add_argument(
        "--cap-band-top-frac",
        type=float,
        default=0.22,
        help="Drop bottom fraction of ROI (keep buoy crown); 0 disables.",
    )
    p.add_argument(
        "--suppress-water",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Suppress low-saturation water/glint pixels inside ROI (recommended).",
    )
    p.add_argument("--water-sat-max", type=int, default=48)
    p.add_argument("--water-val-min", type=int, default=30)
    p.add_argument("--water-desat-blue-max", type=int, default=38)
    p.add_argument(
        "--disable-lab-fusion",
        action="store_true",
        help="Use HSV branch only (no LAB Gaussian fusion from captures/classes).",
    )
    p.add_argument(
        "--fusion-lab-weight",
        type=float,
        default=0.55,
        help="LAB branch weight (paired with --fusion-hsv-weight).",
    )
    p.add_argument(
        "--fusion-hsv-weight",
        type=float,
        default=0.45,
        help="HSV branch weight.",
    )
    p.add_argument(
        "--lr-head-path",
        type=Path,
        default=SCRIPT_DIR / "color_lr_head.joblib",
        help="Optional logistic head weights (from fit_color_lr_head.py).",
    )
    p.add_argument("--disable-lr-head", action="store_true")
    p.add_argument(
        "--lr-min-confidence",
        type=float,
        default=0.52,
        help="Use logistic head only if max class prob >= this; else LAB+HSV fusion (0 = always LR).",
    )
    p.add_argument(
        "--tree-head-path",
        type=Path,
        default=SCRIPT_DIR / "color_tree_head.joblib",
        help="Boosted-tree head from fit_color_boost_head.py.",
    )
    p.add_argument("--disable-tree-head", action="store_true")
    p.add_argument(
        "--tree-geometric-beta",
        type=float,
        default=0.62,
        help="Blend boosted tree probs with LAB/HSV fusion (0 = do not load tree head). Typical 0.5–0.65.",
    )
    p.add_argument(
        "--fusion-rgb-weight",
        type=float,
        default=0.62,
        help="Blend BGR distance-to-reference (0 = off).",
    )
    p.add_argument(
        "--fusion-rgb-sigma",
        type=float,
        default=40.0,
        help="Scale for BGR distance softmax (smaller = sharper toward nearest centroid).",
    )
    p.add_argument(
        "--fusion-temperature",
        type=float,
        default=1.1,
        help="Sharpen final class probs before argmax (>1 = peakier). 1.0 disables.",
    )
    p.add_argument(
        "--roi-inner-crop",
        type=float,
        default=0.88,
        help="Use central fraction of each detection ROI for color (1.0 = full box). Try 0.82–0.92 to drop edges.",
    )
    p.add_argument(
        "--rg-bgr-balance-strength",
        type=float,
        default=0.88,
        help="Nudge red vs green from paint mean R−G imbalance (0 disables).",
    )
    p.add_argument(
        "--rg-balance-hue-gate",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="When R/G nudge disagrees with median hue band, damp the correction (recommended on).",
    )
    return p.parse_args()


def _safe_unlink_or_rmtree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def collect_samples(captures_root: Path, labels_subdir: str) -> list[Sample]:
    samples: list[Sample] = []
    for clip_dir in sorted([p for p in captures_root.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        labels_dir = clip_dir / labels_subdir
        if not labels_dir.is_dir():
            continue
        for image_path in sorted(clip_dir.glob("uav_*.jpg")):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            key = f"{clip_dir.name}/{image_path.name}"
            samples.append(Sample(key=key, image_path=image_path, label_path=label_path))
    return samples


def split_samples(samples: list[Sample], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[list[Sample], list[Sample], list[Sample]]:
    if not samples:
        return [], [], []
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {s:.6f}")

    # Group-aware split: all frames from one clip stay in exactly one split.
    by_clip: dict[str, list[Sample]] = {}
    for smp in samples:
        clip = smp.key.split("/", 1)[0]
        by_clip.setdefault(clip, []).append(smp)

    clip_names = list(by_clip.keys())
    random.Random(seed).shuffle(clip_names)

    n = len(samples)
    train_target = int(round(n * train_ratio))
    val_target = int(round(n * val_ratio))
    train_clips: list[str] = []
    val_clips: list[str] = []
    test_clips: list[str] = []
    train_n = 0
    val_n = 0
    for clip in clip_names:
        clip_n = len(by_clip[clip])
        if train_n < train_target:
            train_clips.append(clip)
            train_n += clip_n
        elif val_n < val_target:
            val_clips.append(clip)
            val_n += clip_n
        else:
            test_clips.append(clip)

    if not train_clips or not val_clips or not test_clips:
        raise RuntimeError(
            f"Clip-group split failed: train_clips={len(train_clips)} "
            f"val_clips={len(val_clips)} test_clips={len(test_clips)}"
        )

    train = [smp for clip in train_clips for smp in by_clip[clip]]
    val = [smp for clip in val_clips for smp in by_clip[clip]]
    test = [smp for clip in test_clips for smp in by_clip[clip]]
    if not train or not val or not test:
        raise RuntimeError(f"Empty split encountered: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


def to_roi_label_lines(label_path: Path) -> list[str]:
    lines: list[str] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        t = raw.strip().split()
        if len(t) != 5:
            continue
        # collapse to single class "buoy"
        lines.append(f"0 {t[1]} {t[2]} {t[3]} {t[4]}")
    return lines


def prepare_split_dir(split_name: str, split_samples: list[Sample], out_root: Path) -> None:
    img_dir = out_root / "dataset_roi" / "images" / split_name
    lbl_dir = out_root / "dataset_roi" / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(split_samples):
        name = f"{i:07d}__{s.image_path.stem}.jpg"
        img_link = img_dir / name
        lbl_path = lbl_dir / f"{Path(name).stem}.txt"
        if img_link.exists() or img_link.is_symlink():
            img_link.unlink()
        os.symlink(s.image_path, img_link)
        roi_lines = to_roi_label_lines(s.label_path)
        lbl_path.write_text("\n".join(roi_lines) + ("\n" if roi_lines else ""), encoding="utf-8")


def write_dataset_yaml(out_root: Path) -> Path:
    yml = out_root / "dataset_roi" / "dataset.yaml"
    yml.write_text(
        "path: ./dataset_roi\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['buoy']\n",
        encoding="utf-8",
    )
    yml_abs = out_root / "dataset_roi" / "dataset_abs.yaml"
    dataset_root_posix = str((out_root / "dataset_roi").resolve()).replace(os.sep, "/")
    yml_abs.write_text(
        f"path: {dataset_root_posix}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['buoy']\n",
        encoding="utf-8",
    )
    return yml_abs


def yolo_line_to_box(parts: list[str], w: int, h: int) -> GTBox | None:
    if len(parts) != 5:
        return None
    cls = int(float(parts[0]))
    cx = float(parts[1]) * w
    cy = float(parts[2]) * h
    bw = float(parts[3]) * w
    bh = float(parts[4]) * h
    x1 = max(0.0, cx - 0.5 * bw)
    y1 = max(0.0, cy - 0.5 * bh)
    x2 = min(float(w - 1), cx + 0.5 * bw)
    y2 = min(float(h - 1), cy + 0.5 * bh)
    return GTBox(cls_id=cls, x1=x1, y1=y1, x2=x2, y2=y2)


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aa + ab - inter
    return inter / denom if denom > 0 else 0.0


def classify_roi_color(
    roi_bgr: np.ndarray,
    roi_hsv: np.ndarray,
    ranges_map: dict[str, list],
    *,
    mask_shape: str,
    ellipse_rx_frac: float,
    ellipse_ry_frac: float,
    circle_r_frac: float,
    inner_shrink_xy: float,
    cap_band_top_frac: float,
    suppress_water: bool,
    water_sat_max: int,
    water_val_min: int,
    water_desat_blue_max: int,
    lab_models: dict[str, LabColorModel] | None,
    fusion_lab_weight: float,
    fusion_hsv_weight: float,
    lr_head: ColorLogisticHead | None,
    lr_min_confidence: float,
    tree_head: ColorTreeHead | None,
    tree_geometric_beta: float,
    bgr_centroids: dict | None,
    fusion_rgb_weight: float,
    fusion_rgb_sigma: float,
    fusion_temperature: float,
    rg_bgr_balance_strength: float,
    rg_balance_hue_gate: bool,
) -> tuple[str, float]:
    rh, rw = roi_hsv.shape[:2]
    if rh < 2 or rw < 2:
        return "unknown", 0.0
    geom_mask = geom_sampling_mask(
        rh,
        rw,
        shape=mask_shape,
        ellipse_rx_frac=ellipse_rx_frac,
        ellipse_ry_frac=ellipse_ry_frac,
        circle_r_frac=circle_r_frac,
        inner_shrink_xy=inner_shrink_xy,
        cap_band_top_frac=cap_band_top_frac,
    )
    return classify_buoy_roi_hsv(
        roi_bgr,
        roi_hsv,
        ranges_map,
        geom_mask,
        suppress_water=suppress_water,
        water_sat_max=water_sat_max,
        water_val_min=water_val_min,
        water_desat_blue_max=water_desat_blue_max,
        lab_models=lab_models,
        fusion_lab_weight=fusion_lab_weight,
        fusion_hsv_weight=fusion_hsv_weight,
        lr_head=lr_head,
        lr_min_confidence=lr_min_confidence,
        tree_head=tree_head,
        tree_geometric_beta=tree_geometric_beta,
        bgr_centroids=bgr_centroids,
        fusion_rgb_weight=fusion_rgb_weight,
        fusion_rgb_sigma=fusion_rgb_sigma,
        fusion_temperature=fusion_temperature,
        rg_bgr_balance_strength=rg_bgr_balance_strength,
        rg_balance_hue_gate=rg_balance_hue_gate,
    )


def find_original_label_for_link(link_path: Path) -> Path:
    # link filename format: 0000001__uav_....jpg ; use symlink target parent clip labels_proj
    target = Path(os.path.realpath(link_path))
    clip = target.parent
    return clip / "labels_proj" / f"{target.stem}.txt"


def run_hsv_test_eval(
    args: argparse.Namespace,
    ranges_map: dict[str, list],
    *,
    best_w: Path,
    yml_abs: Path,
    split_counts: dict[str, int],
    lab_models: dict[str, LabColorModel] | None,
    lr_head: ColorLogisticHead | None,
    tree_head: ColorTreeHead | None,
    bgr_centroids: dict | None,
) -> int:
    det_model = YOLO(str(best_w))
    test_img_dir = args.out_root / "dataset_roi" / "images" / "test"
    eval_dir = args.out_root / "test_eval"
    ann_dir = eval_dir / "annotated_frames"
    ann_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / "roi_hsv_test_results.csv"
    summary_path = eval_dir / "summary.json"
    draw_color = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0), "unknown": (255, 255, 255)}

    total_preds = 0
    matched_preds = 0
    correct_color = 0
    per_cls_total = {"red": 0, "green": 0, "blue": 0}
    per_cls_correct = {"red": 0, "green": 0, "blue": 0}

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image",
                "pred_conf",
                "pred_color_hsv",
                "pred_score_hsv",
                "matched_gt_color",
                "match_iou",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )
        test_paths = sorted(test_img_dir.glob("*.jpg"))
        _mt = int(getattr(args, "max_test_images", 0) or 0)
        if _mt > 0:
            test_paths = test_paths[:_mt]
        n_test_used = len(test_paths)
        for img_path in test_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, ww = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gt_label_path = find_original_label_for_link(img_path)
            gt_boxes: list[GTBox] = []
            if gt_label_path.exists():
                for line in gt_label_path.read_text(encoding="utf-8").splitlines():
                    b = yolo_line_to_box(line.strip().split(), ww, h)
                    if b is not None:
                        gt_boxes.append(b)

            preds = det_model(frame, verbose=False, conf=args.conf_thres)[0]
            boxes = preds.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                x1f, y1f, x2f, y2f = boxes.xyxy[i].tolist()
                x1 = max(0, int(round(x1f)))
                y1 = max(0, int(round(y1f)))
                x2 = min(ww - 1, int(round(x2f)))
                y2 = min(h - 1, int(round(y2f)))
                if x2 <= x1 or y2 <= y1:
                    continue
                total_preds += 1
                roi_hsv = hsv[y1:y2, x1:x2]
                roi_bgr = frame[y1:y2, x1:x2]
                if args.roi_inner_crop < 0.999:
                    roi_bgr, roi_hsv = crop_roi_center(roi_bgr, roi_hsv, args.roi_inner_crop)
                hsv_color, hsv_score = classify_roi_color(
                    roi_bgr,
                    roi_hsv,
                    ranges_map,
                    mask_shape=args.hsv_mask_shape,
                    ellipse_rx_frac=args.ellipse_rx_frac,
                    ellipse_ry_frac=args.ellipse_ry_frac,
                    circle_r_frac=args.circle_r_frac,
                    inner_shrink_xy=args.inner_shrink_xy,
                    cap_band_top_frac=args.cap_band_top_frac,
                    suppress_water=args.suppress_water,
                    water_sat_max=args.water_sat_max,
                    water_val_min=args.water_val_min,
                    water_desat_blue_max=args.water_desat_blue_max,
                    lab_models=lab_models,
                    fusion_lab_weight=args.fusion_lab_weight,
                    fusion_hsv_weight=args.fusion_hsv_weight,
                    lr_head=lr_head,
                    lr_min_confidence=args.lr_min_confidence,
                    tree_head=tree_head,
                    tree_geometric_beta=args.tree_geometric_beta,
                    bgr_centroids=bgr_centroids,
                    fusion_rgb_weight=args.fusion_rgb_weight,
                    fusion_rgb_sigma=args.fusion_rgb_sigma,
                    fusion_temperature=args.fusion_temperature,
                    rg_bgr_balance_strength=args.rg_bgr_balance_strength,
                    rg_balance_hue_gate=args.rg_balance_hue_gate,
                )

                best_iou = 0.0
                gt_color = "unmatched"
                best_gt_idx = -1
                for gi, gt in enumerate(gt_boxes):
                    iou = iou_xyxy((x1, y1, x2, y2), (gt.x1, gt.y1, gt.x2, gt.y2))
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi
                if best_gt_idx >= 0 and best_iou >= args.iou_match:
                    matched_preds += 1
                    gt_color = CLASS_NAME.get(gt_boxes[best_gt_idx].cls_id, "unknown")
                    if gt_color in per_cls_total:
                        per_cls_total[gt_color] += 1
                    if hsv_color == gt_color:
                        correct_color += 1
                        if gt_color in per_cls_correct:
                            per_cls_correct[gt_color] += 1

                draw = draw_color.get(hsv_color, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw, 2)
                cv2.putText(
                    frame,
                    f"{hsv_color} {conf:.2f} iou:{best_iou:.2f}",
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    draw,
                    2,
                    cv2.LINE_AA,
                )
                w.writerow(
                    [
                        img_path.name,
                        f"{conf:.4f}",
                        hsv_color,
                        f"{hsv_score:.4f}",
                        gt_color,
                        f"{best_iou:.4f}",
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )
            cv2.imwrite(str(ann_dir / img_path.name), frame)

    color_acc = (correct_color / matched_preds) if matched_preds else 0.0
    summary = {
        "total_frames": n_test_used,
        "total_predictions": total_preds,
        "matched_predictions_iou": matched_preds,
        "hsv_color_accuracy_on_matched": color_acc,
        "lab_fusion_enabled": lab_models is not None and len(lab_models) >= 3,
        "lr_head_active": lr_head is not None,
        "tree_head_active": tree_head is not None,
        "tree_geometric_beta": args.tree_geometric_beta,
        "max_test_images": getattr(args, "max_test_images", 0),
        "fusion_lab_weight": args.fusion_lab_weight,
        "fusion_hsv_weight": args.fusion_hsv_weight,
        "fusion_rgb_weight": args.fusion_rgb_weight,
        "fusion_rgb_sigma": args.fusion_rgb_sigma,
        "fusion_temperature": args.fusion_temperature,
        "roi_inner_crop": args.roi_inner_crop,
        "rg_bgr_balance_strength": args.rg_bgr_balance_strength,
        "rg_balance_hue_gate": args.rg_balance_hue_gate,
        "per_color_total": per_cls_total,
        "per_color_correct": per_cls_correct,
        "split_counts": split_counts,
        "weights": str(best_w),
        "dataset_yaml_abs": str(yml_abs),
        "test_csv": str(csv_path),
        "annotated_test_dir": str(ann_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    apply_plot_max_subplots()

    ranges_map = derive_class_hsv_ranges(str(args.classes_dir), hue_margin=12, sat_floor=50, val_floor=45)
    if not ranges_map:
        print(f"Failed to derive HSV ranges from {args.classes_dir}")
        return 1

    lab_models: dict[str, LabColorModel] | None = None
    if not args.disable_lab_fusion:
        lab_models = derive_lab_ab_models(str(args.classes_dir))
        if not lab_models or len(lab_models) < 3:
            print("Warning: LAB fusion disabled (need red/green/blue reference images in classes-dir).")
            lab_models = None

    lr_head: ColorLogisticHead | None = None
    if (
        not args.disable_lr_head
        and joblib is not None
        and args.lr_head_path.is_file()
    ):
        blob = joblib.load(args.lr_head_path)
        lr_head = ColorLogisticHead(
            model=blob["model"],
            classes=tuple(blob.get("classes", ("red", "green", "blue"))),
        )
        print(f"Loaded logistic color head: {args.lr_head_path}")

    tree_head: ColorTreeHead | None = None
    if (
        not args.disable_tree_head
        and joblib is not None
        and float(args.tree_geometric_beta) > 0.0
        and args.tree_head_path.is_file()
    ):
        tblob = joblib.load(args.tree_head_path)
        mdl = tblob["model"]
        proba_cls = tblob.get("proba_classes")
        if not proba_cls:
            proba_cls = infer_sklearn_predict_proba_class_order(mdl)
        tree_head = ColorTreeHead(model=mdl, proba_classes=tuple(str(c) for c in proba_cls))
        print(f"Loaded boosted color head (+geometric fuse beta={args.tree_geometric_beta}): {args.tree_head_path}")

    bgr_centroids = derive_bgr_centroids(str(args.classes_dir))
    if not bgr_centroids or len(bgr_centroids) < 3:
        print("Warning: BGR centroid fusion disabled (need reference PNGs).")
        bgr_centroids = None

    if args.hsv_reval_only:
        best_w = args.out_root / "ultralytics_runs" / "roi_detector" / "weights" / "best.pt"
        yml_abs = args.out_root / "dataset_roi" / "dataset_abs.yaml"
        test_img_dir = args.out_root / "dataset_roi" / "images" / "test"
        if not best_w.exists():
            print(f"Missing weights for --hsv-reval-only: {best_w}")
            return 1
        if not yml_abs.exists():
            print(f"Missing dataset yaml: {yml_abs}")
            return 1
        if not test_img_dir.is_dir():
            print(f"Missing test images: {test_img_dir}")
            return 1
        split_counts = {
            "train": len(list((args.out_root / "dataset_roi" / "images" / "train").glob("*.jpg"))),
            "val": len(list((args.out_root / "dataset_roi" / "images" / "val").glob("*.jpg"))),
            "test": len(list(test_img_dir.glob("*.jpg"))),
        }
        return run_hsv_test_eval(
            args,
            ranges_map,
            best_w=best_w,
            yml_abs=yml_abs,
            split_counts=split_counts,
            lab_models=lab_models,
            lr_head=lr_head,
            tree_head=tree_head,
            bgr_centroids=bgr_centroids,
        )

    if args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.captures_root, args.labels_subdir)
    if not samples:
        print(f"No samples found under {args.captures_root}")
        return 1
    train_s, val_s, test_s = split_samples(samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    prepare_split_dir("train", train_s, args.out_root)
    prepare_split_dir("val", val_s, args.out_root)
    prepare_split_dir("test", test_s, args.out_root)
    yml_abs = write_dataset_yaml(args.out_root)

    print(f"Split counts: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    model = YOLO("yolo11n.pt")
    run_dir = args.out_root / "ultralytics_runs"
    train_res = model.train(
        data=str(yml_abs),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(run_dir),
        name="roi_detector",
        exist_ok=True,
        pretrained=True,
        cache=False,
        workers=args.workers,
    )

    best_w = Path(train_res.save_dir) / "weights" / "best.pt"
    if not best_w.exists():
        print(f"Best weights missing: {best_w}")
        return 1

    split_counts = {"train": len(train_s), "val": len(val_s), "test": len(test_s)}
    return run_hsv_test_eval(
        args,
        ranges_map,
        best_w=best_w,
        yml_abs=yml_abs,
        split_counts=split_counts,
        lab_models=lab_models,
        lr_head=lr_head,
        tree_head=tree_head,
        bgr_centroids=bgr_centroids,
    )


if __name__ == "__main__":
    raise SystemExit(main())


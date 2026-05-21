#!/usr/bin/env python3
"""Train YOLOv11n as single-class buoy ROI detector, then classify ROI color with HSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys

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

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from color_utils import (
    build_mask,
    ColorLogisticHead,
    ColorTreeHead,
    FALLBACK_COLOR_RANGES,
    HSVRange,
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

# First token matched wins; covers RobotX UAV clip folder names after robotx_dr_XXX_...
GLARE_ENV_TOKENS: tuple[str, ...] = (
    "clear_blue",
    "sunset_glint",
    "greenish_murky",
    "dark_overcast",
    "silt_haze",
)


def clip_from_symlink(dataset_test_dir: Path, image_name: str) -> str | None:
    p = dataset_test_dir / image_name
    if not p.is_symlink():
        return None
    try:
        raw = os.readlink(str(p))
    except OSError:
        return None
    parts = Path(raw).parts
    if "gazebo_uav_batch" in parts:
        i = parts.index("gazebo_uav_batch")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def glare_setting_from_clip(clip_folder: str) -> str | None:
    """Map capture clip folder name to a coarse glare / water-appearance bucket."""
    for tok in GLARE_ENV_TOKENS:
        if tok in clip_folder:
            return tok
    return None


def bucket_test_images_by_glare(test_img_dir: Path) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = defaultdict(list)
    for img_path in sorted(test_img_dir.glob("*.jpg")):
        clip = clip_from_symlink(test_img_dir, img_path.name)
        if not clip:
            continue
        glare = glare_setting_from_clip(clip)
        if glare:
            buckets[glare].append(img_path)
    return dict(buckets)


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
    p.add_argument(
        "--skip-det-train",
        action="store_true",
        help="Prepare dataset split but skip YOLO training; run HSV eval using --weights (or existing best.pt under out-root).",
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Weights path for --skip-det-train or external eval (e.g. runs/.../best.pt).",
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
        "--hsv-center-circle-cv",
        action="store_true",
        help="Use lightweight traditional CV HSV mode (center circle sampling, no LR/tree/LAB heads).",
    )
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
        "--temporal-stabilize",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable frame-to-frame color label stabilization using nearest-center matching.",
    )
    p.add_argument(
        "--temporal-match-px",
        type=float,
        default=70.0,
        help="Max pixel distance for temporal center matching between adjacent frames.",
    )
    p.add_argument(
        "--temporal-switch-margin",
        type=float,
        default=0.12,
        help="Required confidence margin to switch away from prior frame color label.",
    )
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
    p.add_argument(
        "--generate-glare-demos",
        action="store_true",
        help="With --hsv-reval-only: one detection MP4 + reprojection overlay MP4 per lighting family (requires test symlinks).",
    )
    p.add_argument(
        "--glare-demo-max-images",
        type=int,
        default=200,
        help="Max annotated frames processed per glare setting (speed / video size); 0 = use all bucket images.",
    )
    p.add_argument(
        "--demo-fps",
        type=float,
        default=24.0,
        help="FPS for demo MP4s when --generate-glare-demos is set.",
    )
    p.add_argument(
        "--demo-altitude-m",
        type=float,
        default=12.0,
        help="Nadir projection altitude used for glare reprojection demos.",
    )
    p.add_argument(
        "--demo-heading-deg",
        type=float,
        default=10.0,
        help="Drone heading (degrees) used for glare reprojection demos.",
    )
    p.add_argument(
        "--demo-fx-px",
        type=float,
        default=1319.071398,
        help="Horizontal focal length for reprojection demos (pixels).",
    )
    p.add_argument(
        "--demo-fy-px",
        type=float,
        default=1407.498400,
        help="Vertical focal length for reprojection demos (pixels).",
    )
    p.add_argument(
        "--demo-cx-px",
        type=float,
        default=870.934930,
        help="Principal point cx for demos; <= 0 centers automatically.",
    )
    p.add_argument(
        "--demo-cy-px",
        type=float,
        default=533.095324,
        help="Principal point cy for demos; <= 0 centers automatically.",
    )
    p.add_argument("--demo-drone-lat", type=float, default=32.88010)
    p.add_argument("--demo-drone-lon", type=float, default=-117.23420)
    p.add_argument(
        "--eval-reproj-sim-metrics",
        action="store_true",
        help="Aggregate planar error (m) vs simulator GT buoys using world SDFs (same as offline reprojection_eval).",
    )
    p.add_argument(
        "--eval-worlds-dir",
        type=Path,
        default=REPO_ROOT / "gazebo" / "worlds" / "generated",
        help="RobotX *.sdf directory for simulator GT buoy positions (--eval-reproj-sim-metrics).",
    )
    p.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="After collecting samples, keep at most this many clips (0 = all). Shuffled with --seed before capping.",
    )
    p.add_argument(
        "--subset-max-samples",
        type=int,
        default=0,
        help="Cap total frames used (0 = all). May cut clips in half; prefer --max-clips for clip-integrity subsampling.",
    )
    return p.parse_args()


def _safe_unlink_or_rmtree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def limit_dataset_samples(samples: list[Sample], max_clips: int, max_samples: int, seed: int) -> list[Sample]:
    if not samples:
        return []
    by_clip: dict[str, list[Sample]] = {}
    for smp in samples:
        clip = smp.key.split("/", 1)[0]
        by_clip.setdefault(clip, []).append(smp)

    clip_names = sorted(by_clip.keys())
    if max_clips and max_clips > 0:
        random.Random(seed).shuffle(clip_names)
        clip_names = clip_names[:max_clips]

    out: list[Sample] = []
    for c in clip_names:
        out.extend(sorted(by_clip[c], key=lambda s: s.key))
    if max_samples and max_samples > 0 and len(out) > max_samples:
        out = sorted(out, key=lambda s: s.key)[:max_samples]
    return out


def collect_samples(captures_root: Path, labels_subdir: str) -> list[Sample]:
    if not captures_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {captures_root}\n"
            "Expected clip folders with uav_*.jpg and labels_proj/ (Gazebo batch). "
            "Restore from backup/Time Machine or re-export from sim, or pass --captures-root to a copy on disk."
        )
    samples: list[Sample] = []
    for clip_dir in sorted([p for p in captures_root.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        labels_dir = clip_dir / labels_subdir
        if not labels_dir.is_dir():
            continue
        for image_path in sorted(clip_dir.glob("uav_*.jpg")):
            if not image_path.is_file():
                continue
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

    # With 1–2 clips, clip-group split cannot produce non-empty train/val/test; shuffle frames instead.
    if len(clip_names) < 3:
        rng = random.Random(seed)
        p = list(samples)
        rng.shuffle(p)
        nlen = len(p)
        if nlen == 0:
            return [], [], []
        if nlen == 1:
            return p, p, p
        if nlen == 2:
            return [p[0]], [p[1]], [p[1]]
        nt = max(1, int(round(nlen * train_ratio)))
        nv = max(1, int(round(nlen * val_ratio)))
        nte = max(1, nlen - nt - nv)
        over = nt + nv + nte - nlen
        if over > 0:
            nte = max(1, nte - over)
        elif over < 0:
            nte += -over
        if nt + nv + nte > nlen:
            nte = max(1, nlen - nt - nv)
        return p[:nt], p[nt : nt + nv], p[nt + nv : nlen]

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

    # Greedy fill by sample count can assign every clip to train+val and starve test (or val).
    if not test_clips and len(clip_names) >= 3:
        if len(train_clips) > 1:
            test_clips.append(train_clips.pop())
        elif len(val_clips) > 1:
            test_clips.append(val_clips.pop())
    if not val_clips and len(clip_names) >= 2 and train_clips:
        val_clips.append(train_clips.pop())
    if not train_clips and val_clips:
        train_clips.append(val_clips.pop(0))

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


def _clip_set(items: list[Sample]) -> set[str]:
    return {s.key.split("/", 1)[0] for s in items}


def assert_no_clip_leakage(train: list[Sample], val: list[Sample], test: list[Sample]) -> None:
    tr = _clip_set(train)
    va = _clip_set(val)
    te = _clip_set(test)
    overlap = {
        "train_val": sorted(tr & va),
        "train_test": sorted(tr & te),
        "val_test": sorted(va & te),
    }
    if overlap["train_val"] or overlap["train_test"] or overlap["val_test"]:
        raise RuntimeError(
            "Clip leakage detected across splits: "
            + ", ".join(f"{k}={len(v)}" for k, v in overlap.items())
        )


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
        os.symlink(s.image_path.resolve(), img_link)
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


def greedy_match_predictions_to_gt(
    gt_boxes: list[GTBox],
    preds: list[tuple[float, float, float, float, float, str]],
    iou_thresh: float,
) -> tuple[int, int, int, int]:
    """
    One prediction → at most one GT; greedy by descending confidence.

    Returns (tp, fp, fn, tp_with_correct_color_name).
    """
    if not gt_boxes:
        return 0, len(preds), 0, 0
    if not preds:
        return 0, 0, len(gt_boxes), 0

    order = sorted(range(len(preds)), key=lambda i: preds[i][4], reverse=True)
    unmatched_gt = set(range(len(gt_boxes)))
    tp = fp = tcp = 0
    for ii in order:
        x1, y1, x2, y2, _, color = preds[ii]
        best_j = -1
        best_iou_v = iou_thresh
        box_p = (x1, y1, x2, y2)
        for gj in unmatched_gt:
            gv = gt_boxes[gj]
            v = iou_xyxy(box_p, (gv.x1, gv.y1, gv.x2, gv.y2))
            if v >= best_iou_v:
                best_iou_v = v
                best_j = gj
        if best_j < 0:
            fp += 1
            continue
        tp += 1
        unmatched_gt.remove(best_j)
        gt_nm = CLASS_NAME.get(gt_boxes[best_j].cls_id, "unknown")
        if color == gt_nm:
            tcp += 1
    fn = len(unmatched_gt)
    return tp, fp, fn, tcp


def world_sdf_for_clip(worlds_dir: Path, clip_name: str) -> Path | None:
    m_scn = re.search(r"robotx_dr_\d{3}_(.+)_s\d+$", clip_name)
    if m_scn:
        scenario = m_scn.group(1)
        exact = sorted(worlds_dir.glob(f"robotx_dr_*_{scenario}.sdf"))
        if exact:
            m_id = re.search(r"robotx_dr_(\d{3})", clip_name)
            if m_id:
                same_id = [p for p in exact if f"robotx_dr_{m_id.group(1)}_" in p.name]
                if same_id:
                    return same_id[0]
            return exact[0]
    m = re.search(r"robotx_dr_(\d{3})", clip_name)
    if not m:
        return None
    cands = sorted(worlds_dir.glob(f"robotx_dr_{m.group(1)}_*.sdf"))
    return cands[0] if cands else None


def _nearest_same_color_world_error(
    est_n: float,
    est_e: float,
    color: str,
    gt_pts: list[tuple[float, float, str]],
) -> float | None:
    best_d = float("inf")
    for gn, ge, gc in gt_pts:
        if gc != color:
            continue
        d = math.hypot(est_n - gn, est_e - ge)
        if d < best_d:
            best_d = d
    return best_d if best_d < float("inf") else None


def summarize_reprojection_vs_sim_world(
    csv_path: Path,
    dataset_test_dir: Path,
    worlds_dir: Path,
    *,
    altitude_m: float,
    fx_px: float,
    fy_px: float,
    cx_px: float,
    cy_px: float,
    heading_deg: float,
    drone_x_m: float = 0.0,
    drone_y_m: float = 0.0,
) -> dict[str, object] | None:
    """
    Euclidean error in simulator world horizontal plane vs nearest same-color buoy in matched SDF.
    """
    if not csv_path.is_file() or not dataset_test_dir.is_dir() or not worlds_dir.is_dir():
        return None
    try:
        from reproject_detection_map import parse_buoys_from_world_sdf, project_pixel_to_ground_ned
    except ImportError:
        return None

    errs: list[float] = []
    skipped_world = 0
    skipped_no_gt = 0
    clips_cache: dict[str, tuple[Path | None, list[tuple[float, float, str]] | None]] = {}

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fobj:
        for row in csv.DictReader(fobj):
            rows.append(row)

    for row in rows:
        img_name = row.get("image", "")
        clip = clip_from_symlink(dataset_test_dir, img_name)
        if clip is None:
            skipped_world += 1
            continue
        if clip not in clips_cache:
            ws = world_sdf_for_clip(worlds_dir.resolve(), clip)
            gt_pts = parse_buoys_from_world_sdf(ws) if ws and ws.is_file() else []
            clips_cache[clip] = (ws, gt_pts)

        ws, gt_pts = clips_cache[clip]
        if not ws or gt_pts is None or not gt_pts:
            skipped_world += 1
            continue
        color = row.get("pred_color_hsv", "unknown").strip().lower()
        try:
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
        except (KeyError, ValueError):
            continue
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        north_cam, east_cam = project_pixel_to_ground_ned(
            u, v, altitude_m, fx_px, fy_px, cx_px, cy_px, heading_deg
        )
        wn = drone_y_m + north_cam
        we = drone_x_m + east_cam

        dd = _nearest_same_color_world_error(wn, we, color, gt_pts)
        if dd is None:
            skipped_no_gt += 1
            continue
        errs.append(dd)

    if not errs:
        return {
            "note": "no_errors_computed",
            "rows_seen": len(rows),
            "skipped_clip_or_world": skipped_world,
            "skipped_nearest_same_color_gt": skipped_no_gt,
        }

    errs.sort()
    n = len(errs)
    p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
    return {
        "samples_used": n,
        "error_mean_m": float(sum(errs) / n),
        "error_median_m": float(errs[n // 2]),
        "error_p95_m": float(errs[p95_idx]),
        "rmse_m": float(math.sqrt(sum(e * e for e in errs) / n)),
        "skipped_clip_or_world": skipped_world,
        "skipped_nearest_same_color_gt": skipped_no_gt,
        "projection_params": {
            "altitude_m": altitude_m,
            "fx_px": fx_px,
            "fy_px": fy_px,
            "cx_px": cx_px,
            "cy_px": cy_px,
            "heading_deg": heading_deg,
            "drone_x_m": drone_x_m,
            "drone_y_m": drone_y_m,
        },
    }


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
    center_circle_cv_rules: bool,
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
    color, score = classify_buoy_roi_hsv(
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
    if not center_circle_cv_rules:
        return color, score

    # Deterministic guardrails for CV-only deployment mode:
    # use a rim mask (outside center core) to read buoy paint color and
    # reduce green↔red flicker from low-saturation center pixels.
    gm = geom_mask.astype(bool)
    total = int(np.count_nonzero(gm))
    if total < 10:
        return color, score

    outer_r = min(0.48, max(circle_r_frac * 1.75, 0.36))
    outer_mask = geom_sampling_mask(
        rh,
        rw,
        shape="circle",
        ellipse_rx_frac=ellipse_rx_frac,
        ellipse_ry_frac=ellipse_ry_frac,
        circle_r_frac=outer_r,
        inner_shrink_xy=inner_shrink_xy,
        cap_band_top_frac=cap_band_top_frac,
    ).astype(bool)
    rim = outer_mask & (~gm)
    rim_total = int(np.count_nonzero(rim))

    ratios: dict[str, float] = {}
    rim_ratios: dict[str, float] = {}
    for cname in ("red", "green", "blue"):
        hsv_ranges = [HSVRange(low=low, high=high) for (low, high) in FALLBACK_COLOR_RANGES[cname]]
        m = build_mask(roi_hsv, hsv_ranges) > 0
        ratios[cname] = float(np.count_nonzero(m & gm)) / float(total)
        rim_ratios[cname] = float(np.count_nonzero(m & rim)) / float(max(1, rim_total))

    mean_g = float(np.mean(roi_bgr[:, :, 1][gm]))
    mean_r = float(np.mean(roi_bgr[:, :, 2][gm]))
    med_s = float(np.median(roi_hsv[:, :, 1][gm]))
    rim_v_mean = float(np.mean(roi_hsv[:, :, 2][rim])) if rim_total > 0 else 255.0
    rim_s_mean = float(np.mean(roi_hsv[:, :, 1][rim])) if rim_total > 0 else 255.0

    r = ratios["red"]
    g = ratios["green"]
    b = ratios["blue"]
    rr = rim_ratios["red"]
    rg = rim_ratios["green"]
    rb = rim_ratios["blue"]

    if rim_total >= 20:
        if rb >= 0.11 and rb > max(rr, rg) + 0.03:
            return "blue", max(score, rb)
        if rg >= 0.14 and rg > max(rr, rb) + 0.03:
            return "green", max(score, rg)
        if rr >= 0.14 and rr > max(rg, rb) + 0.03:
            return "red", max(score, rr)

    if b >= 0.08 and b > 1.20 * max(r, g):
        return "blue", max(score, b)
    if max(r, g) < 0.10 and b >= 0.04:
        return "blue", max(score, b)
    # Blue rescue: in this domain blue often looks like low-confidence green with dark/desaturated rim.
    if color == "green" and score <= 0.53 and rim_v_mean <= 69.0 and rim_s_mean <= 52.0:
        return "blue", max(score, 0.56)
    if g >= 0.14 and g >= 1.10 * r and g >= 1.25 * b:
        return "green", max(score, g)
    if color == "red" and g >= 0.10 and g >= 0.85 * r:
        return "green", max(score, g)
    return color, score


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
    eval_subdir: str = "",
    image_paths_override: list[Path] | None = None,
    bbox_mp4_out: Path | None = None,
    demo_subtitle: str = "",
) -> int:
    det_model = YOLO(str(best_w))
    test_img_dir = args.out_root / "dataset_roi" / "images" / "test"
    eval_root = args.out_root / "test_eval"
    sub = (eval_subdir or "").strip().strip("/")
    eval_dir = eval_root / sub if sub else eval_root
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
                "clip",
                "image",
                "box_conf",
                "pred_color_hsv",
                "color_conf",
                "matched_gt_color",
                "match_iou",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )
        if image_paths_override is None:
            test_paths = sorted(test_img_dir.glob("*.jpg"))
            _mt = int(getattr(args, "max_test_images", 0) or 0)
            if _mt > 0:
                test_paths = test_paths[:_mt]
        else:
            test_paths = sorted(image_paths_override)
        n_test_used = len(test_paths)
        bbox_vw = None
        demo_fps = float(getattr(args, "demo_fps", 24.0) or 24.0)
        prev_tracks: list[dict[str, float | str]] = []

        def _subtitle(frame_bgr: np.ndarray, lines: tuple[str, ...]) -> None:
            yy = frame_bgr.shape[0] - 18 - 18 * len(lines)
            for line in lines:
                cv2.putText(
                    frame_bgr,
                    line,
                    (12, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                yy += 22

        for img_path in test_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, ww = frame.shape[:2]
            if bbox_mp4_out is not None and bbox_vw is None:
                bbox_mp4_out.parent.mkdir(parents=True, exist_ok=True)
                bbox_vw = cv2.VideoWriter(
                    str(bbox_mp4_out),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    demo_fps,
                    (ww, h),
                )
                if not bbox_vw.isOpened():
                    print(f"Warning: could not open bbox VideoWriter at {bbox_mp4_out}")
                    bbox_vw = None

            if demo_subtitle:
                _subtitle(frame, (demo_subtitle,))

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gt_label_path = find_original_label_for_link(img_path)
            gt_boxes: list[GTBox] = []
            if gt_label_path.exists():
                for line in gt_label_path.read_text(encoding="utf-8").splitlines():
                    b = yolo_line_to_box(line.strip().split(), ww, h)
                    if b is not None:
                        gt_boxes.append(b)

            gt_overlay_bgr = (0, 220, 255)
            for gt in gt_boxes:
                gx1 = max(0, int(gt.x1))
                gy1 = max(0, int(gt.y1))
                gx2 = min(ww - 1, int(gt.x2))
                gy2 = min(h - 1, int(gt.y2))
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), gt_overlay_bgr, 1, cv2.LINE_AA)
                gnm = CLASS_NAME.get(gt.cls_id, "?")
                cv2.putText(
                    frame,
                    f"GT:{gnm}",
                    (gx1, max(18, gy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    gt_overlay_bgr,
                    1,
                    cv2.LINE_AA,
                )

            preds = det_model(frame, verbose=False, conf=args.conf_thres)[0]
            boxes = preds.boxes
            n_det = len(boxes) if boxes is not None else 0
            cur_tracks: list[dict[str, float | str]] = []
            used_prev: set[int] = set()
            for i in range(n_det):
                conf = float(boxes.conf[i].item())
                x1f, y1f, x2f, y2f = boxes.xyxy[i].tolist()
                x1 = max(0, int(round(x1f)))
                y1 = max(0, int(round(y1f)))
                x2 = min(ww - 1, int(round(x2f)))
                y2 = min(h - 1, int(round(y2f)))
                if x2 <= x1 or y2 <= y1:
                    continue
                total_preds += 1
                cx_det = 0.5 * (x1 + x2)
                cy_det = 0.5 * (y1 + y2)
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
                    center_circle_cv_rules=args.hsv_center_circle_cv,
                )

                if args.temporal_stabilize and prev_tracks:
                    best_j = -1
                    best_d2 = float("inf")
                    for j, tr in enumerate(prev_tracks):
                        if j in used_prev:
                            continue
                        dx = float(cx_det) - float(tr["cx"])
                        dy = float(cy_det) - float(tr["cy"])
                        d2 = dx * dx + dy * dy
                        if d2 < best_d2:
                            best_d2 = d2
                            best_j = j
                    if best_j >= 0 and best_d2 <= float(args.temporal_match_px) ** 2:
                        used_prev.add(best_j)
                        tr = prev_tracks[best_j]
                        prev_label = str(tr["label"])
                        prev_score = float(tr["score"])
                        if hsv_color != prev_label and (hsv_score + float(args.temporal_switch_margin)) < prev_score:
                            hsv_color = prev_label
                            hsv_score = max(hsv_score, prev_score * 0.98)
                cur_tracks.append({"cx": float(cx_det), "cy": float(cy_det), "label": hsv_color, "score": float(hsv_score)})

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
                clip_name = clip_from_symlink(test_img_dir, img_path.name) or ""
                w.writerow(
                    [
                        clip_name,
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
            if bbox_vw is not None:
                bbox_vw.write(frame)
            prev_tracks = cur_tracks

        if bbox_vw is not None:
            bbox_vw.release()

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
    if sub:
        summary["eval_subdir"] = sub
    per_clip_dir = eval_dir / "per_clip"
    per_clip_dir.mkdir(parents=True, exist_ok=True)
    by_clip: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as fobj:
        for row in csv.DictReader(fobj):
            clip = (row.get("clip") or "").strip()
            if not clip:
                clip = clip_from_symlink(test_img_dir, row.get("image", "")) or "_unknown_clip"
            by_clip[clip].append(row)
    clip_counts: dict[str, int] = {}
    for clip, rows in sorted(by_clip.items()):
        out_csv = per_clip_dir / f"{clip}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as fobj:
            wcsv = csv.writer(fobj)
            wcsv.writerow(
                [
                    "image",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "pred_color_hsv",
                    "box_conf",
                    "color_conf",
                    "matched_gt_color",
                    "match_iou",
                ]
            )
            for r in rows:
                wcsv.writerow(
                    [
                        r.get("image", ""),
                        r.get("x1", ""),
                        r.get("y1", ""),
                        r.get("x2", ""),
                        r.get("y2", ""),
                        r.get("pred_color_hsv", ""),
                        r.get("box_conf", ""),
                        r.get("color_conf", ""),
                        r.get("matched_gt_color", ""),
                        r.get("match_iou", ""),
                    ]
                )
        clip_counts[clip] = len(rows)
    summary["per_clip_dir"] = str(per_clip_dir)
    summary["per_clip_detection_rows"] = clip_counts
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


def run_glare_demos(
    args: argparse.Namespace,
    *,
    ranges_map: dict[str, list],
    best_w: Path,
    yml_abs: Path,
    split_counts: dict[str, int],
    lab_models: dict[str, LabColorModel] | None,
    lr_head: ColorLogisticHead | None,
    tree_head: ColorTreeHead | None,
    bgr_centroids: dict | None,
) -> int:
    test_img_dir = args.out_root / "dataset_roi" / "images" / "test"
    buckets = bucket_test_images_by_glare(test_img_dir)
    glare_root = args.out_root / "test_eval" / "glare_demos"
    glare_root.mkdir(parents=True, exist_ok=True)
    cap_lim = int(getattr(args, "glare_demo_max_images", 0) or 0)

    rp_script = SCRIPT_DIR / "reproject_detection_map.py"
    manifest: dict[str, object] = {
        "buckets_found": sorted(buckets.keys()),
        "clips_per_bucket": {k: len(v) for k, v in buckets.items()},
    }

    ordered = tuple(GLARE_ENV_TOKENS)

    for tag in ordered:
        paths = buckets.get(tag)
        if not paths:
            print(f"[glare demo] skip {tag}: no test symlinks mapped to bucket")
            continue
        subset = sorted(paths)
        if cap_lim > 0:
            subset = subset[:cap_lim]

        eval_subdir = f"glare_demos/{tag}"
        bbox_mp4 = glare_root / f"{tag}_detection_demo.mp4"
        run_hsv_test_eval(
            args,
            ranges_map,
            best_w=best_w,
            yml_abs=yml_abs,
            split_counts=split_counts,
            lab_models=lab_models,
            lr_head=lr_head,
            tree_head=tree_head,
            bgr_centroids=bgr_centroids,
            eval_subdir=eval_subdir,
            image_paths_override=subset,
            bbox_mp4_out=bbox_mp4,
            demo_subtitle=f"{tag} | live ROI + fused color pipeline",
        )

        eval_here = args.out_root / "test_eval" / eval_subdir
        subprocess.run(
            [
                sys.executable,
                str(rp_script),
                "--csv",
                str(eval_here / "roi_hsv_test_results.csv"),
                "--frames-dir",
                str(eval_here / "annotated_frames"),
                "--out-dir",
                str(glare_root),
                "--intrinsics",
                str(REPO_ROOT / "calibration" / "camera_intrinsics_latest.json"),
                "--altitude-m",
                str(getattr(args, "demo_altitude_m", 12.0)),
                "--heading-deg",
                str(getattr(args, "demo_heading_deg", 10.0)),
                "--drone-lat",
                str(getattr(args, "demo_drone_lat", 32.88010)),
                "--drone-lon",
                str(getattr(args, "demo_drone_lon", -117.23420)),
                "--fps",
                str(float(getattr(args, "demo_fps", 24.0) or 24.0)),
                "--demo-label",
                f"{tag} — nadir projection (GPS deltas from drone ref)",
            ],
            check=True,
        )
        overlay_default = glare_root / "reprojection_overlay.mp4"
        target_overlay = glare_root / f"{tag}_reprojection_overlay.mp4"
        if overlay_default.is_file():
            shutil.move(str(overlay_default), str(target_overlay))

        sample_default = glare_root / "reprojection_sample.jpg"
        if sample_default.is_file():
            shutil.move(str(sample_default), str(glare_root / f"{tag}_reprojection_sample.jpg"))

    glare_root.joinpath("glare_demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[glare demo] wrote outputs under {glare_root}")
    return 0


def main() -> int:
    args = parse_args()
    apply_plot_max_subplots()

    if args.hsv_center_circle_cv:
        # Lightweight Jetson-friendly preset: detector + traditional HSV.
        args.hsv_mask_shape = "circle"
        args.circle_r_frac = 0.24
        args.roi_inner_crop = 0.92
        args.cap_band_top_frac = 0.12
        args.suppress_water = False
        args.water_sat_max = 60
        args.water_val_min = 28
        args.disable_lab_fusion = True
        args.disable_lr_head = True
        args.disable_tree_head = True
        args.fusion_lab_weight = 0.0
        args.fusion_hsv_weight = 1.0
        # In strict CV mode, avoid RGB/LAB side channels that can skew red↔green.
        args.fusion_rgb_weight = 0.0
        args.rg_bgr_balance_strength = 0.0
        args.fusion_temperature = 1.0
        args.temporal_stabilize = True

    ranges_map = derive_class_hsv_ranges(str(args.classes_dir), hue_margin=12, sat_floor=50, val_floor=45)
    if args.hsv_center_circle_cv:
        # Force canonical HSV thresholds for deployment-style deterministic behavior.
        # Using fallback ranges avoids dependence on potentially stale/corrupted swatch PNGs.
        ranges_map = {
            color: [HSVRange(low=low, high=high) for (low, high) in FALLBACK_COLOR_RANGES[color]]
            for color in ("red", "green", "blue")
        }
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
        if args.generate_glare_demos:
            return run_glare_demos(
                args,
                ranges_map=ranges_map,
                best_w=best_w,
                yml_abs=yml_abs,
                split_counts=split_counts,
                lab_models=lab_models,
                lr_head=lr_head,
                tree_head=tree_head,
                bgr_centroids=bgr_centroids,
            )
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
    samples = limit_dataset_samples(samples, args.max_clips, args.subset_max_samples, args.seed)
    if not samples:
        print(f"No samples found under {args.captures_root}")
        return 1
    train_s, val_s, test_s = split_samples(samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    assert_no_clip_leakage(train_s, val_s, test_s)
    prepare_split_dir("train", train_s, args.out_root)
    prepare_split_dir("val", val_s, args.out_root)
    prepare_split_dir("test", test_s, args.out_root)
    yml_abs = write_dataset_yaml(args.out_root)

    print(f"Split counts: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    if args.skip_det_train:
        best_w = args.weights or (args.out_root / "ultralytics_runs" / "roi_detector" / "weights" / "best.pt")
        if not best_w.exists():
            print(f"Missing weights for --skip-det-train: {best_w}")
            return 1
        split_counts = {"train": len(train_s), "val": len(val_s), "test": len(test_s)}
        if args.generate_glare_demos:
            return run_glare_demos(
                args,
                ranges_map=ranges_map,
                best_w=best_w,
                yml_abs=yml_abs,
                split_counts=split_counts,
                lab_models=lab_models,
                lr_head=lr_head,
                tree_head=tree_head,
                bgr_centroids=bgr_centroids,
            )
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
    if args.generate_glare_demos:
        return run_glare_demos(
            args,
            ranges_map=ranges_map,
            best_w=best_w,
            yml_abs=yml_abs,
            split_counts=split_counts,
            lab_models=lab_models,
            lr_head=lr_head,
            tree_head=tree_head,
            bgr_centroids=bgr_centroids,
        )
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


#!/usr/bin/env python3
"""Shared color range helpers for Stage-A HSV detection pipelines."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HSVRange:
    low: tuple[int, int, int]
    high: tuple[int, int, int]


@dataclass
class LabColorModel:
    """2D Gaussian on OpenCV Lab channels a*, b* (uint8 0–255, neutral ~128)."""

    mean: np.ndarray  # shape (2,)
    inv_cov: np.ndarray  # shape (2, 2)
    log_norm: float  # log(sqrt((2pi)^2 det cov)))

    def log_pdf(self, ab: np.ndarray) -> float:
        d = ab.astype(np.float64) - self.mean
        q = float(d @ self.inv_cov @ d)
        return -0.5 * q - self.log_norm


def derive_bgr_centroids(classes_dir: str, *, min_sat: int = 30) -> dict[str, np.ndarray]:
    """Mean BGR (3,) per class from saturated pixels in reference swatches."""
    centroids: dict[str, np.ndarray] = {}
    if not os.path.isdir(classes_dir):
        return centroids

    for path in list_images(classes_dir):
        color = os.path.splitext(os.path.basename(path))[0].lower()
        if color not in {"red", "green", "blue"}:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(np.float64)
        m = s >= min_sat
        if int(np.count_nonzero(m)) < 80:
            m = np.ones_like(s, dtype=bool)
        pix = img.astype(np.float64)[m]
        centroids[color] = np.mean(pix, axis=0)
    return centroids


def derive_lab_ab_models(classes_dir: str, *, min_sat: int = 30) -> dict[str, LabColorModel]:
    """
    Fit per-class (a,b) means from reference swatches with a **shared pooled** covariance (LDA-style).
    Per-class covariances often make one class (e.g. green) dominate likelihood in mixed lighting.
    """
    models: dict[str, LabColorModel] = {}
    if not os.path.isdir(classes_dir):
        return models

    class_pts: dict[str, list[np.ndarray]] = {"red": [], "green": [], "blue": []}
    for path in list_images(classes_dir):
        color = os.path.splitext(os.path.basename(path))[0].lower()
        if color not in class_pts:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        s = hsv[:, :, 1].astype(np.float64)
        m = s >= min_sat
        if int(np.count_nonzero(m)) < 80:
            m = np.ones_like(s, dtype=bool)
        a = lab[:, :, 1].astype(np.float64)[m]
        b = lab[:, :, 2].astype(np.float64)[m]
        class_pts[color].append(np.stack([a, b], axis=1))

    pooled_list: list[np.ndarray] = []
    means: dict[str, np.ndarray] = {}
    for color in ("red", "green", "blue"):
        if not class_pts[color]:
            return {}
        pts = np.vstack(class_pts[color])
        means[color] = np.mean(pts, axis=0)
        pooled_list.append(pts)

    all_pts = np.vstack(pooled_list)
    cov = np.cov(all_pts, rowvar=False)
    cov = cov + np.eye(2, dtype=np.float64) * 10.0
    inv_cov = np.linalg.inv(cov)
    det = float(np.linalg.det(cov))
    det = max(det, 1e-9)
    log_norm = 0.5 * math.log((2.0 * math.pi) ** 2 * det)

    for color in ("red", "green", "blue"):
        models[color] = LabColorModel(mean=means[color], inv_cov=inv_cov, log_norm=log_norm)

    return models


def _softmax_scores(raw: dict[str, float], *, keys: tuple[str, ...] = ("red", "green", "blue")) -> dict[str, float]:
    xs = np.array([float(raw.get(k, 0.0)) for k in keys], dtype=np.float64)
    xs = xs - np.max(xs)
    ex = np.exp(np.clip(xs, -80.0, 80.0))
    z = float(np.sum(ex)) + 1e-12
    return {k: float(ex[i] / z) for i, k in enumerate(keys)}


FALLBACK_COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 100, 70), (10, 255, 255)), ((170, 100, 70), (179, 255, 255))],
    "green": [((75, 60, 50), (105, 255, 255)),
    ],
    "blue": [((100, 80, 60), (130, 255, 255))],
}


def list_images(folder: str) -> list[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths: list[str] = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(exts):
            paths.append(path)
    return paths


def circular_hue_mean(hues: np.ndarray) -> int:
    angles = (hues.astype(np.float32) / 180.0) * 2.0 * np.pi
    s = np.sin(angles).mean()
    c = np.cos(angles).mean()
    mean_angle = np.arctan2(s, c)
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return int(round((mean_angle / (2.0 * np.pi)) * 180.0)) % 180


def make_ranges_for_hue(hue_center: int, hue_margin: int, s_min: int, v_min: int) -> list[HSVRange]:
    h_low = hue_center - hue_margin
    h_high = hue_center + hue_margin
    if h_low < 0:
        return [
            HSVRange((0, s_min, v_min), (h_high, 255, 255)),
            HSVRange((180 + h_low, s_min, v_min), (179, 255, 255)),
        ]
    if h_high > 179:
        return [
            HSVRange((h_low, s_min, v_min), (179, 255, 255)),
            HSVRange((0, s_min, v_min), (h_high - 180, 255, 255)),
        ]
    return [HSVRange((h_low, s_min, v_min), (h_high, 255, 255))]


def derive_class_hsv_ranges(classes_dir: str, hue_margin: int, sat_floor: int, val_floor: int) -> dict[str, list[HSVRange]]:
    out: dict[str, list[HSVRange]] = {}
    if not os.path.isdir(classes_dir):
        return out

    for path in list_images(classes_dir):
        color = os.path.splitext(os.path.basename(path))[0].lower()
        if color not in {"red", "green", "blue"}:
            continue

        img = cv2.imread(path)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        valid = (s > 25) & (v > 25)
        if int(valid.sum()) < 10:
            valid = np.ones_like(h, dtype=bool)

        hue_center = circular_hue_mean(h[valid])
        s_min = max(sat_floor, int(np.percentile(s[valid], 15)))
        v_min = max(val_floor, int(np.percentile(v[valid], 15)))
        out[color] = make_ranges_for_hue(hue_center, hue_margin, s_min, v_min)

    return out


def geom_sampling_mask(
    rh: int,
    rw: int,
    *,
    shape: str,
    ellipse_rx_frac: float,
    ellipse_ry_frac: float,
    circle_r_frac: float,
    inner_shrink_xy: float,
    cap_band_top_frac: float,
) -> np.ndarray:
    """Boolean mask for HSV sampling inside a detection ROI (fractions of ROI width/height)."""
    yy, xx = np.mgrid[0:rh, 0:rw]
    cx = (rw - 1) / 2.0
    cy = (rh - 1) / 2.0

    if shape == "ellipse":
        rx = max(2.0, ellipse_rx_frac * rw)
        ry = max(2.0, ellipse_ry_frac * rh)
        geom = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    elif shape == "circle":
        r = max(2.0, circle_r_frac * float(min(rw, rh)))
        geom = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)
    elif shape == "inner_rect":
        sx = max(0.05, min(0.95, inner_shrink_xy))
        sy = max(0.05, min(0.95, inner_shrink_xy))
        x0 = int(round(0.5 * (1.0 - sx) * rw))
        y0 = int(round(0.5 * (1.0 - sy) * rh))
        x1 = int(round(rw - x0))
        y1 = int(round(rh - y0))
        geom = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    else:
        rx = max(2.0, 0.38 * rw)
        ry = max(2.0, 0.38 * rh)
        geom = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

    if 0.0 < cap_band_top_frac <= 1.0:
        cut = int(round((1.0 - cap_band_top_frac) * rh))
        geom = geom & (yy < cut)
    return geom


def water_suppress_mask(
    hsv_roi: np.ndarray,
    *,
    sat_max: int,
    val_min: int,
    desat_blue_max: int = 38,
) -> np.ndarray:
    """Mask out low-saturation cyan/blue water and washed-out glints inside an ROI."""
    h = hsv_roi[:, :, 0].astype(np.int16)
    s = hsv_roi[:, :, 1].astype(np.int16)
    v = hsv_roi[:, :, 2].astype(np.int16)
    cyan_blue = (h >= 76) & (h <= 140)
    bright_wash = cyan_blue & (s <= sat_max) & (v >= val_min)
    flat_water = cyan_blue & (s <= desat_blue_max)
    return ~(bright_wash | flat_water)


def crop_roi_center(
    roi_bgr: np.ndarray,
    roi_hsv: np.ndarray,
    inner_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep central ``inner_frac`` of width and height (1.0 = unchanged). Reduces sky/water at box edges."""
    if inner_frac >= 0.999 or inner_frac <= 0.0:
        return roi_bgr, roi_hsv
    rh, rw = roi_bgr.shape[:2]
    if rh < 3 or rw < 3:
        return roi_bgr, roi_hsv
    mx = max(1, int(round(0.5 * rw * (1.0 - inner_frac))))
    my = max(1, int(round(0.5 * rh * (1.0 - inner_frac))))
    x0, y0 = mx, my
    x1, y1 = rw - mx, rh - my
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return roi_bgr, roi_hsv
    sl = np.s_[y0:y1, x0:x1]
    return roi_bgr[sl].copy(), roi_hsv[sl].copy()


def _rgb_distance_softmax(
    mean_bgr: np.ndarray,
    centroids: dict[str, np.ndarray],
    *,
    sigma: float = 55.0,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for c in ("red", "green", "blue"):
        mu = centroids.get(c)
        if mu is None:
            raw[c] = -1e9
        else:
            d = float(np.linalg.norm(mean_bgr - mu))
            raw[c] = -d / max(sigma, 1e-6)
    return _softmax_scores(raw)


TREE_H_BINS = 24
TREE_S_BINS = 10
TREE_V_BINS = 10


def stack_lr_features(evidence: dict[str, np.ndarray | float]) -> np.ndarray:
    """Fixed-order feature vector for `ColorLogisticHead` (must match fit_color_lr_head.py)."""
    pl = evidence["p_lab"]
    ph = evidence["p_hsv"]
    v = np.array(
        [
            pl["red"],
            pl["green"],
            pl["blue"],
            ph["red"],
            ph["green"],
            ph["blue"],
            float(evidence["ma_roi"]) / 255.0,
            float(evidence["mb_roi"]) / 255.0,
            float(evidence["chroma_mean"]) / 85.0,
            float(evidence["med_h"]) / 179.0,
            float(evidence["med_s"]) / 255.0,
            float(evidence["mean_b"]) / 255.0,
            float(evidence["mean_g"]) / 255.0,
            float(evidence["mean_r"]) / 255.0,
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(np.clip(v, -1e6, 1e6), nan=0.0, posinf=1e6, neginf=-1e6)


def stack_tree_features(
    roi_hsv: np.ndarray,
    evidence: dict[str, np.ndarray | float | dict[str, float]],
    *,
    h_bins: int = TREE_H_BINS,
    s_bins: int = TREE_S_BINS,
    v_bins: int = TREE_V_BINS,
) -> np.ndarray:
    """
    Logistic features + masked HSV histograms on `paint` (must match fit_color_boost_head.py).
    """
    lr = stack_lr_features(evidence)
    paint_raw = evidence.get("paint")
    if paint_raw is None:
        m = np.asarray(evidence["gm"], dtype=bool)
    else:
        m = np.asarray(paint_raw, dtype=bool)
    if int(np.count_nonzero(m)) < 4:
        m = np.asarray(evidence["gm"], dtype=bool)
    hh_flat = roi_hsv[:, :, 0].astype(np.float64)[m]
    sh_flat = roi_hsv[:, :, 1].astype(np.float64)[m]
    vh_flat = roi_hsv[:, :, 2].astype(np.float64)[m]
    h_hist, _ = np.histogram(hh_flat, bins=h_bins, range=(0, 180))
    s_hist, _ = np.histogram(sh_flat, bins=s_bins, range=(0, 256))
    v_hist, _ = np.histogram(vh_flat, bins=v_bins, range=(0, 256))
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float64)
    hist = hist / (float(np.sum(hist)) + 1e-9)
    out = np.concatenate([lr, hist])
    return np.nan_to_num(np.clip(out, -1e6, 1e6), nan=0.0, posinf=1e6, neginf=-1e6)


def compute_color_evidence_bundle(
    roi_bgr: np.ndarray,
    roi_hsv: np.ndarray,
    ranges_map: dict[str, list],
    geom_mask: np.ndarray,
    *,
    suppress_water: bool,
    water_sat_max: int,
    water_val_min: int,
    water_desat_blue_max: int,
    lab_models: dict[str, LabColorModel] | None,
) -> dict[str, np.ndarray | float | dict[str, float]] | None:
    """Shared facts for fusion + optional logistic head (LAB pooled models required for p_lab)."""
    rh, rw = roi_hsv.shape[:2]
    if rh < 2 or rw < 2 or not ranges_map:
        return None

    water_ok = (
        water_suppress_mask(
            roi_hsv,
            sat_max=water_sat_max,
            val_min=water_val_min,
            desat_blue_max=water_desat_blue_max,
        )
        if suppress_water
        else np.ones((rh, rw), dtype=bool)
    )

    gm = geom_mask & water_ok
    if int(np.count_nonzero(gm)) < 6:
        return None

    s = roi_hsv[:, :, 1]
    v = roi_hsv[:, :, 2]
    if int(np.count_nonzero(gm)) > 10:
        s_th = max(45, int(np.percentile(s[gm], 55)))
        v_th = max(40, int(np.percentile(v[gm], 35)))
    else:
        s_th = max(45, int(np.percentile(s, 55)))
        v_th = max(40, int(np.percentile(v, 35)))
    fg_mask = (s >= s_th) & (v >= v_th) & gm
    fg_count = int(np.count_nonzero(fg_mask))
    roi_area = float(rh * rw)

    hsv_raw = _hsv_branch_raw_scores(
        roi_bgr, roi_hsv, ranges_map, fg_mask=fg_mask, fg_count=fg_count, roi_area=roi_area
    )
    if not hsv_raw:
        return None

    med_h = float(np.median(roi_hsv[:, :, 0][gm].astype(np.float64)))
    med_s = float(np.median(s[gm].astype(np.float64)))

    if not lab_models or not all(c in lab_models for c in ("red", "green", "blue")):
        return {
            "hsv_raw": hsv_raw,
            "p_lab": {"red": 0.0, "green": 0.0, "blue": 0.0},
            "p_hsv": _softmax_scores({c: hsv_raw[c] for c in ("red", "green", "blue")}),
            "gm": gm,
            "paint": gm,
            "ma_roi": 128.0,
            "mb_roi": 128.0,
            "chroma_mean": 0.0,
            "med_h": med_h,
            "med_s": med_s,
            "mean_b": float(np.mean(roi_bgr[:, :, 0][gm])),
            "mean_g": float(np.mean(roi_bgr[:, :, 1][gm])),
            "mean_r": float(np.mean(roi_bgr[:, :, 2][gm])),
        }

    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    la = lab[:, :, 1].astype(np.float64)
    lb = lab[:, :, 2].astype(np.float64)
    s_roi = roi_hsv[:, :, 1].astype(np.float64)
    paint = gm & water_ok
    if int(np.count_nonzero(paint)) > 12:
        st = max(36.0, float(np.percentile(s_roi[paint], 28)))
        paint_sel = paint & (s_roi >= st)
        if int(np.count_nonzero(paint_sel)) >= 8:
            paint = paint_sel
    if int(np.count_nonzero(paint)) < 6:
        paint = gm & water_ok

    ma_roi = float(np.mean(la[paint]))
    mb_roi = float(np.mean(lb[paint]))
    chroma_mean = float(
        np.mean(np.sqrt((la[paint].astype(np.float64) - 128.0) ** 2 + (lb[paint].astype(np.float64) - 128.0) ** 2))
    )
    x = np.array([ma_roi, mb_roi], dtype=np.float64)
    lab_logp = {c: lab_models[c].log_pdf(x) for c in ("red", "green", "blue")}
    p_lab = _softmax_scores({c: lab_logp[c] for c in ("red", "green", "blue")})
    p_hsv = _softmax_scores({c: hsv_raw[c] for c in ("red", "green", "blue")})

    mean_b = float(np.mean(roi_bgr[:, :, 0][paint]))
    mean_g = float(np.mean(roi_bgr[:, :, 1][paint]))
    mean_r = float(np.mean(roi_bgr[:, :, 2][paint]))

    return {
        "hsv_raw": hsv_raw,
        "p_lab": p_lab,
        "p_hsv": p_hsv,
        "gm": gm,
        "paint": paint,
        "ma_roi": ma_roi,
        "mb_roi": mb_roi,
        "chroma_mean": chroma_mean,
        "med_h": med_h,
        "med_s": med_s,
        "mean_b": mean_b,
        "mean_g": mean_g,
        "mean_r": mean_r,
    }


def _hsv_branch_raw_scores(
    roi_bgr: np.ndarray,
    roi_hsv: np.ndarray,
    ranges_map: dict[str, list],
    *,
    fg_mask: np.ndarray,
    fg_count: int,
    roi_area: float,
) -> dict[str, float]:
    h_fg = roi_hsv[:, :, 0][fg_mask] if fg_count > 30 else roi_hsv[:, :, 0].reshape(-1)
    s_fg = roi_hsv[:, :, 1][fg_mask] if fg_count > 30 else roi_hsv[:, :, 1].reshape(-1)
    v_fg = roi_hsv[:, :, 2][fg_mask] if fg_count > 30 else roi_hsv[:, :, 2].reshape(-1)
    w_fg = (s_fg.astype(np.float32) + 1.0) * (v_fg.astype(np.float32) + 1.0)
    b_fg = roi_bgr[:, :, 0][fg_mask] if fg_count > 30 else roi_bgr[:, :, 0].reshape(-1)
    g_fg = roi_bgr[:, :, 1][fg_mask] if fg_count > 30 else roi_bgr[:, :, 1].reshape(-1)
    r_fg = roi_bgr[:, :, 2][fg_mask] if fg_count > 30 else roi_bgr[:, :, 2].reshape(-1)
    denom = np.maximum(r_fg + g_fg + b_fg, 1.0).astype(np.float32)
    r_norm = r_fg.astype(np.float32) / denom
    g_norm = g_fg.astype(np.float32) / denom
    b_norm = b_fg.astype(np.float32) / denom

    channel_score_map = {
        "red": np.clip(r_norm - np.maximum(g_norm, b_norm) + 0.35, 0.0, 1.0),
        "green": np.clip(g_norm - np.maximum(r_norm, b_norm) + 0.35, 0.0, 1.0),
        "blue": np.clip(b_norm - np.maximum(r_norm, g_norm) + 0.35, 0.0, 1.0),
    }
    hue_centers = {"red": [0, 179], "green": [60, 86], "blue": [106]}
    raw: dict[str, float] = {}
    for color, ranges in ranges_map.items():
        if color not in ("red", "green", "blue"):
            continue
        mask = build_mask(roi_hsv, ranges)
        if fg_count > 30:
            ratio = float(np.count_nonzero((mask > 0) & fg_mask)) / float(fg_count)
        else:
            ratio = float(np.count_nonzero(mask)) / roi_area
        centers = hue_centers.get(color, [90])
        dists = []
        for c0 in centers:
            d = np.abs(h_fg.astype(np.int16) - int(c0))
            d = np.minimum(d, 180 - d)
            dists.append(d.astype(np.float32))
        dmin = np.min(np.stack(dists, axis=0), axis=0)
        hue_score = float(np.average(np.clip(1.0 - (dmin / 26.0), 0.0, 1.0), weights=w_fg))
        ch_vec = channel_score_map.get(color)
        ch_score = float(np.average(ch_vec, weights=w_fg)) if ch_vec is not None else 0.0
        raw[color] = 0.45 * ratio + 0.30 * hue_score + 0.25 * ch_score
    return raw


@dataclass
class ColorLogisticHead:
    """Sklearn multinomial logistic regression on `stack_lr_features` (optional high-accuracy head)."""

    model: object
    classes: tuple[str, ...] = ("red", "green", "blue")

    def predict_label(self, vec: np.ndarray) -> tuple[str, float]:
        proba = self.model.predict_proba(vec.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        return self.classes[i], float(proba[i])


def infer_sklearn_predict_proba_class_order(model: object) -> tuple[str, ...]:
    """Column order of ``predict_proba`` (must match multiclass clf.classes_)."""
    ns = getattr(model, "named_steps", None)
    hgb = None
    if isinstance(ns, dict):
        hgb = ns.get("hgb")
    if hgb is not None and hasattr(hgb, "classes_"):
        return tuple(str(c) for c in hgb.classes_)
    if hasattr(model, "classes_"):
        return tuple(str(c) for c in model.classes_)
    return ("red", "green", "blue")


@dataclass
class ColorTreeHead:
    """Sklearn HistGradientBoosting (or Pipeline) on `stack_tree_features`."""

    model: object
    """Class name for predict_proba column `i`; typically sklearn alphabetical order."""

    proba_classes: tuple[str, ...]

    def predict_label(self, vec: np.ndarray) -> tuple[str, float]:
        proba = self.model.predict_proba(vec.reshape(1, -1))[0]
        i = int(np.argmax(proba))
        return self.proba_classes[i], float(proba[i])

    def class_probabilities(self, vec: np.ndarray) -> dict[str, float]:
        proba = self.model.predict_proba(vec.reshape(1, -1))[0]
        return {self.proba_classes[i]: float(proba[i]) for i in range(len(self.proba_classes))}


def _apply_tree_geometric_fuse(fused: dict[str, float], tree_probs: dict[str, float], beta: float) -> None:
    """In-place: fused[c] *= tree[c]^beta after raising fused by (1-beta). Multiclass calibration blend."""
    b = float(np.clip(beta, 0.0, 1.0))
    if b <= 0.0:
        return
    for c in fused:
        fp = max(1e-15, float(fused[c]))
        tp = max(1e-15, float(tree_probs.get(c, 1e-9)))
        fused[c] = (fp ** (1.0 - b)) * (tp ** b)


def _normalize_fused_and_pick(
    fused: dict[str, float], fusion_temperature: float
) -> tuple[str, float]:
    """Optional temperature sharpening, renormalize, argmax."""
    ft = float(fusion_temperature)
    if ft > 0.0 and abs(ft - 1.0) > 1e-6:
        ft = float(np.clip(ft, 0.15, 12.0))
        for c in fused:
            fused[c] = max(1e-15, fused[c]) ** ft
    tot = sum(fused.values()) + 1e-12
    for c in fused:
        fused[c] /= tot
    best_color = max(fused, key=fused.get)
    return best_color, float(fused[best_color])


def _apply_rg_bgr_balance(
    fused: dict[str, float],
    ev: dict[str, np.ndarray | float | dict[str, float]],
    *,
    strength: float,
    med_h: float,
    hue_gate: bool = True,
) -> None:
    """Nudge red vs green from saturated paint-channel imbalance (helps R/G confusion on buoys)."""
    if strength <= 0.0:
        return
    mr = float(ev["mean_r"])
    mg = float(ev["mean_g"])
    mb = float(ev["mean_b"])
    denom = max(mr + mg + mb, 1.0)
    dom = (mr - mg) / denom
    s = float(np.clip(strength, 0.0, 0.88))
    if dom > 0.042:
        k = min(dom / 0.13, 1.0)
        if hue_gate and 48.0 < med_h < 102.0:
            k *= 0.42
        fused["red"] *= 1.0 + s * k
        fused["green"] *= max(0.35, 1.0 - 0.52 * s * k)
    elif dom < -0.042:
        k = min(-dom / 0.13, 1.0)
        if hue_gate and (med_h < 38.0 or med_h > 168.0):
            k *= 0.42
        fused["green"] *= 1.0 + s * k
        fused["red"] *= max(0.35, 1.0 - 0.52 * s * k)


def _apply_weak_blue_and_rgb(
    fused: dict[str, float],
    ev: dict[str, np.ndarray | float | dict[str, float]],
    *,
    med_h: float,
    med_s: float,
    mb_roi: float,
    blue_hue_lo: float,
    blue_hue_hi: float,
    blue_min_med_sat: float,
    bgr_centroids: dict[str, np.ndarray] | None,
    fusion_rgb_weight: float,
    fusion_rgb_sigma: float,
) -> None:
    """In-place: blue gates + optional BGR-distance blend."""
    weak_blue = (med_h < blue_hue_lo or med_h > blue_hue_hi) or (med_s < blue_min_med_sat)
    if weak_blue:
        fused["blue"] *= 0.06
    if mb_roi > 118.0:
        fused["blue"] *= 0.05
    if (
        bgr_centroids
        and len(bgr_centroids) >= 3
        and fusion_rgb_weight > 0
        and all(c in bgr_centroids for c in ("red", "green", "blue"))
    ):
        mbgr = np.array([ev["mean_b"], ev["mean_g"], ev["mean_r"]], dtype=np.float64)
        sig = float(max(fusion_rgb_sigma, 1e-3))
        p_rgb = _rgb_distance_softmax(mbgr, bgr_centroids, sigma=sig)
        wrgb = float(np.clip(fusion_rgb_weight, 0.0, 1.0))
        for c in fused:
            fused[c] = (1.0 - wrgb) * fused[c] + wrgb * p_rgb[c]


def classify_buoy_roi_hsv(
    roi_bgr: np.ndarray,
    roi_hsv: np.ndarray,
    ranges_map: dict[str, list],
    geom_mask: np.ndarray,
    *,
    suppress_water: bool = True,
    water_sat_max: int = 48,
    water_val_min: int = 30,
    water_desat_blue_max: int = 38,
    lab_models: dict[str, LabColorModel] | None = None,
    fusion_lab_weight: float = 0.55,
    fusion_hsv_weight: float = 0.45,
    blue_hue_lo: float = 94.0,
    blue_hue_hi: float = 132.0,
    blue_min_med_sat: float = 44.0,
    lr_head: ColorLogisticHead | None = None,
    lr_min_confidence: float = 0.52,
    tree_head: ColorTreeHead | None = None,
    tree_geometric_beta: float = 0.0,
    bgr_centroids: dict[str, np.ndarray] | None = None,
    fusion_rgb_weight: float = 0.62,
    fusion_rgb_sigma: float = 40.0,
    fusion_temperature: float = 1.1,
    rg_bgr_balance_strength: float = 0.88,
    rg_balance_hue_gate: bool = True,
) -> tuple[str, float]:
    """
    Optional sklearn logistic head (fit via fit_color_lr_head.py) overrides fusion when provided.
    Otherwise: LAB+HSV fusion with pooled LAB covariances and blue-suppression gates.
    """
    ev = compute_color_evidence_bundle(
        roi_bgr,
        roi_hsv,
        ranges_map,
        geom_mask,
        suppress_water=suppress_water,
        water_sat_max=water_sat_max,
        water_val_min=water_val_min,
        water_desat_blue_max=water_desat_blue_max,
        lab_models=lab_models,
    )
    if ev is None:
        return "unknown", 0.0

    tree_geom = tree_head is not None and float(tree_geometric_beta) > 0.0

    if (
        lr_head is not None
        and lab_models
        and all(c in lab_models for c in ("red", "green", "blue"))
        and not tree_geom
    ):
        vec = stack_lr_features(ev)
        lr_name, lr_conf = lr_head.predict_label(vec)
        if lr_min_confidence <= 0 or lr_conf >= lr_min_confidence:
            return lr_name, lr_conf

    hsv_raw = ev["hsv_raw"]
    med_h = float(ev["med_h"])
    med_s = float(ev["med_s"])
    mb_roi = float(ev["mb_roi"])

    if not lab_models or not all(c in lab_models for c in ("red", "green", "blue")):
        p_hsv_only = _softmax_scores({c: hsv_raw[c] for c in ("red", "green", "blue")})
        fused = {c: p_hsv_only[c] for c in ("red", "green", "blue")}
        _apply_weak_blue_and_rgb(
            fused,
            ev,
            med_h=med_h,
            med_s=med_s,
            mb_roi=mb_roi,
            blue_hue_lo=blue_hue_lo,
            blue_hue_hi=blue_hue_hi,
            blue_min_med_sat=blue_min_med_sat,
            bgr_centroids=bgr_centroids,
            fusion_rgb_weight=fusion_rgb_weight,
            fusion_rgb_sigma=fusion_rgb_sigma,
        )
        _apply_rg_bgr_balance(
            fused,
            ev,
            strength=rg_bgr_balance_strength,
            med_h=med_h,
            hue_gate=rg_balance_hue_gate,
        )
        if tree_geom and tree_head is not None:
            tpm = tree_head.class_probabilities(stack_tree_features(roi_hsv, ev))
            _apply_tree_geometric_fuse(fused, tpm, tree_geometric_beta)
        return _normalize_fused_and_pick(fused, fusion_temperature)

    p_lab = ev["p_lab"]
    p_hsv = ev["p_hsv"]
    chroma_mean = float(ev["chroma_mean"])
    ma_roi = float(ev["ma_roi"])

    fw = fusion_lab_weight + fusion_hsv_weight
    if fw <= 0:
        fw = 1.0
    base_wl = fusion_lab_weight / fw
    base_wh = fusion_hsv_weight / fw
    t = float(np.clip((chroma_mean - 14.0) / 38.0, 0.0, 1.0))
    wl = (1.0 - t) * 0.36 + t * base_wl
    wh = (1.0 - t) * 0.64 + t * base_wh
    wns = wl + wh
    wl, wh = wl / wns, wh / wns

    fused = {c: wl * p_lab[c] + wh * p_hsv[c] for c in ("red", "green", "blue")}

    _apply_weak_blue_and_rgb(
        fused,
        ev,
        med_h=med_h,
        med_s=med_s,
        mb_roi=mb_roi,
        blue_hue_lo=blue_hue_lo,
        blue_hue_hi=blue_hue_hi,
        blue_min_med_sat=blue_min_med_sat,
        bgr_centroids=bgr_centroids,
        fusion_rgb_weight=fusion_rgb_weight,
        fusion_rgb_sigma=fusion_rgb_sigma,
    )

    _apply_rg_bgr_balance(
        fused,
        ev,
        strength=rg_bgr_balance_strength,
        med_h=med_h,
        hue_gate=rg_balance_hue_gate,
    )
    if tree_geom and tree_head is not None:
        tpm = tree_head.class_probabilities(stack_tree_features(roi_hsv, ev))
        _apply_tree_geometric_fuse(fused, tpm, tree_geometric_beta)

    return _normalize_fused_and_pick(fused, fusion_temperature)


def build_mask(
    hsv: np.ndarray,
    ranges: list[HSVRange] | list[tuple[tuple[int, int, int], tuple[int, int, int]]],
) -> np.ndarray:
    acc = None
    for r in ranges:
        if isinstance(r, HSVRange):
            low, high = r.low, r.high
        else:
            low, high = r
        m = cv2.inRange(
            hsv,
            np.array(low, dtype=np.uint8),
            np.array(high, dtype=np.uint8),
        )
        acc = m if acc is None else cv2.bitwise_or(acc, m)
    if acc is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    return acc


def _ranges_to_tuples(ranges_map: dict[str, list[HSVRange]]) -> dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    out: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {}
    for color, ranges in ranges_map.items():
        out[color] = [(r.low, r.high) for r in ranges]
    return out


def _print_tuple_ranges(title: str, ranges_map: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]]) -> None:
    print(title)
    for color in ("red", "green", "blue"):
        for i, (low, high) in enumerate(ranges_map.get(color, [])):
            print(f"  {color}[{i}] low={low} high={high}")


def load_color_ranges(
    classes_dir: str = "captures/classes",
    hue_margin: int = 12,
    sat_floor: int = 50,
    val_floor: int = 45,
) -> dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    derived = derive_class_hsv_ranges(classes_dir, hue_margin, sat_floor, val_floor)
    if derived:
        tuple_ranges = _ranges_to_tuples(derived)
        _print_tuple_ranges("Derived HSV ranges:", tuple_ranges)
        return tuple_ranges

    print(f"Classes directory missing or empty: {classes_dir}")
    print("Using fallback HSV ranges:")
    _print_tuple_ranges("Fallback HSV ranges:", FALLBACK_COLOR_RANGES)
    return {color: list(ranges) for color, ranges in FALLBACK_COLOR_RANGES.items()}

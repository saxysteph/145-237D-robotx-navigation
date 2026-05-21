#!/usr/bin/env python3
"""
Draw YOLO-format label boxes (normalized cx,cy,w,h) onto uav_*.jpg for batch review.

Writes parallel tree under each clip: --out-subdir/default annotated_gt/ mirroring image names.

With --hsv-mean-label, prints mean H,S,V (OpenCV 0–179/0–255) for a center crop of each box
(useful to sanity-check projected ROIs without running the YOLO/HSV trainer).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


# Canonical class names + BGR colors: 0=red, 1=green, 2=blue, 3=unknown
CLASS_NAME = {
    0: "red",
    1: "green",
    2: "blue",
    3: "unknown",
}
CLASS_COLOR_BGR = {
    0: (0, 0, 255),      # red
    1: (0, 255, 0),      # green
    2: (255, 120, 0),    # blue-ish (BGR)
    3: (190, 190, 190),  # gray
}


def yolo_norm_to_xyxy(
    cx: float, cy: float, w: float, h: float, iw: int, ih: int
) -> tuple[int, int, int, int]:
    cxp, cyp, wp, hp = cx * iw, cy * ih, w * iw, h * ih
    x1 = max(0, int(round(cxp - wp / 2)))
    y1 = max(0, int(round(cyp - hp / 2)))
    x2 = min(iw - 1, int(round(cxp + wp / 2)))
    y2 = min(ih - 1, int(round(cyp + hp / 2)))
    return x1, y1, x2, y2


def _inner_roi_xyxy(x1: int, y1: int, x2: int, y2: int, frac: float = 0.35) -> tuple[int, int, int, int]:
    """Center ``frac`` of width/height, clamped to valid ints."""
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    dx = max(1, int(0.5 * w * (1.0 - frac)))
    dy = max(1, int(0.5 * h * (1.0 - frac)))
    nx1 = min(x2 - 1, x1 + dx)
    ny1 = min(y2 - 1, y1 + dy)
    nx2 = max(nx1 + 1, x2 - dx)
    ny2 = max(ny1 + 1, y2 - dy)
    return nx1, ny1, nx2, ny2


def _mean_hsv_center_crop(bgr, x1: int, y1: int, x2: int, y2: int) -> tuple[float, float, float] | None:
    ix1, iy1, ix2, iy2 = _inner_roi_xyxy(x1, y1, x2, y2)
    crop = bgr[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    flat = hsv.reshape(-1, 3).astype("float64")
    return float(flat[:, 0].mean()), float(flat[:, 1].mean()), float(flat[:, 2].mean())


def draw_labels_on_image(
    img_path: Path,
    label_path: Path,
    out_path: Path,
    *,
    hsv_mean_label: bool = False,
) -> bool:
    im = cv2.imread(str(img_path))
    if im is None:
        return False
    h, w = im.shape[:2]
    if not label_path.is_file():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), im)
        return True
    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for li, raw in enumerate(lines):
        parts = raw.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        x1, y1, x2, y2 = yolo_norm_to_xyxy(cx, cy, bw, bh, w, h)
        col = CLASS_COLOR_BGR.get(cls, CLASS_COLOR_BGR[3])
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
        cname = CLASS_NAME.get(cls, f"cls{cls}")
        lbl = cname
        if hsv_mean_label:
            mh = _mean_hsv_center_crop(im, x1, y1, x2, y2)
            if mh is not None:
                lbl = f"{lbl} H={mh[0]:.0f} S={mh[1]:.0f} V={mh[2]:.0f}"
        cv2.putText(
            im,
            lbl,
            (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures-root", type=Path, required=True)
    p.add_argument("--labels-subdir", type=str, default="labels_proj")
    p.add_argument("--out-subdir", type=str, default="annotated_gt")
    p.add_argument(
        "--max-per-clip",
        type=int,
        default=0,
        help="Cap images per clip (0 = all uav_*.jpg that have labels).",
    )
    p.add_argument(
        "--hsv-mean-label",
        action="store_true",
        help="Overlay mean H,S,V from center ROI for each projected box.",
    )
    args = p.parse_args()
    if not args.captures_root.is_dir():
        print(f"Not a directory: {args.captures_root}", file=sys.stderr)
        return 1
    n_img = 0
    for clip in sorted(d for d in args.captures_root.iterdir() if d.is_dir() and not d.name.startswith("_")):
        labels_dir = clip / args.labels_subdir
        if not labels_dir.is_dir():
            continue
        out_base = clip / args.out_subdir
        images = sorted(clip.glob("uav_*.jpg"))
        if args.max_per_clip and args.max_per_clip > 0:
            images = images[: args.max_per_clip]
        for img_path in images:
            lab = labels_dir / f"{img_path.stem}.txt"
            out_path = out_base / img_path.name
            if draw_labels_on_image(
                img_path, lab, out_path, hsv_mean_label=args.hsv_mean_label
            ):
                n_img += 1
        if images:
            print(f"{clip.name}: {len(images)} frames → {out_base}")
    print(f"Total annotated images written: {n_img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

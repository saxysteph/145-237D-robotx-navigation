#!/usr/bin/env python3
"""Create reproducible 80/20 train/val split for honest validation."""

from __future__ import annotations

import os
import random
import shutil


def main() -> int:
    root = os.path.dirname(__file__)
    src_images = os.path.join(root, "path2_dataset", "images")
    src_labels = os.path.join(root, "path2_dataset", "labels")

    dst_root = os.path.join(root, "dataset")
    train_img = os.path.join(dst_root, "images", "train")
    val_img = os.path.join(dst_root, "images", "val")
    train_lbl = os.path.join(dst_root, "labels", "train")
    val_lbl = os.path.join(dst_root, "labels", "val")

    for p in (train_img, val_img, train_lbl, val_lbl):
        os.makedirs(p, exist_ok=True)

    all_images = sorted(
        [n for n in os.listdir(src_images) if n.lower().endswith(".jpg") and os.path.isfile(os.path.join(src_images, n))]
    )
    random.seed(42)
    shuffled = all_images[:]
    random.shuffle(shuffled)

    split_idx = int(0.8 * len(shuffled))
    train_names = set(shuffled[:split_idx])
    val_names = set(shuffled[split_idx:])

    for name in shuffled:
        stem = os.path.splitext(name)[0]
        src_i = os.path.join(src_images, name)
        src_l = os.path.join(src_labels, stem + ".txt")
        if name in train_names:
            shutil.copy2(src_i, os.path.join(train_img, name))
            if os.path.exists(src_l):
                shutil.copy2(src_l, os.path.join(train_lbl, stem + ".txt"))
            else:
                open(os.path.join(train_lbl, stem + ".txt"), "w", encoding="utf-8").close()
        else:
            shutil.copy2(src_i, os.path.join(val_img, name))
            if os.path.exists(src_l):
                shutil.copy2(src_l, os.path.join(val_lbl, stem + ".txt"))
            else:
                open(os.path.join(val_lbl, stem + ".txt"), "w", encoding="utf-8").close()

    yaml_path = os.path.join(dst_root, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: ./dataset\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 3\n")
        f.write("names: ['red', 'green', 'blue']\n")

    print(f"Train set: {len(train_names)} images")
    print(f"Val set:   {len(val_names)} images (NEVER seen during training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

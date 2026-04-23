#!/usr/bin/env python3
"""Path 2B: fine-tune YOLOv11n on auto-labeled dataset, then infer."""

from __future__ import annotations

import csv
import os
from statistics import mean

import cv2
from ultralytics import YOLO


def list_capture_images(captures_dir: str) -> list[str]:
    out: list[str] = []
    for name in sorted(os.listdir(captures_dir)):
        path = os.path.join(captures_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".jpg"):
            out.append(path)
    return out


def extract_map50(results_csv: str) -> float:
    if not os.path.isfile(results_csv):
        return 0.0
    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            last = row
    if not last:
        return 0.0
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/mAP50(M)"):
        if key in last and last[key]:
            try:
                return float(last[key])
            except ValueError:
                continue
    return 0.0


def main() -> int:
    root = os.path.dirname(__file__)
    repo_root = os.path.join(root, "..")
    captures_dir = os.path.join(repo_root, "captures")

    dataset_dir = os.path.join(root, "path2_dataset")
    dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")
    if not os.path.isfile(dataset_yaml):
        print(f"Dataset YAML not found: {dataset_yaml}")
        return 1
    dataset_yaml_abs = os.path.join(dataset_dir, "dataset_abs.yaml")
    with open(dataset_yaml_abs, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_dir.replace('\\', '/')}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("nc: 3\n")
        f.write("names: ['red', 'green', 'blue']\n")

    train_project = os.path.join(root, "path2_training")
    train_name = "balloon_finetune"
    os.makedirs(train_project, exist_ok=True)

    model = YOLO("yolo11n.pt")
    model.train(
        data=dataset_yaml_abs,
        epochs=30,
        imgsz=640,
        batch=8,
        project=train_project,
        name=train_name,
        patience=10,
        exist_ok=True,
    )

    train_dir = os.path.join(train_project, train_name)
    best_weights = os.path.join(train_dir, "weights", "best.pt")
    if not os.path.isfile(best_weights):
        print(f"best.pt not found after training: {best_weights}")
        return 1
    infer_model = YOLO(best_weights)

    out_dir = os.path.join(root, "path2_results")
    ann_dir = os.path.join(out_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "detections.csv")

    images = list_capture_images(captures_dir)
    total_det = 0
    all_conf: list[float] = []
    zero_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "cx", "cy", "x", "y", "w", "h", "confidence"])
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None:
                continue
            result = infer_model(img, verbose=False)[0]
            boxes = result.boxes
            img_count = 0
            if boxes is not None:
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    x = int(max(0, round(x1)))
                    y = int(max(0, round(y1)))
                    w = int(max(0, round(x2 - x1)))
                    h = int(max(0, round(y2 - y1)))
                    cx = x + (w / 2.0)
                    cy = y + (h / 2.0)
                    writer.writerow(
                        [
                            os.path.basename(img_path),
                            f"{cx:.2f}",
                            f"{cy:.2f}",
                            x,
                            y,
                            w,
                            h,
                            f"{conf:.4f}",
                        ]
                    )
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.putText(
                        img,
                        f"conf={conf:.2f}",
                        (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    total_det += 1
                    all_conf.append(conf)
                    img_count += 1
            if img_count == 0:
                zero_count += 1
            cv2.imwrite(os.path.join(ann_dir, os.path.basename(img_path)), img)

    map50 = extract_map50(os.path.join(train_dir, "results.csv"))
    avg_conf = mean(all_conf) if all_conf else 0.0

    summary_lines = [
        "=== PATH 2 SUMMARY ===",
        "Training: 30 epochs on 110 auto-labeled images",
        f"Best mAP50: {map50:.3f} (from training run)",
        f"Inference: Images: {len(images)} | Detections: {total_det} | Avg conf: {avg_conf:.3f}",
        f"Images with 0 detections: {zero_count}",
    ]
    for line in summary_lines:
        print(line)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import csv
from pathlib import Path

import cv2
import numpy as np


def ensure_results_dirs(results_dir):
    subdirs = {
        "t0": results_dir / "t0",
        "tb": results_dir / "tb",
        "enhanced": results_dir / "enhanced",
        "comparisons": results_dir / "comparisons",
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return subdirs


def find_paired_images(input_dir, gt_dir):
    pairs = []

    for input_path in sorted(input_dir.glob("*")):
        if not input_path.is_file():
            continue

        gt_path = gt_dir / input_path.name
        if gt_path.exists() and gt_path.is_file():
            pairs.append((input_path, gt_path))

    return pairs


def read_image_rgb(image_path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"图像读取失败: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb.astype(np.float32) / 255.0


def _float_to_uint8(image):
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def save_gray_image(output_path, gray_image):
    gray_uint8 = _float_to_uint8(gray_image)
    cv2.imwrite(str(output_path), gray_uint8)


def save_color_image(output_path, rgb_image):
    rgb_uint8 = _float_to_uint8(rgb_image)
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), bgr_uint8)


def build_comparison_image(input_image, enhanced_image, gt_image):
    # 按要求仅保存 Input / Enhanced / GT 三图定性对比
    return np.concatenate([input_image, enhanced_image, gt_image], axis=1)


def save_csv(csv_path, rows):
    metric_names = [key for key in rows[0]]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metric_names)
        writer.writeheader()
        writer.writerows(rows)

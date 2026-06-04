from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics import compute_average_metrics, compute_metrics, metrics_rows_append
from pipeline import enhance_low_light
from utils import (
    build_comparison_image,
    build_heatmap_rgb,
    compute_delta_ab_map,
    find_paired_images,
    read_image_rgb,
    save_color_image,
    save_csv,
)

INPUT_DIR = Path("samples/Train/Input")
GT_DIR = Path("samples/Train/GT")
FIGURES_DIR = Path("report_workspace/assets/figures")
TABLES_DIR = Path("report_workspace/assets/tables")

TARGET_FILENAMES = {"00051.png", "00091.png"}
ALPHA_GRID_FILENAME = "00051.png"
ALPHA_MAX_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_GUIDED_PARAMS = {
    "method": "guided",
    "bilateral_diameter": 15,
    "bilateral_sigma_color": 0.1,
    "bilateral_sigma_space": 15,
    "guided_radius": 8,
    "guided_eps": 1e-3,
    "tmin": 0.1,
    "gamma": 0.8,
    "alpha_max": 0.5,
}


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def save_selected_figures(filename, input_image, gt_image, outputs):
    stem = Path(filename).stem

    save_color_image(FIGURES_DIR / f"ccb_input_{stem}.png", input_image)
    save_color_image(FIGURES_DIR / f"ccb_enhanced_{stem}.png", outputs["enhanced"])
    save_color_image(FIGURES_DIR / f"ccb_corrected_{stem}.png", outputs["color_corrected_b"])
    save_color_image(FIGURES_DIR / f"ccb_gt_{stem}.png", gt_image)

    enhanced_delta_ab = compute_delta_ab_map(outputs["enhanced"], gt_image)
    corrected_delta_ab = compute_delta_ab_map(outputs["color_corrected_b"], gt_image)
    save_color_image(
        FIGURES_DIR / f"ccb_deltaab_enhanced_{stem}.png",
        build_heatmap_rgb(enhanced_delta_ab),
    )
    save_color_image(
        FIGURES_DIR / f"ccb_deltaab_corrected_{stem}.png",
        build_heatmap_rgb(corrected_delta_ab),
    )

    enhanced_comparison = build_comparison_image(
        input_image=input_image,
        enhanced_image=outputs["enhanced"],
        gt_image=gt_image,
    )
    corrected_comparison = build_comparison_image(
        input_image=input_image,
        enhanced_image=outputs["color_corrected_b"],
        gt_image=gt_image,
    )
    save_color_image(FIGURES_DIR / f"ccb_compare_enhanced_{stem}.png", enhanced_comparison)
    save_color_image(FIGURES_DIR / f"ccb_compare_corrected_{stem}.png", corrected_comparison)


def run_alpha_experiment(paired_images):
    summary_rows = []
    alpha_grid_images = []

    for alpha_max in ALPHA_MAX_VALUES:
        config_rows = []
        for input_path, gt_path in paired_images:
            input_image = read_image_rgb(input_path)
            gt_image = read_image_rgb(gt_path)
            params = {**DEFAULT_GUIDED_PARAMS, "alpha_max": alpha_max}
            outputs = enhance_low_light(
                input_image=input_image,
                **params,
            )
            corrected_b_metrics = compute_metrics(outputs["color_corrected_b"], gt_image)
            metrics_rows_append(config_rows, input_path.name, corrected_b_metrics)

            if input_path.name == ALPHA_GRID_FILENAME:
                alpha_grid_images.append((alpha_max, outputs["color_corrected_b"]))

        average_metrics = compute_average_metrics(config_rows)
        summary_rows.append(
            {
                "alpha_max": alpha_max,
                **average_metrics,
            }
        )

    alpha_df = pd.DataFrame(summary_rows).sort_values("alpha_max")
    alpha_df.to_csv(TABLES_DIR / "color_correction_b_alpha_max.csv", index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metric_defs = [
        ("psnr", "PSNR"),
        ("ssim", "SSIM"),
        ("delta_ab", r"$\Delta ab$"),
    ]
    for ax, (metric_name, ylabel) in zip(axes, metric_defs):
        ax.plot(alpha_df["alpha_max"], alpha_df[metric_name], marker="o", linewidth=1.8)
        ax.set_xlabel(r"$\alpha_{\max}$")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ccb_alpha_max_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    alpha_grid_images.sort(key=lambda item: item[0])
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax, (alpha_max, image) in zip(axes.flat, alpha_grid_images):
        ax.imshow(image)
        ax.set_title(rf"$\alpha_{{\max}}={alpha_max:.1f}$")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ccb_alpha_max_grid_00051.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()
    paired_images = find_paired_images(INPUT_DIR, GT_DIR)
    if not paired_images:
        raise RuntimeError("未找到可匹配的 Input / GT 图像对。")

    enhanced_rows = []
    corrected_rows = []
    selected_rows = []

    for input_path, gt_path in paired_images:
        input_image = read_image_rgb(input_path)
        gt_image = read_image_rgb(gt_path)

        outputs = enhance_low_light(
            input_image=input_image,
            **DEFAULT_GUIDED_PARAMS,
        )

        enhanced_metrics = compute_metrics(outputs["enhanced"], gt_image)
        corrected_metrics = compute_metrics(outputs["color_corrected_b"], gt_image)

        metrics_rows_append(enhanced_rows, input_path.name, enhanced_metrics)
        metrics_rows_append(corrected_rows, input_path.name, corrected_metrics)

        if input_path.name in TARGET_FILENAMES:
            selected_rows.append(
                {
                    "filename": input_path.name,
                    "variant": "enhanced",
                    **{k: f"{v:.4f}" for k, v in enhanced_metrics.items()},
                }
            )
            selected_rows.append(
                {
                    "filename": input_path.name,
                    "variant": "color_corrected_b",
                    **{k: f"{v:.4f}" for k, v in corrected_metrics.items()},
                }
            )
            save_selected_figures(input_path.name, input_image, gt_image, outputs)

    enhanced_average = compute_average_metrics(enhanced_rows)
    corrected_average = compute_average_metrics(corrected_rows)
    metrics_rows_append(enhanced_rows, "average", enhanced_average)
    metrics_rows_append(corrected_rows, "average", corrected_average)

    save_csv(TABLES_DIR / "color_correction_b_guided_enhanced_metrics.csv", enhanced_rows)
    save_csv(TABLES_DIR / "color_correction_b_guided_corrected_metrics.csv", corrected_rows)
    save_csv(TABLES_DIR / "color_correction_b_selected_metrics.csv", selected_rows)

    average_rows = [
        {"variant": "guided_enhanced", **{k: f"{v:.4f}" for k, v in enhanced_average.items()}},
        {
            "variant": "guided_color_corrected_b",
            **{k: f"{v:.4f}" for k, v in corrected_average.items()},
        },
    ]
    save_csv(TABLES_DIR / "color_correction_b_average_metrics.csv", average_rows)

    run_alpha_experiment(paired_images)
    print("color correction B analysis finished")


if __name__ == "__main__":
    main()

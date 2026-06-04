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
    find_paired_images,
    read_image_rgb,
    save_color_image,
    save_csv,
    save_gray_image,
)

INPUT_DIR = Path("samples/Train/Input")
GT_DIR = Path("samples/Train/GT")
ASSETS_DIR = Path("report_workspace/assets")
EXPERIMENTS_DIR = ASSETS_DIR / "experiments"
FIGURES_DIR = ASSETS_DIR / "figures"
TABLES_DIR = ASSETS_DIR / "tables"

DEFAULTS = {
    "bilateral_diameter": 15,
    "bilateral_sigma_color": 0.1,
    "bilateral_sigma_space": 15,
    "guided_radius": 8,
    "guided_eps": 1e-3,
    "tmin": 0.1,
    "gamma": 0.8,
}

BILATERAL_DIAMETERS = [3, 5, 9, 15, 25, 35]
BILATERAL_SIGMA_COLORS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.40]
BILATERAL_SIGMA_SPACES = [3, 7, 15, 30, 60, 90]
GUIDED_RADII = [2, 4, 8, 16, 24, 32]
GUIDED_EPS_VALUES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
TMIN_VALUES = [0.02, 0.05, 0.08, 0.10, 0.15, 0.25]
GUIDED_GAMMAS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

CONFIGS = []

for diameter in BILATERAL_DIAMETERS:
    CONFIGS.append(
        {
            "name": f"bilateral_d{diameter}",
            "method": "bilateral",
            "params": {
                **DEFAULTS,
                "bilateral_diameter": diameter,
            },
        }
    )

for sigma_color in BILATERAL_SIGMA_COLORS:
    config_name = f"bilateral_sc{int(round(sigma_color * 100)):03d}"
    CONFIGS.append(
        {
            "name": config_name,
            "method": "bilateral",
            "params": {
                **DEFAULTS,
                "bilateral_sigma_color": sigma_color,
            },
        }
    )

for sigma_space in BILATERAL_SIGMA_SPACES:
    CONFIGS.append(
        {
            "name": f"bilateral_ss{sigma_space}",
            "method": "bilateral",
            "params": {
                **DEFAULTS,
                "bilateral_sigma_space": sigma_space,
            },
        }
    )

gamma_configs = []
for gamma in GUIDED_GAMMAS:
    gamma_name = f"guided_gamma{int(round(gamma * 10)):02d}"
    gamma_configs.append(
        {
            "name": gamma_name,
            "method": "guided",
            "params": {
                **DEFAULTS,
                "gamma": gamma,
            },
        }
    )

tmin_configs = []
for tmin in TMIN_VALUES:
    tmin_name = f"guided_tmin{int(round(tmin * 100)):03d}"
    tmin_configs.append(
        {
            "name": tmin_name,
            "method": "guided",
            "params": {
                **DEFAULTS,
                "tmin": tmin,
            },
        }
    )

guided_configs = []
for radius in GUIDED_RADII:
    guided_configs.append(
        {
            "name": f"guided_r{radius}",
            "method": "guided",
            "params": {
                **DEFAULTS,
                "guided_radius": radius,
            },
        }
    )

for eps in GUIDED_EPS_VALUES:
    exp = f"{eps:.0e}".replace("-0", "-")
    guided_configs.append(
        {
            "name": f"guided_eps{exp}",
            "method": "guided",
            "params": {
                **DEFAULTS,
                "guided_eps": eps,
            },
        }
    )

CONFIGS.extend(guided_configs)
CONFIGS.extend(tmin_configs)
CONFIGS.extend(gamma_configs)

REPRESENTATIVE_FILES = ["00017.png", "00051.png", "00091.png"]
BILATERAL_VISUAL_CONFIGS = [
    ("bilateral_d3", "00017.png"),
    ("bilateral_d35", "00017.png"),
    ("bilateral_sc001", "00017.png"),
    ("bilateral_sc040", "00017.png"),
    ("bilateral_ss3", "00017.png"),
    ("bilateral_ss90", "00017.png"),
]
GUIDED_VISUAL_CONFIGS = [
    ("guided_r2", "00017.png"),
    ("guided_r32", "00017.png"),
    ("guided_eps1e-6", "00017.png"),
    ("guided_eps1e-1", "00017.png"),
]


def ensure_report_dirs():
    for directory in [ASSETS_DIR, EXPERIMENTS_DIR, FIGURES_DIR, TABLES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _save_config_outputs(config_name, filename, outputs, input_image, gt_image):
    config_dir = EXPERIMENTS_DIR / config_name
    subdirs = {
        "root": config_dir,
        "t0": config_dir / "t0",
        "smooth": config_dir / "smooth",
        "enhanced": config_dir / "enhanced",
        "comparisons": config_dir / "comparisons",
    }

    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    save_gray_image(subdirs["t0"] / filename, outputs["t0"])
    save_gray_image(subdirs["smooth"] / filename, outputs["smooth"])
    save_color_image(subdirs["enhanced"] / filename, outputs["enhanced"])
    comparison = build_comparison_image(
        input_image=input_image,
        enhanced_image=outputs["enhanced"],
        gt_image=gt_image,
    )
    save_color_image(subdirs["comparisons"] / filename, comparison)


def run_single_config(config, paired_images):
    metrics_rows = []

    for input_path, gt_path in paired_images:
        filename = input_path.name
        input_image = read_image_rgb(input_path)
        gt_image = read_image_rgb(gt_path)

        outputs = enhance_low_light(
            input_image=input_image,
            method=config["method"],
            bilateral_diameter=config["params"]["bilateral_diameter"],
            bilateral_sigma_color=config["params"]["bilateral_sigma_color"],
            bilateral_sigma_space=config["params"]["bilateral_sigma_space"],
            guided_radius=config["params"]["guided_radius"],
            guided_eps=config["params"]["guided_eps"],
            tmin=config["params"]["tmin"],
            gamma=config["params"]["gamma"],
        )

        metrics = compute_metrics(outputs["enhanced"], gt_image)
        metrics_rows_append(metrics_rows, filename, metrics)
        _save_config_outputs(config["name"], filename, outputs, input_image, gt_image)

    average_metrics = compute_average_metrics(metrics_rows)
    metrics_rows_append(metrics_rows, "average", average_metrics)
    save_csv(EXPERIMENTS_DIR / config["name"] / "metrics.csv", metrics_rows)

    return {
        "config": config["name"],
        "method": config["method"],
        **config["params"],
        **average_metrics,
    }


def load_metrics_csv(csv_path):
    data = pd.read_csv(csv_path)
    numeric_cols = [col for col in data.columns if col != "filename"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def write_main_comparison_tables():
    bilateral = load_metrics_csv(PROJECT_ROOT / "results/bilateral/metrics.csv")
    guided = load_metrics_csv(PROJECT_ROOT / "results/guided/metrics.csv")

    bilateral_avg = bilateral[bilateral["filename"] == "average"].iloc[0].to_dict()
    guided_avg = guided[guided["filename"] == "average"].iloc[0].to_dict()
    comparison_rows = [
        {
            "method": "bilateral",
            "psnr": bilateral_avg["psnr"],
            "ssim": bilateral_avg["ssim"],
            "mae": bilateral_avg["mae"],
            "mse": bilateral_avg["mse"],
            "lpips": bilateral_avg["lpips"],
        },
        {
            "method": "guided",
            "psnr": guided_avg["psnr"],
            "ssim": guided_avg["ssim"],
            "mae": guided_avg["mae"],
            "mse": guided_avg["mse"],
            "lpips": guided_avg["lpips"],
        },
    ]
    save_csv(TABLES_DIR / "main_method_comparison.csv", comparison_rows)

    merged = bilateral.merge(guided, on="filename", suffixes=("_bilateral", "_guided"))
    merged = merged[merged["filename"] != "average"].copy()
    merged["psnr_gain"] = merged["psnr_guided"] - merged["psnr_bilateral"]
    merged["ssim_gain"] = merged["ssim_guided"] - merged["ssim_bilateral"]
    merged["lpips_gain"] = merged["lpips_bilateral"] - merged["lpips_guided"]
    merged = merged.sort_values(by="psnr_gain", ascending=False)
    merged.to_csv(TABLES_DIR / "per_image_method_gain.csv", index=False, encoding="utf-8")


def plot_parameter_curve(df, x_col, title, output_name):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    plots = [
        ("psnr", "PSNR", False),
        ("ssim", "SSIM", False),
        ("lpips", "LPIPS", True),
    ]

    for ax, (metric_col, ylabel, invert) in zip(axes, plots):
        ax.plot(df[x_col], df[metric_col], marker="o", linewidth=1.8, color="#1f77b4")
        ax.set_xlabel(x_col)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        if invert:
            best_idx = df[metric_col].idxmin()
        else:
            best_idx = df[metric_col].idxmax()
        best_row = df.loc[best_idx]
        ax.scatter([best_row[x_col]], [best_row[metric_col]], color="#d62728", zorder=3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / output_name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_main_method_bar():
    data = pd.read_csv(TABLES_DIR / "main_method_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    plot_defs = [
        ("psnr", "PSNR", False),
        ("ssim", "SSIM", False),
        ("lpips", "LPIPS", True),
    ]
    colors = ["#4e79a7", "#f28e2b"]

    for ax, (metric_col, ylabel, invert) in zip(axes, plot_defs):
        ax.bar(data["method"], data[metric_col], color=colors, width=0.55)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        if invert:
            ax.set_ylim(0, max(data[metric_col]) * 1.2)
        else:
            ax.set_ylim(min(data[metric_col]) * 0.95, max(data[metric_col]) * 1.05)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "main_method_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_parameter_tables(summary_df):
    bilateral_diameter_df = summary_df[
        summary_df["config"].str.startswith("bilateral_d")
    ].copy()
    bilateral_sigma_color_df = summary_df[
        summary_df["config"].str.startswith("bilateral_sc")
    ].copy()
    bilateral_sigma_space_df = summary_df[
        summary_df["config"].str.startswith("bilateral_ss")
    ].copy()
    guided_radius_df = summary_df[summary_df["config"].str.startswith("guided_r")].copy()
    guided_eps_df = summary_df[summary_df["config"].str.startswith("guided_eps")].copy()
    guided_tmin_df = summary_df[summary_df["config"].str.startswith("guided_tmin")].copy()
    guided_gamma_df = summary_df[summary_df["config"].str.startswith("guided_gamma")].copy()

    bilateral_diameter_df = bilateral_diameter_df.sort_values("bilateral_diameter")
    bilateral_sigma_color_df = bilateral_sigma_color_df.sort_values(
        "bilateral_sigma_color"
    )
    bilateral_sigma_space_df = bilateral_sigma_space_df.sort_values(
        "bilateral_sigma_space"
    )
    guided_radius_df = guided_radius_df.sort_values("guided_radius")
    guided_eps_df = guided_eps_df.sort_values("guided_eps")
    guided_tmin_df = guided_tmin_df.sort_values("tmin")
    guided_gamma_df = guided_gamma_df.sort_values("gamma")

    bilateral_diameter_df.to_csv(
        TABLES_DIR / "bilateral_diameter.csv", index=False, encoding="utf-8"
    )
    bilateral_sigma_color_df.to_csv(
        TABLES_DIR / "bilateral_sigma_color.csv", index=False, encoding="utf-8"
    )
    bilateral_sigma_space_df.to_csv(
        TABLES_DIR / "bilateral_sigma_space.csv", index=False, encoding="utf-8"
    )
    guided_radius_df.to_csv(TABLES_DIR / "guided_radius.csv", index=False, encoding="utf-8")
    guided_eps_df.to_csv(TABLES_DIR / "guided_eps.csv", index=False, encoding="utf-8")
    guided_tmin_df.to_csv(TABLES_DIR / "guided_tmin.csv", index=False, encoding="utf-8")
    guided_gamma_df.to_csv(TABLES_DIR / "guided_gamma.csv", index=False, encoding="utf-8")

    plot_parameter_curve(
        bilateral_diameter_df,
        "bilateral_diameter",
        "Bilateral Diameter Sensitivity",
        "bilateral_diameter_curve.png",
    )
    plot_parameter_curve(
        bilateral_sigma_color_df,
        "bilateral_sigma_color",
        "Bilateral SigmaColor Sensitivity",
        "bilateral_sigma_color_curve.png",
    )
    plot_parameter_curve(
        bilateral_sigma_space_df,
        "bilateral_sigma_space",
        "Bilateral SigmaSpace Sensitivity",
        "bilateral_sigma_space_curve.png",
    )
    plot_parameter_curve(
        guided_radius_df,
        "guided_radius",
        "Guided Filter Radius Sensitivity",
        "guided_radius_curve.png",
    )
    plot_parameter_curve(
        guided_eps_df,
        "guided_eps",
        "Guided Filter Eps Sensitivity",
        "guided_eps_curve.png",
    )
    plot_parameter_curve(
        guided_tmin_df,
        "tmin",
        "Enhancement Tmin Sensitivity",
        "guided_tmin_curve.png",
    )
    plot_parameter_curve(
        guided_gamma_df,
        "gamma",
        "Enhancement Gamma Sensitivity",
        "guided_gamma_curve.png",
    )


def export_representative_figures():
    source_pairs = [
        ("results/bilateral/comparisons", "comparison_bilateral_default"),
        ("results/guided/comparisons", "comparison_guided_default"),
        ("results/bilateral/t0", "t0_default"),
        ("results/bilateral/smooth", "smooth_bilateral_default"),
        ("results/guided/smooth", "smooth_guided_default"),
    ]

    for filename in REPRESENTATIVE_FILES:
        for source_dir, prefix in source_pairs:
            source_path = PROJECT_ROOT / source_dir / filename
            target_path = FIGURES_DIR / f"{prefix}_{filename}"
            target_path.write_bytes(source_path.read_bytes())

    for config_name, filename in BILATERAL_VISUAL_CONFIGS:
        source_path = EXPERIMENTS_DIR / config_name / "comparisons" / filename
        target_path = FIGURES_DIR / f"{config_name}_{filename}"
        target_path.write_bytes(source_path.read_bytes())

    for config_name, filename in GUIDED_VISUAL_CONFIGS:
        source_path = EXPERIMENTS_DIR / config_name / "comparisons" / filename
        target_path = FIGURES_DIR / f"{config_name}_{filename}"
        target_path.write_bytes(source_path.read_bytes())


def main():
    ensure_report_dirs()
    paired_images = find_paired_images(INPUT_DIR, GT_DIR)
    if not paired_images:
        raise RuntimeError("未找到可匹配的输入图像与 GT 图像。")

    summary_rows = []
    for config in CONFIGS:
        print(f"running {config['name']}")
        summary_rows.append(run_single_config(config, paired_images))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "parameter_summary.csv", index=False, encoding="utf-8")

    write_main_comparison_tables()
    write_parameter_tables(summary_df)
    plot_main_method_bar()
    export_representative_figures()

    print("report experiments finished")


if __name__ == "__main__":
    main()

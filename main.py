from pathlib import Path

from metrics import (
    compute_average_metrics,
    compute_metrics,
    metrics_rows_append,
    print_metrics,
)
from pipeline import enhance_low_light
from utils import (
    build_comparison_image,
    ensure_results_dirs,
    find_paired_images,
    read_image_rgb,
    save_color_image,
    save_csv,
    save_gray_image,
)


# 这里集中放置本次作业使用的参数，便于后续调参与写报告说明
INPUT_DIR = Path("samples/Train/Input")
GT_DIR = Path("samples/Train/GT")
RESULTS_DIR = Path("results")

METHODS = ["bilateral", "guided"]

BILATERAL_DIAMETER = 15
BILATERAL_SIGMA_COLOR = 0.1
BILATERAL_SIGMA_SPACE = 15
GUIDED_RADIUS = 8
GUIDED_EPS = 1e-3
TMIN = 0.1
GAMMA = 0.8


def main():
    result_dirs_by_method = ensure_results_dirs(RESULTS_DIR, METHODS)
    paired_images = find_paired_images(INPUT_DIR, GT_DIR)

    if not paired_images:
        raise RuntimeError("未找到可匹配的 Input / GT 图像对。")

    comparison_summary_rows = []

    for method in METHODS:
        print(f"开始处理方法: {method}")
        result_dirs = result_dirs_by_method[method]
        metrics_rows = []

        for input_path, gt_path in paired_images:
            filename = input_path.name

            input_image = read_image_rgb(input_path)
            gt_image = read_image_rgb(gt_path)

            outputs = enhance_low_light(
                input_image=input_image,
                method=method,
                bilateral_diameter=BILATERAL_DIAMETER,
                bilateral_sigma_color=BILATERAL_SIGMA_COLOR,
                bilateral_sigma_space=BILATERAL_SIGMA_SPACE,
                guided_radius=GUIDED_RADIUS,
                guided_eps=GUIDED_EPS,
                tmin=TMIN,
                gamma=GAMMA,
            )

            metrics = compute_metrics(outputs["enhanced"], gt_image)

            save_gray_image(result_dirs["t0"] / filename, outputs["t0"])
            save_gray_image(result_dirs["smooth"] / filename, outputs["smooth"])
            save_color_image(result_dirs["enhanced"] / filename, outputs["enhanced"])

            comparison = build_comparison_image(
                input_image=input_image,
                enhanced_image=outputs["enhanced"],
                gt_image=gt_image,
            )
            save_color_image(result_dirs["comparisons"] / filename, comparison)

            metrics_rows_append(metrics_rows, filename, metrics)
            print_metrics(metrics, f"{method}/{filename}")

        average_metrics = compute_average_metrics(metrics_rows)
        metrics_rows_append(metrics_rows, "average", average_metrics)
        save_csv(result_dirs["root"] / "metrics.csv", metrics_rows)

        metrics_rows_append(comparison_summary_rows, method, average_metrics)

        print(f"方法 {method} 处理完成，共 {len(paired_images)} 张图像。")
        print_metrics(average_metrics, f"{method}/average")
        print(f"指标文件已保存到: {result_dirs['root'] / 'metrics.csv'}")

    save_csv(RESULTS_DIR / "comparison_summary.csv", comparison_summary_rows)
    print(f"两种方法的平均指标汇总已保存到: {RESULTS_DIR / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()

from pathlib import Path

from metrics import compute_psnr
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


# 这里集中放置本次作业使用的参数，便于后续调参和写报告说明
INPUT_DIR = Path("samples/Train/Input")
GT_DIR = Path("samples/Train/GT")
RESULTS_DIR = Path("results")

print(type(INPUT_DIR))

BILATERAL_DIAMETER = 15
BILATERAL_SIGMA_COLOR = 0.1
BILATERAL_SIGMA_SPACE = 15
TMIN = 0.1
GAMMA = 0.8


def main():
    result_dirs = ensure_results_dirs(RESULTS_DIR)
    paired_images = find_paired_images(INPUT_DIR, GT_DIR)

    if not paired_images:
        raise RuntimeError("未找到可匹配的 Input / GT 图像对。")

    metrics_rows = []

    for input_path, gt_path in paired_images:
        filename = input_path.name

        input_image = read_image_rgb(input_path)
        gt_image = read_image_rgb(gt_path)

        outputs = enhance_low_light(
            input_image=input_image,
            bilateral_diameter=BILATERAL_DIAMETER,
            bilateral_sigma_color=BILATERAL_SIGMA_COLOR,
            bilateral_sigma_space=BILATERAL_SIGMA_SPACE,
            tmin=TMIN,
            gamma=GAMMA,
        )

        psnr_value = compute_psnr(outputs["enhanced"], gt_image)

        save_gray_image(result_dirs["t0"] / filename, outputs["t0"])
        save_gray_image(result_dirs["tb"] / filename, outputs["tb"])
        save_color_image(result_dirs["enhanced"] / filename, outputs["enhanced"])

        comparison = build_comparison_image(
            input_image=input_image,
            enhanced_image=outputs["enhanced"],
            gt_image=gt_image,
        )
        save_color_image(result_dirs["comparisons"] / filename, comparison)

        metrics_rows.append(
            {
                "filename": filename,
                "psnr": f"{psnr_value:.4f}",
            }
        )

        print(f"已处理: {filename}, PSNR={psnr_value:.4f}")

    average_psnr = sum(float(row["psnr"]) for row in metrics_rows) / len(metrics_rows)
    metrics_rows.append({"filename": "average", "psnr": f"{average_psnr:.4f}"})

    save_csv(RESULTS_DIR / "metrics.csv", metrics_rows, fieldnames=["filename", "psnr"])
    print(f"处理完成，共 {len(paired_images)} 张图像。")
    print(f"平均 PSNR: {average_psnr:.4f}")
    print(f"指标文件已保存到: {RESULTS_DIR / 'metrics.csv'}")


if __name__ == "__main__":
    main()

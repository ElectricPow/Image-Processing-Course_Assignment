from functools import lru_cache

import numpy as np
from skimage.metrics import structural_similarity


def compute_psnr(pred_image, gt_image):
    # 这里假设输入图像已经归一化到 [0, 1]
    mse = np.mean((pred_image - gt_image) ** 2)

    if mse <= 1e-12:
        return float("inf")

    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(pred_image, gt_image):
    # 使用 scikit-image 的标准 SSIM 实现。
    # 当前项目默认输入图像已归一化到 [0, 1]，因此 data_range 取 1.0。
    channel_axis = -1 if pred_image.ndim == 3 else None
    return structural_similarity(
        pred_image,
        gt_image,
        data_range=1.0,
        channel_axis=channel_axis,
    )


def compute_mae(pred_image, gt_image):
    # 这里假设输入图像已经归一化到 [0, 1]
    return np.mean(np.abs(pred_image - gt_image))


def compute_mse(pred_image, gt_image):
    # 这里假设输入图像已经归一化到 [0, 1]
    return np.mean((pred_image - gt_image) ** 2)


@lru_cache(maxsize=1)
def _get_lpips_model():
    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "缺少 lpips 依赖。请在项目 .venv 中安装 lpips 和 torch 后再运行。"
        ) from exc

    model = lpips.LPIPS(net="alex")
    model.eval()
    return model


def compute_lpips(pred_image, gt_image):
    # 标准 LPIPS 需要将 RGB 图像转换为 [N, C, H, W]、范围 [-1, 1] 的张量。
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "缺少 torch 依赖。请在项目 .venv 中安装 torch 后再运行 LPIPS。"
        ) from exc

    model = _get_lpips_model()

    pred_tensor = torch.from_numpy(pred_image).permute(2, 0, 1).unsqueeze(0).float()
    gt_tensor = torch.from_numpy(gt_image).permute(2, 0, 1).unsqueeze(0).float()

    pred_tensor = pred_tensor * 2.0 - 1.0
    gt_tensor = gt_tensor * 2.0 - 1.0

    with torch.no_grad():
        lpips_value = model(pred_tensor, gt_tensor)

    return float(lpips_value.item())


def compute_metrics(pred_image, gt_image):
    # 需要添加的指标计算函数在这里调用
    psnr_value = compute_psnr(pred_image, gt_image)
    ssim_value = compute_ssim(pred_image, gt_image)
    mae_value = compute_mae(pred_image, gt_image)
    mse_value = compute_mse(pred_image, gt_image)
    lpips_value = compute_lpips(pred_image, gt_image)

    # 在这里调整每一张图的指标计算结果的格式
    return {
        "psnr": psnr_value,
        "ssim": ssim_value,
        "mae": mae_value,
        "mse": mse_value,
        "lpips": lpips_value,
    }


def metrics_rows_append(metrics_rows, filename, metrics):
    metric_names = [key for key in metrics]
    metrics_rows.append(
        {
            "filename": filename,
            **{
                metric_name: f"{metrics[metric_name]:.4f}"
                for metric_name in metric_names
            },
        }
    )


def print_metrics(metrics, filename):
    metric_names = [key for key in metrics]
    for metric_name in metric_names:
        print(f"已处理 {filename}, {metric_name.upper()}={metrics[metric_name]:.4f}")
    print("-" * 40)


def compute_average_metrics(metrics_rows):
    if not metrics_rows:
        return {}

    metric_names = [key for key in metrics_rows[0] if key != "filename"]
    num_rows = len(metrics_rows)

    return {
        metric_name: sum(float(row[metric_name]) for row in metrics_rows) / num_rows
        for metric_name in metric_names
    }

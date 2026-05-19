import numpy as np


def compute_psnr(pred_image, gt_image):
    # 这里假设输入图像已经归一化到 [0, 1]
    mse = np.mean((pred_image - gt_image) ** 2)

    if mse <= 1e-12:
        return float("inf")

    return 10.0 * np.log10(1.0 / mse)

import numpy as np
import cv2


def estimate_initial_illumination(input_image):
    # 按题目要求，使用 RGB 三通道最大值作为初始照明图 T0
    return np.max(input_image, axis=2)


def smooth_illumination_with_bilateral(t0, diameter, sigma_color, sigma_space):
    # OpenCV 双边滤波可直接用于单通道浮点图像
    t0_float32 = t0.astype(np.float32)
    tb = cv2.bilateralFilter(
        src=t0_float32,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )
    return tb


def enhance_low_light(
    input_image,
    bilateral_diameter,
    bilateral_sigma_color,
    bilateral_sigma_space,
    tmin,
    gamma,
):
    t0 = estimate_initial_illumination(input_image)
    tb = smooth_illumination_with_bilateral(
        t0=t0,
        diameter=bilateral_diameter,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space,
    )

    # 根据题目公式 J = I / max(Tb, Tmin)^gamma 进行增强
    denominator = np.maximum(tb, tmin) ** gamma
    enhanced = input_image / denominator[:, :, np.newaxis]

    # 将结果限制到 [0, 1]，便于后续保存和计算指标
    enhanced = np.clip(enhanced, 0.0, 1.0)
    tb = np.clip(tb, 0.0, 1.0)
    t0 = np.clip(t0, 0.0, 1.0)

    return {
        "t0": t0,
        "tb": tb,
        "enhanced": enhanced,
    }

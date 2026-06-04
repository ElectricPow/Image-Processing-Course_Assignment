import cv2
import numpy as np

from utils import lab_to_rgb, rgb_to_lab


def estimate_initial_illumination(input_image):
    # 按题目要求，使用 RGB 三通道最大值作为初始照明图 T0
    return np.max(input_image, axis=2)


def smooth_illumination_with_bilateral(t0, diameter, sigma_color, sigma_space):
    # OpenCV 双边滤波可直接用于单通道浮点图像
    return cv2.bilateralFilter(
        src=t0.astype(np.float32),
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )


def smooth_illumination_with_guided(t0, radius, eps):
    # 根据 Project.md 中的默认方案，使用自引导滤波：G = T0, p = T0
    guide = t0.astype(np.float32)
    src = t0.astype(np.float32)
    window_size = (2 * radius + 1, 2 * radius + 1)

    mean_guide = cv2.boxFilter(guide, ddepth=-1, ksize=window_size)
    mean_src = cv2.boxFilter(src, ddepth=-1, ksize=window_size)

    corr_guide = cv2.boxFilter(guide * guide, ddepth=-1, ksize=window_size)
    corr_guide_src = cv2.boxFilter(guide * src, ddepth=-1, ksize=window_size)

    var_guide = corr_guide - mean_guide * mean_guide
    cov_guide_src = corr_guide_src - mean_guide * mean_src

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=window_size)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=window_size)

    return mean_a * guide + mean_b


def apply_linear_color_correction(input_image, enhanced_image, alpha_max=0.5):
    # 保持增强图亮度不变，根据原图亮度自适应地向原图色度回拉
    input_lab = rgb_to_lab(input_image)
    enhanced_lab = rgb_to_lab(enhanced_image)
    corrected_lab = enhanced_lab.copy()

    original_l = input_lab[:, :, 0]
    alpha = np.clip(alpha_max * (original_l / 100.0), 0.0, alpha_max).astype(np.float32)

    corrected_lab[:, :, 1] = (
        (1.0 - alpha) * enhanced_lab[:, :, 1] + alpha * input_lab[:, :, 1]
    )
    corrected_lab[:, :, 2] = (
        (1.0 - alpha) * enhanced_lab[:, :, 2] + alpha * input_lab[:, :, 2]
    )

    return lab_to_rgb(corrected_lab)


def enhance_low_light(
    input_image,
    method,
    bilateral_diameter,
    bilateral_sigma_color,
    bilateral_sigma_space,
    guided_radius,
    guided_eps,
    tmin,
    gamma,
    linear_alpha_max=0.5,
):
    t0 = estimate_initial_illumination(input_image)

    if method == "bilateral":
        smooth_illumination = smooth_illumination_with_bilateral(
            t0=t0,
            diameter=bilateral_diameter,
            sigma_color=bilateral_sigma_color,
            sigma_space=bilateral_sigma_space,
        )
    elif method == "guided":
        smooth_illumination = smooth_illumination_with_guided(
            t0=t0,
            radius=guided_radius,
            eps=guided_eps,
        )
    else:
        raise ValueError(f"不支持的增强方法: {method}")

    # 根据公式 J = I / max(T, Tmin)^gamma 进行增强
    denominator = np.maximum(smooth_illumination, tmin) ** gamma
    enhanced = np.clip(input_image / denominator[:, :, np.newaxis], 0.0, 1.0)

    linear_color_corrected = apply_linear_color_correction(
        input_image=input_image,
        enhanced_image=enhanced,
        alpha_max=linear_alpha_max,
    )

    return {
        "method": method,
        "t0": np.clip(t0, 0.0, 1.0),
        "smooth": np.clip(smooth_illumination, 0.0, 1.0),
        "enhanced": enhanced,
        "linear_color_corrected": linear_color_corrected,
    }

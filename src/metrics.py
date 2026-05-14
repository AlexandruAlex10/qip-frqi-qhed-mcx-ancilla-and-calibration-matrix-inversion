
"""
Metrics for evaluating image quality, such as PSNR and a simple SSIM-like score.

PSNR (Peak Signal-to-Noise Ratio) is a common metric for measuring the quality of
reconstructed images compared to reference images. It is defined as:
PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
where MAX_I is the maximum possible pixel value of the image (e.g., 255 for 8-bit images)
and MSE is the mean squared error between the two images.
"""

from __future__ import annotations

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def ssim_uint8(a: np.ndarray, b: np.ndarray, *, data_range: float = 255.0) -> float:
    """Structural similarity (SSIM) for 2D grayscale images via scikit-image.

    Expects spatially aligned arrays of the same shape. ``data_range`` should
    match the intensity scale (255 for uint8-style images).
    """
    from skimage.metrics import structural_similarity

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Inputs must have the same shape.")

    min_side = min(a.shape[:2])
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError(
            "Images must be at least 3x3 for SSIM computation."
        )

    return float(
        structural_similarity(
            a,
            b,
            data_range=data_range,
            channel_axis=None,
            win_size=win_size,
        )
    )


def ssim_like(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight similarity score in [-1, 1], not a full SSIM implementation.
    Kept simple to avoid extra dependencies in Week 2.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("Inputs must have the same number of elements.")
    a_mean, b_mean = a.mean(), b.mean()
    a_var, b_var = a.var(), b.var()
    cov = np.mean((a - a_mean) * (b - b_mean))
    c1 = 1e-6
    c2 = 1e-6
    num = (2 * a_mean * b_mean + c1) * (2 * cov + c2)
    den = (a_mean**2 + b_mean**2 + c1) * (a_var + b_var + c2)
    return float(num / den)

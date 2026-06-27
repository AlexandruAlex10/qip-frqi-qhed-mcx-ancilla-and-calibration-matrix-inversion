"""Image quality metrics: PSNR and SSIM variants.

PSNR is ``20 * log10(MAX_I) - 10 * log10(MSE)``, where ``MAX_I`` is the maximum
pixel value (255 for 8-bit images) and ``MSE`` is the mean squared error.
"""

from __future__ import annotations

import numpy as np

__all__ = ["psnr", "ssim_uint8", "ssim_like"]


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Return the peak signal-to-noise ratio in dB (``inf`` for identical inputs)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def ssim_uint8(a: np.ndarray, b: np.ndarray, *, data_range: float = 255.0) -> float:
    """Return the structural similarity of two grayscale images via scikit-image.

    The window size adapts to the (small) image side so SSIM works on 4x4 tiles.

    Parameters
    ----------
    a, b : np.ndarray
        Spatially aligned arrays of identical shape.
    data_range : float, optional
        Intensity scale (255 for uint8-style images).

    Raises
    ------
    ValueError
        If shapes differ or the images are smaller than 3x3.
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
        raise ValueError("Images must be at least 3x3 for SSIM computation.")

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
    """Return a lightweight global similarity score in ``[-1, 1]``.

    This is a dependency-free approximation, not a full windowed SSIM; use
    :func:`ssim_uint8` for the scikit-image metric.

    Raises
    ------
    ValueError
        If the inputs do not have the same number of elements.
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

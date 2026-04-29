
from __future__ import annotations

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


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


"""
Minimal Hadamard-based baseline edge detector

This is a simple, reproducible baseline inspired by Hadamard-transform edge detection
ideas. It is intentionally lightweight so it runs without a full quantum SDK.
"""

from __future__ import annotations

from math import sqrt
from typing import Tuple

import numpy as np

from .frqi import validate_grayscale_image, build_frqi_statevector, reconstruct_image_from_statevector


def _fwht1d(x: np.ndarray) -> np.ndarray:
    """In-place Fast Walsh-Hadamard Transform for 1D arrays of length power of two."""
    a = np.asarray(x, dtype=np.float64).copy()
    n = a.shape[0]
    if n & (n - 1):
        raise ValueError("FWHT input length must be a power of two.")
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            j1 = slice(i, i + h)
            j2 = slice(i + h, i + 2 * h)
            u = a[j1].copy()
            v = a[j2].copy()
            a[j1] = u + v
            a[j2] = u - v
        h *= 2
    return a


def _ifwht1d(x: np.ndarray) -> np.ndarray:
    """Inverse FWHT, normalized by n."""
    a = _fwht1d(x)
    return a / a.shape[0]


def fwht2d(image: np.ndarray) -> np.ndarray:
    """2D FWHT applied over rows and then columns."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("fwht2d expects a square 2D array.")
    n = arr.shape[0]
    if n & (n - 1):
        raise ValueError("fwht2d expects power-of-two image sizes.")

    temp = np.vstack([_fwht1d(row) for row in arr])
    out = np.vstack([_fwht1d(col) for col in temp.T]).T
    return out


def ifwht2d(coeffs: np.ndarray) -> np.ndarray:
    """Inverse 2D FWHT applied over rows and then columns."""
    arr = np.asarray(coeffs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("ifwht2d expects a square 2D array.")
    n = arr.shape[0]
    if n & (n - 1):
        raise ValueError("ifwht2d expects power-of-two image sizes.")

    temp = np.vstack([_ifwht1d(row) for row in arr])
    out = np.vstack([_ifwht1d(col) for col in temp.T]).T
    return out


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale a numeric array to uint8 in [0,255]."""
    x = np.asarray(arr, dtype=np.float64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if np.isclose(hi, lo):
        return np.zeros_like(x, dtype=np.uint8)
    scaled = (x - lo) / (hi - lo)
    return np.rint(255.0 * scaled).astype(np.uint8)


def baseline_qhed_edge_map(image: np.ndarray, low_frequency_block: int = 2) -> np.ndarray:
    """Baseline Hadamard-style edge map.

    Steps:
      1. Validate input.
      2. Apply 2D FWHT.
      3. Take the magnitude of the transform coefficients as a simple interference proxy.
      4. Reconstruct a contrast image from the magnitude spectrum.
      5. Apply a Sobel magnitude to obtain an edge-like output.

    The output is intentionally simple and serves as a baseline for later improvement.
    """
    arr = validate_grayscale_image(image).astype(np.float64)
    coeffs = fwht2d(arr)

    # The low_frequency_block argument is kept for compatibility with the README and
    # future experiments; baseline focuses on a stable non-degenerate output.
    _ = int(low_frequency_block)

    contrast = np.abs(ifwht2d(np.abs(coeffs)))
    return classical_sobel_edge_map(contrast)


def classical_sobel_edge_map(image: np.ndarray) -> np.ndarray:
    """Simple 3x3 Sobel magnitude for comparison."""
    arr = validate_grayscale_image(image).astype(np.float64)

    gx_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=np.float64)

    gy_kernel = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1],
    ], dtype=np.float64)

    pad = 1
    padded = np.pad(arr, pad, mode="edge")
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            patch = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(patch * gx_kernel)
            gy[i, j] = np.sum(patch * gy_kernel)

    mag = np.sqrt(gx**2 + gy**2)
    return normalize_to_uint8(mag)


def qhed_pipeline(image: np.ndarray, low_frequency_block: int = 2):
    """End-to-end baseline pipeline:
      FRQI encode -> Hadamard-style edge map -> return outputs for inspection.
    """
    arr = validate_grayscale_image(image)
    state = build_frqi_statevector(arr)
    reconstructed = reconstruct_image_from_statevector(state, arr.shape[0])
    edge_map = baseline_qhed_edge_map(arr, low_frequency_block=low_frequency_block)
    return {
        "statevector": state,
        "reconstructed": reconstructed,
        "edge_map": edge_map,
    }

"""Minimal Hadamard-based baseline edge detector.

A lightweight, reproducible baseline inspired by Hadamard-transform edge
detection that runs without a full quantum SDK.
"""

from __future__ import annotations

import numpy as np

from .frqi import validate_grayscale_image, build_frqi_statevector, reconstruct_image_from_statevector

__all__ = [
    "fwht2d",
    "ifwht2d",
    "normalize_to_uint8",
    "baseline_qhed_edge_map",
    "classical_sobel_edge_map",
    "qhed_pipeline",
]


def _fwht1d(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform for a 1D array whose length is a power of two.

    Raises
    ------
    ValueError
        If the input length is not a power of two.
    """
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
    """Inverse 1D FWHT, normalized by the array length."""
    a = _fwht1d(x)
    return a / a.shape[0]


def fwht2d(image: np.ndarray) -> np.ndarray:
    """Apply a 2D FWHT over rows and then columns.

    Raises
    ------
    ValueError
        If the input is not a square 2D array with power-of-two side length.
    """
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
    """Apply an inverse 2D FWHT over rows and then columns.

    Raises
    ------
    ValueError
        If the input is not a square 2D array with power-of-two side length.
    """
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
    """Scale a numeric array to uint8 in ``[0, 255]`` (all-zeros if constant)."""
    x = np.asarray(arr, dtype=np.float64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if np.isclose(hi, lo):
        return np.zeros_like(x, dtype=np.uint8)
    scaled = (x - lo) / (hi - lo)
    return np.rint(255.0 * scaled).astype(np.uint8)


def baseline_qhed_edge_map(image: np.ndarray) -> np.ndarray:
    """Compute a baseline Hadamard-style edge map.

    The pipeline applies a 2D FWHT, takes the coefficient magnitude as a simple
    interference proxy, reconstructs a contrast image, and returns its Sobel
    magnitude. The output is intentionally simple, serving as a baseline for
    later improvement.
    """
    arr = validate_grayscale_image(image).astype(np.float64)
    coeffs = fwht2d(arr)
    contrast = np.abs(ifwht2d(np.abs(coeffs)))
    return classical_sobel_edge_map(contrast)


def classical_sobel_edge_map(image: np.ndarray) -> np.ndarray:
    """Return the 3x3 Sobel gradient magnitude as a uint8 edge map."""
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

    padded = np.pad(arr, 1, mode="edge")
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            patch = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(patch * gx_kernel)
            gy[i, j] = np.sum(patch * gy_kernel)

    mag = np.sqrt(gx**2 + gy**2)
    return normalize_to_uint8(mag)


def qhed_pipeline(image: np.ndarray):
    """Run the end-to-end baseline: FRQI encode, reconstruct, and edge-map.

    Returns
    -------
    dict
        Keys ``statevector``, ``reconstructed``, and ``edge_map``.
    """
    arr = validate_grayscale_image(image)
    state = build_frqi_statevector(arr)
    reconstructed = reconstruct_image_from_statevector(state, arr.shape[0])
    edge_map = baseline_qhed_edge_map(arr)
    return {
        "statevector": state,
        "reconstructed": reconstructed,
        "edge_map": edge_map,
    }

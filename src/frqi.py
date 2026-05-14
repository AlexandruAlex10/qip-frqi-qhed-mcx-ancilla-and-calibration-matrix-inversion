
"""
FRQI state preparation and reconstruction utilities.

This module provides a minimal, exact FRQI statevector construction for
grayscale images whose width and height are equal powers of two.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, log2
from pathlib import Path
from typing import Tuple

import numpy as np


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def validate_grayscale_image(image: np.ndarray) -> np.ndarray:
    """Validate and normalize the input image array.

    Accepts a 2D grayscale image with equal power-of-two dimensions.
    Returns a uint8 array clipped to [0, 255].
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("FRQI expects a 2D grayscale image.")
    h, w = arr.shape
    if h != w:
        raise ValueError("FRQI expects a square image.")
    if not is_power_of_two(h):
        raise ValueError("FRQI expects image size to be a power of two.")
    arr = np.clip(arr, 0, 255)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def required_position_qubits(image_size: int) -> int:
    """Return the number of position qubits for an image of size image_size x image_size."""
    if not is_power_of_two(image_size):
        raise ValueError("image_size must be a power of two.")
    return int(2 * log2(image_size))


def image_to_angles(image: np.ndarray) -> np.ndarray:
    """Map grayscale intensities in [0,255] to FRQI angles in [0, pi/2]."""
    arr = validate_grayscale_image(image).astype(np.float64)
    return (pi / 2.0) * (arr / 255.0)


def build_frqi_statevector(image: np.ndarray) -> np.ndarray:
    """Construct the ideal FRQI statevector for a grayscale image.

    Basis convention:
        |position, color>
    with the color qubit as the least significant bit in the statevector index.
    Therefore each pixel p occupies indices 2*p (color=0) and 2*p+1 (color=1).

    The state is:
        1/sqrt(N) sum_p (cos(theta_p)|p,0> + sin(theta_p)|p,1>)
    where N is the number of pixels.
    """
    arr = validate_grayscale_image(image)
    n = arr.shape[0]
    n_pixels = n * n
    angles = image_to_angles(arr).reshape(-1)

    state = np.zeros(2 * n_pixels, dtype=np.complex128)
    norm = np.sqrt(n_pixels)
    for p, theta in enumerate(angles):
        state[2 * p] = np.cos(theta) / norm
        state[2 * p + 1] = np.sin(theta) / norm
    return state


def reconstruct_image_from_statevector(statevector: np.ndarray, image_size: int) -> np.ndarray:
    """Recover the image from an ideal FRQI statevector."""
    if not is_power_of_two(image_size):
        raise ValueError("image_size must be a power of two.")
    state = np.asarray(statevector, dtype=np.complex128).reshape(-1)
    expected = 2 * image_size * image_size
    if state.size != expected:
        raise ValueError(f"Expected a statevector of length {expected}, got {state.size}.")
    n_pixels = image_size * image_size
    intensities = np.zeros(n_pixels, dtype=np.float64)
    for p in range(n_pixels):
        a0 = np.abs(state[2 * p])
        a1 = np.abs(state[2 * p + 1])
        theta = np.arctan2(a1, a0)
        intensities[p] = (theta / (pi / 2.0)) * 255.0
    return np.rint(intensities).astype(np.uint8).reshape((image_size, image_size))


def l2_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean squared error between two equally-shaped grayscale images."""
    a = validate_grayscale_image(original).astype(np.float64)
    b = validate_grayscale_image(reconstructed).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("Images must have the same shape.")
    return float(np.mean((a - b) ** 2))


def maybe_build_qiskit_circuit(image: np.ndarray):
    """Optional helper: build an exact initialization circuit if Qiskit is installed.

    Useful for later work.
    """
    try:
        from qiskit import QuantumCircuit
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Qiskit is not installed.") from exc

    state = build_frqi_statevector(image)
    n_qubits = int(np.log2(state.size))
    qc = QuantumCircuit(n_qubits, name="FRQI")
    qc.initialize(state, qc.qubits)
    return qc

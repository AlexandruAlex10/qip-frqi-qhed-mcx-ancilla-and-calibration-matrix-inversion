"""FRQI state preparation and reconstruction utilities.

Provides an exact FRQI statevector construction for grayscale images whose
width and height are equal powers of two, plus reconstruction helpers.
"""

from __future__ import annotations

from math import pi, log2

import numpy as np

__all__ = [
    "is_power_of_two",
    "validate_grayscale_image",
    "required_position_qubits",
    "image_to_angles",
    "build_frqi_statevector",
    "reconstruct_image_from_reduced_density_matrix",
    "reconstruct_image_from_statevector",
    "l2_error",
    "maybe_build_qiskit_circuit",
]


def is_power_of_two(n: int) -> bool:
    """Return whether ``n`` is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def validate_grayscale_image(image: np.ndarray) -> np.ndarray:
    """Validate and normalize a grayscale image to uint8.

    Parameters
    ----------
    image : np.ndarray
        A 2D grayscale image with equal power-of-two dimensions.

    Returns
    -------
    np.ndarray
        A uint8 array clipped to ``[0, 255]``.

    Raises
    ------
    ValueError
        If the image is not 2D, not square, or not power-of-two sized.
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
    """Return the number of position qubits for an ``image_size``-square image.

    Raises
    ------
    ValueError
        If ``image_size`` is not a power of two.
    """
    if not is_power_of_two(image_size):
        raise ValueError("image_size must be a power of two.")
    return int(2 * log2(image_size))


def image_to_angles(image: np.ndarray) -> np.ndarray:
    """Map grayscale intensities in ``[0, 255]`` to FRQI angles in ``[0, pi/2]``."""
    arr = validate_grayscale_image(image).astype(np.float64)
    return (pi / 2.0) * (arr / 255.0)


def build_frqi_statevector(image: np.ndarray) -> np.ndarray:
    """Construct the ideal FRQI statevector for a grayscale image.

    The basis convention is ``|position, color>`` with the color qubit as the
    least significant bit, so pixel ``p`` occupies indices ``2*p`` (color=0) and
    ``2*p+1`` (color=1). The state is
    ``1/sqrt(N) sum_p (cos(theta_p)|p,0> + sin(theta_p)|p,1>)`` for ``N`` pixels.

    Parameters
    ----------
    image : np.ndarray
        Validated grayscale image (square, power-of-two side).

    Returns
    -------
    np.ndarray
        Complex statevector of length ``2 * N``.
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


def reconstruct_image_from_reduced_density_matrix(rho, image_size: int) -> np.ndarray:
    """Recover a grayscale image from the reduced FRQI density matrix.

    Pixel angles are inferred from the color-qubit diagonal populations at each
    address: ``theta_p = arctan2(sqrt(rho_{2p+1,2p+1}), sqrt(rho_{2p,2p}))``,
    which matches the statevector rule when the state is a pure FRQI state.

    Parameters
    ----------
    rho : DensityMatrix or np.ndarray
        Reduced density matrix over color + position (a Qiskit
        :class:`DensityMatrix` or a complex square array).
    image_size : int
        Side length of the (square) image.

    Returns
    -------
    np.ndarray
        Reconstructed uint8 image of shape ``(image_size, image_size)``.

    Raises
    ------
    ValueError
        If ``image_size`` is not a power of two or ``rho`` has the wrong shape.
    """
    if not is_power_of_two(image_size):
        raise ValueError("image_size must be a power of two.")
    n_pixels = image_size * image_size
    dim = 2 * n_pixels
    if hasattr(rho, "data"):
        mat = np.asarray(rho.data, dtype=np.complex128)
    else:
        mat = np.asarray(rho, dtype=np.complex128)
    mat = mat.reshape(dim, dim)
    if mat.shape != (dim, dim):
        raise ValueError(f"Expected reduced density matrix of shape {(dim, dim)}, got {mat.shape}.")

    intensities = np.zeros(n_pixels, dtype=np.float64)
    for p in range(n_pixels):
        d0 = max(float(np.real(mat[2 * p, 2 * p])), 0.0)
        d1 = max(float(np.real(mat[2 * p + 1, 2 * p + 1])), 0.0)
        a0, a1 = np.sqrt(d0), np.sqrt(d1)
        if a0 == 0.0 and a1 == 0.0:
            theta = 0.0
        else:
            theta = float(np.arctan2(a1, a0))
        intensities[p] = (theta / (pi / 2.0)) * 255.0
    return np.rint(intensities).astype(np.uint8).reshape((image_size, image_size))


def reconstruct_image_from_statevector(statevector: np.ndarray, image_size: int) -> np.ndarray:
    """Recover the image from an ideal FRQI statevector.

    Raises
    ------
    ValueError
        If ``image_size`` is not a power of two or the statevector length is
        not ``2 * image_size**2``.
    """
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
    """Return the mean squared error between two equally-shaped grayscale images.

    Raises
    ------
    ValueError
        If the validated images do not share the same shape.
    """
    a = validate_grayscale_image(original).astype(np.float64)
    b = validate_grayscale_image(reconstructed).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError("Images must have the same shape.")
    return float(np.mean((a - b) ** 2))


def maybe_build_qiskit_circuit(image: np.ndarray):
    """Build an exact FRQI initialization circuit if Qiskit is installed.

    Returns
    -------
    qiskit.QuantumCircuit
        A circuit that prepares the exact FRQI statevector via ``initialize``.

    Raises
    ------
    RuntimeError
        If Qiskit is not importable.
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

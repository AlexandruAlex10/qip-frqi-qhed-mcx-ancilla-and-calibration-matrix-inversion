"""Structural FRQI preparation circuits (multi-controlled address match + Ry).

Wire ordering must match ``build_frqi_statevector`` in ``src/frqi.py``:

- Qubit 0: color (LSB of the computational index).
- Qubits ``1..m``: position register; pixel bit ``(p >> k) & 1`` lives on qubit
  ``k + 1`` (qubit 1 is the LSB of ``p``).
- Qubit ``m + 1``: flag, the target of the "address equals ``p``" MCX.
- Qubits ``m + 2 .. 2m - 1``: v-chain workspace (``m - 2`` dirty ancillas, empty
  when ``m < 3``) required by ``MCXGate(..., mode="v-chain")``.

Total width is ``m + 2 + max(0, m - 2)`` qubits. Each pixel applies a polarity
compare, an MCX into the flag, ``cry(2*theta_p)`` from flag to color, then the
inverse compare. The factor of 2 in ``cry`` matches Qiskit's ``Ry`` convention
to ``build_frqi_statevector``. Preparation starts with a Hadamard layer on the
position register (color stays ``|0>``).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit

from src.frqi import image_to_angles, required_position_qubits, validate_grayscale_image

__all__ = [
    "frqi_structural_num_qubits_vchain",
    "frqi_structural_num_qubits_naive_slice",
    "build_frqi_prep_vchain",
    "build_frqi_prep_naive",
    "build_single_address_ry_slice",
]

# Naive structural MCX decomposition (no dedicated v-chain ancillas on the gate object).
_NAIVE_MCX_MODE = "noancilla"


def _position_qubit_indices(m: int) -> list[int]:
    """Physical qubits holding the ``m``-bit pixel index (see module docstring)."""
    return [k + 1 for k in range(m)]


def _apply_address_compare_x_polarity(qc: QuantumCircuit, m: int, p: int) -> None:
    """Flip controls so that all read |1> if the position register encodes ``p``."""
    pos = _position_qubit_indices(m)
    for k in range(m):
        bit = (p >> k) & 1
        if bit == 0:
            qc.x(pos[k])


def _mcx_address_match(
    qc: QuantumCircuit,
    m: int,
    *,
    flag: int,
    chain_ancilla: Iterable[int],
    mode: str,
) -> None:
    """Multi-controlled X: flip ``flag`` when the position register matches the prepared polarity."""
    controls = _position_qubit_indices(m)
    anc = list(chain_ancilla)
    if mode == "v-chain":
        expected = max(0, m - 2)
        if len(anc) != expected:
            raise ValueError(f"v-chain expects {expected} ancilla qubits, got {len(anc)}")
        if expected == 0:
            qc.mcx(controls, flag, mode="v-chain")
        else:
            qc.mcx(controls, flag, ancilla_qubits=anc, mode="v-chain")
    else:
        qc.mcx(controls, flag, mode=mode)


def frqi_structural_num_qubits_vchain(m: int) -> int:
    """Return total qubit count for the v-chain layout (``m`` position qubits)."""
    if m < 1:
        raise ValueError("m must be >= 1")
    return m + 2 + max(0, m - 2)


def frqi_structural_num_qubits_naive_slice(m: int) -> int:
    """Return qubit count for :func:`build_single_address_ry_slice` with ``mode='noancilla'``."""
    if m < 1:
        raise ValueError("m must be >= 1")
    return m + 2  # color + m position + flag


def build_frqi_prep_vchain(image: np.ndarray) -> QuantumCircuit:
    """FRQI prep via per-pixel v-chain ``MCX`` + ``cry`` + uncompute (reused ancillas)."""
    arr = validate_grayscale_image(image)
    n = int(arr.shape[0])
    m = required_position_qubits(n)
    thetas = image_to_angles(arr).reshape(-1)
    n_pix = n * n
    if n_pix != 2**m:
        raise ValueError("Internal inconsistency: pixel count does not match m.")

    n_qubits = frqi_structural_num_qubits_vchain(m)
    qc = QuantumCircuit(n_qubits, name="FRQI_struct_vchain")
    color = 0
    flag = m + 1
    chain = list(range(m + 2, 2 * m))

    # Uniform superposition over the address register (color stays |0>).
    for q in _position_qubit_indices(m):
        qc.h(q)

    for p in range(n_pix):
        _apply_address_compare_x_polarity(qc, m, p)
        _mcx_address_match(qc, m, flag=flag, chain_ancilla=chain, mode="v-chain")
        qc.cry(2.0 * float(thetas[p]), flag, color)
        _mcx_address_match(qc, m, flag=flag, chain_ancilla=chain, mode="v-chain")
        _apply_address_compare_x_polarity(qc, m, p)

    return qc


def build_frqi_prep_naive(image: np.ndarray) -> QuantumCircuit:
    """Full naive FRQI prep (``noancilla`` ``MCX``); scales poorly — prefer :func:`build_single_address_ry_slice` for large ``m``."""
    arr = validate_grayscale_image(image)
    n = int(arr.shape[0])
    m = required_position_qubits(n)
    thetas = image_to_angles(arr).reshape(-1)
    n_pix = n * n
    if n_pix != 2**m:
        raise ValueError("Internal inconsistency: pixel count does not match m.")

    n_qubits = frqi_structural_num_qubits_naive_slice(m)
    qc = QuantumCircuit(n_qubits, name="FRQI_struct_naive")
    color = 0
    flag = m + 1

    for q in _position_qubit_indices(m):
        qc.h(q)

    for p in range(n_pix):
        _apply_address_compare_x_polarity(qc, m, p)
        _mcx_address_match(qc, m, flag=flag, chain_ancilla=(), mode=_NAIVE_MCX_MODE)
        qc.cry(2.0 * float(thetas[p]), flag, color)
        _mcx_address_match(qc, m, flag=flag, chain_ancilla=(), mode=_NAIVE_MCX_MODE)
        _apply_address_compare_x_polarity(qc, m, p)

    return qc


def build_single_address_ry_slice(
    m: int,
    address_bits: int,
    theta: float,
    *,
    mode: str = "v-chain",
) -> QuantumCircuit:
    """Build one pixel slice: match ``address_bits``, ``cry(2*theta)`` on color, uncompute.

    For reporting, multiplying this slice's transpiled depth/CX by ``N_pix = 2**m``
    estimates the full prep, since every slice shares the same multi-controlled-X
    structure and differs only in the ``cry`` angle.

    Raises
    ------
    ValueError
        If ``m < 1``, ``address_bits`` is out of range, or ``mode`` is unknown.
    """
    if m < 1:
        raise ValueError("m must be >= 1")
    if address_bits < 0 or address_bits >= 2**m:
        raise ValueError("address_bits must be in [0, 2**m).")

    if mode == "v-chain":
        n_qubits = frqi_structural_num_qubits_vchain(m)
        flag = m + 1
        chain = list(range(m + 2, 2 * m))
    elif mode == _NAIVE_MCX_MODE:
        n_qubits = frqi_structural_num_qubits_naive_slice(m)
        flag = m + 1
        chain = ()
    else:
        raise ValueError(f"Unsupported MCX decomposition mode: {mode!r}")

    qc = QuantumCircuit(n_qubits, name=f"FRQI_slice_m{m}_{mode}")
    _apply_address_compare_x_polarity(qc, m, address_bits)
    _mcx_address_match(qc, m, flag=flag, chain_ancilla=chain, mode=mode)
    qc.cry(2.0 * float(theta), flag, 0)
    _mcx_address_match(qc, m, flag=flag, chain_ancilla=chain, mode=mode)
    _apply_address_compare_x_polarity(qc, m, address_bits)
    return qc

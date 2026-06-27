"""Structural FRQI prep (v-chain / naive) vs ideal statevector."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from qiskit.quantum_info import Statevector, state_fidelity

from src.frqi import build_frqi_statevector, required_position_qubits
from src.improved import (
    build_frqi_prep_naive,
    build_frqi_prep_vchain,
    frqi_structural_num_qubits_vchain,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "test_images"


def _data_sub_statevector(sv: Statevector, m: int) -> Statevector:
    """Project onto the data Hilbert space (qubits ``0..m``); ancilla must be |0…0⟩."""
    block = np.asarray(sv.data[: 2 ** (m + 1)], dtype=np.complex128)
    nrm = float(np.linalg.norm(block))
    if nrm == 0:
        raise AssertionError("Empty data block in statevector.")
    return Statevector(block / nrm)


def _ancilla_all_zero_probability(sv: Statevector, m: int) -> float:
    """Probability that qubits ``m+1 .. n-1`` are all |0⟩ (computational)."""
    probs = np.abs(sv.data) ** 2
    return float(sum(probs[k] for k in range(len(probs)) if (k >> (m + 1)) == 0))


@pytest.mark.parametrize("factory", [build_frqi_prep_vchain, build_frqi_prep_naive])
def test_frqi_structural_fidelity_4x4(factory):
    img = np.load(DATA / "test_4x4.npy")
    m = required_position_qubits(int(img.shape[0]))
    ref = Statevector(build_frqi_statevector(img))
    qc = factory(img)
    sv = Statevector.from_instruction(qc)
    sub = _data_sub_statevector(sv, m)
    fid = state_fidelity(sub, ref)
    assert fid >= 1.0 - 1e-6


def test_frqi_vchain_width_and_ancilla_reset_4x4():
    img = np.load(DATA / "test_4x4.npy")
    m = required_position_qubits(int(img.shape[0]))
    qc = build_frqi_prep_vchain(img)
    assert qc.num_qubits == frqi_structural_num_qubits_vchain(m)
    sv = Statevector.from_instruction(qc)
    assert _ancilla_all_zero_probability(sv, m) >= 1.0 - 1e-9


@pytest.mark.slow
def test_frqi_vchain_fidelity_8x8():
    img = np.load(DATA / "test_8x8.npy")
    m = required_position_qubits(int(img.shape[0]))
    ref = Statevector(build_frqi_statevector(img))
    sv = Statevector.from_instruction(build_frqi_prep_vchain(img))
    sub = _data_sub_statevector(sv, m)
    assert state_fidelity(sub, ref) >= 1.0 - 1e-6


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_FRQI_16") == "1",
    reason="16×16 statevector simulation is optional (set RUN_FRQI_16=1).",
)
def test_frqi_vchain_fidelity_16x16():
    img = np.load(DATA / "test_16x16.npy")
    m = required_position_qubits(int(img.shape[0]))
    ref = Statevector(build_frqi_statevector(img))
    sv = Statevector.from_instruction(build_frqi_prep_vchain(img))
    sub = _data_sub_statevector(sv, m)
    assert state_fidelity(sub, ref) >= 1.0 - 1e-5

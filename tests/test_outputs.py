"""
Skimage SSIM, transpiled circuit stats (4×4 only, fast).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

import numpy as np

from src.frqi import (
    build_frqi_statevector,
    maybe_build_qiskit_circuit,
    reconstruct_image_from_statevector,
)
from src.metrics import psnr, ssim_uint8
from src.resources import transpiled_circuit_stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "test_images"
TEST_IMG = DATA / "test_4x4.npy"
BASIS_GATES = ["cx", "rz", "sx"]
OPT_LEVEL = 3


def test_metrics_4x4():
    img = np.load(TEST_IMG)
    state = build_frqi_statevector(img)
    recon = reconstruct_image_from_statevector(state, img.shape[0])

    p = psnr(img, recon)
    assert np.isfinite(p) or p == float("inf")

    s = ssim_uint8(img, recon)
    assert np.isfinite(s)
    assert -1.0 <= s <= 1.0

    s_identical = ssim_uint8(img, img)
    assert np.isfinite(s_identical)
    assert 0.0 <= s_identical <= 1.0


def test_transpiled_circuit_stats_sanity_4x4():
    img = np.load(TEST_IMG)
    qc = maybe_build_qiskit_circuit(img)
    stats = transpiled_circuit_stats(qc, BASIS_GATES, OPT_LEVEL)

    assert isinstance(stats["num_qubits"], int)
    assert stats["num_qubits"] > 0
    assert isinstance(stats["depth"], int)
    assert stats["depth"] >= 0
    assert isinstance(stats["size"], int)
    assert stats["size"] >= 0
    assert isinstance(stats["cx"], int)
    assert stats["cx"] >= 0
    assert isinstance(stats["single_qubit_gates"], int)
    assert stats["single_qubit_gates"] >= 0

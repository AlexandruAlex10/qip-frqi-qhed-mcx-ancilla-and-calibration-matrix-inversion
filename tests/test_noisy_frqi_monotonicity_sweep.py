"""
Monotonicity of noisy FRQI metrics vs noise scale.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

pytest.importorskip("qiskit_aer")

import numpy as np

from src.frqi import required_position_qubits
from src.noise_models import build_noise_model, noisy_frqi_metrics_row


@pytest.mark.parametrize("method", ["naive", "vchain"])
def test_fidelity_nonincreasing_with_scale_4x4(method: str):
    img = np.load(os.path.join("data", "test_images", "test_4x4.npy"))
    _ = required_position_qubits(int(img.shape[0]))
    scales = (0.0, 0.05, 0.1, 0.15)
    fids: list[float] = []
    for s in scales:
        nm = build_noise_model(p_depol_1q=0.004 * s, p_depol_2q=0.02 * s)
        r = noisy_frqi_metrics_row(img, method=method, noise_model=nm, seed_simulator=0)
        fids.append(float(r["fidelity"]))
    for i in range(len(scales) - 1):
        assert fids[i] >= fids[i + 1] - 1e-6, (method, scales[i], fids[i], scales[i + 1], fids[i + 1])


def test_vchain_fewer_transpiled_cx_than_scaled_naive_at_8x8():
    """Structural CSV narrative: v-chain full prep CX < naive slice-scaled at N=8."""
    import pandas as pd

    path = os.path.join("outputs", "frqi_structural_metrics.csv")
    if not os.path.isfile(path):
        pytest.skip(f"Missing {path}; run scripts/frqi_structural_resources.py")
    df = pd.read_csv(path)
    v = df[(df["kind"] == "struct_vchain") & (df["image_size"] == 8)]["cx"].iloc[0]
    n = df[(df["kind"] == "struct_naive_slice_scaled") & (df["image_size"] == 8)]["cx"].iloc[0]
    assert int(v) < int(n)

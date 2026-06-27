"""Noisy FRQI (Aer density matrix): noiseless limit, monotonicity, readout demo smoke."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

pytest.importorskip("qiskit_aer")

import numpy as np

from src.frqi import reconstruct_image_from_reduced_density_matrix, required_position_qubits
from src.improved import build_frqi_prep_naive, build_frqi_prep_vchain, frqi_structural_num_qubits_vchain
from src.metrics import psnr
from src.noise_models import (
    build_noise_model,
    frqi_reduced_state_fidelity,
    noisy_frqi_metrics_row,
    reduced_frqi_density_matrix,
    run_density_matrix,
    run_readout_mitigation_slice_demo,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "test_images"


@pytest.mark.parametrize("factory", [build_frqi_prep_vchain, build_frqi_prep_naive])
def test_noisy_frqi_noiseless_density_matrix_fidelity(factory):
    img = np.load(DATA / "test_4x4.npy")
    m = required_position_qubits(int(img.shape[0]))
    qc = factory(img)
    total = qc.num_qubits
    nm = build_noise_model()
    rho = run_density_matrix(qc, nm, seed_simulator=0)
    fid = frqi_reduced_state_fidelity(rho, img, m=m, total_qubits=total)
    assert fid >= 1.0 - 1e-5


def test_reconstruct_from_reduced_dm_matches_statevector_rule_noiseless():
    img = np.load(DATA / "test_4x4.npy")
    n = int(img.shape[0])
    m = required_position_qubits(n)
    qc = build_frqi_prep_vchain(img)
    total = frqi_structural_num_qubits_vchain(m)
    rho = run_density_matrix(qc, build_noise_model(), seed_simulator=1)
    red = reduced_frqi_density_matrix(rho, m, total_qubits=total)
    recon = reconstruct_image_from_reduced_density_matrix(red, n)
    assert psnr(img, recon) >= 60.0


def test_depol_monotonic_fidelity_4x4_vchain():
    img = np.load(DATA / "test_4x4.npy")
    m = required_position_qubits(int(img.shape[0]))
    qc = build_frqi_prep_vchain(img)
    total = frqi_structural_num_qubits_vchain(m)
    probs = (0.002, 0.008, 0.020)
    fids: list[float] = []
    for p in probs:
        nm = build_noise_model(p_depol_1q=p, p_depol_2q=5.0 * p)
        rho = run_density_matrix(qc, nm, seed_simulator=123)
        fids.append(frqi_reduced_state_fidelity(rho, img, m=m, total_qubits=total))
    assert fids[0] > fids[1] > fids[2]


def test_readout_mitigation_slice_demo_smoke():
    img = np.load(DATA / "test_4x4.npy")
    m = required_position_qubits(int(img.shape[0]))
    nm = build_noise_model(readout_prob_01=0.06, readout_prob_10=0.06)
    raw, mit, conf, invc = run_readout_mitigation_slice_demo(
        m,
        address_bits=0,
        theta=float(np.pi / 4.0),
        mode="v-chain",
        noise_model=nm,
        shots=6000,
        seed_simulator=7,
    )
    assert 0.0 <= raw <= 1.0
    assert 0.0 <= mit <= 1.0
    assert conf.shape == (2, 2)
    assert invc.shape == (2, 2)

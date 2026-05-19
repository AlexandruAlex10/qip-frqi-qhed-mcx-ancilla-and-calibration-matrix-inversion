
"""Demo script for FRQI, QHED, and Sobel edge detection.

This script loads test images, constructs their FRQI statevectors,
reconstructs the images from the statevectors, and computes edge maps
using both the QHED baseline and classical Sobel method.
It then compares the results using MSE, PSNR, and similarity metrics,
saves comparison figures, FRQI reconstruction panels, and a CSV of metrics
including transpiled FRQI ``initialize`` circuit resources.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.frqi import (
    build_frqi_statevector,
    l2_error,
    maybe_build_qiskit_circuit,
    reconstruct_image_from_statevector
)
from src.metrics import psnr, ssim_like, ssim_uint8
from src.qhed import baseline_qhed_edge_map, classical_sobel_edge_map
from src.resources import transpiled_circuit_stats
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "test_images"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Transpiled resource semantics — exact state-prep via ``initialize``.
BASIS_GATES = ["cx", "rz", "sx"]
OPT_LEVEL = 3
FRQI_CIRCUIT_KIND = "initialize"


def _load(name: str) -> np.ndarray:
    return np.load(DATA / f"{name}.npy")


def run_case(name: str) -> dict:
    img = _load(name)
    n = int(img.shape[0])
    state = build_frqi_statevector(img)
    recon = reconstruct_image_from_statevector(state, n)
    edge_q = baseline_qhed_edge_map(img)
    edge_c = classical_sobel_edge_map(img)

    frqi_mse = l2_error(img, recon)
    frqi_psnr = psnr(img, recon)
    frqi_ssim = ssim_uint8(img, recon)
    qhed_sobel_ssim = ssim_uint8(edge_q, edge_c)

    print(f"=== {name} ===")
    print("FRQI reconstruction MSE:", frqi_mse)
    print("FRQI reconstruction PSNR:", frqi_psnr)
    print("FRQI reconstruction similarity (ssim_like):", ssim_like(img, recon))
    print("FRQI reconstruction SSIM (skimage):", frqi_ssim)
    print("QHED vs Sobel similarity (ssim_like):", ssim_like(edge_q, edge_c))
    print("QHED vs Sobel SSIM (skimage):", qhed_sobel_ssim)

    fig_cmp, axes = plt.subplots(1, 3, figsize=(8, 3))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(edge_q, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("QHED baseline")
    axes[2].imshow(edge_c, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Classical Sobel")
    for ax in axes:
        ax.axis("off")
    fig_cmp.tight_layout()
    fig_cmp.savefig(OUT / f"{name}_comparison.png", dpi=160)
    plt.close(fig_cmp)

    fig_recon, rax = plt.subplots(1, 2, figsize=(6, 3))
    rax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    rax[0].set_title("Original")
    rax[1].imshow(recon, cmap="gray", vmin=0, vmax=255)
    rax[1].set_title("FRQI reconstruction")
    for ax in rax:
        ax.axis("off")
    fig_recon.tight_layout()
    fig_recon.savefig(OUT / f"{name}_frqi_recon.png", dpi=160)
    plt.close(fig_recon)

    row: dict = {
        "image": name,
        "size": n,
        "frqi_mse": frqi_mse,
        "frqi_psnr": frqi_psnr,
        "frqi_ssim_skimage": frqi_ssim,
        "qhed_vs_sobel_ssim_skimage": qhed_sobel_ssim,
        "frqi_circuit_kind": FRQI_CIRCUIT_KIND,
        "basis_gates": ",".join(BASIS_GATES),
        "transpile_optimization_level": OPT_LEVEL,
        "frqi_qubits": np.nan,
        "frqi_depth_transpiled": np.nan,
        "frqi_cx_transpiled": np.nan,
        "frqi_transpiled_gate_count": np.nan,
    }

    try:
        qc = maybe_build_qiskit_circuit(img)
        stats = transpiled_circuit_stats(qc, BASIS_GATES, OPT_LEVEL)
        row["frqi_qubits"] = stats["num_qubits"]
        row["frqi_depth_transpiled"] = stats["depth"]
        row["frqi_cx_transpiled"] = stats["cx"]
        row["frqi_transpiled_gate_count"] = stats["size"]
    except RuntimeError:
        pass

    return row


if __name__ == "__main__":
    rows = []
    for case_name in ["test_4x4", "test_8x8", "test_16x16"]:
        rows.append(run_case(case_name))

    df = pd.DataFrame(rows)
    csv_path = OUT / "frqi_qhed_sobel_metrics.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved figures and {csv_path.name} to {OUT}")

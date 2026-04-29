
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.frqi import build_frqi_statevector, reconstruct_image_from_statevector, l2_error
from src.qhed import baseline_qhed_edge_map, classical_sobel_edge_map, qhed_pipeline
from src.metrics import psnr, ssim_like


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "test_images"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def _load(name: str) -> np.ndarray:
    return np.load(DATA / f"{name}.npy")


def run_case(name: str):
    img = _load(name)
    state = build_frqi_statevector(img)
    recon = reconstruct_image_from_statevector(state, img.shape[0])
    edge_q = baseline_qhed_edge_map(img)
    edge_c = classical_sobel_edge_map(img)

    print(f"=== {name} ===")
    print("FRQI reconstruction MSE:", l2_error(img, recon))
    print("FRQI reconstruction PSNR:", psnr(img, recon))
    print("FRQI reconstruction similarity:", ssim_like(img, recon))
    print("QHED vs Sobel similarity:", ssim_like(edge_q, edge_c))

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[1].imshow(edge_q, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("QHED baseline")
    axes[2].imshow(edge_c, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Classical Sobel")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for name in ["test_4x4", "test_8x8", "test_16x16"]:
        run_case(name)
    print(f"Saved figures to {OUT}")

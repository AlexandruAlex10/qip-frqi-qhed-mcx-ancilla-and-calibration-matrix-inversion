"""Edge-map quality after noisy FRQI reconstruction (classical QHED on recon vs Sobel on original).

This does **not** run a noisy quantum edge circuit; it evaluates how a noisy FRQI state
preparation (density matrix) degrades the **classical** QHED baseline applied to the
reconstructed image, using the classical Sobel magnitude map of the clean image as
reference.

Writes ``outputs/noisy_recon_qhed_edges.csv`` and one figure per image with PSNR/SSIM
of edge maps vs ``noise_scale``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.frqi import required_position_qubits  # noqa: E402
from src.improved import frqi_structural_num_qubits_naive_slice, frqi_structural_num_qubits_vchain  # noqa: E402
from src.metrics import psnr, ssim_uint8  # noqa: E402
from src.noise_models import build_noise_model, noisy_frqi_metrics_row  # noqa: E402
from src.qhed import baseline_qhed_edge_map, classical_sobel_edge_map  # noqa: E402

DATA = ROOT / "data" / "test_images"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

IMAGES_DEFAULT = "test_4x4,test_8x8,test_16x16"


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _dm_qubits_for_method(img: np.ndarray, method: str) -> int:
    n = int(img.shape[0])
    m = required_position_qubits(n)
    if method == "vchain":
        return frqi_structural_num_qubits_vchain(m)
    if method == "naive":
        return frqi_structural_num_qubits_naive_slice(m)
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser(description="Noisy recon → QHED edges vs Sobel on original.")
    ap.add_argument("--images", default=IMAGES_DEFAULT, help="Comma-separated test image stems.")
    ap.add_argument("--methods", default="naive,vchain", help="naive,vchain")
    ap.add_argument("--scales", default="0,0.05,0.1,0.15,0.2", help="Noise scale grid (same as noisy_frqi_sweep).")
    ap.add_argument("--dm-max-qubits", type=int, default=14, help="Skip pair if qubit count exceeds this.")
    ap.add_argument("--allow-heavy-dm", action="store_true", help="Disable qubit-count cap.")
    ap.add_argument("--p1", type=float, default=0.004)
    ap.add_argument("--p2", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = [x.strip() for x in args.images.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    scales = _parse_float_list(args.scales)
    max_q = 10_000 if args.allow_heavy_dm else int(args.dm_max_qubits)

    rows: list[dict] = []
    for img_name in images:
        img = np.load(DATA / f"{img_name}.npy")
        ref_edges = classical_sobel_edge_map(img)
        for s in scales:
            nm = build_noise_model(p_depol_1q=args.p1 * s, p_depol_2q=args.p2 * s)
            for method in methods:
                nq = _dm_qubits_for_method(img, method)
                if nq > max_q:
                    print(f"Skip {img_name} {method}: {nq} qubits > {max_q}")
                    continue
                r = noisy_frqi_metrics_row(
                    img,
                    method=method,
                    noise_model=nm,
                    seed_simulator=args.seed,
                    return_recon=True,
                )
                recon = r.pop("recon")
                qhed_edges = baseline_qhed_edge_map(recon)
                ep = psnr(ref_edges.astype(np.float64), qhed_edges.astype(np.float64))
                try:
                    es = ssim_uint8(ref_edges, qhed_edges)
                except ValueError:
                    es = float("nan")
                rows.append(
                    {
                        "image": img_name,
                        "method": method,
                        "noise_scale": s,
                        "edge_psnr": ep,
                        "edge_ssim": es,
                        **{k: v for k, v in r.items() if k != "recon"},
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = OUT / "noisy_recon_qhed_edges.csv"
    df.to_csv(csv_path, index=False)

    for img_name in images:
        sub = df[df["image"] == img_name]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for method in methods:
            msub = sub[sub["method"] == method].sort_values("noise_scale")
            if msub.empty:
                continue
            axes[0].plot(msub["noise_scale"], msub["edge_psnr"], marker="o", label=method)
            axes[1].plot(msub["noise_scale"], msub["edge_ssim"], marker="o", label=method)
        axes[0].set_title("Edge-map PSNR (QHED(recon) vs Sobel(orig))")
        axes[0].set_xlabel("noise scale")
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].set_title("Edge-map SSIM")
        axes[1].set_xlabel("noise scale")
        axes[1].set_ylabel("SSIM")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.suptitle(f"Noisy recon → QHED edges — {img_name}")
        fig.tight_layout()
        fig.savefig(OUT / f"noisy_recon_qhed_edges_{img_name}.png", dpi=160)
        plt.close(fig)

    print("Wrote", csv_path)


if __name__ == "__main__":
    main()

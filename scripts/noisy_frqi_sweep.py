"""
Sweep noisy FRQI preparation (naive vs v-chain): CSV + matplotlib curves.

Uses density-matrix simulation for depolarizing / thermal gate noise. Readout parameters
scale with ``--scales`` for API consistency; readout does not affect the saved-state
density matrix unless you use the optional ``--readout-demo`` shot branch.

Default (``--images``): ``test_4x4,test_8x8,test_16x16`` with the same ``--scales``
grid for every image that is actually simulated.

**16×16 and Aer density matrices:** v-chain FRQI uses ``m+2+max(0,m-2)`` qubits (16 for
``N=16``). A full density matrix then has dimension ``2^16``, i.e. ``(2^16)^2`` complex
numbers — often tens of GiB of RAM and long wall times. This script therefore skips any
``(image, method)`` pair whose simulated qubit count exceeds ``--dm-max-qubits`` (default
14), unless ``--allow-heavy-dm`` is set. In practice, ``test_16x16`` + ``vchain`` is
skipped by default while ``test_16x16`` + ``naive`` (10 qubits) still runs. For a full
16×16 v-chain grid on a large machine, run with ``--allow-heavy-dm`` and prefer a **coarser**
``--scales`` list (for example ``0,0.1,0.2``) to limit total simulations.
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
from src.noise_models import (  # noqa: E402
    build_noise_model,
    noisy_frqi_metrics_row,
    run_readout_mitigation_slice_demo,
)

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
    ap = argparse.ArgumentParser(
        description="Noisy FRQI metrics sweep (density matrix).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--images",
        default=IMAGES_DEFAULT,
        help=f"Comma-separated stems under data/test_images (without .npy). Default: {IMAGES_DEFAULT!r}.",
    )
    ap.add_argument("--methods", default="naive,vchain", help="Comma-separated: naive,vchain")
    ap.add_argument(
        "--scales",
        default="0,0.05,0.1,0.15,0.2",
        help="Noise scale factors applied to baseline depolarizing / readout knobs.",
    )
    ap.add_argument(
        "--dm-max-qubits",
        type=int,
        default=14,
        help="Skip (image, method) if simulated qubit count exceeds this (RAM safety). Default 14.",
    )
    ap.add_argument(
        "--allow-heavy-dm",
        action="store_true",
        help="Disable the qubit-count skip (may require huge RAM for 16×16 v-chain).",
    )
    ap.add_argument("--p1", type=float, default=0.004, help="Baseline 1Q depolarizing at scale 1.")
    ap.add_argument("--p2", type=float, default=0.02, help="Baseline 2Q depolarizing at scale 1.")
    ap.add_argument("--t1", type=float, default=None, help="T1 (same unit as gate times).")
    ap.add_argument("--t2", type=float, default=None, help="T2 (same unit as gate times).")
    ap.add_argument("--tg1", type=float, default=None, help="Single-qubit gate time.")
    ap.add_argument("--tg2", type=float, default=None, help="Ignored by build_noise_model (API hook).")
    ap.add_argument("--r01", type=float, default=0.0, help="Baseline readout P(1|0) at scale 1.")
    ap.add_argument("--r10", type=float, default=0.0, help="Baseline readout P(0|1) at scale 1.")
    ap.add_argument("--seed", type=int, default=42, help="Simulator seed (determinism).")
    ap.add_argument(
        "--readout-demo",
        action="store_true",
        help="Run a small shot-based readout calibration + matrix inversion demo on a slice.",
    )
    args = ap.parse_args()

    images = [x.strip() for x in args.images.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    scales = _parse_float_list(args.scales)
    max_q = 10_000 if args.allow_heavy_dm else int(args.dm_max_qubits)

    rows: list[dict] = []
    for img_name in images:
        img = np.load(DATA / f"{img_name}.npy")
        for s in scales:
            nm = build_noise_model(
                p_depol_1q=args.p1 * s,
                p_depol_2q=args.p2 * s,
                t1=args.t1,
                t2=args.t2,
                gate_time_1q=args.tg1,
                gate_time_2q=args.tg2,
                readout_prob_01=args.r01 * s,
                readout_prob_10=args.r10 * s,
            )
            for method in methods:
                nq = _dm_qubits_for_method(img, method)
                if nq > max_q:
                    print(
                        f"Skip {img_name} {method}: {nq} qubits > dm-max {max_q}. "
                        f"Use --allow-heavy-dm or raise --dm-max-qubits."
                    )
                    continue
                r = noisy_frqi_metrics_row(
                    img,
                    method=method,
                    noise_model=nm,
                    seed_simulator=args.seed,
                )
                rows.append(
                    {
                        "image": img_name,
                        "method": method,
                        "noise_scale": s,
                        "p_depol_1q": args.p1 * s,
                        "p_depol_2q": args.p2 * s,
                        "t1": args.t1 if args.t1 is not None else "",
                        "t2": args.t2 if args.t2 is not None else "",
                        "readout_prob_01": args.r01 * s,
                        "readout_prob_10": args.r10 * s,
                        **r,
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = OUT / "noisy_frqi_metrics.csv"
    df.to_csv(csv_path, index=False)

    for img_name in images:
        sub = df[df["image"] == img_name]
        if sub.empty:
            print(f"No rows for {img_name}; skipping figure.")
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for method in methods:
            msub = sub[sub["method"] == method].sort_values("noise_scale")
            if msub.empty:
                continue
            axes[0].plot(msub["noise_scale"], msub["fidelity"], marker="o", label=method)
            psnr_vals = msub["psnr"].mask(np.isinf(msub["psnr"]))
            axes[1].plot(msub["noise_scale"], psnr_vals, marker="o", label=method)
            axes[2].plot(msub["noise_scale"], msub["ssim"], marker="o", label=method)
        axes[0].set_title("Fidelity vs noise scale")
        axes[0].set_xlabel("scale")
        axes[0].set_ylabel("fidelity")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].set_title("PSNR vs noise scale")
        axes[1].set_xlabel("scale")
        axes[1].set_ylabel("PSNR (dB)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[2].set_title("SSIM vs noise scale")
        axes[2].set_xlabel("scale")
        axes[2].set_ylabel("SSIM")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        fig.suptitle(f"Noisy FRQI — {img_name}")
        fig.tight_layout()
        fig.savefig(OUT / f"noisy_frqi_{img_name}_curves.png", dpi=160)
        plt.close(fig)

    if args.readout_demo:
        img = np.load(DATA / "test_4x4.npy")
        m = required_position_qubits(int(img.shape[0]))
        nm_ro = build_noise_model(readout_prob_01=0.08, readout_prob_10=0.05)
        raw, mit, conf, _ = run_readout_mitigation_slice_demo(
            m,
            address_bits=0,
            theta=float(np.pi / 4.0),
            mode="v-chain",
            noise_model=nm_ro,
            shots=24000,
            seed_simulator=args.seed,
        )
        print("Readout mitigation demo (color qubit, single structural slice):")
        print("  empirical confusion (rows=measured, cols=true):\n", conf)
        print("  P(meas=1) raw:", raw, " mitigated:", mit)

    print("Wrote", csv_path)


if __name__ == "__main__":
    main()

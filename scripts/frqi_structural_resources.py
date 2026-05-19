"""
Emit transpiled resource stats for structural FRQI circuits (CSV + figures).

Uses the same basis and optimization level as ``scripts/frqi_qhed_sobel.py``:
``basis_gates = ['cx', 'rz', 'sx']``, ``optimization_level = 3``.

Rows include:
- ``struct_vchain``: full :func:`src.improved.build_frqi_prep_vchain` per test image.
- ``struct_naive_full``: full :func:`src.improved.build_frqi_prep_naive` (4×4 only; scales poorly).
- ``struct_naive_slice``: one representative ``noancilla`` MCX + ``cry`` slice at address 0,
  angle ``theta = pi/4`` (transpiled ``cry`` cost is angle-dependent; the MCX skeleton is shared).
- ``struct_naive_slice_scaled``: integer fields ``depth``, ``cx``, ``size`` multiplied by
  ``N_pix = 2**m`` for linear extrapolation (same caveat as the thesis design note).

After writing ``outputs/frqi_structural_metrics.csv``, plots
``outputs/fig_cx_vs_image_size.png`` and ``outputs/fig_depth_vs_image_size.png`` unless
``--no-plot`` is set. ``--plot-only`` reads an existing CSV and only emits figures.
"""

from __future__ import annotations

import argparse
import sys
from math import pi
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.frqi import required_position_qubits  # noqa: E402
from src.improved import (  # noqa: E402
    build_frqi_prep_naive,
    build_frqi_prep_vchain,
    build_single_address_ry_slice,
)
from src.resources import transpiled_circuit_stats  # noqa: E402

BASIS_GATES = ["cx", "rz", "sx"]
OPT_LEVEL = 3
DATA = ROOT / "data" / "test_images"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def _row(
    *,
    image: str,
    size: int,
    kind: str,
    qc,
    notes: str = "",
    scale: int = 1,
) -> dict:
    stats = transpiled_circuit_stats(qc, BASIS_GATES, OPT_LEVEL)
    nq = int(stats["num_qubits"])
    return {
        "image": image,
        "image_size": size,
        "m": required_position_qubits(size),
        "kind": kind,
        "notes": notes,
        "scale": scale,
        "basis_gates": ",".join(BASIS_GATES),
        "transpile_optimization_level": OPT_LEVEL,
        "num_qubits": nq,
        "depth": int(stats["depth"]) * scale,
        "cx": int(stats["cx"]) * scale,
        "transpiled_gate_count": int(stats["size"]) * scale,
        "single_qubit_gates": int(stats["single_qubit_gates"]) * scale,
    }


def plot_structural_metrics_csv(csv_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Plot transpiled CX and depth vs linear image side length from ``frqi_structural_metrics.csv``."""
    df = pd.read_csv(csv_path)
    vc = df[df["kind"] == "struct_vchain"].sort_values("image_size")
    scaled = df[df["kind"] == "struct_naive_slice_scaled"].sort_values("image_size")
    if vc.empty or scaled.empty:
        raise ValueError(f"CSV {csv_path} missing struct_vchain or struct_naive_slice_scaled rows.")

    x = vc["image_size"].to_numpy(dtype=float)
    naive_full = df[df["kind"] == "struct_naive_full"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in (
        (axes[0], "cx", "Transpiled CX vs image size"),
        (axes[1], "depth", "Transpiled depth vs image size"),
    ):
        ax.plot(x, vc[col], marker="o", label="v-chain (full prep)")
        ax.plot(scaled["image_size"], scaled[col], marker="s", linestyle="--", label="naive (slice × N_pix)")
        if not naive_full.empty and col in naive_full.columns:
            row = naive_full.iloc[0]
            ax.scatter([row["image_size"]], [row[col]], marker="D", s=80, zorder=5, label="naive full (4×4 only)")
        ax.set_xlabel("Image side length N (pixels)")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Structural FRQI (basis cx,rz,sx, optimization_level=3)")
    fig.tight_layout()
    cx_path = out_dir / "fig_cx_vs_image_size.png"
    depth_path = out_dir / "fig_depth_vs_image_size.png"
    # First panel is CX; save combined figure for CX file name used in thesis, and a second file for depth-only reuse.
    fig.savefig(cx_path, dpi=160)
    plt.close(fig)

    # Second standalone depth figure (same data as right panel) for explicit thesis path naming
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.plot(x, vc["depth"], marker="o", label="v-chain (full prep)")
    ax2.plot(scaled["image_size"], scaled["depth"], marker="s", linestyle="--", label="naive (slice × N_pix)")
    if not naive_full.empty:
        row = naive_full.iloc[0]
        ax2.scatter([row["image_size"]], [row["depth"]], marker="D", s=80, zorder=5, label="naive full (4×4 only)")
    ax2.set_xlabel("Image side length N (pixels)")
    ax2.set_ylabel("depth")
    ax2.set_title("Transpiled depth vs image size")
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(depth_path, dpi=160)
    plt.close(fig2)

    print(f"Wrote {cx_path} (combined CX+depth panels) and {depth_path}")
    return cx_path, depth_path


def _emit_csv() -> Path:
    rows: list[dict] = []
    for case in ("test_4x4", "test_8x8", "test_16x16"):
        img = np.load(DATA / f"{case}.npy")
        n = int(img.shape[0])
        rows.append(
            _row(
                image=case,
                size=n,
                kind="struct_vchain",
                qc=build_frqi_prep_vchain(img),
                notes="full prep + H on position",
            )
        )

    img4 = np.load(DATA / "test_4x4.npy")
    rows.append(
        _row(
            image="test_4x4",
            size=4,
            kind="struct_naive_full",
            qc=build_frqi_prep_naive(img4),
            notes="full naive prep (4×4 only)",
        )
    )

    for n in (4, 8, 16):
        m = required_position_qubits(n)
        n_pix = 2**m
        sl = build_single_address_ry_slice(m, 0, pi / 4.0, mode="noancilla")
        base = _row(
            image=f"test_{n}x{n}",
            size=n,
            kind="struct_naive_slice",
            qc=sl,
            notes="one address-0 slice, theta=pi/4",
        )
        rows.append(base)
        scaled = dict(base)
        scaled["kind"] = "struct_naive_slice_scaled"
        scaled["notes"] = f"slice stats × N_pix={n_pix} (linear MCX scaling estimate)"
        scaled["scale"] = n_pix
        for k in ("depth", "cx", "transpiled_gate_count", "single_qubit_gates"):
            scaled[k] = int(base[k]) * n_pix
        rows.append(scaled)

    df = pd.DataFrame(rows)
    path = OUT / "frqi_structural_metrics.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Structural FRQI transpiled metrics (CSV + plots).")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip circuit transpilation; read outputs/frqi_structural_metrics.csv and plot only.",
    )
    ap.add_argument("--no-plot", action="store_true", help="Emit CSV only.")
    args = ap.parse_args()

    csv_path = OUT / "frqi_structural_metrics.csv"
    if args.plot_only:
        if not csv_path.is_file():
            raise SystemExit(f"Missing {csv_path}; run without --plot-only first.")
        if not args.no_plot:
            plot_structural_metrics_csv(csv_path, OUT)
        return

    _emit_csv()
    if not args.no_plot:
        plot_structural_metrics_csv(csv_path, OUT)


if __name__ == "__main__":
    main()

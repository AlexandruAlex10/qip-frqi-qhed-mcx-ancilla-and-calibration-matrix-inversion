"""Emit transpiled resource stats for structural FRQI circuits (CSV).

Uses the same basis and optimization level as ``scripts/demo_frqi_qhed_sobel.py``:
``basis_gates = ['cx', 'rz', 'sx']``, ``optimization_level = 3``.

Rows include:
- ``struct_vchain``: full :func:`src.improved.build_frqi_prep_vchain` per test image.
- ``struct_naive_full``: full :func:`src.improved.build_frqi_prep_naive` (4×4 only; scales poorly).
- ``struct_naive_slice``: one representative ``noancilla`` MCX + ``cry`` slice at address 0,
  angle ``theta = pi/4`` (transpiled ``cry`` cost is angle-dependent; the MCX skeleton is shared).
- ``struct_naive_slice_scaled``: integer fields ``depth``, ``cx``, ``size`` multiplied by
  ``N_pix = 2**m`` for linear extrapolation (same caveat as the thesis design note).
"""

from __future__ import annotations

import sys
from math import pi
from pathlib import Path

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


def main() -> None:
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


if __name__ == "__main__":
    main()

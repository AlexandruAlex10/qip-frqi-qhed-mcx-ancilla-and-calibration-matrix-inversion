"""Readout calibration + linear-inversion mitigation on a single structural FRQI slice.

Expands ``run_readout_mitigation_slice_demo`` into a small study over multiple
simulator seeds, with separate calibration/data shot budgets and optional
bootstrap confidence bands for the mitigated ``P(meas=1)``. Writes a CSV, a
figure, and an ``experiment_manifest_*.json`` sidecar under ``outputs/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.transpiler import CouplingMap  # noqa: E402

from src.experiment_manifest import build_base_manifest, write_manifest_json  # noqa: E402
from src.frqi import required_position_qubits  # noqa: E402
from src.improved import build_single_address_ry_slice  # noqa: E402
from src.nisq_mock import linear_coupling_edges, load_named_fake_backend_v2, make_generic_linear_backend  # noqa: E402
from src.noise_models import (  # noqa: E402
    build_noise_model,
    counts_to_prob_vector,
    estimate_single_qubit_readout_matrix,
    invert_readout_matrix,
    mitigate_linear,
    run_shots_counts,
    transpile_for_aer_noise,
)

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _resolve_backend(mock_backend: str, nq: int, *, seed: int) -> Tuple[Any, str]:
    if mock_backend == "generic_linear":
        be = make_generic_linear_backend(nq, seed=seed)
        return be, f"GenericBackendV2(linear,n={nq},seed={seed})"
    be = load_named_fake_backend_v2(mock_backend)
    if int(be.num_qubits) < nq:
        raise ValueError(f"{mock_backend} too small for nq={nq}; use generic_linear.")
    return be, f"{mock_backend}(n_backend={int(be.num_qubits)})"


def _build_transpile_kwargs(
    *,
    topology: str,
    nq: int,
    mock_backend: str,
    mock_backend_seed: int,
    optimization_level: int,
    transpile_seed: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta: Dict[str, Any] = {"topology": topology, "mock_backend": mock_backend, "mock_backend_seed": int(mock_backend_seed)}
    if topology == "full":
        return (
            {
                "transpile_backend": None,
                "transpile_coupling_map": None,
                "transpile_basis_gates": ("id", "rz", "sx", "x", "cx"),
                "transpile_optimization_level": int(optimization_level),
                "seed_transpiler": transpile_seed,
            },
            meta,
        )
    if topology == "mock_backend":
        be, bid = _resolve_backend(mock_backend, nq, seed=int(mock_backend_seed))
        meta["mock_backend_resolved"] = bid
        return (
            {
                "transpile_backend": be,
                "transpile_coupling_map": None,
                "transpile_basis_gates": None,
                "transpile_optimization_level": int(optimization_level),
                "seed_transpiler": transpile_seed,
            },
            meta,
        )
    cmap = CouplingMap(linear_coupling_edges(nq))
    return (
        {
            "transpile_backend": None,
            "transpile_coupling_map": cmap,
            "transpile_basis_gates": ("id", "rz", "sx", "x", "cx"),
            "transpile_optimization_level": int(optimization_level),
            "seed_transpiler": transpile_seed,
        },
        meta,
    )


def _bootstrap_mitigated_p1(
    *,
    shots: int,
    p0_hat: float,
    p1_hat: float,
    inv_matrix: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap mitigated P(meas=1) using multinomial resampling of outcome counts (fixed inverse)."""
    p = np.asarray([p0_hat, p1_hat], dtype=np.float64)
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s <= 0:
        return np.zeros(int(n_boot), dtype=np.float64)
    p = p / s

    out = np.zeros(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        c0, c1 = rng.multinomial(int(shots), p)
        probs = np.asarray([c0, c1], dtype=np.float64) / float(shots)
        mit = mitigate_linear(probs, inv_matrix)
        out[b] = float(mit[1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Readout mitigation shot sweep (multi-seed + optional bootstrap).")
    ap.add_argument("--image", default="test_4x4", help="Stem under data/test_images (without .npy).")
    ap.add_argument("--topology", choices=("full", "linear", "mock_backend"), default="full")
    ap.add_argument("--mock-backend", default="generic_linear")
    ap.add_argument("--mock-backend-seed", type=int, default=42)
    ap.add_argument("--transpile-optimization-level", type=int, default=0)
    ap.add_argument("--transpile-seed", type=int, default=None)
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated Aer shot seeds.")
    ap.add_argument("--shots-cal", type=int, default=8000, help="Shots per calibration column (|0>,|1> prep).")
    ap.add_argument("--shots-data", type=int, default=12000, help="Shots for the structural slice measurement.")
    ap.add_argument("--r01", type=float, default=0.08)
    ap.add_argument("--r10", type=float, default=0.05)
    ap.add_argument("--rcond", type=float, default=1e-6, help="rcond for numpy.linalg.pinv on readout matrix.")
    ap.add_argument("--bootstrap", type=int, default=400, help="Bootstrap resamples (0 disables).")
    ap.add_argument("--bootstrap-seed", type=int, default=999, help="RNG seed for bootstrap resampling.")
    ap.add_argument("--csv-name", default="readout_mitigation_shot_sweep.csv")
    ap.add_argument("--manifest-name", default="experiment_manifest_readout_mitigation_shot_sweep.json")
    args = ap.parse_args()

    img = np.load(ROOT / "data" / "test_images" / f"{args.image}.npy")
    m = required_position_qubits(int(img.shape[0]))
    base0 = build_single_address_ry_slice(m, 0, float(np.pi / 4.0), mode="v-chain")
    nq = int(base0.num_qubits)

    nm = build_noise_model(readout_prob_01=float(args.r01), readout_prob_10=float(args.r10))
    tkwargs, sweep_meta = _build_transpile_kwargs(
        topology=str(args.topology),
        nq=nq,
        mock_backend=str(args.mock_backend),
        mock_backend_seed=int(args.mock_backend_seed),
        optimization_level=int(args.transpile_optimization_level),
        transpile_seed=args.transpile_seed,
    )

    seeds = _parse_int_list(args.seeds)
    rng = np.random.default_rng(int(args.bootstrap_seed))

    rows: list[dict] = []

    for sd in seeds:
        confusion = estimate_single_qubit_readout_matrix(
            nq,
            0,
            nm,
            shots=int(args.shots_cal),
            seed_simulator=int(sd),
            **tkwargs,
        )
        inv_c = invert_readout_matrix(confusion, rcond=float(args.rcond))

        base = build_single_address_ry_slice(m, 0, float(np.pi / 4.0), mode="v-chain")
        qc = QuantumCircuit(nq, 1)
        qc.compose(base, qubits=list(range(nq)), inplace=True)
        qc.measure(0, 0)
        bg = tuple(tkwargs.get("transpile_basis_gates") or ("id", "rz", "sx", "x", "cx"))
        tqc = transpile_for_aer_noise(
            qc,
            basis_gates=bg,
            optimization_level=int(tkwargs["transpile_optimization_level"]),
            coupling_map=tkwargs.get("transpile_coupling_map"),
            backend=tkwargs.get("transpile_backend"),
            seed_transpiler=tkwargs.get("seed_transpiler"),
        )
        cts = run_shots_counts(
            tqc,
            nm,
            shots=int(args.shots_data),
            seed_simulator=int(sd),
            transpile_first=False,
        )
        probs = counts_to_prob_vector(cts, 1, int(args.shots_data))
        mit = mitigate_linear(probs, inv_c)

        p1_raw = float(probs[1])
        p1_mit = float(mit[1])

        row = {
            "image": args.image,
            "seed": int(sd),
            "shots_cal": int(args.shots_cal),
            "shots_data": int(args.shots_data),
            "rcond": float(args.rcond),
            "p1_raw": p1_raw,
            "p1_mitigated": p1_mit,
            "topology": args.topology,
            "mock_backend": args.mock_backend,
            "transpile_optimization_level": int(args.transpile_optimization_level),
        }
        for i in range(2):
            for j in range(2):
                row[f"A_{i}{j}"] = float(confusion[i, j])

        if int(args.bootstrap) > 0:
            boot = _bootstrap_mitigated_p1(
                shots=int(args.shots_data),
                p0_hat=float(probs[0]),
                p1_hat=float(probs[1]),
                inv_matrix=inv_c,
                n_boot=int(args.bootstrap),
                rng=rng,
            )
            row["p1_mitigated_boot_p05"] = float(np.quantile(boot, 0.05))
            row["p1_mitigated_boot_p50"] = float(np.quantile(boot, 0.50))
            row["p1_mitigated_boot_p95"] = float(np.quantile(boot, 0.95))
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUT / str(args.csv_name)
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    xs = np.arange(len(seeds), dtype=float)
    ax.plot(xs, df["p1_raw"], marker="o", label="P1 raw")
    ax.plot(xs, df["p1_mitigated"], marker="o", label="P1 mitigated")
    if int(args.bootstrap) > 0:
        ax.fill_between(
            xs,
            df["p1_mitigated_boot_p05"],
            df["p1_mitigated_boot_p95"],
            alpha=0.2,
            label="Mitigated bootstrap 5–95%",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(s)) for s in seeds], rotation=45, ha="right")
    ax.set_xlabel("seed_simulator")
    ax.set_ylabel("probability")
    ax.set_title(f"Readout mitigation slice demo — {args.image} (topology={args.topology})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path = OUT / "readout_mitigation_shot_sweep.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    manifest = build_base_manifest(
        script="scripts/readout_mitigation_shot_sweep.py",
        argv=sys.argv[1:],
        repo_root=ROOT,
        extra={
            "outputs": {
                "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                "figure": str(fig_path.relative_to(ROOT)).replace("\\", "/"),
            },
            "readout": {
                "r01": float(args.r01),
                "r10": float(args.r10),
                "shots_cal": int(args.shots_cal),
                "shots_data": int(args.shots_data),
                "rcond": float(args.rcond),
                "bootstrap": int(args.bootstrap),
                "bootstrap_seed": int(args.bootstrap_seed),
            },
            "transpile": sweep_meta,
            "seeds": {"shot_seeds": seeds, "transpile": args.transpile_seed},
        },
    )
    write_manifest_json(OUT / str(args.manifest_name), manifest)

    print("Wrote", csv_path)
    print("Wrote", fig_path)
    print("Wrote", OUT / str(args.manifest_name))


if __name__ == "__main__":
    main()

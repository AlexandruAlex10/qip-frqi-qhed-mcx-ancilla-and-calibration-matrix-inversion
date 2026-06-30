"""
Sweep noisy FRQI preparation (naive vs v-chain): CSV + matplotlib curves.

Uses density-matrix simulation for depolarizing / thermal gate noise. Readout parameters
scale with ``--scales`` for API consistency; readout does not affect the saved-state
density matrix unless you use the optional ``--readout-demo`` shot branch.

Default (``--images``): ``test_4x4,test_8x8,test_16x16`` with the same ``--scales``
grid for every image that is actually simulated.

**Mock NISQ + layout:** use ``--noise-mode from_backend`` with
``--mock-backend generic_linear`` (default) to build ``NoiseModel.from_backend`` on a
``GenericBackendV2`` line topology with ``num_qubits`` matching each circuit, and
transpile with ``--transpile-backend`` semantics (``optimization_level`` recommended: 3).
In ``from_backend`` mode, ``--scales`` is ignored (a single snapshot is used).

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
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from qiskit.transpiler import CouplingMap  # noqa: E402

from src.experiment_manifest import build_base_manifest, write_manifest_json  # noqa: E402
from src.frqi import required_position_qubits  # noqa: E402
from src.improved import (  # noqa: E402
    build_single_address_ry_slice,
    frqi_structural_num_qubits_naive_slice,
    frqi_structural_num_qubits_vchain,
)
from src.nisq_mock import (  # noqa: E402
    build_noise_model_from_preset,
    linear_coupling_edges,
    load_named_fake_backend_v2,
    load_yaml_preset,
    make_generic_linear_backend,
    noise_model_from_backend,
)
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


def _resolve_backend_for_nqubits(mock_backend: str, nq: int, *, seed: int) -> Tuple[Any, str]:
    if mock_backend == "generic_linear":
        be = make_generic_linear_backend(nq, seed=seed)
        bid = f"GenericBackendV2(linear,n={nq},seed={seed})"
        return be, bid
    be = load_named_fake_backend_v2(mock_backend)
    n_backend = int(be.num_qubits)
    if n_backend < nq:
        raise ValueError(
            f"Mock backend {mock_backend!r} has {n_backend} qubits but the circuit needs {nq}. "
            f"Use --mock-backend generic_linear for arbitrary widths."
        )
    bid = f"{mock_backend}(n_backend={n_backend})"
    return be, bid


def _transpile_kwargs_from_backend(
    be: Any,
    bid: str,
    *,
    optimization_level: int,
    transpile_seed: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "noise_mode": "from_backend",
        "topology": "mock_backend",
        "mock_backend_resolved": bid,
        "transpile_optimization_level": int(optimization_level),
        "seed_transpiler": transpile_seed,
    }
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


def _transpile_kwargs_for_sweep(
    *,
    noise_mode: str,
    topology: str,
    nq: int,
    mock_backend: str,
    mock_backend_seed: int,
    optimization_level: int,
    transpile_seed: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (noisy_frqi_metrics_row kwargs subset, manifest-ish transpile fields)."""
    meta: Dict[str, Any] = {
        "noise_mode": noise_mode,
        "topology": topology,
        "mock_backend": mock_backend,
        "mock_backend_seed": int(mock_backend_seed),
        "transpile_optimization_level": int(optimization_level),
        "seed_transpiler": transpile_seed,
    }

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
        help="Noise scale factors applied to baseline depolarizing / readout knobs (synthetic/yaml). Ignored for from_backend.",
    )
    ap.add_argument(
        "--noise-mode",
        choices=("synthetic", "from_backend", "yaml_preset"),
        default="synthetic",
        help="synthetic: hand-built Aer noise; from_backend: NoiseModel.from_backend(mock); yaml_preset: data/experiment_presets/*.yaml",
    )
    ap.add_argument(
        "--topology",
        choices=("full", "linear"),
        default="full",
        help="Transpiler coupling topology for synthetic/yaml_preset modes (ignored for from_backend, which uses the mock backend graph).",
    )
    ap.add_argument(
        "--mock-backend",
        default="generic_linear",
        help="For --noise-mode from_backend: generic_linear or a Fake*V2 class name from qiskit.providers.fake_provider (must fit qubit count).",
    )
    ap.add_argument("--mock-backend-seed", type=int, default=42, help="Seed for GenericBackendV2 calibration sampling.")
    ap.add_argument(
        "--yaml-preset",
        type=str,
        default="",
        help="Path to YAML preset (repo-relative or absolute). Required for yaml_preset noise mode.",
    )
    ap.add_argument(
        "--transpile-optimization-level",
        type=int,
        default=0,
        help="Qiskit transpiler optimization_level (3 matches scripts/frqi_structural_resources.py).",
    )
    ap.add_argument(
        "--transpile-seed",
        type=int,
        default=None,
        help="Optional seed_transpiler for deterministic layout/routing when not using trivial layout.",
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
    ap.add_argument(
        "--csv-name",
        default="noisy_frqi_metrics.csv",
        help="Output CSV filename under outputs/ (default keeps thesis paths stable).",
    )
    ap.add_argument(
        "--manifest-name",
        default="experiment_manifest_noisy_frqi_sweep.json",
        help="Provenance JSON filename under outputs/.",
    )
    ap.add_argument(
        "--mock-nisq-bundle",
        action="store_true",
        help="Convenience: run from_backend + opt=3 + write noisy_frqi_metrics_mock_nisq_bundle.csv + matching PNGs + manifest.",
    )
    args = ap.parse_args()

    if args.mock_nisq_bundle:
        args.noise_mode = "from_backend"
        args.mock_backend = "generic_linear"
        args.transpile_optimization_level = 3
        args.csv_name = "noisy_frqi_metrics_mock_nisq_bundle.csv"
        args.manifest_name = "experiment_manifest_noisy_frqi_mock_nisq_bundle.json"

    if args.noise_mode == "yaml_preset" and not str(args.yaml_preset).strip():
        raise SystemExit("--yaml-preset is required for --noise-mode yaml_preset")

    preset_doc: Optional[Dict[str, Any]] = None
    if args.noise_mode == "yaml_preset":
        preset_path = Path(args.yaml_preset)
        if not preset_path.is_file():
            cand = ROOT / args.yaml_preset
            if cand.is_file():
                preset_path = cand
            else:
                raise SystemExit(f"Missing YAML preset: {args.yaml_preset}")
        preset_doc = load_yaml_preset(preset_path)

    images = [x.strip() for x in args.images.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    scales = _parse_float_list(args.scales)
    max_q = 10_000 if args.allow_heavy_dm else int(args.dm_max_qubits)

    if args.noise_mode == "from_backend":
        if not args.mock_nisq_bundle and scales != [1.0]:
            print("Note: --noise-mode from_backend ignores noise_scale sweeps; forcing scales=[1.0].", flush=True)
        scales = [1.0]

    total_steps = len(images) * len(scales) * len(methods)
    print(
        f"Starting noisy FRQI sweep: {len(images)} image(s) x {len(scales)} scale(s) x "
        f"{len(methods)} method(s) = {total_steps} planned simulation(s) "
        f"[noise_mode={args.noise_mode}].",
        flush=True,
    )

    rows: list[dict] = []
    step = 0
    for img_name in images:
        img = np.load(DATA / f"{img_name}.npy")
        for s in scales:
            for method in methods:
                step += 1
                nq = _dm_qubits_for_method(img, method)
                if nq > max_q:
                    print(
                        f"[{step}/{total_steps}] Skip {img_name} {method}: {nq} qubits > dm-max {max_q}. "
                        f"Use --allow-heavy-dm or raise --dm-max-qubits.",
                        flush=True,
                    )
                    continue

                print(
                    f"[{step}/{total_steps}] Simulating {img_name} scale={s} method={method} ({nq} qubits)...",
                    flush=True,
                )

                if args.noise_mode == "from_backend":
                    be, bid = _resolve_backend_for_nqubits(str(args.mock_backend), nq, seed=int(args.mock_backend_seed))
                    nm = noise_model_from_backend(be)
                    tkwargs, tmeta = _transpile_kwargs_from_backend(
                        be,
                        bid,
                        optimization_level=int(args.transpile_optimization_level),
                        transpile_seed=args.transpile_seed,
                    )
                    noise_tags = {"from_backend": bid}
                else:
                    tkwargs, tmeta = _transpile_kwargs_for_sweep(
                        noise_mode=str(args.noise_mode),
                        topology=str(args.topology),
                        nq=nq,
                        mock_backend=str(args.mock_backend),
                        mock_backend_seed=int(args.mock_backend_seed),
                        optimization_level=int(args.transpile_optimization_level),
                        transpile_seed=args.transpile_seed,
                    )
                    if args.noise_mode == "synthetic":
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
                        noise_tags = {
                            "p_depol_1q": args.p1 * s,
                            "p_depol_2q": args.p2 * s,
                            "t1": args.t1 if args.t1 is not None else "",
                            "t2": args.t2 if args.t2 is not None else "",
                            "readout_prob_01": args.r01 * s,
                            "readout_prob_10": args.r10 * s,
                        }
                    else:
                        assert preset_doc is not None
                        nm, preset_meta = build_noise_model_from_preset(preset_doc, scale=float(s))
                        noise_tags = {**preset_meta, "yaml_preset": str(args.yaml_preset)}

                r = noisy_frqi_metrics_row(
                    img,
                    method=method,
                    noise_model=nm,
                    seed_simulator=args.seed,
                    **tkwargs,
                )
                rows.append(
                    {
                        "image": img_name,
                        "method": method,
                        "noise_mode": args.noise_mode,
                        "noise_scale": float(s),
                        **noise_tags,
                        **tmeta,
                        **r,
                    }
                )
                print(
                    f"[{step}/{total_steps}] -> done {img_name} scale={s} method={method}: "
                    f"fidelity={r.get('fidelity')}, psnr={r.get('psnr')}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    csv_path = OUT / str(args.csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}", flush=True)

    stem = csv_path.stem
    print(f"Writing curve figures for {len(images)} image(s)...", flush=True)
    for img_name in images:
        sub = df[df["image"] == img_name]
        if sub.empty:
            print(f"No rows for {img_name}; skipping figure.", flush=True)
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
        if str(args.csv_name) == "noisy_frqi_metrics.csv":
            fig_path = OUT / f"noisy_frqi_{img_name}_curves.png"
        else:
            fig_path = OUT / f"{stem}_{img_name}_curves.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {fig_path}", flush=True)

    if args.readout_demo:
        img = np.load(DATA / "test_4x4.npy")
        m = required_position_qubits(int(img.shape[0]))
        nm_ro = build_noise_model(readout_prob_01=0.08, readout_prob_10=0.05)
        sl = build_single_address_ry_slice(m, 0, float(np.pi / 4.0), mode="v-chain")
        nq_demo = int(sl.num_qubits)
        demo_tkwargs, _ = _transpile_kwargs_for_sweep(
            noise_mode="synthetic",
            topology=str(args.topology),
            nq=nq_demo,
            mock_backend=str(args.mock_backend),
            mock_backend_seed=int(args.mock_backend_seed),
            optimization_level=int(args.transpile_optimization_level),
            transpile_seed=args.transpile_seed,
        )
        raw, mit, conf, _ = run_readout_mitigation_slice_demo(
            m,
            address_bits=0,
            theta=float(np.pi / 4.0),
            mode="v-chain",
            noise_model=nm_ro,
            shots=24000,
            seed_simulator=args.seed,
            **demo_tkwargs,
        )
        print("Readout mitigation demo (color qubit, single structural slice):", flush=True)
        print("  empirical confusion (rows=measured, cols=true):\n", conf, flush=True)
        print("  P(meas=1) raw:", raw, " mitigated:", mit, flush=True)

    manifest = build_base_manifest(
        script="scripts/noisy_frqi_sweep.py",
        argv=sys.argv[1:],
        repo_root=ROOT,
        extra={
            "outputs": {
                "csv": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                "figures_glob": f"outputs/{stem}_*_curves.png",
            },
            "flags": {
                "allow_heavy_dm": bool(args.allow_heavy_dm),
                "dm_max_qubits": int(args.dm_max_qubits),
            },
            "seeds": {"simulator": int(args.seed), "transpile": args.transpile_seed},
            "sweep": {
                "images": images,
                "methods": methods,
                "scales": scales,
                "noise_mode": args.noise_mode,
                "topology": args.topology,
                "mock_backend": args.mock_backend,
                "mock_backend_seed": int(args.mock_backend_seed),
                "transpile_optimization_level": int(args.transpile_optimization_level),
                "yaml_preset": str(args.yaml_preset) if args.yaml_preset else "",
            },
        },
    )
    write_manifest_json(OUT / str(args.manifest_name), manifest)

    print("Wrote", csv_path, flush=True)
    print("Wrote", OUT / str(args.manifest_name), flush=True)
    print("Noisy FRQI sweep complete.", flush=True)


if __name__ == "__main__":
    main()

"""Edge-map quality after noisy FRQI reconstruction (classical QHED on recon vs Sobel on original).

This does **not** run a noisy quantum edge circuit; it evaluates how a noisy FRQI state
preparation (density matrix) degrades the **classical** QHED baseline applied to the
reconstructed image, using the classical Sobel magnitude map of the clean image as
reference.

Writes ``outputs/noisy_recon_qhed_edges.csv`` and one figure per image with PSNR/SSIM
of edge maps vs ``noise_scale``.

Supports the same ``--noise-mode`` / topology / mock-backend / transpile flags as
``scripts/noisy_frqi_sweep.py`` (see that script's help text).
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
from src.improved import frqi_structural_num_qubits_naive_slice, frqi_structural_num_qubits_vchain  # noqa: E402
from src.metrics import psnr, ssim_uint8  # noqa: E402
from src.nisq_mock import (  # noqa: E402
    build_noise_model_from_preset,
    linear_coupling_edges,
    load_named_fake_backend_v2,
    load_yaml_preset,
    make_generic_linear_backend,
    noise_model_from_backend,
)
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
    meta = {
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
    meta = {
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
    ap = argparse.ArgumentParser(description="Noisy recon → QHED edges vs Sobel on original.")
    ap.add_argument("--images", default=IMAGES_DEFAULT, help="Comma-separated test image stems.")
    ap.add_argument("--methods", default="naive,vchain", help="naive,vchain")
    ap.add_argument("--scales", default="0,0.05,0.1,0.15,0.2", help="Noise scale grid (synthetic/yaml).")
    ap.add_argument(
        "--noise-mode",
        choices=("synthetic", "from_backend", "yaml_preset"),
        default="synthetic",
    )
    ap.add_argument("--topology", choices=("full", "linear"), default="full")
    ap.add_argument("--mock-backend", default="generic_linear")
    ap.add_argument("--mock-backend-seed", type=int, default=42)
    ap.add_argument("--yaml-preset", type=str, default="")
    ap.add_argument("--transpile-optimization-level", type=int, default=0)
    ap.add_argument("--transpile-seed", type=int, default=None)
    ap.add_argument("--dm-max-qubits", type=int, default=14, help="Skip pair if qubit count exceeds this.")
    ap.add_argument("--allow-heavy-dm", action="store_true", help="Disable qubit-count cap.")
    ap.add_argument("--p1", type=float, default=0.004)
    ap.add_argument("--p2", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv-name", default="noisy_recon_qhed_edges.csv")
    ap.add_argument("--manifest-name", default="experiment_manifest_noisy_recon_qhed_edges.json")
    args = ap.parse_args()

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
        if scales != [1.0]:
            print("Note: --noise-mode from_backend forces scales=[1.0].", flush=True)
        scales = [1.0]

    total_steps = len(images) * len(scales) * len(methods)
    print(
        f"Starting noisy recon -> QHED edge sweep: {len(images)} image(s) x {len(scales)} scale(s) x "
        f"{len(methods)} method(s) = {total_steps} planned simulation(s) "
        f"[noise_mode={args.noise_mode}].",
        flush=True,
    )

    rows: list[dict] = []
    step = 0
    for img_name in images:
        img = np.load(DATA / f"{img_name}.npy")
        ref_edges = classical_sobel_edge_map(img)
        for s in scales:
            for method in methods:
                step += 1
                nq = _dm_qubits_for_method(img, method)
                if nq > max_q:
                    print(f"[{step}/{total_steps}] Skip {img_name} {method}: {nq} qubits > {max_q}", flush=True)
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
                        nm = build_noise_model(p_depol_1q=args.p1 * s, p_depol_2q=args.p2 * s)
                        noise_tags = {"p_depol_1q": args.p1 * s, "p_depol_2q": args.p2 * s}
                    else:
                        assert preset_doc is not None
                        nm, preset_meta = build_noise_model_from_preset(preset_doc, scale=float(s))
                        noise_tags = {**preset_meta, "yaml_preset": str(args.yaml_preset)}

                r = noisy_frqi_metrics_row(
                    img,
                    method=method,
                    noise_model=nm,
                    seed_simulator=args.seed,
                    return_recon=True,
                    **tkwargs,
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
                        "noise_mode": args.noise_mode,
                        "noise_scale": float(s),
                        **noise_tags,
                        **tmeta,
                        "edge_psnr": ep,
                        "edge_ssim": es,
                        **{k: v for k, v in r.items() if k != "recon"},
                    }
                )
                print(
                    f"[{step}/{total_steps}] -> done {img_name} scale={s} method={method}: "
                    f"edge_psnr={ep}, edge_ssim={es}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    csv_path = OUT / str(args.csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}", flush=True)

    stem = csv_path.stem
    print(f"Writing edge-metric figures for {len(images)} image(s)...", flush=True)
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
        if str(args.csv_name) == "noisy_recon_qhed_edges.csv":
            fig_path = OUT / f"noisy_recon_qhed_edges_{img_name}.png"
        else:
            fig_path = OUT / f"{stem}_{img_name}.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {fig_path}", flush=True)

    manifest = build_base_manifest(
        script="scripts/noisy_recon_qhed_edge_metrics.py",
        argv=sys.argv[1:],
        repo_root=ROOT,
        extra={
            "outputs": {"csv": str(csv_path.relative_to(ROOT)).replace("\\", "/")},
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
            "seeds": {"simulator": int(args.seed), "transpile": args.transpile_seed},
        },
    )
    write_manifest_json(OUT / str(args.manifest_name), manifest)

    print("Wrote", csv_path, flush=True)
    print("Wrote", OUT / str(args.manifest_name), flush=True)
    print("Noisy recon -> QHED edge sweep complete.", flush=True)


if __name__ == "__main__":
    main()

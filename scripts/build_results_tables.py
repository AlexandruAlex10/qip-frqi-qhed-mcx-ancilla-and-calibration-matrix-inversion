"""
Single ingestion path for thesis results: load outputs/ per thesis/results_bundle.yaml,
validate columns, optional manifest freshness checks, emit tables/stats/figures under thesis/generated/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


@dataclass
class LoadReport:
    loaded: dict[str, str] = field(default_factory=dict)
    missing_primary: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    manifests: dict[str, Any] = field(default_factory=dict)


def _parse_manifest_time(iso: str) -> datetime:
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    return datetime.fromisoformat(iso)


def _check_stale(csv_path: Path, manifest: Mapping[str, Any], stale_seconds: float) -> str | None:
    exp = manifest.get("experiment") or {}
    gen = manifest.get("generated_utc")
    if not gen:
        return "manifest missing generated_utc; stale check skipped"
    mtime = datetime.fromtimestamp(csv_path.stat().st_mtime, tz=timezone.utc)
    gen_t = _parse_manifest_time(str(gen))
    if gen_t.tzinfo is None:
        gen_t = gen_t.replace(tzinfo=timezone.utc)
    delta = (mtime - gen_t).total_seconds()
    if delta < -stale_seconds:
        return (
            f"{csv_path.name}: CSV mtime {mtime.isoformat()} is **before** manifest "
            f"generated_utc {gen_t.isoformat()} by {abs(delta):.0f}s (clock skew or wrong manifest?)"
        )
    if delta > stale_seconds:
        return (
            f"{csv_path.name}: CSV mtime {mtime.isoformat()} is **after** manifest "
            f"generated_utc {gen_t.isoformat()} by {delta:.0f}s -- regenerate manifest or confirm bundle."
        )
    return None


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> list[str]:
    miss = [c for c in required if c not in df.columns]
    if miss:
        return [f"{name}: missing columns {miss}; have {list(df.columns)}"]
    return []


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ingest(
    repo_root: Path,
    config: dict[str, Any],
    *,
    stale_seconds: float,
    require_figures: bool,
    fail_on_stale: bool,
) -> tuple[LoadReport, dict[str, pd.DataFrame]]:
    report = LoadReport()
    out_dir = repo_root / config["paths"]["outputs_dir"]
    if not out_dir.is_dir():
        report.errors.append(f"outputs directory not found: {out_dir}")
        return report, {}

    frames: dict[str, pd.DataFrame] = {}
    primary = config.get("primary") or {}
    for key, spec in primary.items():
        csv_name = spec["csv"]
        csv_path = out_dir / csv_name
        if not csv_path.is_file():
            report.missing_primary.append(str(csv_path.relative_to(repo_root)))
            report.errors.append(f"Missing required dataset {key}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        report.errors.extend(_validate_columns(df, spec["required_columns"], key))
        man_name = spec.get("manifest")
        if man_name:
            mpath = out_dir / man_name
            if not mpath.is_file():
                report.warnings.append(f"Manifest listed for {key} but missing: {mpath}")
            else:
                with mpath.open("r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
                report.manifests[key] = manifest
                st = _check_stale(csv_path, manifest, stale_seconds)
                if st:
                    if fail_on_stale:
                        report.errors.append(st)
                    else:
                        report.warnings.append(st)
        report.loaded[key] = str(csv_path.relative_to(repo_root))
        frames[key] = df

    sup = config.get("supplementary") or {}
    for key, spec in sup.items():
        if not spec.get("optional"):
            report.warnings.append(f"supplementary.{key} should set optional: true")
        csv_path = out_dir / spec["csv"]
        if not csv_path.is_file():
            report.missing_optional.append(str(csv_path.relative_to(repo_root)))
            continue
        df = pd.read_csv(csv_path)
        errs = _validate_columns(df, spec["required_columns"], f"supplementary.{key}")
        if errs:
            report.warnings.extend(errs)
            continue
        frames[f"supplementary:{key}"] = df
        report.loaded[f"supplementary:{key}"] = str(csv_path.relative_to(repo_root))

    for fig_rel in config.get("expected_figures") or []:
        fp = (repo_root / Path(fig_rel)).resolve()
        if not fp.is_file():
            msg = f"Expected figure missing: {fig_rel}"
            if require_figures:
                report.errors.append(msg)
            else:
                report.warnings.append(msg)

    return report, frames


def table_encoding_baseline(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "image",
        "size",
        "frqi_qubits",
        "frqi_depth_transpiled",
        "frqi_cx_transpiled",
        "frqi_ssim_skimage",
        "qhed_vs_sobel_ssim_skimage",
        "transpile_optimization_level",
    ]
    return df[cols].sort_values("size")


def table_structural(df: pd.DataFrame) -> pd.DataFrame:
    kinds = {
        "struct_vchain",
        "struct_naive_full",
        "struct_naive_slice",
        "struct_naive_slice_scaled",
    }
    sub = df[df["kind"].isin(kinds)].copy()
    sub = sub.sort_values(["image_size", "kind"])
    return sub[
        [
            "image",
            "image_size",
            "m",
            "kind",
            "num_qubits",
            "depth",
            "cx",
            "topology",
            "transpile_optimization_level",
        ]
    ]


def pivot_noisy_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per (image, noise_scale, ...) with naive vs vchain side-by-side."""
    keys = [
        c
        for c in [
            "image",
            "noise_mode",
            "noise_scale",
            "topology",
            "mock_backend",
            "transpile_optimization_level",
        ]
        if c in df.columns
    ]
    metrics = [c for c in ["edge_psnr", "edge_ssim", "psnr", "ssim", "fidelity"] if c in df.columns]
    pieces = []
    for met in metrics:
        p = df.pivot_table(
            index=keys,
            columns="method",
            values=met,
            aggfunc="first",
        )
        p.columns = [f"{met}_{str(c)}" for c in p.columns]
        pieces.append(p)
    wide = pd.concat(pieces, axis=1).reset_index()
    for m in metrics:
        n, v = f"{m}_naive", f"{m}_vchain"
        if n in wide.columns and v in wide.columns:
            wide[f"delta_{m}_vchain_minus_naive"] = wide[v] - wide[n]
    return wide


def paired_naive_vchain_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Paired comparisons by (image, noise_scale, noise_mode, topology) on positive noise scales."""
    req = {"image", "method", "noise_scale", "edge_ssim", "edge_psnr"}
    if not req.issubset(df.columns):
        return pd.DataFrame(), {"error": "noisy_recon_qhed_edges missing columns for pairing"}

    sub = df[df["method"].isin(["naive", "vchain"])].copy()
    keys = [k for k in ["image", "noise_mode", "noise_scale", "topology"] if k in sub.columns]
    metrics = [c for c in ["edge_ssim", "edge_psnr", "ssim", "psnr", "fidelity"] if c in sub.columns]

    rows = []
    for metric in metrics:
        p = sub.pivot_table(index=keys, columns="method", values=metric, aggfunc="first")
        if "naive" not in p.columns or "vchain" not in p.columns:
            continue
        paired = p.dropna(subset=["naive", "vchain"], how="any")
        paired = paired[paired.index.get_level_values("noise_scale") > 0]
        if paired.empty:
            continue
        d = paired["vchain"] - paired["naive"]
        row: dict[str, Any] = {
            "metric": metric,
            "n_pairs": int(len(d)),
            "median_delta": float(np.median(d.values)),
            "mean_delta": float(np.mean(d.values)),
        }
        if stats is not None and len(d) >= 6:
            # Wilcoxon on paired differences (exploratory; no multiple-comparison adjustment).
            try:
                w = stats.wilcoxon(d.values, alternative="two-sided", zero_method="wilcox", mode="auto")
                row["wilcoxon_statistic"] = float(w.statistic)
                row["wilcoxon_pvalue"] = float(w.pvalue)
            except ValueError as e:
                row["wilcoxon_error"] = str(e)
        rows.append(row)

    summary = pd.DataFrame(rows)
    text: dict[str, Any] = {
        "note": "Exploratory paired Wilcoxon on vchain−naive differences (noise_scale>0 only).",
        "primary_metric_recommendation": "edge_ssim under synthetic noise",
    }
    return summary, text


def readout_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for img, g in df.groupby("image"):
        r = {
            "image": img,
            "n_seeds": len(g),
            "p1_raw_mean": g["p1_raw"].mean(),
            "p1_mitigated_mean": g["p1_mitigated"].mean(),
            "mean_delta_mitigated_minus_raw": (g["p1_mitigated"] - g["p1_raw"]).mean(),
        }
        if {"p1_mitigated_boot_p05", "p1_mitigated_boot_p95"}.issubset(g.columns):
            r["boot_p50_median_of_medians"] = g["p1_mitigated_boot_p50"].median()
            r["boot_p05_min"] = g["p1_mitigated_boot_p05"].min()
            r["boot_p95_max"] = g["p1_mitigated_boot_p95"].max()
        rows.append(r)
    return pd.DataFrame(rows)


def readout_paired_seed_test(df: pd.DataFrame) -> dict[str, Any]:
    """Per seed, paired p1_raw vs p1_mitigated (same circuit draws) aggregated via Wilcoxon across seeds."""
    if stats is None or df.empty:
        return {}
    d = df["p1_mitigated"].values - df["p1_raw"].values
    try:
        w = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox", mode="auto")
        return {
            "n": len(d),
            "median_delta_mitigated_minus_raw": float(np.median(d)),
            "wilcoxon_statistic": float(w.statistic),
            "wilcoxon_pvalue": float(w.pvalue),
        }
    except ValueError as e:
        return {"n": len(d), "median_delta_mitigated_minus_raw": float(np.median(d)), "wilcoxon_error": str(e)}


def _df_to_markdown(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    """GitHub-style pipe table without optional ``tabulate`` dependency."""
    d = df if max_rows is None else df.head(max_rows)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                cells.append(str(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _df_to_latex(df: pd.DataFrame, caption: str, label: str, *, float_spec: str = "htbp") -> str:
    # Escape underscores in the visible caption; keep \label{} as a plain ASCII key (no \_).
    cap = caption.replace("_", r"\_")
    body = df.to_latex(index=False, float_format="%.4g", escape=True)
    return f"% auto-generated\n\\begin{{table}}[{float_spec}]\n\\centering\n\\caption{{{cap}}}\n\\label{{{label}}}\n{body}\\end{{table}}\n"


def _write_bachelors_tex_fragment(repo_root: Path, filename: str, tex: str) -> None:
    """Mirror key tables into thesis/bachelors_en/generated for stable \\input{} from the PDF."""
    dest = repo_root / "thesis" / "bachelors_en" / "generated" / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(tex.rstrip() + "\n", encoding="utf-8")


def noisy_edge_excerpt_df(wide: pd.DataFrame) -> pd.DataFrame:
    """Narrow excerpt for Results chapter: paired edge SSIM at noise_scale>0 on small test images."""
    cols = ["image", "noise_scale", "edge_ssim_naive", "edge_ssim_vchain", "delta_edge_ssim_vchain_minus_naive"]
    miss = [c for c in cols if c not in wide.columns]
    if miss:
        return pd.DataFrame()
    sub = wide[(wide["noise_scale"] > 0) & (wide["image"].isin(["test_4x4", "test_8x8"]))].copy()
    if sub.empty:
        return sub
    out = sub[cols].sort_values(["image", "noise_scale"])
    return out.reset_index(drop=True)


def write_provenance_snapshot(repo_root: Path, report: LoadReport, dest: Path) -> None:
    lines = ["# Provenance snapshot (auto-generated)", ""]
    lines.append(f"- **Generated (ingestion run):** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    for key, man in report.manifests.items():
        lines.append(f"## {key}")
        lines.append("")
        lines.append(f"- **script:** `{man.get('script')}`")
        lines.append(f"- **generated_utc:** `{man.get('generated_utc')}`")
        lines.append(f"- **git_commit:** `{man.get('git_commit')}`")
        lines.append(f"- **python:** `{man.get('python')}`")
        vers = man.get("versions") or {}
        lines.append(f"- **qiskit:** `{vers.get('qiskit')}`  **qiskit_aer:** `{vers.get('qiskit_aer')}`")
        exp = man.get("experiment") or {}
        if "sweep" in exp:
            lines.append(f"- **sweep:** `{json.dumps(exp['sweep'], sort_keys=True)}`")
        if "readout" in exp:
            lines.append(f"- **readout:** `{json.dumps(exp['readout'], sort_keys=True)}`")
        lines.append("")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")


def emit_figures(
    repo_root: Path,
    frames: dict[str, pd.DataFrame],
    out_fig_dir: Path,
) -> list[str]:
    written: list[str] = []
    if plt is None:
        return written
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    if "frqi_structural_metrics" in frames:
        df = frames["frqi_structural_metrics"]
        v = df[df["kind"] == "struct_vchain"].set_index("image_size").sort_index()
        n = df[df["kind"] == "struct_naive_slice_scaled"].set_index("image_size").sort_index()
        if not v.empty and not n.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(v.index, v["cx"].values, marker="o", label="v-chain (struct_vchain)")
            ax.plot(n.index, n["cx"].values, marker="s", label="naive scaled (struct_naive_slice_scaled)")
            ax.set_xlabel("Image size (pixels per side)")
            ax.set_ylabel("Transpiled CX count")
            ax.legend()
            ax.set_title("Structural resource comparison (from CSV)")
            fig.tight_layout()
            p = out_fig_dir / "summary_structural_cx_vs_size.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            written.append(str(p.relative_to(repo_root)))

    if "noisy_recon_qhed_edges" in frames:
        df = frames["noisy_recon_qhed_edges"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for img, g in df.groupby("image"):
            gn = g[g["method"] == "naive"].sort_values("noise_scale").replace([np.inf, -np.inf], np.nan)
            gv = g[g["method"] == "vchain"].sort_values("noise_scale").replace([np.inf, -np.inf], np.nan)
            axes[0].plot(gn["noise_scale"], gn["psnr"], marker="o", label=f"{img} naive")
            axes[0].plot(gv["noise_scale"], gv["psnr"], marker="^", linestyle="--", label=f"{img} vchain")
            axes[1].plot(gn["noise_scale"], gn["edge_ssim"], marker="o", label=f"{img} naive")
            axes[1].plot(gv["noise_scale"], gv["edge_ssim"], marker="^", linestyle="--", label=f"{img} vchain")
        axes[0].set_xlabel("noise_scale")
        axes[0].set_ylabel("PSNR")
        axes[1].set_xlabel("noise_scale")
        axes[1].set_ylabel("edge SSIM")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        fig.suptitle("Noisy recon + QHED (ingested CSV summary)")
        fig.tight_layout()
        p = out_fig_dir / "summary_noisy_psnr_edge_ssim.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(str(p.relative_to(repo_root)))

    if "readout_mitigation_shot_sweep" in frames:
        df = frames["readout_mitigation_shot_sweep"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df["p1_raw"], df["p1_mitigated"], c=df["seed"], cmap="viridis", alpha=0.85)
        lims = [min(df["p1_raw"].min(), df["p1_mitigated"].min()), max(df["p1_raw"].max(), df["p1_mitigated"].max())]
        ax.plot(lims, lims, "k--", alpha=0.4, label="y=x")
        ax.set_xlabel("p1 raw")
        ax.set_ylabel("p1 mitigated")
        ax.legend()
        ax.set_title("Readout mitigation (per seed)")
        fig.tight_layout()
        p = out_fig_dir / "summary_readout_raw_vs_mitigated.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(str(p.relative_to(repo_root)))

    if "noisy_frqi_metrics" in frames:
        df = frames["noisy_frqi_metrics"]
        fig, ax = plt.subplots(figsize=(6, 4))
        for img, g in df.groupby("image"):
            gn = g[g["method"] == "naive"].sort_values("noise_scale")
            gv = g[g["method"] == "vchain"].sort_values("noise_scale")
            ax.plot(gn["noise_scale"], gn["fidelity"], marker="o", label=f"{img} naive")
            ax.plot(gv["noise_scale"], gv["fidelity"], marker="^", linestyle="--", label=f"{img} vchain")
        ax.set_xlabel("noise_scale")
        ax.set_ylabel("fidelity")
        ax.legend(fontsize=7)
        ax.set_title("Noisy FRQI fidelity (ingested CSV)")
        fig.tight_layout()
        p = out_fig_dir / "summary_noisy_frqi_fidelity.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(str(p.relative_to(repo_root)))

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest outputs/ and emit thesis/generated fragments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML bundle (default: thesis/results_bundle.yaml under repo root)",
    )
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument(
        "--stale-seconds",
        type=float,
        default=600.0,
        help="If CSV mtime and manifest generated_utc differ by more than this many seconds, warn (or error with --strict).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat stale CSV/manifest pairs and missing expected_figures as hard errors.",
    )
    parser.add_argument(
        "--require-figures",
        action="store_true",
        help="Fail if expected_figures from config are missing (implies strict figure checks).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Generated output directory (default: thesis/generated)",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    cfg_path = args.config or (repo_root / "thesis" / "results_bundle.yaml")
    out_dir = args.output_dir or (repo_root / "thesis" / "generated")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cfg_path.is_file():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    config = load_yaml_config(cfg_path)
    fail_on_stale = bool(args.strict)
    require_figures = bool(args.require_figures or args.strict)
    report, frames = ingest(
        repo_root,
        config,
        stale_seconds=args.stale_seconds,
        require_figures=require_figures,
        fail_on_stale=fail_on_stale,
    )

    for w in report.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if report.errors:
        for e in report.errors:
            print(f"ERROR: {e}", file=sys.stderr)
        (out_dir / "ingestion_report.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "loaded": report.loaded,
                    "missing_primary": report.missing_primary,
                    "errors": report.errors,
                    "warnings": report.warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1

    # --- tables ---
    md_chunks: list[str] = ["# Generated results tables\n"]
    tex_chunks: list[str] = ["% Generated results tables\n"]

    if "frqi_qhed_sobel_metrics" in frames:
        t = table_encoding_baseline(frames["frqi_qhed_sobel_metrics"])
        t.to_csv(out_dir / "table_encoding_baseline.csv", index=False)
        md_chunks.append("## Table: encoding / ideal baseline\n\n" + _df_to_markdown(t) + "\n")
        tex_chunks.append(_df_to_latex(t, "Ideal FRQI encoding and QHED vs Sobel (transpiled initialize)", "tab:encoding_baseline"))

    if "frqi_structural_metrics" in frames:
        t = table_structural(frames["frqi_structural_metrics"])
        t.to_csv(out_dir / "table_structural_resources.csv", index=False)
        md_chunks.append("## Table: structural resources\n\n" + _df_to_markdown(t) + "\n")
        struct_tex = _df_to_latex(
            t,
            "Structural FRQI preparation: transpiled depth and CX (optimization level 3)",
            "tab:structural_resources",
        )
        tex_chunks.append(struct_tex)

    if "noisy_recon_qhed_edges" in frames:
        wide = pivot_noisy_edges(frames["noisy_recon_qhed_edges"])
        wide.to_csv(out_dir / "table_noisy_edges_paired_wide.csv", index=False)
        md_chunks.append("## Table: noisy edge metrics (wide)\n\n" + _df_to_markdown(wide) + "\n")
        tex_chunks.append(
            _df_to_latex(
                wide.head(40),
                "Noisy recon + QHED edge metrics naive vs vchain (excerpt)",
                "tab:noisy_edges_wide",
            )
        )
        ex = noisy_edge_excerpt_df(wide)
        if not ex.empty:
            ex_tex = _df_to_latex(
                ex,
                "Noisy reconstruction: edge SSIM naive vs v-chain (excerpt, noise scale $>0$, linear topology).",
                "tab:noisy_edges_excerpt",
            )
            _write_bachelors_tex_fragment(repo_root, "results_table_noisy_edges_excerpt.tex", ex_tex)

    if "readout_mitigation_shot_sweep" in frames:
        rs = readout_summary(frames["readout_mitigation_shot_sweep"])
        rs.to_csv(out_dir / "table_readout_summary_by_image.csv", index=False)
        md_chunks.append("## Table: readout mitigation summary\n\n" + _df_to_markdown(rs) + "\n")
        tex_chunks.append(
            _df_to_latex(
                rs,
                r"Readout calibration-matrix mitigation on the color qubit ($4\times 4$, 10 seeds)",
                "tab:readout_summary",
            )
        )

    if "noisy_frqi_metrics" in frames:
        sub = frames["noisy_frqi_metrics"]
        sub.to_csv(out_dir / "table_noisy_frqi_long.csv", index=False)
        md_chunks.append("## Table: noisy FRQI (long, optional)\n\n" + _df_to_markdown(sub.head(24)) + "\n")

    (out_dir / "tables.md").write_text("\n".join(md_chunks), encoding="utf-8")
    (out_dir / "tables.tex").write_text("\n".join(tex_chunks), encoding="utf-8")

    bundle = (
        "% auto-generated bundle (structural, encoding, readout).\n"
        "\\input{generated/results_table_structural_resources}\n"
        "\\input{generated/results_table_encoding_baseline}\n"
        "\\input{generated/results_table_readout_summary}\n"
    )
    _write_bachelors_tex_fragment(repo_root, "results_tables.tex", bundle)

    # --- stats ---
    stats_md: list[str] = ["# Paired statistics (exploratory)\n"]
    stats_json: dict[str, Any] = {}

    if "noisy_recon_qhed_edges" in frames:
        ps_df, ps_meta = paired_naive_vchain_stats(frames["noisy_recon_qhed_edges"])
        if not ps_df.empty:
            ps_df.to_csv(out_dir / "stats_paired_naive_vchain.csv", index=False)
            stats_md.append("## Naive vs v-chain (paired on image×noise_scale, noise_scale>0)\n\n")
            stats_md.append(_df_to_markdown(ps_df) + "\n\n")
            stats_md.append(f"*Note:* {ps_meta.get('note')}\n")
        stats_json["paired_naive_vchain"] = ps_meta | {"rows": ps_df.to_dict(orient="records")}

    if "readout_mitigation_shot_sweep" in frames:
        rdf = frames["readout_mitigation_shot_sweep"]
        rt = readout_paired_seed_test(rdf)
        stats_json["readout_paired_across_seeds"] = rt
        stats_md.append("## Readout raw vs mitigated (paired per seed)\n\n")
        stats_md.append(f"- `median_delta_mitigated_minus_raw`: {rt.get('median_delta_mitigated_minus_raw')}\n")
        stats_md.append(f"- `wilcoxon_pvalue`: {rt.get('wilcoxon_pvalue')}\n")
        stats_md.append(f"- `n_seeds`: {rt.get('n')}\n")

    (out_dir / "stats.md").write_text("\n".join(stats_md), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats_json, indent=2), encoding="utf-8")

    # --- figures ---
    fig_dir = out_dir / "figures"
    fig_written = emit_figures(repo_root, frames, fig_dir)
    fig_index = ["# Generated summary figures (from ingested CSVs)\n"]
    for rel in fig_written:
        fig_index.append(f"- `{rel}` (relative to repository root)\n")
    fig_index.append("\n## Canonical `outputs/` figures (from experiment scripts)\n\n")
    for fig_rel in config.get("expected_figures") or []:
        exists = (repo_root / Path(fig_rel)).is_file()
        fig_index.append(f"- `{fig_rel}` — **exists:** {exists}\n")
    (out_dir / "figures_index.md").write_text("".join(fig_index), encoding="utf-8")

    write_provenance_snapshot(repo_root, report, out_dir / "provenance_snapshot.md")

    (out_dir / "ingestion_report.json").write_text(
        json.dumps(
            {
                "ok": True,
                "loaded": report.loaded,
                "missing_optional": report.missing_optional,
                "warnings": report.warnings,
                "figures_emitted": fig_written,
                "config": str(cfg_path.relative_to(repo_root)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"OK: wrote artifacts under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

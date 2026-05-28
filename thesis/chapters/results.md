# Results (draft)

This chapter is written against the **primary** artifacts declared in [`../results_bundle.yaml`](../results_bundle.yaml) and validated by [`../../scripts/build_results_tables.py`](../../scripts/build_results_tables.py). **Regenerate** the fragments under `thesis/generated/` after any change to `outputs/`:

```powershell
python scripts\build_results_tables.py
```

Frozen provenance for the noisy and readout sweeps (Git commit, Qiskit versions, seeds, transpile level) lives in [`../results_configuration.md`](../results_configuration.md). If the ingestion run reports CSV/manifest **staleness**, re-run the corresponding script so the manifest `generated_utc` matches the CSV, or pass a larger `--stale-seconds` while iterating.

## Overview

This work compares **structural FRQI** preparation with a **naive** multi-controlled decomposition to a **v-chain ancilla** pattern for multi-controlled rotations, then studies **noisy simulator** reconstructions with **QHED** edge metrics, and a **readout mitigation** slice based on **calibration matrix inversion**. All quantitative claims below are tied to the CSV files named in the bundle; **no IBM hardware** runs are included here—evidence is **simulator-only** (synthetic noise and mock backends), which limits external validity to real devices.

**Primary loaded artifacts (contract):**

- `outputs/frqi_qhed_sobel_metrics.csv` — ideal encoding and Sobel reference metrics; transpiled `initialize` resource columns.
- `outputs/frqi_structural_metrics.csv` — transpiled depth/CX for structural `struct_vchain` vs naive baselines.
- `outputs/noisy_recon_qhed_edges.csv` — paired `naive` vs `vchain` rows across `noise_scale` for each test image.
- `outputs/readout_mitigation_shot_sweep.csv` — per-seed `p1_raw` vs `p1_mitigated` with bootstrap quantiles when present.
- `outputs/noisy_frqi_metrics.csv` — optional noisy FRQI sweep mirroring the edge study.

Supplementary paths such as `noisy_recon_qhed_edges_mock_nisq.csv` are **not** part of the default bundle until you generate them and enable loading in the YAML.

## Structural resources

Transpiled **qubit count**, **depth**, and **CX** comparisons between the v-chain structural preparation and naive structural estimates are summarized from `frqi_structural_metrics.csv` (see generated `table_structural_resources.csv` / `tables.md`). At small sizes, the v-chain layout trades ancillas for different multi-controlled gate structure; at larger sizes the resource story is dominated by state preparation width and synthesis choices—**honest reporting** requires showing both regimes.

**Figures:** prefer committed experiment plots `outputs/fig_cx_vs_image_size.png` and `outputs/fig_depth_vs_image_size.png` from `scripts/frqi_structural_resources.py`. The ingestion pipeline also emits `thesis/generated/figures/summary_structural_cx_vs_size.png` for a compact thesis-side recap.

## Encoding baseline and QHED vs Sobel

Ideal FRQI reconstruction matches the classical image at the tested resolutions; QHED versus Sobel similarity drops as resolution increases, as recorded in `frqi_qhed_sobel_metrics.csv` (generated `table_encoding_baseline.csv`). Transpiled `initialize` depth/CX columns are a **synthesis reference**, not the hand-structured circuit used in the improvement chapter.

**Figures:** `outputs/<image>_comparison.png` and `outputs/<image>_frqi_recon.png` from `scripts/frqi_qhed_sobel.py`.

## Noisy reconstruction and edge metrics

Using `noisy_recon_qhed_edges.csv`, we pair **naive** and **vchain** at identical `(image, noise_mode, noise_scale, topology, …)` and compare reconstruction **PSNR/SSIM** and QHED **edge_psnr/edge_ssim**, plus **fidelity** where reported. The wide pivot and Δ columns are in `table_noisy_edges_paired_wide.csv`.

**Qualitative regime dependence:** for some `(image, noise_scale)` cells the v-chain preparation improves edge SSIM; for others the ranking flips—summaries in `stats_paired_naive_vchain.csv` are **exploratory** (paired Wilcoxon on \( \Delta = \) vchain−naive across image×scale pairs with `noise_scale>0`). Treat multiplicity cautiously if you emphasize more than one metric; the bundle recommends **`edge_ssim` under noise** as the primary narrative metric.

**Figures:** canonical per-image curves `outputs/noisy_recon_qhed_edges_<image>.png`; ingestion recap `thesis/generated/figures/summary_noisy_psnr_edge_ssim.png`.

## Readout mitigation slice

For the controlled readout error model in `readout_mitigation_shot_sweep.csv`, mitigation shifts the inferred **\(p_1\)** distribution toward lower error on average. `table_readout_summary_by_image.csv` aggregates means per image; bootstrap columns `p1_mitigated_boot_p05/p50/p95` provide **per-seed** uncertainty bands that can be summarized as min/median/max across seeds in the same table.

**Exploratory test:** across seeds, paired differences `p1_mitigated − p1_raw` are tested with a Wilcoxon signed-rank statistic (see `stats.json`).

**Figures:** `outputs/readout_mitigation_shot_sweep.png` when present; recap `thesis/generated/figures/summary_readout_raw_vs_mitigated.png`.

## Optional noisy FRQI sweep

`noisy_frqi_metrics.csv` supports PSNR/SSIM/fidelity curves aligned with the noisy recon experiment flags (see its manifest). Generated `summary_noisy_frqi_fidelity.png` mirrors the CSV for quick inclusion.

## Reproducibility and limitations

- **Versions and seeds** are recorded in `outputs/experiment_manifest_noisy_recon_qhed_edges.json`, `experiment_manifest_readout_mitigation_shot_sweep.json`, and `experiment_manifest_noisy_frqi_sweep.json` (excerpted in `results_configuration.md`).
- **Simulator-only** evidence: results do not include device calibration drift, spatial correlations, or crosstalk beyond what the mock/noise models encode.
- **Density-matrix / memory** constraints for large circuits are discussed in the repository `README.md`; the 16×16 noisy rows depend on those engineering choices.

## Generated tables and statistics

After a successful ingestion run, consult:

- `thesis/generated/tables.md` and `thesis/generated/tables.tex`
- `thesis/generated/stats.md` and `thesis/generated/stats.json`
- `thesis/generated/figures_index.md` and `thesis/generated/provenance_snapshot.md`
- `thesis/generated/ingestion_report.json` for the exact list of files loaded in the last run

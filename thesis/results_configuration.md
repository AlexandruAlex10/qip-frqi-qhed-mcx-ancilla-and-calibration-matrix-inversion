# Results bundle — frozen provenance

This note pins the **primary** CSV + manifest pairs referenced by [`results_bundle.yaml`](results_bundle.yaml). After re-running experiments, refresh the excerpts below, or copy fields from the auto-generated `thesis/generated/provenance_snapshot.md` produced by `python scripts/build_results_tables.py`.

> **Housekeeping:** If `outputs/*.csv` was touched after the manifest (for example a bulk file timestamp refresh), `build_results_tables.py` warns by default; re-run the experiment script to regenerate JSON, or use `--stale-seconds` / non-`--strict` while editing.

## `noisy_recon_qhed_edges.csv` ↔ `experiment_manifest_noisy_recon_qhed_edges.json`

- **CSV path:** `outputs/noisy_recon_qhed_edges.csv`
- **Manifest `generated_utc`:** 2026-05-23T21:40:05.499867+00:00
- **`git_commit`:** `0db96e6ae7eeffbccbc65c86b3945af69188a501`
- **`python`:** 3.11.3
- **`versions`:** qiskit 1.3.3, qiskit_aer 0.16.4
- **`script`:** `scripts/noisy_recon_qhed_edge_metrics.py`
- **Sweep (manifest `experiment.sweep`):** images `test_4x4`, `test_8x8`, `test_16x16`; methods `naive`, `vchain`; `noise_mode` synthetic; scales `0.0`–`0.2` step 0.05; `topology` full; `mock_backend` generic_linear; `mock_backend_seed` 42; simulator seed 42; `transpile_optimization_level` **0** (re-run with level 3 if you adopt the Week 9 runbook).

## `readout_mitigation_shot_sweep.csv` ↔ `experiment_manifest_readout_mitigation_shot_sweep.json`

- **CSV path:** `outputs/readout_mitigation_shot_sweep.csv`
- **Manifest `generated_utc`:** 2026-05-23T21:49:24.713645+00:00
- **`git_commit`:** `0db96e6ae7eeffbccbc65c86b3945af69188a501`
- **`python`:** 3.11.3
- **`versions`:** qiskit 1.3.3, qiskit_aer 0.16.4
- **`script`:** `scripts/readout_mitigation_shot_sweep.py`
- **Readout slice:** `shots_cal` 8000, `shots_data` 12000, `rcond` 1e-06, `bootstrap` 400, `bootstrap_seed` 999, synthetic readout errors r01=0.08, r10=0.05; shot seeds 1–10; topology full, mock_backend generic_linear.

## `noisy_frqi_metrics.csv` ↔ `experiment_manifest_noisy_frqi_sweep.json`

- **CSV path:** `outputs/noisy_frqi_metrics.csv`
- **Manifest `generated_utc`:** 2026-05-23T20:12:47.561799+00:00
- **`git_commit`:** `0db96e6ae7eeffbccbc65c86b3945af69188a501`
- **`python`:** 3.11.3
- **`versions`:** qiskit 1.3.3, qiskit_aer 0.16.4
- **`script`:** `scripts/noisy_frqi_sweep.py`
- **Sweep:** same image/method/noise grid as noisy recon manifest above; `transpile_optimization_level` **0**.

## Other primary tables (no manifest in repo)

- **`frqi_qhed_sobel_metrics.csv`**, **`frqi_structural_metrics.csv`:** regenerate with `scripts/frqi_qhed_sobel.py` and `scripts/frqi_structural_resources.py`; record `transpile_optimization_level` and Qiskit versions in thesis Methods if not captured in JSON.

## Supplementary (optional) paths

- **`noisy_recon_qhed_edges_mock_nisq.csv`** + **`experiment_manifest_noisy_recon_mock_nisq.json`:** not present in the repository snapshot; enable in `results_bundle.yaml` / supplementary loader when you generate them.

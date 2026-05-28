# Outputs contract (`outputs/`)

Single source of truth for quantitative results cited in the thesis. The machine-readable twin of this document is [`results_bundle.yaml`](results_bundle.yaml), consumed by [`scripts/build_results_tables.py`](../scripts/build_results_tables.py).

## Primary CSV artifacts

| Filename | Produced by | Required columns (minimum) | Thesis subsection |
|----------|-------------|----------------------------|-------------------|
| `frqi_qhed_sobel_metrics.csv` | `scripts/frqi_qhed_sobel.py` | `image`, `size`, `frqi_qubits`, `frqi_depth_transpiled`, `frqi_cx_transpiled`, `frqi_ssim_skimage`, `qhed_vs_sobel_ssim_skimage`, `transpile_optimization_level` | Encoding / ideal baseline; transpiled `initialize` resource reference |
| `frqi_structural_metrics.csv` | `scripts/frqi_structural_resources.py` | `image`, `image_size`, `m`, `kind`, `num_qubits`, `depth`, `cx`, `topology`, `transpile_optimization_level` | Structural resources (naive vs v-chain MCX story) |
| `noisy_recon_qhed_edges.csv` | `scripts/noisy_recon_qhed_edge_metrics.py` | `image`, `method`, `noise_mode`, `noise_scale`, `topology`, `edge_psnr`, `edge_ssim`, `psnr`, `ssim`, `fidelity`, `m`, `total_qubits` | Noisy reconstruction + QHED edge metrics; paired `naive` vs `vchain` |
| `readout_mitigation_shot_sweep.csv` | `scripts/readout_mitigation_shot_sweep.py` | `image`, `seed`, `p1_raw`, `p1_mitigated`, `shots_cal`, `shots_data`; if present: `p1_mitigated_boot_p05`, `p1_mitigated_boot_p50`, `p1_mitigated_boot_p95` | Readout calibration matrix inversion slice study |
| `noisy_frqi_metrics.csv` | `scripts/noisy_frqi_sweep.py` | `image`, `method`, `noise_mode`, `noise_scale`, `fidelity`, `psnr`, `ssim`, `m`, `total_qubits` | Optional noisy FRQI sweep (supports curves cited alongside edge study) |

## Optional / supplementary CSVs

| Filename | Notes |
|----------|--------|
| `frqi_structural_metrics_constrained.csv` | From `scripts/frqi_structural_resources.py --emit-layout-csv`; layout-constrained linear coupling story |
| `noisy_recon_qhed_edges_mock_nisq.csv` | Alternate `--csv-name` for mock NISQ snapshot (`--noise-mode from_backend`, bundle flags) |
| `noisy_frqi_metrics_mock_nisq_bundle.csv` (or stem from `--mock-nisq-bundle`) | Separate artifact; does not overwrite default sweep CSV |

## Manifest JSON (`experiment_manifest_*.json`)

Each primary noisy/readout experiment should ship a manifest with at least:

- `schema`, `script`, `generated_utc`, `git_commit`, `python`, `versions.qiskit`, `versions.qiskit_aer` (or Aer key variants)
- `experiment.outputs.csv` path matching the CSV on disk
- Sweep parameters you will quote (seeds, topology, `transpile_optimization_level`, noise mode, scales)

The ingestion pipeline can treat a CSV as **stale** when its modification time is inconsistent with the linked manifest `generated_utc` (see script `--stale-seconds`).

## Expected figures under `outputs/` (from experiment scripts)

| PNG path | Source script |
|----------|----------------|
| `fig_cx_vs_image_size.png`, `fig_depth_vs_image_size.png` | `scripts/frqi_structural_resources.py` |
| `{name}_comparison.png`, `{name}_frqi_recon.png` per test image | `scripts/frqi_qhed_sobel.py` |
| `noisy_recon_qhed_edges_<image>.png` (or custom `--csv-name` stem) | `scripts/noisy_recon_qhed_edge_metrics.py` |
| `noisy_frqi_metrics_<image>_curves.png` (or custom stem) | `scripts/noisy_frqi_sweep.py` |
| `readout_mitigation_shot_sweep.png` | `scripts/readout_mitigation_shot_sweep.py` |

Thesis text should cite these paths when the files exist after your runs. The build script also writes reproducible summary plots under `thesis/generated/figures/` when figures are missing or for dissertation-only bundles.

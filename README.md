# QIP: Baseline FRQI + QHED & Improvement MCX ancilla + Calibration Matrix Inversion

## Repository description

This repository contains the following dissertation idea:

Baseline: FRQI encoding + QHED edge detection implemented with multi-controlled rotations

Improvement: Replace naive multi-controlled Ry with ancilla-assisted decomposition (MCX ancilla), which reduces CNOT chain depth for small images and results in fewer depth-sensitive two-qubit gates; pair with simple readout error mitigation (calibration matrix inversion)


## Repository content

- FRQI implementation
- QHED baseline and Classical Sobel comparison
- Metrics utilities (PSNR and SSIM)
- Unit tests for FRQI and QHED
- Demo scripts
- Test images for 4x4, 8x8, and 16x16 pixels


## Install

Create and activate a virtual environment, then install dependencies:

### Windows (powershell)
```
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### MacOS/Linux (bash)
```
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Run all tests

```powershell
pytest
```

## Thesis (LaTeX)

The UPT English template lives under `thesis/masters_en/`. Chapters: **Introduction** (`chapters/4-Introduction.tex`), **State of the art and theoretical background** (`chapters/5-RelatedWork.tex`), **Methods and experimental setup** (`chapters/6-Methods.tex`), **Results** (`chapters/7-Results.tex`), **Conclusions** (`chapters/8-Conclusions.tex`), **Appendices** (`chapters/9-Appendices.tex`), plus abstracts and `customs.tex`. Numeric tables are split under `thesis/masters_en/generated/results_table_*.tex` (one fragment per Results section); `results_tables.tex` bundles structural, encoding, and readout for convenience. Sync encoding/structural/readout fragments from `thesis/generated/tables.tex` after `build_results_tables.py` if numbers change; the script auto-refreshes `results_table_noisy_edges_excerpt.tex` and `results_tables.tex`.

From `thesis/masters_en/`, build with MiKTeX or TeX Live:

```powershell
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_thesis.ps1
```

`main.tex` loads **hyperref** and **csquotes** (with **cleveref** after hyperref) for PDF links and biblatex integration, uses **Latin Modern** instead of the template Nimbus fonts for portable installs, and uses the **biber** backend for `biblatex` (run `biber main` between the first and second `pdflatex` passes). If you must use the legacy BibTeX backend instead, change `backend=biber` to `backend=bibtex` in `main.tex` and run `bibtex main` in place of `biber main`.

Methods figures `toy_naive_vs_vchain.pdf` and `block_frqi_prep.pdf` live under `thesis/design/figures/`; if they are missing, run `python scripts/toy_and_block_level_diagrams.py` before building.

## Defence slides (Beamer)

Source lives under [`slides/main.tex`](slides/main.tex). Figures are loaded from `slides/assets/` (not committed if absent). After generating plots and diagrams, sync copies from `outputs/` and `thesis/design/figures/`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\sync_slide_assets.ps1
```

Then build from `slides/` with two `pdflatex` passes so references settle:

```powershell
cd slides
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Speaker notes use Beamer's `\note{...}` (open the PDF in a notes-capable viewer, or use two screens). Backup frames after the thank-you slide list the thesis Appendix~B script order.

## License

The repository is licensed under the [MIT License](LICENSE).

## Thesis results ingestion

The `outputs/` contract is documented in `thesis/outputs_registry.md` (see `thesis/results_bundle.yaml` for the loader). After you populate `outputs/` with CSVs and manifests, run:

```powershell
python scripts\build_results_tables.py
```

This validates required columns, warns when CSV timestamps disagree with linked `experiment_manifest_*.json` (use `--strict` to fail on those checks and on missing `expected_figures`), and writes Markdown/LaTeX fragments plus plots under `thesis/generated/` (gitignored). It also writes `thesis/masters_en/generated/results_table_noisy_edges_excerpt.tex` and refreshes `results_tables.tex`. Use `--require-figures` if PNGs under `outputs/` must exist before building the thesis.


## Run demos

### FRQI, QHED and Sobel Edge Detection

```powershell
python scripts\frqi_qhed_sobel.py
```

This writes the following artifacts in folder `outputs/`:

- **Figures**: for each test image, a comparison image is being created (original, QHED, Sobel) and a reconstruction image (original vs FRQI reconstruction).
- **Metrics**: a CSV file with one row per image. Columns include important statistics gathered while running the demo.

The printed demo output includes both the legacy function `ssim_like` score and **scikit-image SSIM** (function `ssim_uint8`).

### FRQI resource counts

Transpiled depth and CX counts in the CSV refer to Qiskit synthesis of the **exact** FRQI statevector via `QuantumCircuit.initialize()` (`frqi_circuit_kind` = `initialize`), using basis gates `cx`, `rz`, `sx` at optimization level 3. They do **not** yet describe a hand-built FRQI circuit with explicit multi-controlled rotations; that structural baseline is intended for the improvement chapter.

### Structural and Noisy FRQI

Generate artifacts:

```powershell
python scripts\frqi_structural_resources.py
```

Optional: emit a **layout-constrained** transpile-only CSV using a linear-chain `GenericBackendV2` whose width matches each structural circuit (pairs with the mock-device story in the thesis):

```powershell
python scripts\frqi_structural_resources.py --emit-layout-csv
```

This additionally writes `outputs/frqi_structural_metrics_constrained.csv`.

Noisy FRQI sweep (defaults: `test_4x4`, `test_8x8`, `test_16x16`; **16×16 v-chain** is skipped unless `--allow-heavy-dm` because density-matrix simulation at 16 qubits needs very large RAM):

```powershell
python scripts\noisy_frqi_sweep.py
```

“Mock NISQ” bundle (Aer `NoiseModel.from_backend` on a `GenericBackendV2` line topology + `optimization_level=3`, plus provenance JSON):

```powershell
python scripts\noisy_frqi_sweep.py --mock-nisq-bundle
```

This writes `outputs/noisy_frqi_metrics_mock_nisq_bundle.csv`, matching curve PNGs, and `outputs/experiment_manifest_noisy_frqi_mock_nisq_bundle.json`.

Synthetic ablations can load YAML presets from `data/experiment_presets/` (weak/medium/strong), for example:

```powershell
python scripts\noisy_frqi_sweep.py --noise-mode yaml_preset --yaml-preset data/experiment_presets/medium.yaml --topology linear --transpile-optimization-level 3
```

Optional: edge-map quality after noisy reconstruction (QHED on the recon vs Sobel on the original):

```powershell
python scripts\noisy_recon_qhed_edge_metrics.py
```

Readout calibration + inversion slice study (multi-seed + bootstrap bands + figure):

```powershell
python scripts\readout_mitigation_shot_sweep.py
```

### Experiment provenance (manifest JSON)

The sweep scripts write small JSON manifests under `outputs/` (Qiskit/Aer versions, topology/noise mode, seeds, flags, output paths, and `git rev-parse HEAD` when available), for example:

- `experiment_manifest_noisy_frqi_sweep.json` (default noisy sweep)
- `experiment_manifest_noisy_frqi_mock_nisq_bundle.json` (when using `--mock-nisq-bundle`)
- `experiment_manifest_noisy_recon_qhed_edges.json`
- `experiment_manifest_readout_mitigation_shot_sweep.json`

Artifacts land in `outputs/` (`frqi_structural_metrics.csv`, `fig_cx_vs_image_size.png`, `noisy_frqi_metrics.csv`, `noisy_frqi_*_curves.png`, etc.). For a full **16×16 v-chain** noisy grid on a capable machine, use e.g. `python scripts\noisy_frqi_sweep.py --allow-heavy-dm --scales 0,0.1,0.2` to keep runtime manageable.

**Reproducibility note:** `requirements.txt` pins compatible Qiskit 1.2.x / Aer 0.14.x ranges; exact runtime versions are also captured in the manifest JSON files.
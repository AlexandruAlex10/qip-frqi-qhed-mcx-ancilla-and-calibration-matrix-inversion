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
- Demo script
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


## Run tests

```powershell
pytest
```


## Run demo

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

Noisy FRQI sweep (defaults: `test_4x4`, `test_8x8`, `test_16x16`; **16×16 v-chain** is skipped unless `--allow-heavy-dm` because density-matrix simulation at 16 qubits needs very large RAM):

```powershell
python scripts\noisy_frqi_sweep.py
```

Optional: edge-map quality after noisy reconstruction (QHED on the recon vs Sobel on the original):

```powershell
python scripts\noisy_recon_qhed_edge_metrics.py
```

Artifacts land in `outputs/` (`frqi_structural_metrics.csv`, `fig_cx_vs_image_size.png`, `noisy_frqi_metrics.csv`, `noisy_frqi_*_curves.png`, etc.). For a full **16×16 v-chain** noisy grid on a capable machine, use e.g. `python scripts\noisy_frqi_sweep.py --allow-heavy-dm --scales 0,0.1,0.2` to keep runtime manageable.
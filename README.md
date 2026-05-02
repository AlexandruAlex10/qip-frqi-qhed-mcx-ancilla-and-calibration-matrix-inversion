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
python scripts\demo_frqi_qhed_sobel.py
```

This saves comparison figures in `outputs/`.
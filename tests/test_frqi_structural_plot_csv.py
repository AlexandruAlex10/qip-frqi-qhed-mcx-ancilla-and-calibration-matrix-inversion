"""
Smoke test: structural metrics plotter reads CSV and writes PNGs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_plotter():
    path = ROOT / "scripts" / "frqi_structural_resources.py"
    spec = importlib.util.spec_from_file_location("frqi_structural_resources", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["frqi_structural_resources"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_plot_structural_metrics_csv_smoke(tmp_path: Path) -> None:
    mod = _load_plotter()
    csv_path = tmp_path / "m.csv"
    rows = [
        {"image_size": 4, "kind": "struct_vchain", "cx": 100, "depth": 200},
        {"image_size": 8, "kind": "struct_vchain", "cx": 400, "depth": 800},
        {"image_size": 16, "kind": "struct_vchain", "cx": 900, "depth": 1800},
        {"image_size": 4, "kind": "struct_naive_slice_scaled", "cx": 300, "depth": 600},
        {"image_size": 8, "kind": "struct_naive_slice_scaled", "cx": 1200, "depth": 2400},
        {"image_size": 16, "kind": "struct_naive_slice_scaled", "cx": 5000, "depth": 10000},
        {"image_size": 4, "kind": "struct_naive_full", "cx": 250, "depth": 500},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    cx_p, d_p = mod.plot_structural_metrics_csv(csv_path, tmp_path)
    assert cx_p.is_file() and d_p.is_file()

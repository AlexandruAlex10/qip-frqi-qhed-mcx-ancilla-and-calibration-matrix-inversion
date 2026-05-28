# Thesis figures — traceability

Canonical raster outputs live under **`outputs/`** after you run the experiment scripts. Prefer citing those paths in LaTeX (`\\includegraphics` relative to your main `.tex` location). The ingestion pipeline (`scripts/build_results_tables.py`) checks several of these paths and can emit reproducible copies under `thesis/generated/figures/` when originals are absent.

| Thesis intent | Expected file under `outputs/` | Source script |
|---------------|-------------------------------|---------------|
| CNOT count vs image size (structural) | `fig_cx_vs_image_size.png` | `scripts/frqi_structural_resources.py` |
| Circuit depth vs image size | `fig_depth_vs_image_size.png` | `scripts/frqi_structural_resources.py` |
| Ideal FRQI vs QHED vs Sobel panel | `<image>_comparison.png` | `scripts/frqi_qhed_sobel.py` |
| FRQI reconstruction | `<image>_frqi_recon.png` | `scripts/frqi_qhed_sobel.py` |
| Noisy edge metrics curves | `noisy_recon_qhed_edges_<image>.png` | `scripts/noisy_recon_qhed_edge_metrics.py` |
| Noisy FRQI PSNR/SSIM curves | `noisy_frqi_metrics_<image>_curves.png` | `scripts/noisy_frqi_sweep.py` |
| Readout mitigation sweep | `readout_mitigation_shot_sweep.png` | `scripts/readout_mitigation_shot_sweep.py` |

Replace `<image>` with identifiers such as `test_4x4`, `test_8x8`, `test_16x16` matching your CSV rows.

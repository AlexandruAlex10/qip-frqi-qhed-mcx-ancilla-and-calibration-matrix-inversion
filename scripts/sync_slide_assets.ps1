# Copy figures expected by slides/main.tex into slides/assets/.
# Run from repo root after generating outputs / thesis design figures.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "slides\assets"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$pairs = @(
    @{ Src = "outputs\fig_cx_vs_image_size.png"; Dest = "fig_cx_vs_image_size.png" },
    @{ Src = "outputs\fig_depth_vs_image_size.png"; Dest = "fig_depth_vs_image_size.png" },
    @{ Src = "outputs\noisy_frqi_test_4x4_curves.png"; Dest = "noisy_frqi_test_4x4_curves.png" },
    @{ Src = "outputs\readout_mitigation_shot_sweep.png"; Dest = "readout_mitigation_shot_sweep.png" },
    @{ Src = "outputs\stub_mcx_vchain.png"; Dest = "stub_mcx_vchain.png" },
    @{ Src = "thesis\design\figures\toy_naive_vs_vchain.pdf"; Dest = "toy_naive_vs_vchain.pdf" },
    @{ Src = "thesis\design\figures\block_frqi_prep.pdf"; Dest = "block_frqi_prep.pdf" }
)

$missing = @()
foreach ($p in $pairs) {
    $srcPath = Join-Path $root $p.Src
    $dstPath = Join-Path $dest $p.Dest
    if (Test-Path $srcPath) {
        Copy-Item -Force $srcPath $dstPath
        Write-Host "OK: $($p.Src) -> slides\assets\$($p.Dest)"
    } else {
        $missing += $p.Src
    }
}

if ($missing.Count -gt 0) {
    Write-Warning "Missing sources (build slides after generating these):"
    $missing | ForEach-Object { Write-Warning "  $_" }
    exit 1
}
Write-Host "All slide assets synced."
exit 0

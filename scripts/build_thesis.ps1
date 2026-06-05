# Build thesis PDF from repo root (biblatex + biber + hyperref).
$ErrorActionPreference = "Stop"
$thesisDir = Join-Path $PSScriptRoot ".." "thesis" "bachelors_en" | Resolve-Path
Push-Location $thesisDir
try {
  pdflatex -interaction=nonstopmode main.tex
  if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 1 failed" }
  biber main
  if ($LASTEXITCODE -ne 0) { throw "biber failed" }
  pdflatex -interaction=nonstopmode main.tex
  if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 2 failed" }
  pdflatex -interaction=nonstopmode main.tex
  if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 3 failed" }
}
finally {
  Pop-Location
}

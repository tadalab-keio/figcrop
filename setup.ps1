# figcrop setup (Windows / PowerShell)
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Device xpu   # cpu | cuda | xpu
# Creates a project-local .venv and installs requirements.txt + device-specific torch.
# Nothing is installed system-wide. Verified: Python 3.13 / mineru[core] / torch 2.12 / openvino 2026 / pymupdf.
param([ValidateSet("cpu","cuda","xpu")][string]$Device = "cpu")
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venv = Join-Path $root ".venv"
$vpy  = Join-Path $venv "Scripts\python.exe"

# 1) uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  if (Get-Command scoop -ErrorAction SilentlyContinue) { scoop install uv }
  else { powershell -c "irm https://astral.sh/uv/install.ps1 | iex" }
}
# 2) Python 3.13 (MinerU supports 3.10-3.13)
if (-not (py -3.13 -c "print(1)" 2>$null)) {
  winget install --id Python.Python.3.13 -e --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
}
$py313 = (py -3.13 -c "import sys;print(sys.executable)")
# 3) venv + base deps
uv venv --python $py313 $venv
uv pip install --python $vpy -r (Join-Path $root "requirements.txt")
# 4) device-specific torch (same version, +cuda/+xpu variant -> keeps MinerU compatible)
$tv  = (& $vpy -c "import torch;print(torch.__version__.split('+')[0])")
$tvv = (& $vpy -c "import torchvision;print(torchvision.__version__.split('+')[0])")
switch ($Device) {
  "cuda" { uv pip install --python $vpy --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple "torch==$tv" "torchvision==$tvv" }
  "xpu"  { uv pip install --python $vpy --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/xpu  --extra-index-url https://pypi.org/simple "torch==$tv+xpu" "torchvision==$tvv+xpu" }
  "cpu"  { Write-Output "CPU: using torch from requirements" }
}
Write-Output "=== done. run: ==="
Write-Output ("  " + $vpy + " " + (Join-Path $root "figtools.py") + " serve auto")
Write-Output "  (then POST http://127.0.0.1:8077/extract  body: pdf,out_dir,figs?,top?)"

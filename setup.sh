#!/usr/bin/env bash
# figcrop setup (Linux/macOS).   bash setup.sh [cpu|cuda]
# Creates a project-local .venv and installs requirements.txt + device-specific torch.
set -euo pipefail
DEVICE="${1:-cpu}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"; VPY="$VENV/bin/python"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13 "$VENV"
uv pip install --python "$VPY" -r "$ROOT/requirements.txt"

TV=$("$VPY" -c "import torch;print(torch.__version__.split('+')[0])")
TVV=$("$VPY" -c "import torchvision;print(torchvision.__version__.split('+')[0])")
case "$DEVICE" in
  cuda) uv pip install --python "$VPY" --index-strategy unsafe-best-match \
        --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
        "torch==$TV" "torchvision==$TVV" ;;
  cpu)  echo "CPU: using torch from requirements" ;;
  *)    echo "device must be cpu|cuda (xpu/NPU is Windows+Intel: use setup.ps1 -Device xpu)"; exit 1 ;;
esac
echo "=== done. run: ==="
echo "  $VPY $ROOT/figtools.py serve auto    # OpenVINO server on 127.0.0.1:8077 (IR auto-built on first run)"

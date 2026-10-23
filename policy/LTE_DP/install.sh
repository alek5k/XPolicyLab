#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${XPL_ROOT}" 'numpy>=1.23,<2' 'opencv-python-headless>=4.8,<4.12' \
    'huggingface-hub>=0.34,<1' 'torch>=2.4' 'torchvision>=0.19' 'diffusers==0.35.2' \
    'einops>=0.8,<0.9' 'pytest>=8,<9' 'hydra-core==1.3.2' 'omegaconf==2.3.0' \
    'dill>=0.3.8,<0.4' 'zarr>=2.12,<3' 'numba>=0.60,<0.64' 'pandas>=2,<3'
# Native DP declares Hydra 1.2. Use the Python-compatible Hydra 1.3 runtime
# above while installing the actual package without changing its source metadata.
python -m pip install -e "${XPL_ROOT}/policy/DP" --no-deps
python -c 'from XPolicyLab.policy.LTE_DP.workspace import LTEWorkspace; print("LTE_DP native DP import OK")'

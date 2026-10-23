#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Share the baseline adapter's exact source tree, ref, environment and dependencies.
bash "${POLICY_DIR}/../SmolVLA/install.sh"
source "${POLICY_DIR}/../SmolVLA/conda_init.sh"
smolvla_activate_conda "${SMOVLA_CONDA_ENV:-smolvla}"
export PYTHONPATH="$(cd "${POLICY_DIR}/../../.." && pwd):${PYTHONPATH:-}"
python -c 'from XPolicyLab.policy.LTE_SmolVLA.configuration_lte_smolvla import LTESmolVLAConfig; from XPolicyLab.policy.LTE_SmolVLA.modeling_lte_smolvla import LTESmolVLAPolicy; import lerobot; print("LTE_SmolVLA base:", lerobot.__file__)'

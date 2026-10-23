#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 bench_name ckpt_name env_cfg_type action_type seed gpu_id [options] [native.key=value ...]" >&2
    exit 2
fi
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd -- "${XPL_ROOT}/.." && pwd)"
dimension_args=()
if [[ "${action_type}" == joint ]]; then
    action_dim=$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
    dimension_args=(--action-dim "${action_dim}")
fi
export CUDA_VISIBLE_DEVICES="${gpu_id}"
exec python -m XPolicyLab.policy.LTE_DP.train "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}" "${dimension_args[@]}" "$@"

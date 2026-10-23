#!/usr/bin/env bash
set -Eeuo pipefail
bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# Both standard Conda names and an explicit virtualenv path are supported.
if [[ -f "${policy_env}/bin/activate" ]]; then
    source "${policy_env}/bin/activate"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_env}"
fi
export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
exec python "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${SCRIPT_DIR}/deploy.yml" --overrides \
    port="${policy_server_port}" host="${policy_server_host}" \
    bench_name="${bench_name}" task_name="${task_name}" ckpt_name="${ckpt_name}" \
    env_cfg_type="${env_cfg_type}" seed="${seed}" policy_name=LTE_DP action_type="${action_type}"

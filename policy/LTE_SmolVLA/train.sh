#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 bench_name ckpt_name env_cfg_type joint seed gpu_id [trainer overrides...]" >&2
    exit 1
fi
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6
shift 6
[[ "$action_type" == joint ]] || { echo 'Only joint actions are supported' >&2; exit 1; }
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
source "${POLICY_DIR}/../SmolVLA/conda_init.sh"
smolvla_activate_conda "${SMOVLA_CONDA_ENV:-smolvla}"
export PYTHONPATH="$(dirname "$XPL_ROOT"):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$gpu_id"
action_dim=$(bash "$XPL_ROOT/utils/get_action_dim.sh" "$(dirname "$XPL_ROOT")" "$env_cfg_type")
echo "[LTE_SmolVLA] robot action dim=${action_dim} (validated against dataset before training)"
export LTE_EXPECTED_ACTION_DIM="$action_dim"
run_name=$(python -c 'import sys; from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name; print(build_run_dir_name(dict(zip(("bench_name", "ckpt_name", "env_cfg_type", "action_type", "seed"), sys.argv[1:]))))' "$bench_name" "$ckpt_name" "$env_cfg_type" "$action_type" "$seed")
dataset_root="${LTE_DATASET_ROOT:-${POLICY_DIR}/data/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
exec python -m XPolicyLab.policy.LTE_SmolVLA.train \
    --policy.type=lte_smolvla \
    --policy.base_policy_path="${LTE_BASE_POLICY:-lerobot/smolvla_base}" \
    --policy.device="${LTE_DEVICE:-cuda}" --policy.push_to_hub=false \
    --dataset.repo_id="${SMOVLA_REPO_ID:-RoboDojo_sim_${ckpt_name}_v30}" \
    --dataset.root="$dataset_root" --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=false \
    --output_dir="${POLICY_DIR}/checkpoints/${run_name}" \
    --batch_size=8 --steps=100000 --save_freq=10000 --log_freq=10 \
    --num_workers=4 --wandb.enable=false --eval_freq=0 --seed="$seed" \
    --rename_map='{"observation.images.cam_high":"observation.images.camera1","observation.images.cam_left_wrist":"observation.images.camera2","observation.images.cam_right_wrist":"observation.images.camera3"}' \
    "$@"

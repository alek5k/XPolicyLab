#!/usr/bin/env bash
# Run with the pinned SmolVLA environment active; root contains make_smoke_fixture output.
set -euo pipefail
root=$1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="$root/hf-datasets-cache"
export OMP_NUM_THREADS=2
exec python -m XPolicyLab.policy.LTE_SmolVLA.train \
    --policy.type=lte_smolvla --policy.base_policy_path=null \
    --policy.device=cpu --policy.push_to_hub=false --policy.vlm_model_name="$root/vlm" \
    --policy.num_vlm_layers=2 --policy.num_expert_layers=2 --policy.expert_width_multiplier=1.0 \
    --policy.temporal_latent_dim=8 --policy.temporal_hidden_dim=32 --policy.history_decoder_hidden_dim=32 \
    --policy.resize_imgs_with_padding='[16,16]' --policy.chunk_size=3 --policy.n_action_steps=2 --policy.num_steps=2 \
    --dataset.repo_id=local/lte-smoke --dataset.root="$root/dataset" --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=false --output_dir="$root/run" \
    --steps=2 --batch_size=2 --num_workers=0 --save_freq=2 --log_freq=1 --eval_freq=0 --wandb.enable=false \
    --rename_map='{"observation.images.cam_high":"observation.images.camera1","observation.images.cam_left_wrist":"observation.images.camera2","observation.images.cam_right_wrist":"observation.images.camera3"}'

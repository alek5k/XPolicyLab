"""Scoped integration with the trainer shipped by XPolicyLab's SmolVLA.

Run as python -m XPolicyLab.policy.LTE_SmolVLA.train. No global LTE-DP
plugins or modifications to the baseline adapter/LeRobot installation.
"""

from pathlib import Path
import os
from dataclasses import fields

from .configuration_lte_smolvla import LTESmolVLAConfig

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts import lerobot_train
from lerobot.processor import RenameObservationsProcessorStep
from .multi_dataset import LTEMultiLeRobotDataset


CAMERA_RENAME = {
    "observation.images.cam_high": "observation.images.camera1",
    "observation.images.cam_left_wrist": "observation.images.camera2",
    "observation.images.cam_right_wrist": "observation.images.camera3",
}


def inherit_base_config(config, explicit_fields):
    """Keep pretrained SmolVLA settings unless the user explicitly overrides them."""
    base = PreTrainedConfig.from_pretrained(config.base_policy_path)
    if base.type != "smolvla":
        raise ValueError("base_policy_path must be a baseline SmolVLA; use policy.path for LTE finetuning")
    base_fields = {field.name for field in fields(SmolVLAConfig) if field.init}
    values = {name: getattr(base, name) for name in base_fields}
    for field in fields(config):
        if field.init and (field.name not in base_fields or field.name in explicit_fields):
            values[field.name] = getattr(config, field.name)
    for name in ("device", "push_to_hub", "repo_id", "pretrained_path"):
        values[name] = getattr(config, name)
    values["load_vlm_weights"] = False
    if "temporal_embedding_cache_refresh_steps" not in explicit_fields:
        values["temporal_embedding_cache_refresh_steps"] = 5000
    return LTESmolVLAConfig(**values)


def configure_history(config, dataset, rename_map):
    episodes = dataset.meta.episodes
    starts = torch.as_tensor(episodes["dataset_from_index"], dtype=torch.long)
    ends = torch.as_tensor(episodes["dataset_to_index"], dtype=torch.long)
    if starts.numel() == 0 or starts[0] != 0 or not torch.equal(starts[1:], ends[:-1]):
        raise ValueError("Expected contiguous absolute episode boundaries in LeRobot v3 metadata")
    longest = int((ends - starts).max())
    if longest > config.max_causal_history_length:
        raise ValueError(f"Episode length {longest} exceeds max_causal_history_length={config.max_causal_history_length}; increase it (history is never truncated)")
    if config.history_max_episode_length not in (None, longest):
        raise ValueError("Checkpoint history lag scale disagrees with this dataset")
    config.history_max_episode_length = longest
    config._runtime_causal_prefix_length = longest
    config._runtime_episode_ends = ends.tolist()
    config._runtime_dataset = dataset
    config._runtime_rename_map = dict(rename_map)
    features = dataset_to_policy_features(dataset.meta.features)
    images = {rename_map.get(key, key): value for key, value in features.items() if value.type == FeatureType.VISUAL}
    expected = list(rename_map.values())
    if set(images) != set(expected):
        raise ValueError(f"Configured image rename_map expects {expected}, got {list(images)}")
    expected_input_features = {
        "observation.state": features["observation.state"],
        **{key: images[key] for key in expected},
    }
    expected_output_features = {"action": features["action"]}
    if config.input_features != expected_input_features:
        raise ValueError(
            "Configured policy.input_features disagrees with the dataset schema after rename_map: "
            f"configured={config.input_features}, dataset={expected_input_features}"
        )
    if config.output_features != expected_output_features:
        raise ValueError(
            "Configured policy.output_features disagrees with the dataset schema: "
            f"configured={config.output_features}, dataset={expected_output_features}"
        )
    expected = os.environ.get("LTE_EXPECTED_ACTION_DIM")
    if expected is not None and features["action"].shape[0] != int(expected):
        raise ValueError(f"Dataset action dimension disagrees with configured robot ({expected})")
    if features["action"].shape[0] > config.max_action_dim or features["observation.state"].shape[0] > config.max_state_dim:
        raise ValueError("Robot dimensions exceed the SmolVLA checkpoint's padded dimensions")


def load_base_weights(policy, base_path):
    """Allow only newly introduced LTE tensors to be absent from base weights."""
    path = Path(base_path).expanduser()
    weights = str(path / "model.safetensors") if path.is_dir() else hf_hub_download(base_path, "model.safetensors")
    missing, unexpected = load_model(policy, weights, strict=False)
    allowed = ("model.temporal_encoder.", "model.history_decoder.", "model.temporal_to_vlm.",
               "model._temporal_embedding_", "model._lte_optimizer_step")
    incompatible = [key for key in missing if not key.startswith(allowed)]
    if incompatible or unexpected:
        raise ValueError(f"Incompatible SmolVLA base checkpoint: missing={incompatible}, unexpected={unexpected}")


def main():
    original_dataset = lerobot_train.make_dataset
    original_policy = lerobot_train.make_policy
    original_processors = lerobot_train.make_pre_post_processors
    original_update_policy = lerobot_train.update_policy

    def make_dataset(cfg):
        if not isinstance(cfg.policy, LTESmolVLAConfig):
            raise ValueError("Use --policy.type=lte_smolvla for fresh runs, or --policy.path for an LTE checkpoint")
        if cfg.dataset.streaming or cfg.dataset.image_transforms.enable:
            raise ValueError("LTE history requires a finite dataset with image augmentation disabled")
        if cfg.env is not None or cfg.peft is not None or cfg.use_rabc:
            raise ValueError("Use XPolicyLab eval.sh for evaluation; PEFT and RA-BC are unsupported")
        if not cfg.rename_map:
            raise ValueError("LTE SmolVLA requires an explicit image rename_map")
        if cfg.dataset.repo_id.startswith("lte-multi:"):
            repo_ids = [repo_id for repo_id in cfg.dataset.repo_id.removeprefix("lte-multi:").split(",") if repo_id]
            if len(repo_ids) < 2:
                raise ValueError("lte-multi dataset spec requires at least two repository IDs")
            dataset = LTEMultiLeRobotDataset(
                repo_ids,
                root=cfg.dataset.root,
                video_backend=cfg.dataset.video_backend,
                policy=cfg.policy,
                image_transforms=None,
                revision=cfg.dataset.revision,
            )
        else:
            dataset = original_dataset(cfg)
        configure_history(cfg.policy, dataset, cfg.rename_map)
        return dataset

    def make_policy(cfg, **kwargs):
        policy = original_policy(cfg=cfg, **kwargs)
        if cfg.pretrained_path is None and cfg.base_policy_path:
            load_base_weights(policy, cfg.base_policy_path)
        # Build exactly one of the two mutually exclusive temporal stores.
        # Query-only training keeps only uint8 RGB on GPU and never builds an
        # embedding cache.
        if policy.config.temporal_embedding_cache_enabled:
            # Keep this separate from the embedding-table timer. The raw cache
            # removes source-image decoding from the full-table build, while
            # W&B reports both one-time costs independently.
            if policy.config.temporal_image_cache_on_gpu or policy.config.temporal_image_cache_in_ram:
                policy.model._temporal_image_cache.preload(
                    torch.device(cfg.device),
                    chunk_size=policy.config.temporal_image_cache_build_chunk_size,
                )
            policy.model._rebuild_embedding_cache(int(policy.model._lte_optimizer_step), torch.device(cfg.device))
        elif policy.config.temporal_image_cache_on_gpu or policy.config.temporal_image_cache_in_ram:
            policy.model._temporal_image_cache.preload(
                torch.device(cfg.device),
                chunk_size=policy.config.temporal_image_cache_build_chunk_size,
            )
        return policy

    def update_policy_with_lte_timing(*args, **kwargs):
        accelerator = kwargs.get("accelerator")
        if accelerator is None and len(args) >= 6:
            accelerator = args[5]
        device = accelerator.device if accelerator is not None else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = __import__("time").perf_counter()
        tracker, output = original_update_policy(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_step_seconds = __import__("time").perf_counter() - started
        policy = args[1] if len(args) > 1 else kwargs["policy"]
        model = accelerator.unwrap_model(policy, keep_fp32_wrapper=True).model if accelerator is not None else policy.model
        query_seconds = float(model._last_temporal_vision_seconds)
        output["temporal_query_total_step_seconds"] = total_step_seconds
        output["temporal_query_vision_pct_of_step"] = 100.0 * query_seconds / max(total_step_seconds, 1e-12)
        for metric in (
            "temporal_vision_encoder_seconds",
            "temporal_encoder_seconds",
            "temporal_decoder_loss_seconds",
            "action_forward_seconds",
            "action_loss_seconds",
        ):
            output[f"{metric}_pct_of_step"] = 100.0 * float(output[metric]) / max(total_step_seconds, 1e-12)
        return tracker, output

    def make_processors(policy_cfg, **kwargs):
        pre, post = original_processors(policy_cfg, **kwargs)
        for step in pre.steps:
            if isinstance(step, RenameObservationsProcessorStep):
                step.rename_map = dict(policy_cfg._runtime_rename_map)
        return pre, post

    lerobot_train.make_dataset = make_dataset
    lerobot_train.make_policy = make_policy
    lerobot_train.make_pre_post_processors = make_processors
    lerobot_train.update_policy = update_policy_with_lte_timing
    @parser.wrap()
    def run(cfg: TrainPipelineConfig):
        if (not cfg.resume and not parser.get_path_arg("policy")
                and not parser.parse_arg("config_path") and cfg.policy is not None
                and isinstance(cfg.policy, LTESmolVLAConfig) and cfg.policy.base_policy_path):
            overrides = parser.get_cli_overrides("policy") or []
            explicit = {arg.removeprefix("--").split("=", 1)[0].split(".", 1)[0] for arg in overrides}
            cfg.policy = inherit_base_config(cfg.policy, explicit)
        lerobot_train.train(cfg)
    try:
        run()
    finally:
        lerobot_train.make_dataset = original_dataset
        lerobot_train.make_policy = original_policy
        lerobot_train.make_pre_post_processors = original_processors
        lerobot_train.update_policy = original_update_policy


if __name__ == "__main__":
    main()

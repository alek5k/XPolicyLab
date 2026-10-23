"""XPolicyLab adapter: native SmolVLA I/O with per-environment LTE memory."""

from pathlib import Path

import torch

from .configuration_lte_smolvla import LTESmolVLAConfig
from .modeling_lte_smolvla import LTESmolVLAPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from XPolicyLab.policy.SmolVLA.model import Model as SmolVLAModel, make_lerobot_observation
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import unpack_robot_state

_POLICY_DIR = Path(__file__).resolve().parent


class Model(SmolVLAModel):
    def __init__(self, model_cfg):
        if not (model_cfg.get("env_cfg_type") or model_cfg.get("env_cfg")):
            raise ValueError("env_cfg_type is required for XPolicyLab robot dimensions")
        super().__init__(model_cfg)
        expected = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        if self.policy.config.action_feature.shape[0] != expected:
            raise ValueError(f"Checkpoint action dimension does not match robot dimension {expected}")
        if self.policy.config.robot_state_feature.shape[0] != expected:
            raise ValueError(f"Checkpoint state dimension does not match robot dimension {expected}")
        if not 1 <= self.actions_per_chunk <= self.policy.config.chunk_size:
            raise ValueError("actions_per_chunk must be within the checkpoint chunk_size")
        self.reset()

    def _resolve_pretrained_path_from_candidates(self):
        candidates = candidate_checkpoint_roots(
            self.model_cfg, _POLICY_DIR / "checkpoints", policy_dir=_POLICY_DIR,
            explicit_keys=("pretrained_path", "model_path", "checkpoint_path"),
        )
        for root in candidates:
            try:
                return self._resolve_pretrained_path(root)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"No LTE_SmolVLA checkpoint found in {candidates}")

    def _load_policy(self):
        config = PreTrainedConfig.from_pretrained(self.pretrained_path)
        if not isinstance(config, LTESmolVLAConfig):
            raise ValueError("Evaluation requires a trained LTE_SmolVLA checkpoint")
        config.device = str(self.device)
        config.load_vlm_weights = False
        if self.model_cfg.get("smovla_vlm_path"):
            assets = Path(self.model_cfg["smovla_vlm_path"]).expanduser()
            assets = assets if assets.is_absolute() else _POLICY_DIR / assets
            if not assets.is_dir():
                raise FileNotFoundError(f"VLM assets not found: {assets}")
            config.vlm_model_name = str(assets.resolve())
        policy = LTESmolVLAPolicy.from_pretrained(self.pretrained_path, config=config, strict=True)
        if not bool(policy.model._temporal_embedding_stats_ready):
            raise ValueError("Checkpoint has no fitted LTE embedding statistics; train before evaluation")
        return policy.to(self.device).eval()

    def _build_processors(self):
        overrides = {
            "device_processor": {"device": str(self.device)},
            "rename_observations_processor": {"rename_map": {}},
        }
        if self.model_cfg.get("smovla_vlm_path"):
            overrides["tokenizer_processor"] = {"tokenizer_name": self.policy.config.vlm_model_name}
        return make_pre_post_processors(
            self.policy.config, pretrained_path=self.pretrained_path,
            preprocessor_overrides=overrides,
            postprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    @torch.inference_mode()
    def update_obs(self, obs):
        self.update_obs_batch([obs])

    @torch.inference_mode()
    def update_obs_batch(self, obs_list):
        if not obs_list:
            self._latest_env_idx_list = []
            return
        ids = [int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate env_idx values in observation batch")
        # XPolicyLab's instruction is authoritative; preserve its plural fallback.
        obs_list = [dict(obs, prompt=obs.get("instruction") or obs.get("instructions")
                         or obs.get("prompt") or self.default_prompt) for obs in obs_list]
        super().update_obs_batch(obs_list)
        for env_idx in ids:
            batch = self.preprocessor(make_lerobot_observation(self._latest_payloads[env_idx]))
            self._temporal_states[env_idx] = self.policy.advance_temporal_state(
                batch, self._temporal_states.get(env_idx)
            )
            self._processed_observations[env_idx] = batch

    @torch.inference_mode()
    def get_action(self):
        if not self._latest_env_idx_list:
            raise RuntimeError("update_obs must precede get_action")
        return self.get_action_batch([self._latest_env_idx_list[0]])[0]

    @torch.inference_mode()
    def get_action_batch(self, env_idx_list=None, **kwargs):
        ids = self._latest_env_idx_list if env_idx_list is None else [int(i) for i in env_idx_list]
        result = []
        # Independent calls preserve per-environment token padding and recurrence;
        # shrinking/reordered active batches cannot exchange memory.
        for env_idx in ids:
            if env_idx not in self._processed_observations:
                raise KeyError(f"No observation for env_idx={env_idx}")
            actions = self.policy.predict_with_latent(
                self._processed_observations[env_idx], self._temporal_states[env_idx]
            )
            actions = self._postprocess_action_chunk(actions)[0]
            result.append(unpack_robot_state(actions, self.action_type, self.robot_action_dim_info, source_type="obs"))
        return result

    def reset_envs(self, env_idx_list):
        ids = {int(env_idx) for env_idx in env_idx_list}
        for env_idx in ids:
            self._temporal_states.pop(env_idx, None)
            self._processed_observations.pop(env_idx, None)
            self._latest_payloads.pop(env_idx, None)
        self._latest_env_idx_list = [
            env_idx for env_idx in self._latest_env_idx_list if env_idx not in ids
        ]
        if self._latest_payloads:
            self._latest_payload = next(iter(self._latest_payloads.values()))
        else:
            self._latest_payload = None

    def reset(self):
        super().reset()
        self._latest_env_idx_list = []
        self._temporal_states = {}
        self._processed_observations = {}

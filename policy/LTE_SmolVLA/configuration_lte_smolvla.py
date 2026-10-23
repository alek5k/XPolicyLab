"""LTE configuration layered on XPolicyLab's SmolVLA v0.4.4 dependency."""

from dataclasses import dataclass

from .runtime import use_xpolicy_smolvla

use_xpolicy_smolvla()

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("lte_smolvla")
@dataclass
class LTESmolVLAConfig(SmolVLAConfig):
    temporal_latent_dim: int = 32
    temporal_hidden_dim: int = 128
    history_decoder_hidden_dim: int = 128
    num_history_queries: int = 16
    lambda_history: float = 1.0
    max_causal_history_length: int = 512
    history_max_episode_length: int | None = None
    temporal_num_images: int | None = 1
    temporal_embedding_cache_refresh_steps: int = 5000
    # Per-GPU number of frames encoded at a time while constructing the
    # detached embedding table. This is intentionally independent from the
    # on-demand temporal encoding chunk size used when no table is present.
    temporal_embedding_cache_build_batch_size: int = 32
    temporal_encode_chunk_size: int = 32
    # Raw RGB and persistent visual embeddings are intentionally separate.
    # Query-only training evaluates the vision tower every update and never
    # creates an embedding table.
    temporal_embedding_cache_enabled: bool = True
    temporal_image_cache_on_gpu: bool = False
    temporal_image_cache_in_ram: bool = False
    # torchrun defaults OMP_NUM_THREADS to one.  Raw-cache construction is a
    # one-time CPU image-conversion job, so temporarily restore useful CPU
    # parallelism while it runs, then return the trainer to its normal limit.
    temporal_image_cache_build_num_threads: int = 32
    temporal_image_cache_build_chunk_size: int = 2048
    temporal_query_history_only: bool = False
    temporal_embedding_normalize: bool = True
    # When the SmolVLA VLM is unfrozen, follow SimpleMemVLA's differential
    # optimization: its backbone uses a learning rate below the action-facing
    # modules.  Disabled by default to preserve existing LTE checkpoints.
    differential_lr: bool = False
    backbone_lr_scale: float = 0.1
    # Used only for a fresh training run. Resume/eval load their own full checkpoint.
    base_policy_path: str | None = "lerobot/smolvla_base"

    def __post_init__(self):
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError("XPolicyLab SmolVLA uses n_obs_steps=1; LTE supplies full visual history")
        if self.adapt_to_pi_aloha or self.use_delta_joint_actions_aloha:
            raise ValueError("This adapter uses XPolicyLab joint coordinates, without Aloha conversions")
        if self.rtc_config is not None or self.compile_model or self.use_peft:
            raise ValueError("RTC, torch.compile and PEFT are not supported by LTE_SmolVLA")
        for key in ("temporal_latent_dim", "temporal_hidden_dim", "num_history_queries",
                    "max_causal_history_length", "temporal_embedding_cache_build_batch_size",
                    "temporal_encode_chunk_size",
                    "temporal_image_cache_build_num_threads", "temporal_image_cache_build_chunk_size"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.history_decoder_hidden_dim <= self.temporal_latent_dim + 1:
            raise ValueError("history_decoder_hidden_dim must exceed temporal_latent_dim + 1")
        if self.lambda_history < 0 or self.temporal_embedding_cache_refresh_steps < 0:
            raise ValueError("Loss weight and cache refresh interval must be nonnegative")
        if not 0 < self.backbone_lr_scale <= 1:
            raise ValueError("backbone_lr_scale must be in (0, 1]")
        if self.history_max_episode_length is not None and self.history_max_episode_length <= 0:
            raise ValueError("history_max_episode_length must be positive")
        if self.train_expert_only:
            self.temporal_embedding_cache_refresh_steps = 0
        if self.temporal_image_cache_on_gpu and self.temporal_image_cache_in_ram:
            raise ValueError("Choose either a GPU-resident or shared-RAM temporal image cache, not both")

    def temporal_camera_indices(self):
        available = len(self.image_features)
        count = available if self.temporal_num_images is None else self.temporal_num_images
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= available:
            raise ValueError("temporal_num_images must select at least one available configured camera")
        return list(range(count))

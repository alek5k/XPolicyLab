"""Image-only LTE ported from LTE-DP onto XPolicyLab's pinned SmolVLA.

The native SmolVLA action model remains intact.  LTE mean-pools each selected
camera's post-connector visual tokens, reconstructs detached historical
features during training, and appends one projected temporal token to the VLM
prefix for both training and rollout.
"""

from __future__ import annotations

import math
import os
import time as wall_time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .configuration_lte_smolvla import LTESmolVLAConfig

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    resize_with_pad,
)
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from .temporal import (
    HistoryDecoder,
    ImageTemporalEncoder,
    causal_history_frame_indices,
    interpolate_history,
    sample_stratified_lags,
    should_refresh_embedding_cache,
)

from .dataset_cache import DatasetCache


class LTESmolVLAFlowMatching(VLAFlowMatching):
    """Native SmolVLA flow matcher with a visual LTE prefix token."""

    def __init__(self, config: LTESmolVLAConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.config: LTESmolVLAConfig = config
        self.temporal_camera_indices = config.temporal_camera_indices()
        self.temporal_image_keys = tuple(list(config.image_features)[i] for i in self.temporal_camera_indices)
        self.vlm_embedding_dim = int(self.vlm_with_expert.config.text_config.hidden_size)
        self.temporal_embedding_dim = self.vlm_embedding_dim * len(self.temporal_camera_indices)
        self.temporal_encoder = ImageTemporalEncoder(
            self.temporal_embedding_dim, config.temporal_latent_dim, config.temporal_hidden_dim
        )
        self.history_decoder = HistoryDecoder(
            config.temporal_latent_dim,
            self.temporal_embedding_dim,
            config.history_decoder_hidden_dim,
        )
        self.temporal_to_vlm = nn.Linear(config.temporal_latent_dim, self.vlm_embedding_dim)
        self._active_temporal_token: Tensor | None = None

        self.register_buffer(
            "_lte_episode_ends",
            torch.as_tensor(getattr(config, "_runtime_episode_ends", ()), dtype=torch.long),
            persistent=False,
        )
        self._temporal_image_cache = DatasetCache(
            getattr(config, "_runtime_dataset", None),
            list(self.temporal_image_keys),
            rename_map=getattr(config, "_runtime_rename_map", None),
            cache_on_gpu=config.temporal_image_cache_on_gpu,
            cache_in_ram=config.temporal_image_cache_in_ram,
            build_num_threads=config.temporal_image_cache_build_num_threads,
        )
        self._embedding_cache: Tensor | None = None
        self._embedding_cache_step = -1
        self._last_embedding_cache_build_seconds = 0.0
        # Fixed dataset-wide statistics for the detached visual target.  They
        # are fitted from the complete initial cache, saved in checkpoints,
        # and reused on cache rebuilds and live rollout observations.
        self.register_buffer(
            "_temporal_embedding_mean",
            torch.zeros(self.temporal_embedding_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "_temporal_embedding_std",
            torch.ones(self.temporal_embedding_dim, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "_temporal_embedding_stats_ready",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
        # Updated by ``LTESmolVLAPolicy.update`` only after a synchronized
        # optimizer update. Unlike a forward counter, this remains correct
        # under gradient accumulation and survives resume checkpoints.
        self.register_buffer("_lte_optimizer_step", torch.zeros((), dtype=torch.long), persistent=True)
        self._last_embedding_cache_hit_rate = 0.0
        self._last_temporal_vision_seconds = 0.0
        self._last_temporal_vision_encoder_seconds = 0.0
        self._last_temporal_encoder_seconds = 0.0
        self._last_temporal_decoder_loss_seconds = 0.0
        self._last_action_forward_seconds = 0.0
        self._last_action_loss_seconds = 0.0
        self._last_temporal_vision_batch_size = 0
        self._last_temporal_vision_encoder_calls = 0

    @contextmanager
    def use_temporal_token(self, token: Tensor) -> Iterator[None]:
        """Make one LTE token available to the unchanged SmolVLA forward path."""
        if token.ndim != 2 or token.shape[-1] != self.vlm_embedding_dim:
            raise ValueError("LTE token must have shape (batch, SmolVLA VLM hidden size)")
        previous = self._active_temporal_token
        self._active_temporal_token = token
        try:
            yield
        finally:
            self._active_temporal_token = previous

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state: Tensor | None = None):
        embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
        token = self._active_temporal_token
        if token is None:
            return embs, pad_masks, att_masks
        token = token.to(device=embs.device, dtype=embs.dtype).unsqueeze(1)
        token_mask = torch.ones(token.shape[:2], dtype=torch.bool, device=embs.device)
        # Match SmolVLA's state token: this is action-facing context, rather
        # than an image/language token which is prevented from seeing actions.
        token_attention = torch.ones(token.shape[:2], dtype=att_masks.dtype, device=embs.device)
        return (
            torch.cat((embs, token), dim=1),
            torch.cat((pad_masks, token_mask), dim=1),
            torch.cat((att_masks, token_attention), dim=1),
        )

    def temporal_token(self, latent: Tensor) -> Tensor:
        return self.temporal_to_vlm(latent)

    def _prepare_temporal_image(self, image: Tensor) -> Tensor:
        """Exactly SmolVLA's per-camera resize and SigLIP input conversion."""
        if image.dtype == torch.uint8:
            image = image.to(torch.float32).div(255.0)
        else:
            image = image.to(torch.float32)
        if self.config.resize_imgs_with_padding is not None:
            image = resize_with_pad(
                image,
                *self.config.resize_imgs_with_padding,
                pad_value=0,
            )
        return image.mul(2.0).sub(1.0)

    @torch.no_grad()
    def _pooled_temporal_features_from_raw(self, images: Tensor) -> Tensor:
        """Mean-pool post-connector tokens and concatenate selected cameras.

        ``images`` is ``(*prefix, selected_cameras, C, H, W)`` and comes from
        either the compact raw dataset cache or a current rollout observation.
        """
        if images.ndim < 5 or images.shape[-4] != len(self.temporal_camera_indices):
            raise ValueError("LTE images must end with (selected_cameras, C, H, W)")
        prefix = images.shape[:-4]
        flattened = images.reshape(-1, *images.shape[-4:])
        if flattened.device.type == "cuda":
            torch.cuda.synchronize(flattened.device)
        started = wall_time.perf_counter()
        self._last_temporal_vision_batch_size = int(flattened.shape[0])
        self._last_temporal_vision_encoder_calls = int(flattened.shape[1])
        per_camera: list[Tensor] = []
        for camera in range(flattened.shape[1]):
            visual_tokens = self.vlm_with_expert.embed_image(
                self._prepare_temporal_image(flattened[:, camera])
            )
            # SmolVLA scales post-connector features before sending them to
            # the VLM. Reconstruct that same representation, not raw SigLIP.
            visual_tokens = visual_tokens * math.sqrt(visual_tokens.shape[-1])
            per_camera.append(visual_tokens.mean(dim=1))
        if flattened.device.type == "cuda":
            torch.cuda.synchronize(flattened.device)
        self._last_temporal_vision_encoder_seconds += wall_time.perf_counter() - started
        return torch.cat(per_camera, dim=-1).reshape(*prefix, self.temporal_embedding_dim).detach()

    @torch.no_grad()
    def pooled_temporal_features_from_prepared(self, images: list[Tensor]) -> Tensor:
        """Pool already-prepared SmolVLA image tensors for online recurrence."""
        selected = [images[i] for i in self.temporal_camera_indices]
        per_camera: list[Tensor] = []
        for image in selected:
            tokens = self.vlm_with_expert.embed_image(image)
            tokens = tokens * math.sqrt(tokens.shape[-1])
            per_camera.append(tokens.mean(dim=1))
        return torch.cat(per_camera, dim=-1).detach()

    @torch.no_grad()
    def _normalize_temporal_embeddings(self, embeddings: Tensor, *, fit: bool = False) -> Tensor:
        """Apply fixed dataset normalization to detached visual embeddings.

        The first complete embedding-cache build fits one mean and standard
        deviation per feature over all selected dataset frames.  Refreshes do
        not change this coordinate system, and online observations use the
        same stored statistics as the history encoder and decoder targets.
        """
        if embeddings.shape[-1] != self.temporal_embedding_dim:
            raise ValueError("LTE embedding has an unexpected final dimension")
        if fit:
            reduce_dims = tuple(range(embeddings.ndim - 1))
            mean = embeddings.mean(dim=reduce_dims)
            std = embeddings.std(dim=reduce_dims, correction=0).clamp_min(1e-6)
            self._temporal_embedding_mean.copy_(mean.to(self._temporal_embedding_mean))
            self._temporal_embedding_std.copy_(std.to(self._temporal_embedding_std))
            self._temporal_embedding_stats_ready.fill_(True)
            print(
                "Fitted fixed dataset normalization for detached LTE SmolVLA embeddings.",
                flush=True,
            )
        if not self.config.temporal_embedding_normalize:
            return embeddings
        if not bool(self._temporal_embedding_stats_ready.item()):
            raise RuntimeError("LTE embedding normalization statistics have not been fitted")
        return (embeddings - self._temporal_embedding_mean) / self._temporal_embedding_std

    def _cache_step_for_batch(self) -> int | None:
        if not self.training or not self.config.temporal_embedding_cache_enabled:
            return None
        if not self._temporal_image_cache.available:
            return None
        return int(self._lte_optimizer_step.item())

    @torch.no_grad()
    def _rebuild_embedding_cache(self, step: int, device: torch.device) -> None:
        if not self._temporal_image_cache.available:
            raise RuntimeError("LTE SmolVLA requires local dataset metadata to build its history cache")
        started = wall_time.perf_counter()
        cache = torch.zeros(
            (self._temporal_image_cache.frame_count, self.temporal_embedding_dim),
            device=device,
            dtype=torch.float32,
        )
        absolute = self._temporal_image_cache.absolute_frame_indices
        requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        requested_rank = int(os.environ.get("RANK", "0"))
        if requested_world_size > 1 and not torch.distributed.is_initialized():
            # The pinned trainer constructs its Accelerator before the policy
            # but only prepares DDP after policy creation. Initialise here so
            # the expensive pre-DDP cache can still be sharded correctly.
            torch.distributed.init_process_group(backend="nccl")
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if requested_world_size > 1 and not distributed:
            raise RuntimeError("torchrun did not initialize the LTE embedding-cache process group")
        if distributed and (
            torch.distributed.get_world_size() != requested_world_size
            or torch.distributed.get_rank() != requested_rank
        ):
            raise RuntimeError("Distributed rank disagrees with torchrun environment")
        # torchrun is the authority for the pre-DDP cache partition. Do not
        # let a trainer-specific distributed-state timing detail collapse this
        # back to one full cache per worker.
        world_size = requested_world_size
        rank = requested_rank
        # Every DDP process needs the complete table for local history lookup,
        # but encoding it independently on each rank duplicates the expensive
        # video decode.  Encode disjoint positions, then sum the disjoint
        # rows so every rank receives the identical complete cache.
        # Use contiguous ranges rather than striding through the dataset. The
        # source is video-backed, so contiguous indices preserve decoder
        # locality while still assigning each frame to exactly one rank.
        shard_start = rank * absolute.numel() // world_size
        shard_end = (rank + 1) * absolute.numel() // world_size
        positions = torch.arange(shard_start, shard_end, dtype=torch.long)
        print(
            f"Building LTE embedding-cache shard rank {rank}/{world_size}: "
            f"{positions.numel():,} of {absolute.numel():,} frames.",
            flush=True,
        )
        build_batch_size = self.config.temporal_embedding_cache_build_batch_size
        for start in range(0, positions.numel(), build_batch_size):
            cache_ids = positions[start : start + build_batch_size].to(device)
            frame_ids = absolute[cache_ids.cpu()].to(device)
            # The frozen VLM may run in bfloat16 while the persistent cache
            # deliberately uses float32.  Keep the table stable and cast each
            # detached visual target at the storage boundary.
            cache[cache_ids] = self._pooled_temporal_features_from_raw(
                self._temporal_image_cache.gather(frame_ids, device)
            ).to(dtype=cache.dtype)
            completed = min(start + cache_ids.numel(), positions.numel())
            if completed == positions.numel() or completed % 4096 < cache_ids.numel():
                print(
                    f"LTE embedding-cache shard rank {rank}/{world_size}: "
                    f"{completed:,}/{positions.numel():,} frames.",
                    flush=True,
                )
        if distributed:
            torch.distributed.all_reduce(cache, op=torch.distributed.ReduceOp.SUM)
        cache = self._normalize_temporal_embeddings(
            cache, fit=not bool(self._temporal_embedding_stats_ready.item())
        )
        self._embedding_cache = cache
        self._embedding_cache_step = step
        self._last_embedding_cache_build_seconds = wall_time.perf_counter() - started
        print(
            f"Refreshed detached LTE SmolVLA embedding cache at optimizer step {step} "
            f"({absolute.numel():,} selected frames, "
            f"{self._last_embedding_cache_build_seconds:.1f}s).",
            flush=True,
        )

    @torch.no_grad()
    def _history_embeddings(self, frame_ids: Tensor, valid_mask: Tensor, device: torch.device) -> Tensor:
        step = self._cache_step_for_batch()
        use_cache = step is not None and self._temporal_image_cache.available
        if use_cache and should_refresh_embedding_cache(
            step,
            per_step_until=0,
            middle_until=0,
            middle_refresh_steps=0,
            refresh_steps=self.config.temporal_embedding_cache_refresh_steps,
            cache_exists=self._embedding_cache is not None,
        ) and self._embedding_cache_step != step:
            self._rebuild_embedding_cache(step, device)
        # An existing detached table is also useful for validation. It is not
        # refreshed outside training, but prevents full-prefix re-encoding.
        if self._embedding_cache is not None and (use_cache or not self.training):
            cache_ids = self._temporal_image_cache.cached_indices(frame_ids).to(device)
            self._last_embedding_cache_hit_rate = 1.0
            return self._embedding_cache[cache_ids].masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        if not self._temporal_image_cache.available:
            raise RuntimeError("LTE SmolVLA full causal history requires a local LeRobot dataset cache")
        if self.config.temporal_query_history_only and self._embedding_cache is None:
            # Preserve all B x Q query positions, including repeated early
            # episode frames. This is deliberately not deduplicated: the
            # timing recipe measures one vision-encoder invocation over the
            # complete requested query tensor.
            images = self._temporal_image_cache.gather(frame_ids, device)
            embeddings = self._pooled_temporal_features_from_raw(images)
            self._last_embedding_cache_hit_rate = 0.0
            return self._normalize_temporal_embeddings(embeddings).masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        result = torch.zeros(*frame_ids.shape, self.temporal_embedding_dim, device=device)
        valid_ids = frame_ids[valid_mask]
        if valid_ids.numel() == 0:
            return result
        unique, inverse = torch.unique(valid_ids, sorted=True, return_inverse=True)
        chunks: list[Tensor] = []
        for start in range(0, unique.numel(), self.config.temporal_encode_chunk_size):
            ids = unique[start : start + self.config.temporal_encode_chunk_size]
            chunks.append(self._pooled_temporal_features_from_raw(self._temporal_image_cache.gather(ids, device)))
        result[valid_mask] = self._normalize_temporal_embeddings(torch.cat(chunks, dim=0))[inverse]
        self._last_embedding_cache_hit_rate = 0.0
        return result

    def temporal_history(self, sample_indices: Tensor, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        prefix = int(getattr(self.config, "_runtime_causal_prefix_length", 0))
        if prefix <= 0 or self._lte_episode_ends.numel() == 0:
            raise RuntimeError("LTE SmolVLA causal history metadata was not installed before policy construction")
        sample_indices = sample_indices.to(device=device, dtype=torch.long)
        episode_ends = self._lte_episode_ends.to(device)
        self._last_temporal_vision_encoder_seconds = 0.0
        self._last_temporal_encoder_seconds = 0.0
        # The raw-image timing recipe expects one live B×Q encoder call. A
        # populated detached embedding table intentionally replaces that call
        # with lookup, so it must bypass the raw-forward assertion.
        if self.config.temporal_query_history_only and self._embedding_cache is None:
            # Fixed 16-point history: raw images are gathered from a cache or
            # decoded on demand, then encoded in one B*Q vision invocation.
            episode_ids = torch.bucketize(sample_indices, episode_ends, right=True)
            starts = torch.cat((torch.zeros(1, device=device, dtype=torch.long), episode_ends[:-1]))[episode_ids]
            elapsed = (sample_indices - starts).clamp_min(0)
            fractions = torch.linspace(0, 1, self.config.num_history_queries, device=device)
            frames = starts[:, None] + torch.round(elapsed[:, None] * fractions).to(torch.long)
            valid = torch.ones_like(frames, dtype=torch.bool)
        else:
            frames, valid = causal_history_frame_indices(sample_indices, episode_ends, prefix)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = wall_time.perf_counter()
        embeddings = self._history_embeddings(frames, valid, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._last_temporal_vision_seconds = wall_time.perf_counter() - started
        if self.config.temporal_query_history_only and self._embedding_cache is None:
            expected = sample_indices.shape[0] * self.config.num_history_queries
            if self._last_temporal_vision_batch_size != expected:
                raise RuntimeError(
                    "LTE query forward did not encode all num_history_queries × batch images in one vision batch: "
                    f"expected {expected}, got {self._last_temporal_vision_batch_size}"
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoder_started = wall_time.perf_counter()
        latents = self.temporal_encoder.encode_history(embeddings, valid)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self._last_temporal_encoder_seconds = wall_time.perf_counter() - encoder_started
        return latents, embeddings, valid

    def history_loss(self, latents: Tensor, embeddings: Tensor, valid_mask: Tensor) -> Tensor:
        if latents.device.type == "cuda":
            torch.cuda.synchronize(latents.device)
        started = wall_time.perf_counter()
        batch, length = valid_mask.shape
        last = torch.full((batch, 1), length - 1, device=latents.device, dtype=torch.long)
        first = valid_mask.long().argmax(dim=1, keepdim=True)
        elapsed = (last - first).clamp_min(0).to(latents.dtype)
        lags = sample_stratified_lags(elapsed, self.config.num_history_queries)
        positions = (last.unsqueeze(-1) - lags).maximum(first.unsqueeze(-1))
        targets = interpolate_history(embeddings, positions.reshape(batch, -1)).reshape(
            batch, 1, self.config.num_history_queries, self.temporal_embedding_dim
        )
        max_length = self.config.history_max_episode_length
        if max_length is None:
            raise RuntimeError("LTE SmolVLA requires the dataset-wide history_max_episode_length")
        decoder_lags = lags * (2.0 / (max(max_length - 1, 1) + 1e-8))
        current = latents[:, -1:].unsqueeze(-2).expand(-1, -1, self.config.num_history_queries, -1)
        prediction = self.history_decoder(current, decoder_lags.unsqueeze(-1))
        loss = F.mse_loss(prediction, targets)
        if latents.device.type == "cuda":
            torch.cuda.synchronize(latents.device)
        self._last_temporal_decoder_loss_seconds = wall_time.perf_counter() - started
        return loss


class LTESmolVLAPolicy(SmolVLAPolicy):
    """LeRobot policy wrapper that adds LTE while retaining SmolVLA's action model."""

    config_class = LTESmolVLAConfig
    name = "lte_smolvla"

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, strict=True, **kwargs):
        return super().from_pretrained(pretrained_name_or_path, strict=strict, **kwargs)

    def __init__(self, config: LTESmolVLAConfig, **kwargs) -> None:
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = LTESmolVLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        self._online_temporal_latent: Tensor | None = None

    def update(self) -> None:
        """Advance the cache schedule after one real optimizer update.

        LeRobot calls this hook only on synchronized gradient-accumulation
        boundaries, so ``refresh_steps`` is genuinely expressed in optimizer
        updates rather than forward passes.
        """
        self.model._lte_optimizer_step.add_(1)

    def get_optim_params(self):
        """Return SimpleMemVLA-style VLM and action-facing LR groups.

        The pinned SmolVLA implementation has one optimizer learning-rate
        field.  With an unfrozen VLM, the LTE recipe instead gives the VLM
        (including its vision encoder) a lower rate while retaining the base
        rate for the action expert, state projector, and LTE modules.  Frozen
        tensors are deliberately omitted.  SmolVLA's own safety exclusions
        (its output head, final norm, and selected final layers) remain
        controlled by ``set_requires_grad`` in the base model.
        """
        if not self.config.differential_lr:
            return super().get_optim_params()

        # Match the SimpleMemVLA trainer's AdamW grouping: decay weights but
        # not bias or normalization parameters.  The helper is the same
        # Transformers utility used by Hugging Face Trainer internally.
        from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
        from transformers.trainer_pt_utils import get_parameter_names

        decay_names = {
            name for name in get_parameter_names(self, ALL_LAYERNORM_LAYERS)
            if not name.endswith(".bias")
        }
        backbone_params = {"decay": [], "no_decay": []}
        action_facing_params = {"decay": [], "no_decay": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            decay_group = "decay" if name in decay_names else "no_decay"
            if name.startswith("model.vlm_with_expert.vlm."):
                backbone_params[decay_group].append(parameter)
            else:
                action_facing_params[decay_group].append(parameter)

        if not backbone_params["decay"] and not backbone_params["no_decay"]:
            raise RuntimeError("Differential LR requested but no SmolVLA VLM parameters are trainable")
        if not action_facing_params["decay"] and not action_facing_params["no_decay"]:
            raise RuntimeError("Differential LR requested but no LTE/action parameters are trainable")

        groups = []
        for parameter_groups, lr in (
            (backbone_params, self.config.optimizer_lr * self.config.backbone_lr_scale),
            (action_facing_params, self.config.optimizer_lr),
        ):
            if parameter_groups["decay"]:
                groups.append({
                    "params": parameter_groups["decay"],
                    "lr": lr,
                    "weight_decay": self.config.optimizer_weight_decay,
                })
            if parameter_groups["no_decay"]:
                groups.append({
                    "params": parameter_groups["no_decay"], "lr": lr, "weight_decay": 0.0,
                })
        return groups

    @torch.no_grad()
    def _advance_temporal_state(self, batch: dict[str, Tensor]) -> Tensor:
        self._online_temporal_latent = self.advance_temporal_state(batch, self._online_temporal_latent)
        return self.model.temporal_token(self._online_temporal_latent)

    @torch.no_grad()
    def advance_temporal_state(self, batch: dict[str, Tensor], previous: Tensor | None) -> Tensor:
        """One observation update; callers may own separate states per environment."""
        images, _ = self.prepare_images(batch)
        # ``embed_image`` inherits the VLM's bfloat16 compute dtype, whereas
        # LTE's small recurrence is deliberately float32.  Match the encoder
        # parameters before creating or advancing its online state.
        encoder_dtype = next(self.model.temporal_encoder.parameters()).dtype
        embedding = self.model._normalize_temporal_embeddings(
            self.model.pooled_temporal_features_from_prepared(images)
        ).to(dtype=encoder_dtype)
        if (
            previous is None
            or previous.shape[0] != embedding.shape[0]
        ):
            previous = torch.zeros(
                embedding.shape[0],
                self.config.temporal_latent_dim,
                device=embedding.device,
                dtype=encoder_dtype,
            )
        else:
            previous = previous.to(
                device=embedding.device, dtype=encoder_dtype
            )
        return self.model.temporal_encoder(embedding, previous)

    @torch.no_grad()
    def predict_with_latent(self, batch, latent, noise=None):
        """Generate a chunk without observing the current frame a second time."""
        self.eval()
        batch = self._prepare_batch(dict(batch))
        with self.model.use_temporal_token(self.model.temporal_token(latent)):
            return SmolVLAPolicy._get_action_chunk(self, batch, noise=noise)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        self.eval()
        batch = self._prepare_batch(dict(batch))
        token = self._advance_temporal_state(batch)
        with self.model.use_temporal_token(token):
            return SmolVLAPolicy._get_action_chunk(self, batch, noise=noise, **kwargs)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        self.eval()
        batch = self._prepare_batch(dict(batch))
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        # Update LTE at every control observation, including steps served from
        # an already-generated action chunk.
        token = self._advance_temporal_state(batch)
        if not self._check_get_actions_condition():
            return self._queues[ACTION].popleft()
        with self.model.use_temporal_token(token):
            actions = SmolVLAPolicy._get_action_chunk(self, batch, noise=noise, **kwargs)
        self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        if reduction != "mean":
            raise NotImplementedError("LTE SmolVLA does not support per-sample weighting yet")
        batch = self._prepare_batch(dict(batch))
        if "index" not in batch:
            raise ValueError("LTE SmolVLA training batches must include the absolute LeRobot frame index")
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        latents, embeddings, valid = self.model.temporal_history(batch["index"], actions.device)
        token = self.model.temporal_token(latents[:, -1])
        if actions.device.type == "cuda":
            torch.cuda.synchronize(actions.device)
        action_forward_started = wall_time.perf_counter()
        with self.model.use_temporal_token(token):
            losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        if actions.device.type == "cuda":
            torch.cuda.synchronize(actions.device)
        self.model._last_action_forward_seconds = wall_time.perf_counter() - action_forward_started
        if actions.device.type == "cuda":
            torch.cuda.synchronize(actions.device)
        action_loss_started = wall_time.perf_counter()
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        action_pad = batch.get("action_is_pad")
        valid_actions = torch.ones(losses.shape[:2], dtype=torch.bool, device=losses.device)
        if action_pad is not None:
            valid_actions &= ~action_pad
        dimension_mask = batch.get("action_dimension_mask")
        if dimension_mask is None:
            dimension_mask = torch.ones(losses.shape[0], losses.shape[-1], dtype=torch.bool, device=losses.device)
        dimension_mask = dimension_mask.to(device=losses.device, dtype=torch.bool).unsqueeze(1)
        loss_mask = valid_actions.unsqueeze(-1) & dimension_mask
        action_loss = losses.mul(loss_mask).sum() / loss_mask.sum().clamp_min(1)
        if actions.device.type == "cuda":
            torch.cuda.synchronize(actions.device)
        self.model._last_action_loss_seconds = wall_time.perf_counter() - action_loss_started
        history_loss = self.model.history_loss(latents, embeddings, valid)
        total = action_loss + self.config.lambda_history * history_loss
        return total, {
            "loss": float(total.detach()),
            "action_loss": float(action_loss.detach()),
            "history_loss": float(history_loss.detach()),
            "embedding_cache_hit_rate": self.model._last_embedding_cache_hit_rate,
            "temporal_embedding_cache_build_seconds": self.model._last_embedding_cache_build_seconds,
            # The raw GPU cache is built before optimizer step zero.  Keep its
            # immutable per-rank duration in the normal training metrics so
            # W&B records the setup cost alongside the steady-state timings.
            "temporal_image_cache_build_seconds": self.model._temporal_image_cache.build_seconds,
            "temporal_query_vision_seconds": self.model._last_temporal_vision_seconds,
            "temporal_vision_encoder_seconds": self.model._last_temporal_vision_encoder_seconds,
            "temporal_encoder_seconds": self.model._last_temporal_encoder_seconds,
            "temporal_decoder_loss_seconds": self.model._last_temporal_decoder_loss_seconds,
            "action_forward_seconds": self.model._last_action_forward_seconds,
            "action_loss_seconds": self.model._last_action_loss_seconds,
            "temporal_query_vision_batch_size": self.model._last_temporal_vision_batch_size,
            "temporal_query_vision_encoder_calls": self.model._last_temporal_vision_encoder_calls,
        }

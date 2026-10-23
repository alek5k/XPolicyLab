"""A thin temporal extension of XPolicyLab's actual DiffusionUnetImagePolicy.

The native policy owns the U-Net, scheduler, diffusion objective, normalization
and action sampling. This module only appends LTE conditioning and history MSE.
"""

import torch
from torch import nn

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

from .temporal import HistoryDecoder, ImageTemporalEncoder, history_loss


FRAME_INDICES = "_lte_frame_indices"
STREAM_LATENTS = "_lte_stream_latents"


class TemporalObsEncoder(nn.Module):
    """Append prepared temporal states to native MultiImageObsEncoder output."""
    def __init__(self, base_encoder, latent_dim):
        super().__init__()
        self.base_encoder = base_encoder
        self.latent_dim = latent_dim
        self.context = None

    def output_shape(self):
        return (self.base_encoder.output_shape()[0] + self.latent_dim,)

    def forward(self, obs):
        live = self.base_encoder(obs)
        if self.context is None:
            raise RuntimeError("LTE conditioning was not prepared")
        latent = self.context.reshape(-1, self.latent_dim)
        if live.shape[0] != latent.shape[0]:
            raise ValueError("LTE conditioning does not match DP's observation window")
        return torch.cat((live, latent), dim=-1)


class EmbeddingCache:
    """Runtime-only table of detached native-encoder features."""
    def __init__(self, scope="training"):
        self.values, self.built_at = None, None
        self.rebuild_count = 0
        self.scope = scope

    @torch.no_grad()
    def ensure(self, policy, dataset, update, force=False):
        if self.values is not None and not force and update - self.built_at < policy.embedding_cache_refresh_steps:
            return False
        was_training = policy.training
        policy.eval()
        try:
            table = torch.empty(dataset.replay_buffer.n_steps, policy.temporal_embedding_dim, device=policy.device)
            for start in range(0, len(table), policy.encode_chunk_size):
                stop = min(start + policy.encode_chunk_size, len(table))
                rgb = {
                    key: torch.as_tensor(dataset.replay_buffer[dataset.rgb_data_keys[key]][start:stop], device=policy.device).float() / 255.0
                    for key in policy.temporal_image_keys
                }
                table[start:stop] = policy.encode_temporal_observation(rgb)
            self.values, self.built_at = table.detach(), int(update)
            self.rebuild_count += 1
            print(f"LTE {self.scope} cache rebuilt at optimizer update {update}: {tuple(table.shape)}", flush=True)
        finally:
            policy.train(was_training)
        return True


class LTEDiffusionPolicy(DiffusionUnetImagePolicy):
    def __init__(self, *args, obs_encoder, temporal_num_images=1,
                 temporal_latent_dim=32, temporal_hidden_dim=128,
                 history_decoder_hidden_dim=128, num_history_queries=16,
                 lambda_history=1.0, history_max_episode_length=None,
                 embedding_cache_refresh_steps=5000, encode_chunk_size=64, **kwargs):
        if not kwargs.get("obs_as_global_cond", True):
            raise ValueError("LTE extends DP's global observation conditioning")
        image_keys = list(obs_encoder.rgb_keys)
        count = len(image_keys) if temporal_num_images is None else temporal_num_images
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(image_keys):
            raise ValueError("temporal_num_images must be a valid camera count or null")
        if embedding_cache_refresh_steps <= 0 or encode_chunk_size <= 0 or num_history_queries <= 0:
            raise ValueError("Cache intervals, chunks and history queries must be positive")
        # The original constructor creates its own native U-Net with the
        # larger conditioning width. Its implementation is otherwise untouched.
        super().__init__(*args, obs_encoder=TemporalObsEncoder(obs_encoder, temporal_latent_dim), **kwargs)
        self.temporal_image_keys = tuple(image_keys[:count])
        self.temporal_num_images = count
        # Native ResNet18's AdaptiveAvgPool2d yields 512 features per view.
        models = [obs_encoder.key_model_map["rgb"]] if obs_encoder.share_rgb_model else [obs_encoder.key_model_map[k] for k in image_keys]
        if not all(isinstance(getattr(model, "avgpool", None), nn.AdaptiveAvgPool2d) for model in models):
            raise ValueError("LTE_DP requires native ResNet spatial mean pooling")
        self.temporal_embedding_dim = 512 * count
        self.temporal_encoder = ImageTemporalEncoder(self.temporal_embedding_dim, temporal_latent_dim, temporal_hidden_dim)
        self.history_decoder = HistoryDecoder(temporal_latent_dim, self.temporal_embedding_dim, history_decoder_hidden_dim)
        self.num_history_queries, self.lambda_history = num_history_queries, lambda_history
        self.history_max_episode_length = history_max_episode_length
        self.embedding_cache_refresh_steps, self.encode_chunk_size = embedding_cache_refresh_steps, encode_chunk_size
        self._dataset = None
        self._optimizer_update = 0
        self._training_cache = EmbeddingCache()
        self._evaluation_cache = EmbeddingCache("evaluation")
        self.last_lte_losses = {}

    def bind_dataset(self, dataset):
        self._dataset = dataset
        self.history_max_episode_length = dataset.max_episode_length
        self._training_cache = EmbeddingCache()
        self._evaluation_cache = EmbeddingCache("evaluation")

    def set_normalizer(self, normalizer):
        super().set_normalizer(normalizer)
        dataset = getattr(normalizer, "_lte_dataset", None)
        if dataset is not None:
            self.bind_dataset(dataset)

    def set_optimizer_update(self, update):
        self._optimizer_update = int(update)

    def encode_temporal_observation(self, rgb):
        """Same normalization, transforms and ResNet weights as native DP.

        Callers supply normal DP floats in [0,1]. This preserves DP's image
        normalizer + imagenet_norm pipeline, including its exact ordering.
        """
        encoder = self.obs_encoder.base_encoder
        features = []
        for key in self.temporal_image_keys:
            image = encoder.key_transform_map[key](self.normalizer[key].normalize(rgb[key]))
            model = encoder.key_model_map["rgb" if encoder.share_rgb_model else key]
            features.append(model(image))
        return torch.cat(features, dim=-1)

    def _history_features(self, indices, reconstruct):
        if self._dataset is None:
            raise RuntimeError("Bind the training dataset before computing causal prefixes")
        indices = indices.to(device=self.device, dtype=torch.long)
        ends = torch.as_tensor(self._dataset.replay_buffer.episode_ends, device=self.device)
        episodes = torch.bucketize(indices[:, -1].contiguous(), ends, right=True)
        starts = torch.cat((ends.new_zeros(1), ends[:-1]))[episodes]
        obs_times = indices - starts[:, None]
        if (obs_times < 0).any() or (indices >= ends[episodes, None]).any():
            raise ValueError("An observation window crossed an episode boundary")
        elapsed = obs_times[:, -1]
        evaluation = not torch.is_grad_enabled()
        cache = self._evaluation_cache if evaluation else self._training_cache
        # Validation and native EMA train-batch sampling use independent
        # tables rebuilt once per evaluation update, never the training clock.
        cache.ensure(self, self._dataset, self._optimizer_update,
                     force=evaluation and cache.built_at != self._optimizer_update)
        positions = torch.arange(int(elapsed.max()) + 1, device=self.device)
        frames = starts[:, None] + positions[None, :].minimum(elapsed[:, None])
        embeddings = cache.values[frames].detach()
        if reconstruct:
            return history_loss(self.temporal_encoder, self.history_decoder, embeddings,
                elapsed, self.n_obs_steps, self.history_max_episode_length, self.num_history_queries,
                observation_times=obs_times)
        valid = positions[None, :] <= elapsed[:, None]
        latents = self.temporal_encoder.encode_history(embeddings, valid)
        return latents.gather(1, obs_times[..., None].expand(-1, -1, self.temporal_encoder.latent_dim)), None

    def compute_loss(self, batch):
        obs = dict(batch["obs"])
        indices = obs.pop(FRAME_INDICES)
        latent, reconstruction = self._history_features(indices, reconstruct=True)
        self.obs_encoder.context = latent
        try:
            # Native noise sampling, mask, U-Net and diffusion MSE unchanged.
            diffusion = super().compute_loss(dict(batch, obs=obs))
        finally:
            self.obs_encoder.context = None
        total = diffusion + self.lambda_history * reconstruction
        self.last_lte_losses = {"diffusion_loss": float(diffusion.detach()),
            "history_loss": float(reconstruction.detach()),
            "lte_optimizer_updates": self._optimizer_update,
            "lte_cache_built_at": self._training_cache.built_at if torch.is_grad_enabled() else self._evaluation_cache.built_at}
        return total

    @torch.no_grad()
    def predict_action(self, obs_dict):
        obs = dict(obs_dict)
        if STREAM_LATENTS in obs:
            latent = obs.pop(STREAM_LATENTS)
        elif FRAME_INDICES in obs:
            latent, _ = self._history_features(obs.pop(FRAME_INDICES), reconstruct=False)
        else:
            raise ValueError("Provide full-prefix frame indices or streaming LTE states")
        self.obs_encoder.context = latent
        try:
            # Native DDPM scheduler, sampler and action slicing unchanged.
            return super().predict_action(obs)
        finally:
            self.obs_encoder.context = None

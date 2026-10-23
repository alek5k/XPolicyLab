"""Port of LTE-DP/ltedp/temporal.py (image-only LTE from TemporalDiffusionPolicy).

Encoder/decoder equations and stratified interpolation follow that reference.
No imports from the original repositories are needed at runtime.
"""

import torch
from torch import nn


class ImageTemporalEncoder(nn.Module):
    def __init__(self, embedding_dim, latent_dim=32, hidden_dim=128):
        super().__init__()
        if not 0 < hidden_dim < embedding_dim + latent_dim:
            raise ValueError("LTE hidden width must be smaller than its input width")
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, embedding, previous):
        return torch.tanh(previous + self.mlp(torch.cat((embedding, previous), dim=-1)))

    def encode_history(self, embeddings, valid_mask):
        latent = embeddings.new_zeros(embeddings.shape[0], self.latent_dim)
        outputs = []
        for embedding, valid in zip(embeddings.unbind(1), valid_mask.unbind(1)):
            candidate = self(embedding, latent)
            latent = torch.where(valid[:, None], candidate, latent)
            outputs.append(latent)
        return torch.stack(outputs, dim=1)


class HistoryDecoder(nn.Module):
    def __init__(self, latent_dim, embedding_dim, hidden_dim=128):
        super().__init__()
        if hidden_dim <= latent_dim + 1:
            raise ValueError("Decoder hidden width must exceed latent_dim + 1")
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, latent, normalized_lag):
        return self.mlp(torch.cat((latent, normalized_lag), dim=-1))


def sample_stratified_lags(elapsed, num_queries):
    bins = torch.arange(num_queries, device=elapsed.device, dtype=elapsed.dtype) / num_queries
    jitter = torch.rand(*elapsed.shape, num_queries, device=elapsed.device) / num_queries
    return (bins + jitter) * elapsed.unsqueeze(-1)


def interpolate_history(values, positions):
    lower = positions.floor().long().clamp(0, values.shape[1] - 1)
    upper = (lower + 1).clamp(0, values.shape[1] - 1)
    weight = (positions - lower).unsqueeze(-1)
    low = values.gather(1, lower[..., None].expand(-1, -1, values.shape[-1]))
    high = values.gather(1, upper[..., None].expand(-1, -1, values.shape[-1]))
    return low + weight * (high - low)


def history_loss(encoder, decoder, embeddings, elapsed, n_obs_steps, max_episode_length, queries, observation_times=None):
    """Full prefix; early observation padding repeats frame zero's LTE state.

    ``embeddings`` is left-aligned at the real episode boundary. Query lags
    are scaled by 2/(dataset_max_episode_length-1), as in the aligned repos.
    """
    embeddings = embeddings.detach()
    positions = torch.arange(embeddings.shape[1], device=embeddings.device)
    valid = positions[None, :] <= elapsed[:, None]
    latents = encoder.encode_history(embeddings, valid)
    obs_times = (elapsed[:, None] + torch.arange(
        1 - n_obs_steps, 1, device=embeddings.device
    )).clamp_min(0)
    if observation_times is not None:
        obs_times = observation_times
    current = latents.gather(1, obs_times[..., None].expand(-1, -1, encoder.latent_dim))
    lags = sample_stratified_lags(obs_times.float(), queries)
    query_positions = obs_times[..., None] - lags
    targets = interpolate_history(embeddings, query_positions.flatten(1)).reshape(
        *lags.shape, embeddings.shape[-1]
    )
    normalized = lags * (2.0 / max(max_episode_length - 1, 1))
    predictions = decoder(current[:, :, None, :].expand(-1, -1, queries, -1), normalized[..., None])
    return current, nn.functional.mse_loss(predictions, targets)

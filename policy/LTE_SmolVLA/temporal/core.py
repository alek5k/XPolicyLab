"""Pure LTE recurrence and reconstruction helpers shared by policy adapters."""

import torch
from torch import Tensor, nn


class ImageTemporalEncoder(nn.Module):
    """Causal residual recurrence over detached per-frame visual embeddings."""

    def __init__(self, image_embedding_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        input_dim = int(image_embedding_dim) + latent_dim
        if hidden_dim >= input_dim:
            raise ValueError("ImageTemporalEncoder hidden_dim must be smaller than its input layer")
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, image_embedding: Tensor, previous_latent: Tensor) -> Tensor:
        return torch.tanh(previous_latent + self.mlp(torch.cat((image_embedding, previous_latent), dim=-1)))

    def encode_history(self, image_embeddings: Tensor, valid_mask: Tensor) -> Tensor:
        if image_embeddings.ndim != 3:
            raise ValueError("image_embeddings must have shape (B, T, D)")
        if valid_mask.shape != image_embeddings.shape[:2]:
            raise ValueError("valid_mask must have shape (B, T)")
        latent = image_embeddings.new_zeros(image_embeddings.shape[0], self.latent_dim)
        outputs: list[Tensor] = []
        for embedding, active in zip(image_embeddings.unbind(1), valid_mask.unbind(1), strict=True):
            candidate = self(embedding, latent)
            latent = torch.where(active.unsqueeze(-1), candidate, latent)
            outputs.append(latent)
        return torch.stack(outputs, dim=1)


class HistoryDecoder(nn.Module):
    """Training-only decoder from an LTE latent and a normalized query lag."""

    def __init__(self, latent_dim: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= latent_dim + 1:
            raise ValueError("HistoryDecoder hidden_dim must be larger than its input layer")
        self.mlp = nn.Sequential(nn.Linear(latent_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, embedding_dim))

    def forward(self, latent: Tensor, normalized_lag: Tensor) -> Tensor:
        return self.mlp(torch.cat((latent, normalized_lag), dim=-1))


def sample_stratified_lags(elapsed: Tensor, num_queries: int) -> Tensor:
    offsets = torch.arange(num_queries, device=elapsed.device, dtype=elapsed.dtype) / num_queries
    jitter = torch.rand(*elapsed.shape, num_queries, device=elapsed.device, dtype=elapsed.dtype) / num_queries
    return (offsets + jitter) * elapsed.unsqueeze(-1)


def interpolate_history(values: Tensor, query_positions: Tensor) -> Tensor:
    sequence_length = values.shape[1]
    lower = query_positions.floor().long().clamp(0, sequence_length - 1)
    upper = (lower + 1).clamp(0, sequence_length - 1)
    weight = (query_positions - lower).unsqueeze(-1)
    shape = (*lower.shape, values.shape[-1])
    low = values.gather(1, lower.unsqueeze(-1).expand(shape))
    high = values.gather(1, upper.unsqueeze(-1).expand(shape))
    return low + weight * (high - low)

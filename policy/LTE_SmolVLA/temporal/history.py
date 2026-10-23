"""History-index and cache-schedule logic independent of a policy architecture."""

import torch
from torch import Tensor


def should_refresh_embedding_cache(
    step_idx: int,
    *,
    per_step_until: int,
    middle_until: int,
    middle_refresh_steps: int,
    refresh_steps: int,
    cache_exists: bool,
) -> bool:
    """Return whether a detached embedding cache should be rebuilt.

    A zero refresh interval deliberately means "initial build only".  This is
    the correct schedule when the module that produces the detached targets is
    frozen: the missing-cache case below still builds it at update zero.
    """
    if not cache_exists:
        return True
    if step_idx < per_step_until:
        return True
    if step_idx < middle_until:
        if middle_refresh_steps == 0:
            return False
        return (step_idx - per_step_until) % middle_refresh_steps == 0
    if refresh_steps == 0:
        return False
    return (step_idx - middle_until) % refresh_steps == 0


def trim_batch_padding(frame_ids: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Remove leading history positions that are invalid for the whole batch."""
    if frame_ids.shape != valid_mask.shape or frame_ids.ndim != 2:
        raise ValueError("frame_ids and valid_mask must have matching shape (B, T)")
    active_columns = valid_mask.any(dim=0)
    return frame_ids[:, active_columns], valid_mask[:, active_columns]


def causal_history_frame_indices(
    sample_indices: Tensor, episode_ends: Tensor, prefix_length: int
) -> tuple[Tensor, Tensor]:
    """Return left-padded absolute causal frame IDs and their valid mask."""
    if sample_indices.ndim != 1:
        raise ValueError("sample_indices must have shape (batch,)")
    if episode_ends.ndim != 1 or episode_ends.numel() == 0:
        raise ValueError("episode_ends must be a non-empty one-dimensional tensor")
    if prefix_length <= 0:
        raise ValueError("prefix_length must be positive")
    indices = sample_indices.to(dtype=episode_ends.dtype)
    episode_ids = torch.searchsorted(episode_ends, indices, right=True).clamp_max(episode_ends.numel() - 1)
    starts = torch.cat((episode_ends.new_zeros(1), episode_ends[:-1]))[episode_ids]
    lengths = indices - starts + 1
    if (lengths <= 0).any() or (lengths > prefix_length).any():
        raise ValueError("sample frame index is outside its configured causal prefix")
    positions = torch.arange(prefix_length, device=indices.device).unsqueeze(0)
    first_valid = prefix_length - lengths.unsqueeze(1)
    valid = positions >= first_valid
    offsets = (positions - first_valid).clamp_min(0)
    return starts.unsqueeze(1) + offsets, valid

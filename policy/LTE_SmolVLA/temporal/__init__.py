"""Shared learned-temporal-encoder primitives.

Policy adapters own image preprocessing and the connection from the temporal
latent to their action model.  This package deliberately contains neither
Diffusion Policy nor SmolVLA imports.
"""

from .core import HistoryDecoder, ImageTemporalEncoder, interpolate_history, sample_stratified_lags
from .history import causal_history_frame_indices, should_refresh_embedding_cache, trim_batch_padding

__all__ = [
    "HistoryDecoder",
    "ImageTemporalEncoder",
    "causal_history_frame_indices",
    "interpolate_history",
    "sample_stratified_lags",
    "should_refresh_embedding_cache",
    "trim_batch_padding",
]

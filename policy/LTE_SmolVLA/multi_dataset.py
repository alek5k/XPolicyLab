"""Small, LTE-specific concatenation wrapper for compatible LeRobot datasets."""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.compute_stats import aggregate_stats


def _resolve_delta_timestamps(policy, metadata):
    """Local equivalent of LeRobot's factory helper without importing trainer code."""
    result = {}
    for key in metadata.features:
        if key == "action" and policy.action_delta_indices is not None:
            result[key] = [index / metadata.fps for index in policy.action_delta_indices]
        elif key.startswith("observation.") and policy.observation_delta_indices is not None:
            result[key] = [index / metadata.fps for index in policy.observation_delta_indices]
    return result or None


class LTEMultiLeRobotDataset(torch.utils.data.Dataset):
    """Concatenate compatible datasets while retaining a global frame index.

    LeRobot's upstream multi-dataset path is intentionally disabled in the
    pinned release. LTE needs global episode boundaries for causal history, so
    this wrapper also exposes its source datasets for the raw-frame cache.
    """

    def __init__(self, repo_ids, *, root, video_backend, policy, image_transforms=None, revision=None):
        base_root = Path(root).expanduser() if root else None
        self.lte_component_datasets = []
        for repo_id in repo_ids:
            repo_root = (base_root / repo_id) if base_root is not None else None
            metadata = LeRobotDatasetMetadata(repo_id, root=repo_root, revision=revision)
            self.lte_component_datasets.append(
                LeRobotDataset(
                    repo_id,
                    root=repo_root,
                    image_transforms=image_transforms,
                    delta_timestamps=_resolve_delta_timestamps(policy, metadata),
                    revision=revision,
                    video_backend=video_backend,
                )
            )
        if not self.lte_component_datasets:
            raise ValueError("LTE multi-dataset training requires at least one repository")
        reference = deepcopy(self.lte_component_datasets[0].meta.features)
        image_shapes = {tuple(dataset.meta.features["observation.image"]["shape"]) for dataset in self.lte_component_datasets}
        if len(image_shapes) != 1:
            raise ValueError("LTE multi-dataset training requires matching image shapes")
        self.state_action_dim = max(
            max(dataset.meta.features[key]["shape"][0] for dataset in self.lte_component_datasets)
            for key in ("observation.state", "action")
        )
        for key in ("observation.state", "action"):
            reference[key] = deepcopy(reference[key])
            reference[key]["shape"] = [self.state_action_dim]

        self.lte_component_offsets = [0]
        starts, ends = [], []
        for dataset in self.lte_component_datasets:
            offset = self.lte_component_offsets[-1]
            episodes = dataset.meta.episodes
            starts.extend((torch.as_tensor(episodes["dataset_from_index"], dtype=torch.long) + offset).tolist())
            ends.extend((torch.as_tensor(episodes["dataset_to_index"], dtype=torch.long) + offset).tolist())
            self.lte_component_offsets.append(offset + len(dataset))
        padded_stats = [self._pad_stats(dataset.meta.stats) for dataset in self.lte_component_datasets]
        self.meta = SimpleNamespace(
            features=reference,
            stats=aggregate_stats(padded_stats),
            episodes={"dataset_from_index": starts, "dataset_to_index": ends},
            total_frames=self.lte_component_offsets[-1],
        )
        # Match the public fields consumed by the generic LeRobot trainer and
        # metrics tracker. The wrapper owns all frames and episodes from each
        # child, so these are straightforward joint totals.
        self.num_frames = self.lte_component_offsets[-1]
        self.num_episodes = len(starts)
        self.episodes = None

    def _pad_stats(self, stats):
        stats = deepcopy(stats)
        for key in ("observation.state", "action"):
            for stat_name, value in stats[key].items():
                value = np.asarray(value)
                if stat_name == "count" or value.ndim == 0 or value.shape[-1] >= self.state_action_dim:
                    continue
                fill = 1.0 if stat_name == "std" else 0.0
                padding = [(0, 0)] * value.ndim
                padding[-1] = (0, self.state_action_dim - value.shape[-1])
                stats[key][stat_name] = np.pad(value, padding, constant_values=fill)
        return stats

    def __len__(self):
        return self.lte_component_offsets[-1]

    def __getitem__(self, index):
        index = int(index)
        dataset_index = bisect_right(self.lte_component_offsets, index) - 1
        if dataset_index < 0 or dataset_index >= len(self.lte_component_datasets):
            raise IndexError(index)
        local_index = index - self.lte_component_offsets[dataset_index]
        item = dict(self.lte_component_datasets[dataset_index][local_index])
        action_width = int(item["action"].shape[-1])
        # WaitAtGoal has 2-D actions/states and LiftQA has 3-D.  Right-pad to
        # a shared SmolVLA feature schema. Its artificial action component is
        # masked from the diffusion loss below.
        for key in ("observation.state", "action"):
            value = item[key]
            width = value.shape[-1]
            if width < self.state_action_dim:
                item[key] = torch.nn.functional.pad(value, (0, self.state_action_dim - width))
        item["action_dimension_mask"] = torch.arange(self.state_action_dim) < action_width
        keep = {"observation.state", "observation.image", "action", "action_is_pad", "action_dimension_mask", "task", "index"}
        item = {key: value for key, value in item.items() if key in keep}
        item["index"] = torch.tensor(index, dtype=torch.long)
        return item

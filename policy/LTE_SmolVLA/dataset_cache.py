"""Raw-frame caches used by LTE SmolVLA training.

The raw cache deliberately stores compact uint8 RGB frames, never visual
embeddings. This lets a trainable vision tower process selected temporal
queries every update while avoiding repeated video decoding.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


class DatasetCache:
    """Gather temporal camera frames from GPU or shared host-RAM raw RGB."""

    def __init__(
        self,
        dataset,
        image_keys,
        rename_map=None,
        *,
        cache_on_gpu: bool = False,
        cache_in_ram: bool = False,
        build_num_threads: int = 0,
    ):
        self.dataset = dataset
        inverse = {value: key for key, value in (rename_map or {}).items()}
        self.image_keys = [inverse.get(key, key) for key in image_keys]
        self.cache_on_gpu = bool(cache_on_gpu)
        self.cache_in_ram = bool(cache_in_ram)
        self.build_num_threads = int(build_num_threads)
        if self.cache_on_gpu and self.cache_in_ram:
            raise ValueError("A temporal image cache cannot be on both GPU and shared RAM")
        self._gpu_cache = None
        self._ram_cache = None
        self._image_only_dataset = None
        # Measured locally on every DDP rank.  For a combined dataset this is
        # the sum of its component caches, which is also the serial wall time
        # spent preloading them on that rank.
        self._build_seconds = 0.0
        self._components = None
        self._component_offsets = None

        children = getattr(dataset, "lte_component_datasets", None)
        if children is not None:
            self._components = [
                DatasetCache(
                    child,
                    image_keys,
                    rename_map,
                    cache_on_gpu=cache_on_gpu,
                    cache_in_ram=cache_in_ram,
                    build_num_threads=build_num_threads,
                )
                for child in children
            ]
            offsets = getattr(dataset, "lte_component_offsets", None)
            if offsets is None:
                offsets = [0]
                for child in self._components:
                    offsets.append(offsets[-1] + child.frame_count)
            self._component_offsets = torch.as_tensor(offsets, dtype=torch.long)
            self.absolute_frame_indices = torch.arange(int(self._component_offsets[-1]), dtype=torch.long)
            self._absolute_to_relative = {int(index): int(index) for index in self.absolute_frame_indices.tolist()}
            return

        self.absolute_frame_indices = torch.empty(0, dtype=torch.long)
        self._absolute_to_relative = {}
        if dataset is not None:
            self.absolute_frame_indices = torch.as_tensor(dataset.hf_dataset["index"], dtype=torch.long).flatten()
            self._absolute_to_relative = {
                int(frame): index for index, frame in enumerate(self.absolute_frame_indices.tolist())
            }

    @property
    def available(self):
        return self.dataset is not None

    @property
    def frame_count(self):
        return self.absolute_frame_indices.numel()

    @property
    def build_seconds(self):
        """One-time raw-image-cache build duration for this local rank."""
        if self._components is not None:
            return sum(component.build_seconds for component in self._components)
        return self._build_seconds

    def cached_indices(self, frame_ids):
        if self._components is not None:
            frame_ids = frame_ids.to(dtype=torch.long)
            if frame_ids.numel() and (frame_ids.min() < 0 or frame_ids.max() >= self.frame_count):
                raise ValueError("History references a frame outside the combined LTE dataset")
            return frame_ids
        try:
            indices = [self._absolute_to_relative[int(frame)] for frame in frame_ids.flatten().tolist()]
        except KeyError as error:
            raise ValueError(f"History references a frame outside the selected episodes: {error}") from error
        return torch.tensor(indices, dtype=torch.long, device=frame_ids.device).reshape(frame_ids.shape)

    @staticmethod
    def _as_uint8(image):
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(np.asarray(image))
        if image.ndim == 4:
            image = image[-1]
        if image.ndim == 3 and image.shape[-1] in {1, 3, 4} and image.shape[0] not in {1, 3, 4}:
            image = image.movedim(-1, 0)
        if image.dtype == torch.uint8:
            return image.contiguous()
        image = image.to(dtype=torch.float32)
        if image.numel() and float(image.max()) <= 1.0:
            image = image.mul(255.0)
        return image.clamp(0, 255).to(dtype=torch.uint8).contiguous()

    def _decode(self, frame_ids, device):
        """Decode selected source frames using LeRobot's camera-selective reader."""
        owner = self.dataset
        ensure_loaded = getattr(owner, "_ensure_hf_dataset_loaded", None)
        if ensure_loaded is None:
            ensure_reader = getattr(owner, "_ensure_reader", None)
            if ensure_reader is not None:
                owner = ensure_reader()
        else:
            ensure_loaded()
        relative_indices = self.cached_indices(frame_ids).flatten().cpu().tolist()
        # The temporal benchmark datasets carry images directly in Parquet;
        # RMBench carries them in video. Support both official LeRobot forms.
        if not hasattr(owner, "meta") or not set(self.image_keys).intersection(getattr(owner.meta, "video_keys", ())):
            # Image-feature datasets (such as local WaitAtGoal/LiftQA) are
            # stored column-wise in Parquet.  Reading `owner[index]` for every
            # frame unnecessarily materialises action, state and task fields
            # too. Read only the selected image columns for a complete cache
            # chunk; it retains LeRobot's standard image decoding while
            # removing thousands of Python-level full-example reads.
            if self._image_only_dataset is None:
                self._image_only_dataset = owner.hf_dataset.select_columns(self.image_keys)
            rows = self._image_only_dataset[relative_indices]
            cameras = []
            for key in self.image_keys:
                values = rows[key]
                if isinstance(values, torch.Tensor):
                    cameras.append(self._as_uint8(values))
                else:
                    cameras.append(torch.stack([self._as_uint8(value) for value in values]))
            return torch.stack(cameras, dim=1).to(device)
        rows = owner.hf_dataset[relative_indices]
        episode_indices = [int(value.item()) for value in rows["episode_index"]]
        timestamps = [float(value.item()) for value in rows["timestamp"]]
        offsets_by_episode = {}
        for offset, episode_index in enumerate(episode_indices):
            offsets_by_episode.setdefault(episode_index, []).append(offset)
        decoded = [None] * len(relative_indices)
        for episode_index, offsets in offsets_by_episode.items():
            requested_timestamps = [timestamps[offset] for offset in offsets]
            videos = owner._query_videos({key: requested_timestamps for key in self.image_keys}, episode_index)
            for local_offset, offset in enumerate(offsets):
                images = []
                for key in self.image_keys:
                    image = videos[key][local_offset]
                    if image.ndim == 2:
                        image = image.unsqueeze(0)
                    if image.shape[0] == 1:
                        image = image.expand(3, -1, -1)
                    elif image.shape[0] == 4:
                        image = image[:3]
                    images.append(self._as_uint8(image))
                decoded[offset] = torch.stack(images)
        return torch.stack(decoded).to(device)

    def _ram_cache_paths(self):
        """Stable tmpfs paths so all local DDP ranks open the same cache."""
        repo_id = str(getattr(self.dataset, "repo_id", "lte-dataset"))
        indices = self.absolute_frame_indices
        identity = "|".join(
            (
                repo_id,
                str(self.frame_count),
                str(int(indices[0])) if self.frame_count else "empty",
                str(int(indices[-1])) if self.frame_count else "empty",
                ",".join(self.image_keys),
            )
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        root = Path(os.environ.get("LTE_TEMPORAL_RAM_CACHE_DIR", "/dev/shm/lte_smolvla"))
        return root / f"{digest}.uint8", root / f"{digest}.json"

    @staticmethod
    def _distributed_rank():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    @staticmethod
    def _distributed_barrier():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _open_ram_cache(self, data_path, metadata_path):
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        shape = tuple(metadata["shape"])
        cache = np.memmap(data_path, mode="r+", dtype=np.uint8, shape=shape)
        self._ram_cache = torch.from_numpy(cache)

    def _build_ram_cache(self, data_path, metadata_path, *, chunk_size: int):
        data_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        first = self._decode(self.absolute_frame_indices[:1], device="cpu")
        shape = (self.frame_count, *first.shape[1:])
        temporary_data = data_path.with_name(f"{data_path.name}.{os.getpid()}.building")
        temporary_metadata = metadata_path.with_name(f"{metadata_path.name}.{os.getpid()}.building")
        cache = np.memmap(temporary_data, mode="w+", dtype=np.uint8, shape=shape)
        report_every = max(chunk_size, ((self.frame_count + 9) // 10 // chunk_size) * chunk_size)
        for start in range(0, self.frame_count, chunk_size):
            end = min(start + chunk_size, self.frame_count)
            cache[start:end] = self._decode(self.absolute_frame_indices[start:end], device="cpu").numpy()
            if end == self.frame_count or end % report_every == 0:
                print(
                    f"Building LTE shared-RAM image cache: {end:,}/{self.frame_count:,} frames "
                    f"({time.perf_counter() - started:.1f}s).",
                    flush=True,
                )
        cache.flush()
        del cache
        os.replace(temporary_data, data_path)
        with temporary_metadata.open("w") as handle:
            json.dump({"shape": shape, "dtype": "uint8"}, handle)
        os.replace(temporary_metadata, metadata_path)
        self._open_ram_cache(data_path, metadata_path)
        gib = np.prod(shape) / 1024**3
        print(
            f"Built LTE shared-RAM image cache: {self.frame_count:,} frames, {gib:.2f} GiB, "
            f"{time.perf_counter() - started:.1f}s.",
            flush=True,
        )
        self._build_seconds = time.perf_counter() - started

    def _preload_ram(self, *, chunk_size: int):
        if self._ram_cache is not None:
            return
        data_path, metadata_path = self._ram_cache_paths()
        rank = self._distributed_rank()
        if rank == 0 and not (data_path.exists() and metadata_path.exists()):
            self._build_ram_cache(data_path, metadata_path, chunk_size=chunk_size)
        self._distributed_barrier()
        if self._ram_cache is None:
            self._open_ram_cache(data_path, metadata_path)
        if rank != 0:
            print(f"Opened LTE shared-RAM image cache: {data_path.name}.", flush=True)

    def _preload_gpu(self, device, *, chunk_size: int):
        """Build one local-GPU raw cache with fast, temporary CPU conversion."""
        if device.type != "cuda":
            raise RuntimeError("temporal_image_cache_on_gpu requires CUDA")
        if self.frame_count == 0:
            raise RuntimeError("Cannot build a raw-image cache from an empty dataset")
        # torchrun's safe global default is OMP_NUM_THREADS=1.  That makes the
        # uint8 conversion of every cached image serial and was responsible for
        # several-minute builds. Keep the higher setting local to this one-time
        # setup; the ordinary training loop returns to the launcher setting.
        previous_threads = torch.get_num_threads()
        target_threads = self.build_num_threads or previous_threads
        if target_threads != previous_threads:
            torch.set_num_threads(target_threads)
        try:
            started = time.perf_counter()
            first = self._decode(self.absolute_frame_indices[:1], device="cpu")
            cache = torch.empty((self.frame_count, *first.shape[1:]), dtype=torch.uint8, device=device)
            for start in range(0, self.frame_count, chunk_size):
                end = min(start + chunk_size, self.frame_count)
                cache[start:end].copy_(
                    self._decode(self.absolute_frame_indices[start:end], device="cpu"), non_blocking=False
                )
            torch.cuda.synchronize(device)
            self._gpu_cache = cache
            self._build_seconds = time.perf_counter() - started
        finally:
            if target_threads != previous_threads:
                torch.set_num_threads(previous_threads)
        gib = cache.numel() * cache.element_size() / 1024**3
        print(
            f"Built LTE raw-image GPU cache: {self.frame_count:,} frames, {gib:.2f} GiB, "
            f"{self._build_seconds:.1f}s ({target_threads} CPU conversion threads).",
            flush=True,
        )

    def preload(self, device, *, chunk_size: int = 256):
        """Build a raw uint8 cache once, before optimisation begins."""
        if not (self.cache_on_gpu or self.cache_in_ram):
            return
        if self._components is not None:
            for component in self._components:
                component.preload(device, chunk_size=chunk_size)
            return
        if self._gpu_cache is not None or self._ram_cache is not None:
            return
        if self.cache_in_ram:
            self._preload_ram(chunk_size=chunk_size)
            return
        self._preload_gpu(device, chunk_size=chunk_size)

    def _gather_combined(self, frame_ids, device):
        flat = frame_ids.to(device=device, dtype=torch.long).reshape(-1)
        boundaries = self._component_offsets.to(device=device)
        component_ids = torch.bucketize(flat, boundaries[1:], right=True)
        result = None
        for component_id, component in enumerate(self._components):
            locations = torch.nonzero(component_ids == component_id, as_tuple=False).flatten()
            if not locations.numel():
                continue
            local_positions = flat[locations] - boundaries[component_id]
            source_ids = component.absolute_frame_indices[local_positions.cpu()].to(device=device)
            values = component.gather(source_ids, device)
            if result is None:
                result = torch.empty((flat.numel(), *values.shape[1:]), dtype=values.dtype, device=device)
            result[locations] = values
        if result is None:
            raise ValueError("Cannot gather an empty temporal frame set")
        return result.reshape(*frame_ids.shape, *result.shape[1:])

    def gather(self, frame_ids, device):
        """Return raw frames as ``(*frame_ids.shape, cameras, C, H, W)``."""
        if self._components is not None:
            return self._gather_combined(frame_ids, device)
        if self.cache_in_ram:
            self.preload(device)
            indices = self.cached_indices(frame_ids).cpu()
            return self._ram_cache[indices].to(device=device, non_blocking=False)
        if self.cache_on_gpu:
            self.preload(device)
            return self._gpu_cache[self.cached_indices(frame_ids).to(device=device, dtype=torch.long)]
        frames = self._decode(frame_ids, device)
        return frames.reshape(*frame_ids.shape, *frames.shape[1:])

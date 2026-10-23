"""Native RobotImageDataset plus absolute frame IDs for full-prefix LTE."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import zarr

from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info, pack_robot_state

from .policy import FRAME_INDICES


POLICY_DIR = Path(__file__).resolve().parent
# These are the existing DP dataset/encoder schema, not physical camera names.
# Physical source names are discovered, ordered and persisted during conversion.
RGB_DATA_KEYS = {"head_cam": "head_camera", "left_cam": "left_camera", "right_cam": "right_camera"}


def robot_dimensions(env_cfg_type, action_type):
    info = dict(get_robot_action_dim_info(env_cfg_type))
    if action_type == "ee":
        info["arm_dim"] = [7] * len(info["arm_dim"])
    return info


class LTEImageDataset(RobotImageDataset):
    """Reuse native replay buffer, sampling, split, normalizer and RGB handling."""
    rgb_data_keys = RGB_DATA_KEYS

    def __init__(self, zarr_path, n_obs_steps=3, **kwargs):
        root = zarr.open_group(str(zarr_path), mode="r")
        self.metadata = dict(root.attrs.get("lte_metadata", {}))
        if not self.metadata.get("camera_names"):
            raise ValueError("Dataset needs camera-order metadata; use LTE_DP/process_data.sh to create the native DP zarr")
        self.n_obs_steps = n_obs_steps
        super().__init__(zarr_path=str(zarr_path), **kwargs)
        ends = np.asarray(self.replay_buffer.episode_ends)
        self.max_episode_length = int(np.diff(np.r_[0, ends]).max())

    def __getitem__(self, index):
        result = dict(super().__getitem__(index))
        rows = self.sampler.indices[index]
        start, end, sample_start = rows[..., 0], rows[..., 1], rows[..., 2]
        positions = start[..., None] - sample_start[..., None] + np.arange(self.n_obs_steps)
        frames = np.clip(positions, start[..., None], end[..., None] - 1).astype(np.int64)
        result[FRAME_INDICES] = torch.from_numpy(frames)
        return result

    def postprocess(self, samples, device):
        result = super().postprocess(samples, device)
        result["obs"][FRAME_INDICES] = samples[FRAME_INDICES].to(device=device)
        return result

    def get_normalizer(self, *args, **kwargs):
        normalizer = super().get_normalizer(*args, **kwargs)
        # Native RobotWorkspace already passes this object to both online
        # and EMA policies. Carry a runtime-only dataset reference through
        # that existing interface; numerical normalization is unchanged.
        # This Python attribute is not part of the normalizer state_dict.
        object.__setattr__(normalizer, "_lte_dataset", self)
        return normalizer


def convert(source, output, env_cfg_type, action_type, episodes=None, image_size=(240, 320), max_frames=None):
    paths = sorted(Path(source).glob("episode_*.hdf5"))
    if episodes is not None:
        if episodes <= 0 or episodes > len(paths):
            raise ValueError(f"Requested {episodes} episodes, available {len(paths)}")
        paths = paths[:episodes]
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 under {source}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Preserving existing dataset: {output}")
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"Partial conversion exists: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dest = zarr.open_group(str(partial), mode="w")
    data, meta = dest.create_group("data"), dest.create_group("meta")
    robot_info = robot_dimensions(env_cfg_type, action_type)
    ends, names = [], None
    h, w = image_size
    if h < 1 or w < 1:
        raise ValueError("Image dimensions must be positive")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    for episode, path in enumerate(paths):
        demo = load_hdf5(str(path))
        available = sorted(k for k, v in demo["vision"].items() if "colors" in v)
        if len(available) < len(RGB_DATA_KEYS):
            raise ValueError("Native DP's dataset requires three RGB views")
        selected = available[:len(RGB_DATA_KEYS)]
        if names is None:
            names = selected
        if selected != names:
            raise ValueError(f"Camera order changed in {path}")
        state = pack_robot_state(demo, action_type, robot_info, source_type="dataset", state_type="state").astype(np.float32)
        action = pack_robot_state(demo, action_type, robot_info, source_type="dataset", state_type="action").astype(np.float32)
        length = len(state) if max_frames is None else min(len(state), max_frames)
        if length == 0 or len(action) != len(state):
            raise ValueError(f"Invalid state/action lengths in {path}")
        if episode == 0:
            for key in RGB_DATA_KEYS.values():
                data.create_dataset(key, shape=(0, 3, h, w), chunks=(16, 3, h, w), dtype="u1", compressor=compressor)
            for key, values in (("state", state), ("action", action)):
                data.create_dataset(key, shape=(0, values.shape[-1]), chunks=(100, values.shape[-1]), dtype="f4", compressor=compressor)
        start = ends[-1] if ends else 0
        end = start + length
        for array in data.values():
            array.resize((end, *array.shape[1:]))
        data["state"][start:end], data["action"][start:end] = state[:length], action[:length]
        for source_name, data_key in zip(names, RGB_DATA_KEYS.values()):
            frames = []
            for frame in range(length):
                rgb = decode_image_bit(demo["vision"][source_name]["colors"][frame])
                frames.append(np.moveaxis(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), -1, 0))
                if len(frames) == 16 or frame == length - 1:
                    offset = start + frame + 1 - len(frames)
                    data[data_key][offset:start + frame + 1] = np.stack(frames)
                    frames = []
        ends.append(end)
        print(f"Converted {episode + 1}/{len(paths)}: {path.name}, {length} frames", flush=True)
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    dest.attrs["lte_metadata"] = dict(camera_names=names, obs_keys=list(RGB_DATA_KEYS),
        image_size=list(image_size), env_cfg_type=env_cfg_type, action_type=action_type,
        state_dim=int(data["state"].shape[-1]), action_dim=int(data["action"].shape[-1]),
        source=str(Path(source).resolve()), smoke_max_frames=max_frames)
    partial.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("action_type", choices=("joint", "ee"))
    parser.add_argument("expert_data_num", nargs="?", type=int)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-size", nargs=2, type=int, default=(240, 320), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--max-frames", type=int, help="Smoke tests only")
    args = parser.parse_args()
    setting = f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.action_type}"
    source = args.source or POLICY_DIR.parents[2] / "data" / args.bench_name / args.ckpt_name / args.env_cfg_type / "data"
    output = args.output or POLICY_DIR / "data" / f"{setting}.zarr"
    print(convert(source, output, args.env_cfg_type, args.action_type, args.expert_data_num, tuple(args.image_size), args.max_frames))


if __name__ == "__main__":
    main()

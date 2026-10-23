"""Compose the actual DP config with LTE settings and invoke its workspace."""

import argparse
from pathlib import Path

import hydra
import numpy as np
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name

from .data import POLICY_DIR
from .workspace import LTEWorkspace


NATIVE_CONFIG = POLICY_DIR.parent / "DP" / "diffusion_policy" / "config"


def load_config(overlay=None):
    with initialize_config_dir(version_base=None, config_dir=str(NATIVE_CONFIG)):
        config = compose(config_name="robot_dp")
    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, OmegaConf.load(POLICY_DIR / "config.yaml"))
    OmegaConf.set_struct(config, True)
    if overlay:
        config = OmegaConf.merge(config, OmegaConf.load(overlay))
    return config


def configure_run(dataset_path, output, args, overlay=None, overrides=()):
    config = load_config(overlay)
    root = zarr.open_group(str(dataset_path), mode="r")
    metadata = dict(root.attrs["lte_metadata"])
    if metadata["env_cfg_type"] != args.env_cfg_type or metadata["action_type"] != args.action_type:
        raise ValueError("Dataset robot/action type differs from training arguments")
    ends = root["meta/episode_ends"][:]
    config.bench_name = args.bench_name
    config.task.name = args.ckpt_name
    config.training.seed = args.seed
    config.task.dataset.zarr_path = str(Path(dataset_path).resolve())
    config.task.shape_meta.action.shape = [metadata["action_dim"]]
    config.task.shape_meta.obs.agent_pos.shape = [metadata["state_dim"]]
    for key in metadata["obs_keys"]:
        config.task.shape_meta.obs[key].shape = [3, *metadata["image_size"]]
    config.policy.history_max_episode_length = int(np.diff(np.r_[0, ends]).max())
    config.lte.checkpoint_dir = str(Path(output).resolve())
    config.lte.dataset_metadata = metadata
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    if config.training.gradient_accumulate_every != 1:
        # The inherited loop's accumulation condition is offset by one for
        # values >1. Keep the standard value until native DP fixes that path.
        raise ValueError("LTE uses native DP's standard gradient_accumulate_every=1")
    if config.dataloader.num_workers != 0 or config.val_dataloader.num_workers != 0:
        raise ValueError("Native DP's shared-buffer loader uses num_workers=0")
    OmegaConf.resolve(config)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_name")
    parser.add_argument("ckpt_name")
    parser.add_argument("env_cfg_type")
    parser.add_argument("action_type", choices=("joint", "ee"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--action-dim", type=int)
    args, overrides = parser.parse_known_args()
    if any("=" not in item for item in overrides):
        parser.error("Extra arguments must be native Hydra-style key=value overrides")
    setting = f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.action_type}"
    dataset = args.dataset or POLICY_DIR / "data" / f"{setting}.zarr"
    output = args.output or POLICY_DIR / "checkpoints" / build_run_dir_name(vars(args))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserving existing run: {output}; use a new run directory")
    config = configure_run(dataset, output, args, args.config, overrides)
    if args.action_dim is not None and config.task.shape_meta.action.shape != [args.action_dim]:
        raise ValueError("Training action dimension disagrees with get_action_dim.sh")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml")
    print(f"Native RobotWorkspace: epochs={config.training.num_epochs}, batch={config.dataloader.batch_size}, "
          f"horizon/obs/action={config.horizon}/{config.n_obs_steps}/{config.n_action_steps}", flush=True)
    cls = hydra.utils.get_class(config._target_)
    workspace: LTEWorkspace = cls(config, output_dir=str(output))
    workspace.run()


if __name__ == "__main__":
    main()

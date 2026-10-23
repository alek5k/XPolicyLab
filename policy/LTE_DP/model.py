"""Native DP checkpoint selection and action generation, with streaming LTE."""

from collections import deque
from pathlib import Path
import os

import cv2
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from XPolicyLab.policy.DP.model import Model as DPModel
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

from .data import robot_dimensions
from .policy import STREAM_LATENTS


class Model(DPModel):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = model_cfg["action_type"]
        self.robot_action_dim_info = robot_dimensions(model_cfg["env_cfg_type"], self.action_type)
        policy_dir = Path(__file__).resolve().parent
        root = resolve_checkpoint_root(model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir)
        checkpoint = self._resolve_checkpoint_file(root, model_cfg.get("checkpoint_num", "latest"))
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
        if "state_dicts" not in payload:
            raise ValueError("Expected a native DP workspace .ckpt; old standalone LTE .pt files are incompatible")
        cfg = payload["cfg"]
        self.metadata = OmegaConf.to_container(cfg.lte.dataset_metadata, resolve=True)
        for key in ("env_cfg_type", "action_type"):
            if model_cfg[key] != self.metadata[key]:
                raise ValueError(f"Checkpoint {key} differs from deployment")
        self.model = hydra.utils.instantiate(cfg.policy)
        weights = "ema_model" if cfg.training.use_ema and model_cfg.get("use_ema", True) else "model"
        self.model.load_state_dict(payload["state_dicts"][weights])
        device = model_cfg.get("device") or os.environ.get("LTE_DP_DEVICE", "cuda:0")
        torch.set_num_threads(min(8, torch.get_num_threads()))
        self.model.to(device).eval()
        self.config = {"n_obs_steps": self.model.n_obs_steps,
                       "temporal_latent_dim": self.model.temporal_encoder.latent_dim}
        self.reset()

    @torch.no_grad()
    def update_obs(self, obs):
        self.update_obs_batch([obs])

    @torch.no_grad()
    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("Expected at least one observation")
        env_indices = [int(obs.get("env_idx", 0)) for obs in obs_list]
        if len(set(env_indices)) != len(env_indices):
            raise ValueError("A batch must contain distinct env_idx values")
        h, w = self.metadata["image_size"]
        images, states = [], []
        for obs in obs_list:
            # The server supplies decoded RGB arrays; physical names come
            # from conversion metadata, not assumptions about camera keys.
            rgb = [cv2.resize(np.asarray(obs["vision"][key]["color"]), (w, h),
                    interpolation=cv2.INTER_AREA) for key in self.metadata["camera_names"]]
            if any(image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 for image in rgb):
                raise ValueError("Expected decoded HxWx3 uint8 RGB observations")
            images.append(np.stack([np.moveaxis(image, -1, 0) for image in rgb]))
            if len(self.robot_action_dim_info["arm_dim"]) == 1 and self.action_type == "joint":
                state = dict(obs["state"])
                if "joint_state" not in state and "arm_joint_state" in state:
                    state["joint_state"] = state["arm_joint_state"]
                obs = dict(obs, state=state)
            states.append(pack_robot_state(obs, self.action_type, self.robot_action_dim_info, source_type="obs"))
        images = torch.as_tensor(np.stack(images), device=self.model.device).float() / 255.0
        states = torch.as_tensor(np.stack(states), device=self.model.device, dtype=torch.float32)
        native_obs = {key: images[:, i] for i, key in enumerate(self.metadata["obs_keys"])}
        embeddings = self.model.encode_temporal_observation(native_obs)
        previous = torch.stack([self._latents.get(index, embeddings.new_zeros(self.config["temporal_latent_dim"])) for index in env_indices])
        latent = self.model.temporal_encoder(embeddings, previous)
        for row, index in enumerate(env_indices):
            self._latents[index] = latent[row]
            if index not in self._windows:
                self._windows[index] = deque(maxlen=self.config["n_obs_steps"])
            self._windows[index].append((images[row], states[row], latent[row]))
        self._latest = env_indices

    def get_action(self):
        if not self._latest:
            raise RuntimeError("Call update_obs before get_action")
        return self.get_action_batch([self._latest[0]])[0]

    @torch.no_grad()
    def get_action_batch(self, env_idx_list=None):
        indices = self._latest if env_idx_list is None else list(env_idx_list)
        if not indices:
            raise RuntimeError("Call update_obs_batch before get_action_batch")
        windows = []
        for index in indices:
            if index not in self._windows:
                raise KeyError(f"No observation for environment {index}")
            window = list(self._windows[index])
            windows.append([window[0]] * (self.config["n_obs_steps"] - len(window)) + window)
        images, states, latents = [torch.stack([torch.stack([step[k] for step in window]) for window in windows]) for k in range(3)]
        native_obs = {key: images[:, :, i] for i, key in enumerate(self.metadata["obs_keys"])}
        native_obs.update(agent_pos=states, **{STREAM_LATENTS: latents})
        actions = self.model.predict_action(native_obs)["action"].cpu().numpy()
        unpacked = [unpack_robot_state(chunk, self.action_type, self.robot_action_dim_info, source_type="obs") for chunk in actions]
        if len(self.robot_action_dim_info["arm_dim"]) == 1 and self.action_type == "joint":
            for chunk in unpacked:
                for action in chunk:
                    action["arm_joint_state"] = action.pop("joint_state")
        return unpacked

    def reset(self, env_idx_list=None):
        if env_idx_list is None:
            self._latents, self._windows, self._latest = {}, {}, []
        else:
            for index in env_idx_list:
                self._latents.pop(index, None)
                self._windows.pop(index, None)
            self._latest = [index for index in self._latest if index not in env_idx_list]

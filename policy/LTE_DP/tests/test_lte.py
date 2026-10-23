"""Check that LTE adds temporal state while native DP owns everything else."""

from pathlib import Path
from types import SimpleNamespace

import dill
import hydra
import numpy as np
import pytest
import torch
import zarr
from omegaconf import OmegaConf
from torch import nn

from diffusion_policy.dataset.robot_image_dataset import RobotImageDataset
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.workspace.robotworkspace import RobotWorkspace

from XPolicyLab.policy.LTE_DP.data import LTEImageDataset, RGB_DATA_KEYS
from XPolicyLab.policy.LTE_DP.policy import FRAME_INDICES, STREAM_LATENTS, EmbeddingCache, LTEDiffusionPolicy
from XPolicyLab.policy.LTE_DP.temporal import HistoryDecoder, ImageTemporalEncoder, history_loss, interpolate_history, sample_stratified_lags
from XPolicyLab.policy.LTE_DP.train import NATIVE_CONFIG, configure_run, load_config
from XPolicyLab.policy.LTE_DP.workspace import LTEWorkspace


@pytest.fixture(autouse=True, scope="module")
def threads():
    torch.set_num_threads(2)


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "tiny.zarr"
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    rng = np.random.default_rng(1)
    for key in RGB_DATA_KEYS.values():
        data[key] = rng.integers(0, 256, (16, 3, 32, 32), dtype=np.uint8)
    data["state"] = rng.normal(size=(16, 14)).astype(np.float32)
    data["action"] = rng.normal(size=(16, 14)).astype(np.float32)
    root.create_group("meta")["episode_ends"] = np.array([8, 16], dtype=np.int64)
    root.attrs["lte_metadata"] = dict(camera_names=["unusual_a", "other_b", "random_c"],
        obs_keys=list(RGB_DATA_KEYS), image_size=[32, 32], env_cfg_type="aloha_agilex", action_type="joint",
        state_dim=14, action_dim=14)
    args = SimpleNamespace(bench_name="RMBench", ckpt_name="tiny", env_cfg_type="aloha_agilex", action_type="joint", seed=0)
    config = configure_run(path, tmp_path / "run", args,
        overrides=["policy.down_dims=[32,64,128]", "policy.diffusion_step_embed_dim=32",
            "policy.num_inference_steps=2", "policy.encode_chunk_size=4", "training.device=cpu",
            "dataloader.batch_size=2", "val_dataloader.batch_size=2"])
    return config, path


def make_policy(config):
    dataset = hydra.utils.instantiate(config.task.dataset)
    policy = hydra.utils.instantiate(config.policy)
    policy.set_normalizer(dataset.get_normalizer())
    assert policy._dataset is dataset
    return policy, dataset


def test_defaults_inherit_native_dp():
    config = load_config()
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=str(NATIVE_CONFIG)):
        native = compose(config_name="robot_dp")
    for section in ("training", "optimizer", "ema", "dataloader", "val_dataloader", "checkpoint"):
        assert OmegaConf.to_container(config[section], resolve=True) == OmegaConf.to_container(native[section], resolve=True)
    assert list(native.task.shape_meta.obs) == ["head_cam"]
    assert not hasattr(RobotWorkspace, "on_dataset_ready")
    assert config.horizon == 8 and config.n_obs_steps == 3 and config.n_action_steps == 6
    assert config.training.num_epochs == config.training.checkpoint_every == 600
    assert config.dataloader.batch_size == 128
    assert config.task.dataset.val_ratio == 0.02
    assert config.optimizer.lr == 1e-4 and config.ema.max_value == 0.9999
    assert list(config.policy.down_dims) == [256, 512, 1024]
    assert list(config.task.shape_meta.obs) == ["head_cam", "left_cam", "right_cam", "agent_pos"]


def test_native_policy_classes_and_mean_pooling(setup):
    config, _ = setup
    policy, _ = make_policy(config)
    assert isinstance(policy, DiffusionUnetImagePolicy)
    assert LTEDiffusionPolicy.conditional_sample is DiffusionUnetImagePolicy.conditional_sample
    assert type(policy.model).__module__ == "diffusion_policy.model.diffusion.conditional_unet1d"
    assert type(policy.obs_encoder.base_encoder).__module__ == "diffusion_policy.model.vision.multi_image_obs_encoder"
    assert all(isinstance(model.avgpool, nn.AdaptiveAvgPool2d) for model in policy.obs_encoder.base_encoder.key_model_map.values())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in policy.modules())


def test_dataset_preserves_native_sampling_values_and_normalizer(setup):
    config, _ = setup
    lte = hydra.utils.instantiate(config.task.dataset)
    kwargs = OmegaConf.to_container(config.task.dataset, resolve=True)
    kwargs.pop("_target_")
    kwargs.pop("n_obs_steps")
    native = RobotImageDataset(**kwargs)
    indices = np.array([0, 3])
    native_batch = native.postprocess(native[indices], "cpu")
    lte_batch = lte.postprocess(lte[indices], "cpu")
    assert np.array_equal(native.sampler.indices, lte.sampler.indices)
    assert np.array_equal(native.train_mask, lte.train_mask)
    assert lte_batch["obs"][FRAME_INDICES].tolist() == [[0, 0, 0], [1, 2, 3]]
    assert torch.equal(native_batch["action"], lte_batch["action"])
    assert all(torch.equal(value, lte_batch["obs"][key]) for key, value in native_batch["obs"].items())
    first, second = native.get_normalizer().state_dict(), lte.get_normalizer().state_dict()
    assert first.keys() == second.keys() and all(torch.equal(first[key], second[key]) for key in first)
    val = lte.get_validation_dataset()
    assert val.postprocess(val[np.array([0, 1])], "cpu")["obs"][FRAME_INDICES].tolist() == [[8, 8, 8], [8, 8, 9]]


def test_diffusion_objective_is_parent_implementation(setup, monkeypatch):
    config, _ = setup
    policy, dataset = make_policy(config)
    batch = dataset.postprocess(dataset[np.array([0, 3])], "cpu")
    latent = torch.randn(2, 3, 32)
    monkeypatch.setattr(policy, "_history_features", lambda indices, reconstruct: (latent, torch.tensor(1.75)))
    torch.manual_seed(42)
    actual = policy.compute_loss(batch)
    obs = dict(batch["obs"])
    obs.pop(FRAME_INDICES)
    policy.obs_encoder.context = latent
    torch.manual_seed(42)
    expected = DiffusionUnetImagePolicy.compute_loss(policy, dict(batch, obs=obs))
    policy.obs_encoder.context = None
    assert torch.equal(actual, expected + 1.75)


def test_cache_refresh_and_validation_does_not_touch_training(setup):
    config, _ = setup
    policy, dataset = make_policy(config)
    cache = policy._training_cache
    assert cache.ensure(policy, dataset, 0)
    original = cache.values.clone()
    assert not cache.values.requires_grad
    for step in (1, 4999):
        assert not cache.ensure(policy, dataset, step)
    batch = dataset.postprocess(dataset[np.array([0, 3])], "cpu")
    with torch.no_grad():
        policy.compute_loss(batch)
    assert cache.rebuild_count == 1 and torch.equal(cache.values, original)
    assert policy._evaluation_cache.rebuild_count == 1
    assert cache.ensure(policy, dataset, 5000)
    assert not cache.ensure(policy, dataset, 5000)
    assert cache.rebuild_count == 2


def test_native_forward_gradients_and_action_sampling(setup):
    config, _ = setup
    policy, dataset = make_policy(config)
    batch = dataset.postprocess(dataset[np.array([2, 5])], "cpu")
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(next(backbone.parameters()).grad.abs().sum() > 0 for backbone in policy.obs_encoder.base_encoder.key_model_map.values())
    assert next(policy.temporal_encoder.parameters()).grad.abs().sum() > 0
    assert next(policy.history_decoder.parameters()).grad.abs().sum() > 0
    assert policy._training_cache.values.grad is None
    policy.eval()
    result = policy.predict_action(batch["obs"])
    assert result["action"].shape == (2, 6, 14) and torch.isfinite(result["action"]).all()
    assert result["action_pred"].shape == (2, 8, 14)


def test_stratified_lags_and_fixed_dataset_normalization():
    torch.manual_seed(1)
    lags = sample_stratified_lags(torch.tensor([[200.0]]), 16)
    assert torch.equal((lags / 12.5).floor().long(), torch.arange(16)[None, None])
    values = torch.arange(201.0)[None, :, None]
    assert torch.allclose(interpolate_history(values, (200-lags).flatten(1))[..., 0], (200-lags).flatten(1))
    encoder, decoder = ImageTemporalEncoder(512), HistoryDecoder(32, 512)
    captured = []
    handle = decoder.register_forward_pre_hook(lambda _, args: captured.append(args[1]))
    embeddings = torch.randn(1, 201, 512, requires_grad=True)
    _, loss = history_loss(encoder, decoder, embeddings, torch.tensor([200]), 3, 401, 16)
    loss.backward()
    handle.remove()
    assert embeddings.grad is None
    assert 0.9 < captured[0].max() <= 1.0


def test_temporal_causality_and_padding():
    encoder, decoder = ImageTemporalEncoder(512), HistoryDecoder(32, 512)
    embeddings = torch.randn(2, 6, 512)
    elapsed = torch.tensor([2, 5])
    torch.manual_seed(8)
    latent, loss = history_loss(encoder, decoder, embeddings, elapsed, 3, 201, 16)
    embeddings[0, 3:] = 1000
    torch.manual_seed(8)
    other, other_loss = history_loss(encoder, decoder, embeddings, elapsed, 3, 201, 16)
    assert torch.equal(latent, other) and torch.equal(loss, other_loss)
    latent, _ = history_loss(encoder, decoder, embeddings[:, :1], torch.zeros(2, dtype=torch.long), 3, 201, 16)
    assert torch.equal(latent[:, 0], latent[:, 2])


@pytest.mark.parametrize("count", [1, 2, 3, None])
def test_temporal_camera_concatenation(setup, count):
    config, _ = setup
    config.policy.temporal_num_images = count
    policy, _ = make_policy(config)
    assert policy.temporal_embedding_dim == 512 * (3 if count is None else count)
    assert policy.temporal_encoder.mlp[0].in_features == policy.temporal_embedding_dim + 32


@pytest.mark.parametrize("count", [0, 4, True])
def test_invalid_camera_counts(setup, count):
    config, _ = setup
    config.policy.temporal_num_images = count
    with pytest.raises((ValueError, hydra.errors.InstantiationException)):
        make_policy(config)


def test_native_workspace_checkpoint_selection_and_update_clock(setup):
    config, _ = setup
    output = Path(config.lte.checkpoint_dir)
    output.mkdir()
    config.training.num_epochs = 2
    config.training.max_train_steps = 1
    config.training.max_val_steps = 1
    config.training.checkpoint_every = 1
    config.training.sample_every = 1
    config.training.resume = False
    workspace = LTEWorkspace(config, output_dir=str(output))
    workspace.run()
    assert workspace.lte_optimizer_updates == 2
    assert workspace.model._training_cache.built_at == 0
    assert workspace.model._training_cache.rebuild_count == 1
    assert workspace.model._dataset is not None and workspace.ema_model._dataset is workspace.model._dataset
    assert (output / "lte_metrics.jsonl").is_file()
    assert sorted(path.name for path in output.glob("*.ckpt")) == ["1.ckpt", "2.ckpt"]
    assert not (output / "best.pt").exists()
    payload = torch.load(output / "2.ckpt", map_location="cpu", pickle_module=dill, weights_only=False)
    assert "ema_model" in payload["state_dicts"] and "optimizer" in payload["state_dicts"]
    assert not any("cache" in key for key in payload["state_dicts"]["model"])
    from XPolicyLab.policy.LTE_DP.model import Model
    adapter = Model(dict(ckpt_name=str(output), env_cfg_type="aloha_agilex", action_type="joint", device="cpu"))
    assert adapter._resolve_checkpoint_file(output, "latest").endswith("2.ckpt")
    def obs(index):
        return dict(env_idx=index, vision={key:dict(color=np.full((32,32,3), 120, np.uint8)) for key in adapter.metadata["camera_names"]},
            state={"left_arm_joint_state":np.zeros(6), "right_arm_joint_state":np.zeros(6),
                   "left_ee_joint_state":np.zeros(1), "right_ee_joint_state":np.zeros(1)})
    adapter.update_obs_batch([obs(103), obs(9)])
    before = {key:value.clone() for key,value in adapter._latents.items()}
    chunks = adapter.get_action_batch([9, 103])
    assert len(chunks)==2 and len(chunks[0])==6
    assert all(torch.equal(before[key], value) for key,value in adapter._latents.items())
    adapter.update_obs(obs(9))
    assert not torch.equal(before[9], adapter._latents[9])
    adapter.reset([9])
    assert 103 in adapter._latents and 9 not in adapter._latents
    adapter.reset()
    assert not adapter._latents
    restored = LTEWorkspace(config, output_dir=str(output))
    restored.load_payload(payload)
    assert restored.lte_optimizer_updates == 2 and restored.model._optimizer_update == 2

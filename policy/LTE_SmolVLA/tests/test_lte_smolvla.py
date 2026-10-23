"""CPU regression tests using the pinned SmolVLA flow matcher and a tiny VLM."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from XPolicyLab.policy.LTE_SmolVLA.configuration_lte_smolvla import LTESmolVLAConfig
from XPolicyLab.policy.LTE_SmolVLA.modeling_lte_smolvla import LTESmolVLAPolicy, LTESmolVLAFlowMatching
from XPolicyLab.policy.LTE_SmolVLA.temporal import (
    HistoryDecoder,
    ImageTemporalEncoder,
    causal_history_frame_indices,
    trim_batch_padding,
)
from XPolicyLab.policy.LTE_SmolVLA.dataset_cache import DatasetCache
from XPolicyLab.policy.LTE_SmolVLA.model import Model
from XPolicyLab.policy.LTE_SmolVLA.train import configure_history
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla import smolvlm_with_expert
from transformers import SmolVLMConfig


def config(**kwargs):
    return LTESmolVLAConfig(
        device="cpu", push_to_hub=False, chunk_size=3, n_action_steps=2,
        max_state_dim=4, max_action_dim=4, resize_imgs_with_padding=(16, 16),
        num_vlm_layers=2, num_expert_layers=2, expert_width_multiplier=1.0,
        temporal_latent_dim=8, temporal_hidden_dim=32, history_decoder_hidden_dim=32,
        num_history_queries=3, num_steps=2, history_max_episode_length=3,
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (2,)),
                        **{f"observation.images.camera{i}": PolicyFeature(FeatureType.VISUAL, (3, 16, 16)) for i in range(1, 4)}},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (2,))},
        **kwargs,
    )


def tiny_vlm_config(vocab_size=32):
    return SmolVLMConfig(
        text_config={"model_type": "llama", "hidden_size": 64, "intermediate_size": 128,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
                     "head_dim": 16, "vocab_size": vocab_size, "pad_token_id": 0},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                       "num_attention_heads": 4, "image_size": 16, "patch_size": 4},
        scale_factor=2,
    )


@pytest.fixture
def tiny_policy(monkeypatch):
    monkeypatch.setattr(smolvlm_with_expert.AutoConfig, "from_pretrained", lambda *a, **k: tiny_vlm_config())
    monkeypatch.setattr(smolvlm_with_expert.AutoProcessor, "from_pretrained", lambda *a, **k: SimpleNamespace(
        tokenizer=SimpleNamespace(fake_image_token_id=1, global_image_token_id=2)))
    policy = LTESmolVLAPolicy(config())
    policy.model._normalize_temporal_embeddings(torch.randn(6, 64), fit=True)
    return policy


def batch():
    return {"observation.state": torch.randn(2, 2),
            **{f"observation.images.camera{i}": torch.rand(2, 3, 16, 16) for i in range(1, 4)},
            "observation.language.tokens": torch.ones(2, 3, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "action": torch.randn(2, 3, 2), "action_is_pad": torch.tensor([[False, False, True], [False, True, True]]),
            "index": torch.tensor([1, 4])}


def test_actual_smolvla_flow_backward_and_online_memory(tiny_policy):
    policy = tiny_policy
    flow = policy.model
    flow._lte_episode_ends = torch.tensor([3, 6])
    flow.config._runtime_causal_prefix_length = 3
    flow._embedding_cache = torch.randn(6, 64)
    flow._embedding_cache_step = 0
    flow._temporal_image_cache = DatasetCache(SimpleNamespace(hf_dataset={"index": list(range(6))}), ["x"])
    policy.train()
    loss, metrics = policy(batch())
    assert torch.isfinite(loss) and metrics["history_loss"] > 0
    loss.backward()
    for module in (flow.temporal_encoder, flow.history_decoder, flow.temporal_to_vlm, flow.action_out_proj):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in flow.vlm_with_expert.vlm.parameters())
    before = flow._lte_optimizer_step.clone()
    policy(batch())
    assert flow._lte_optimizer_step == before
    padded = batch()
    padded["action_is_pad"] = torch.ones(2, 3, dtype=torch.bool)
    _, masked_metrics = policy(padded)
    assert masked_metrics["action_loss"] == 0
    policy.update()
    assert flow._lte_optimizer_step == before + 1
    policy.eval()
    observation = batch()
    state = policy.advance_temporal_state(observation, None)
    actions = policy.predict_with_latent(observation, state)
    assert actions.shape == (2, 3, 2) and torch.isfinite(actions).all()
    assert not torch.equal(state, policy.advance_temporal_state(observation, state))


def test_history_matches_online_and_respects_episode_boundaries():
    encoder = ImageTemporalEncoder(16, 4, 8)
    frames, mask = causal_history_frame_indices(torch.tensor([0, 2, 3, 4]), torch.tensor([3, 5]), 3)
    assert frames.tolist() == [[0, 0, 0], [0, 1, 2], [3, 3, 3], [3, 3, 4]]
    embeddings = torch.randn(5, 16)
    latent = encoder.encode_history(embeddings[frames], mask)[:, -1]
    for row, frame in enumerate([0, 2, 3, 4]):
        online = torch.zeros(1, 4)
        for index in range(0 if frame < 3 else 3, frame + 1):
            online = encoder(embeddings[index:index+1], online)
        torch.testing.assert_close(latent[row], online[0])


def test_batch_padding_trim_preserves_final_latent_and_history_loss():
    torch.manual_seed(0)
    encoder = ImageTemporalEncoder(16, 4, 8)
    decoder = HistoryDecoder(4, 16, 8)
    frames, valid = causal_history_frame_indices(
        torch.tensor([1, 4]), torch.tensor([3, 6]), 6
    )
    embeddings = torch.randn(2, 6, 16).masked_fill(~valid.unsqueeze(-1), 0.0)
    full_latents = encoder.encode_history(embeddings, valid)

    active_columns = valid.any(dim=0)
    trimmed_frames, trimmed_valid = trim_batch_padding(frames, valid)
    trimmed_embeddings = embeddings[:, active_columns]
    trimmed_latents = encoder.encode_history(trimmed_embeddings, trimmed_valid)

    assert trimmed_frames.shape[1] < frames.shape[1]
    assert trimmed_valid.any(dim=0).all()
    torch.testing.assert_close(trimmed_latents[:, -1], full_latents[:, -1])

    history_model = SimpleNamespace(
        config=SimpleNamespace(
            num_history_queries=3,
            history_max_episode_length=6,
            temporal_embedding_dim=16,
        ),
        history_decoder=decoder,
    )
    torch.manual_seed(1)
    full_loss = LTESmolVLAFlowMatching.history_loss(
        history_model, full_latents, embeddings, valid
    )
    torch.manual_seed(1)
    trimmed_loss = LTESmolVLAFlowMatching.history_loss(
        history_model, trimmed_latents, trimmed_embeddings, trimmed_valid
    )
    torch.testing.assert_close(trimmed_loss, full_loss)


def test_query_history_encodes_batch_times_queries_in_one_vision_call(tiny_policy):
    flow = tiny_policy.model
    flow.config.temporal_query_history_only = True
    flow.config.temporal_embedding_cache_enabled = False
    flow.config.temporal_image_cache_on_gpu = True
    flow.config.temporal_embedding_normalize = False
    flow._lte_episode_ends = torch.tensor([3, 6])
    flow.config._runtime_causal_prefix_length = 3
    captured = {}

    class Cache:
        available = True

        def gather(self, frame_ids, device):
            captured["frame_ids"] = frame_ids.clone()
            return torch.rand(*frame_ids.shape, 1, 3, 16, 16, device=device)

    flow._temporal_image_cache = Cache()
    latents, _, _ = flow.temporal_history(torch.tensor([1, 4]), torch.device("cpu"))
    assert latents.shape[:2] == (2, flow.config.num_history_queries)
    assert captured["frame_ids"].shape == (2, flow.config.num_history_queries)
    assert flow._last_temporal_vision_batch_size == 2 * flow.config.num_history_queries
    assert flow._last_temporal_vision_encoder_calls == 1


def test_normalization_fixed_and_saved(tiny_policy, tmp_path):
    flow = tiny_policy.model
    original = flow._temporal_embedding_mean.clone()
    flow._normalize_temporal_embeddings(torch.randn(2, 64) + 100)
    torch.testing.assert_close(original, flow._temporal_embedding_mean)
    tiny_policy.save_pretrained(tmp_path)
    loaded = LTESmolVLAPolicy.from_pretrained(tmp_path, strict=True)
    torch.testing.assert_close(original, loaded.model._temporal_embedding_mean)
    assert loaded.model._temporal_embedding_stats_ready


def test_camera_selection_and_non_square_resize(tiny_policy):
    assert config(temporal_num_images=None).temporal_camera_indices() == [0, 1, 2]
    with pytest.raises(ValueError):
        config(temporal_num_images=4).temporal_camera_indices()
    flow = tiny_policy.model
    flow.config.resize_imgs_with_padding = (24, 16)
    image = torch.rand(2, 3, 8, 12)
    prepared, _ = tiny_policy.prepare_images({key: image for key in flow.config.image_features})
    torch.testing.assert_close(flow._prepare_temporal_image(image), prepared[0])


def test_adapter_loads_registered_checkpoint_config(tiny_policy, tmp_path, monkeypatch):
    from XPolicyLab.policy.SmolVLA import model as base
    tiny_policy.save_pretrained(tmp_path)
    monkeypatch.setattr(base, "get_robot_action_dim_info", lambda _: {"arm_dim": [1], "ee_dim": [1]})
    monkeypatch.setattr(Model, "_build_processors", lambda _: (lambda x: x, lambda x: x))
    model = Model({"pretrained_path": str(tmp_path), "env_cfg_type": "test", "device": "cpu"})
    assert isinstance(model.policy.config, LTESmolVLAConfig)


def test_selected_episode_cache_and_rename():
    class Dataset:
        hf_dataset = {"index": [3, 4]}
        def __getitem__(self, index):
            return {"source_camera": torch.full((1, 3, 4, 4), float(index))}
    cache = DatasetCache(Dataset(), ["camera1"], {"source_camera": "camera1"})
    assert cache.cached_indices(torch.tensor([[4, 3]])).tolist() == [[1, 0]]
    assert cache.gather(torch.tensor([4]), "cpu").shape == (1, 1, 3, 4, 4)
    with pytest.raises(ValueError):
        cache.cached_indices(torch.tensor([0]))


def test_adapter_observes_between_chunks_and_keeps_env_identity(monkeypatch):
    from XPolicyLab.policy.LTE_SmolVLA import model as adapter
    model = Model.__new__(Model)
    model.policy = SimpleNamespace(reset=lambda: None,
        advance_temporal_state=lambda batch, previous: batch["value"] + (0 if previous is None else previous),
        predict_with_latent=lambda batch, latent: latent.reshape(1, 1, 1).expand(1, 2, 2))
    model.default_prompt = "fallback"
    model.action_type = "joint"
    model.robot_action_dim_info = {"arm_dim": [1], "ee_dim": [1]}
    model.preprocessor = lambda value: value
    model.postprocessor = lambda value: value
    model.actions_per_chunk = 2
    model.reset()
    def update_base(self, observations):
        self._latest_env_idx_list = [o["env_idx"] for o in observations]
        self._latest_payloads = {o["env_idx"]: o for o in observations}
    monkeypatch.setattr(adapter.SmolVLAModel, "update_obs_batch", update_base)
    monkeypatch.setattr(adapter, "make_lerobot_observation", lambda obs: obs)
    model.update_obs_batch([{"env_idx": 9, "value": torch.tensor([1.0])}, {"env_idx": 2, "value": torch.tensor([10.0])}])
    model.update_obs_batch([{"env_idx": 2, "value": torch.tensor([20.0])}])
    actions = model.get_action_batch([2, 9])
    assert actions[0][0]["joint_state"].item() == 30
    assert actions[1][0]["joint_state"].item() == 1
    model.get_action_batch([9])
    assert model._temporal_states[9].item() == 1
    model.reset_envs([2])
    assert 2 not in model._temporal_states
    assert 2 not in model._processed_observations
    assert 2 not in model._latest_payloads
    assert model._temporal_states[9].item() == 1
    assert model.get_action_batch([9])[0][0]["joint_state"].item() == 1
    with pytest.raises(KeyError):
        model.get_action_batch([2])
    model.reset()
    assert not model._temporal_states
    with pytest.raises(KeyError):
        model.get_action_batch([9])


def test_invalid_configuration():
    for kwargs in ({"temporal_embedding_cache_refresh_steps": -1}, {"n_obs_steps": 2}, {"adapt_to_pi_aloha": True}):
        with pytest.raises(ValueError):
            config(**kwargs)


def test_training_metadata_enforces_full_history_and_camera_order():
    from XPolicyLab.policy.LTE_SmolVLA.train import CAMERA_RENAME
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,)},
        "action": {"dtype": "float32", "shape": (2,)},
        **{key: {"dtype": "image", "shape": (3, 16, 16), "names": ["channels", "height", "width"]}
           for key in reversed(CAMERA_RENAME)},
    }
    dataset = SimpleNamespace(meta=SimpleNamespace(features=features, episodes={
        "dataset_from_index": [0, 3], "dataset_to_index": [3, 5]}))
    cfg = config()
    configured_input_features = dict(cfg.input_features)
    configured_output_features = dict(cfg.output_features)
    configure_history(cfg, dataset, CAMERA_RENAME)
    assert cfg.input_features == configured_input_features
    assert cfg.output_features == configured_output_features
    assert list(cfg.image_features) == list(CAMERA_RENAME.values())
    assert cfg._runtime_episode_ends == [3, 5]
    assert cfg._runtime_causal_prefix_length == cfg.history_max_episode_length == 3
    cfg.max_causal_history_length = 2
    with pytest.raises(ValueError, match="never truncated"):
        configure_history(cfg, dataset, CAMERA_RENAME)


def test_training_metadata_rejects_configured_feature_mismatch():
    from XPolicyLab.policy.LTE_SmolVLA.train import CAMERA_RENAME
    features = {
        "observation.state": {"dtype": "float32", "shape": (2,)},
        "action": {"dtype": "float32", "shape": (2,)},
        **{key: {"dtype": "image", "shape": (3, 16, 16), "names": ["channels", "height", "width"]}
           for key in CAMERA_RENAME},
    }
    dataset = SimpleNamespace(meta=SimpleNamespace(features=features, episodes={
        "dataset_from_index": [0], "dataset_to_index": [3]}))
    cfg = config()
    cfg.input_features = {**cfg.input_features, "observation.state": PolicyFeature(FeatureType.STATE, (3,))}
    with pytest.raises(ValueError, match="input_features disagrees"):
        configure_history(cfg, dataset, CAMERA_RENAME)
    cfg = config()
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (3,))}
    with pytest.raises(ValueError, match="output_features disagrees"):
        configure_history(cfg, dataset, CAMERA_RENAME)


def test_base_settings_preserved_except_explicit_overrides(monkeypatch):
    from XPolicyLab.policy.LTE_SmolVLA.train import inherit_base_config, PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    base = SmolVLAConfig(device="cpu", pad_language_to="max_length", prefix_length=0,
                        chunk_size=50, n_action_steps=50, train_expert_only=False)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda _: base)
    merged = inherit_base_config(
        config(), {"chunk_size", "n_action_steps", "input_features", "output_features"}
    )
    assert merged.chunk_size == 3 and merged.n_action_steps == 2
    assert merged.input_features == config().input_features
    assert merged.output_features == config().output_features
    assert merged.pad_language_to == "max_length" and merged.prefix_length == 0
    assert merged.temporal_embedding_cache_refresh_steps == 5000
    assert not merged.load_vlm_weights

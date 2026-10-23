# LTE_SmolVLA

**Contributor:** Local LTE-DP integration | **Paper:** LTE-DP / SmolVLA | **arXiv:** Not specified | **Original code:** LTE-DP (`ltedp/modeling_lte_smolvla.py`, `common/temporal/`) and [XPolicyLab SmolVLA](../SmolVLA/README.md)

Adds LTE's image-only temporal recurrence, history reconstruction objective, and one projected prefix token to **XPolicyLab's SmolVLA**. The base classes come exclusively from `policy/SmolVLA/smovla`, installed by the baseline adapter at LeRobot `v0.4.4` (commit `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`). There is no import of LTE-DP or its separate LeRobot installation. The baseline adapter and its dependency source are unchanged.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd ~/Repos/XPolicyLab/policy/LTE_SmolVLA
bash install.sh
conda activate smolvla
```

This delegates installation to `../SmolVLA/install.sh` and shares its environment, source checkout and dependency pin. `SMOVLA_CONDA_ENV` overrides the environment name. Keep the pinned serializer (`draccus==0.10.0`) and Transformers 4.x requirements; do not install LTE-DP into this environment. See the baseline [installation notes](../SmolVLA/INSTALLATION.md) for video/system dependencies. The installer does not download trained LTE weights.

## Data Processing

Training consumes **LeRobot v3.0** exports from `scripts/transform_lerobot_v30_format.py`, with the official keys unchanged on disk. Use that converter in your benchmark workspace, or supply an existing official export. This adapter's `process_data.sh` validates and links the export; normalization is fitted from its dataset statistics during training, not rewritten into the dataset.

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> joint <official_export_path>
# Example with an existing official export:
bash process_data.sh RoboDojo stack_bowls arx_x5 joint /data/RoboDojo_sim_stack_bowls_v30
```

The link is `data/<bench>-<ckpt>-<robot>-joint`. Alternatively set `LTE_DATASET_ROOT` to the export directly. All three cameras are required: `observation.images.cam_high`, `cam_left_wrist`, `cam_right_wrist`. The trainer renames these to `camera1`, `camera2`, `camera3` in that order, matching the baseline adapter. State and action keys are `observation.state` and `action`. No custom image decoding or channel swapping is added.

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> joint <seed> <gpu_id> [trainer overrides...]

LTE_BASE_POLICY=lerobot/smolvla_base \
SMOVLA_REPO_ID=RoboDojo_sim_stack_bowls_v30 \
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0 \
  --policy.chunk_size=16 --policy.n_action_steps=8 \
  --policy.max_causal_history_length=2048 --batch_size=8 --steps=20000
```

`LTE_BASE_POLICY` accepts a local baseline SmolVLA `pretrained_model` directory or a Hub ID; default `lerobot/smolvla_base`. Fresh CLI runs inherit that checkpoint's architecture/tokenizer settings, with explicit policy arguments taking precedence; a supplied `--config_path` is treated as a complete configuration. Only LTE parameters may be absent from base weights; incompatible shapes/keys fail explicitly. `--policy.base_policy_path=null` initializes randomly and is intended for tests. Use `--policy.path=/path/to/LTE/pretrained_model` for LTE finetuning instead. Dataset action dimensions are checked against `utils/get_action_dim.sh` before training.

The entry point uses the pinned SmolVLA trainer, optimizer, processors and checkpoint machinery. LTE-specific dataset and processor wiring exists only in that training process. Episodes longer than `max_causal_history_length` cause an error: raise the limit to retain complete history. Streaming data and random image augmentation are disabled because reconstruction targets need a fixed causal visual history.

Checkpoints use the shared resolver naming under:

```text
policy/LTE_SmolVLA/checkpoints/<bench>-<ckpt>-<robot>-joint-<seed>/
  checkpoints/<step>/pretrained_model/
    config.json
    model.safetensors
    policy_preprocessor.json
    policy_postprocessor.json
    ... processor statistics and train_config.json
```

To resume optimizer/scheduler state, activate the same environment and run from the checkout's parent:

```bash
python -m XPolicyLab.policy.LTE_SmolVLA.train \
  --config_path=/path/to/run/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=40000
```

## Evaluation

```bash
bash eval.sh <bench> <task> <ckpt> <robot> joint <seed> <policy_gpu> <env_gpu> <policy_env> <eval_env>
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 joint 0 0 0 smolvla RMBench
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 joint 0 0 0 smolvla RMBench
```

Argument 9 also accepts a virtual-environment directory with `bin/activate`; argument 10 remains a conda environment. For simulation use `EVAL_ENV_TYPE=sim`. `ckpt` can be an absolute run or `pretrained_model` path. Evaluation requires a complete LTE checkpoint with fitted temporal embedding statistics; a plain SmolVLA checkpoint is only a training initializer.

**Workspace requirement:** XPolicyLab's shared robot helper reads `../env_cfg/` outside the checkout. A standalone `~/Repos/XPolicyLab` therefore needs a benchmark workspace arrangement with `env_cfg` in its parent. The separate `RMBench/XPolicyLab` Git worktree does not automatically receive these uncommitted files. Commit and update that worktree before evaluating there, or prepare a separate workspace containing this adapter and RMBench's `env_cfg`. The integration tests use a temporary workspace; they do not modify your parent directories or the RMBench worktree.

Supported action type: **joint**. Supported robot schema: registered single- or dual-arm robots within SmolVLA's configured state/action widths, providing the three baseline cameras. The debug loop is tested with **arx_x5**; other robots and real simulator task performance require validation. Robot dimensions come from shared XPolicyLab helpers.

`update_obs` and `update_obs_batch` advance LTE once for each received observation, including intermediate observations during chunk execution. `get_action` only uses the accumulated state, without observing the same frame again. Memory is keyed by `env_idx`, so shrinking or reordered batches cannot mix histories. `reset` clears all episode memory. Batch action inference currently runs each active environment sequentially. The deployment loop matches `demo_policy`, including the websocket client's `obs=env_idx_list` argument convention.

## Configuration

Core LTE defaults: `temporal_latent_dim=32`, `temporal_hidden_dim=128`, `history_decoder_hidden_dim=128`, `num_history_queries=16`, `lambda_history=1`, `temporal_num_images=1`. Set the latter to `null` for all cameras (concatenation, not learned fusion). All cameras still enter normal SmolVLA conditioning.

Detached targets are mean-pooled post-connector tokens, scaled as in SmolVLA. Dataset-wide normalization is fitted once and persisted in the checkpoint. The recurrence sees only these visual embeddings, never state, language or elapsed time. The decoder reconstructs sampled historical embeddings using a dataset-fixed lag scale. Action loss excludes padded timesteps and padded action dimensions.

The frozen-VLM default builds the embedding cache once. When `train_expert_only=false`, `temporal_embedding_cache_refresh_steps=5000` refreshes on optimizer updates, not forward calls. `temporal_embedding_cache_build_batch_size` controls the per-GPU vision-encoder batch only while constructing that table; `temporal_encode_chunk_size` controls uncached on-demand temporal encoding. The detached embedding table lives on the model device; raw frames are read through the training dataset when rebuilding, rather than retaining LTE-DP's separate raw-image RAM/GPU cache.

Additional `deploy.yml` keys: `checkpoint_num` selects a saved step using the baseline resolver; `pretrained_path` overrides checkpoint search; `smovla_vlm_path` optionally supplies local VLM config/tokenizer assets; `device` defaults to automatic selection; `actions_per_chunk` defaults to the checkpoint's `n_action_steps`; `prompt` is a fallback instruction. `result_dir` and `obs_transform_pipeline` retain XPolicyLab's deployment conventions.

## Notes and validation

The pinned base uses one current observation (`n_obs_steps=1`); LTE supplies full visual history. RTC, PEFT, RA-BC, `torch.compile`, Aloha coordinate conversions and streaming/augmented datasets are explicitly rejected. LTE-DP checkpoints from its newer LeRobot version are not assumed compatible and no automatic migration is performed. Start from XPolicyLab's SmolVLA weights and train LTE here.

```bash
# From the checkout's parent, in the SmolVLA environment:
python -m pytest XPolicyLab/policy/LTE_SmolVLA/tests -q
# Optional synthetic CPU training fixture (requires cached SmolVLM processor assets):
HF_HUB_OFFLINE=1 python -m XPolicyLab.policy.LTE_SmolVLA.tests.make_smoke_fixture /tmp/lte-smoke
bash XPolicyLab/policy/LTE_SmolVLA/tests/smoke_train.sh /tmp/lte-smoke
```

The regression suite uses tiny random VLM weights with the actual pinned SmolVLA flow matcher. Ten tests, synthetic training/resume, both websocket debug modes, and one CPU optimizer update using the full-size pretrained base have passed; see [VALIDATION.md](VALIDATION.md) for commands and limits. Synthetic training and debug checkpoints validate software wiring only and must never be used to control a real robot. No production training run, task success evaluation or leaderboard submission is claimed.

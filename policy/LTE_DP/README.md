# LTE_DP: native XPolicyLab DP + LTE

This adapter extends **the actual `policy/DP` implementation**. It does not
maintain a separate diffusion policy or training loop. The image-only temporal
encoder and reconstruction objective follow TemporalDiffusionPolicy / LTE-DP.

## What is reused

- `DiffusionUnetImagePolicy`: U-Net construction, diffusion scheduler, noise
  sampling, masking, diffusion loss, normalization, sampling and action slicing.
- `MultiImageObsEncoder` and `get_resnet`: the same independent ResNet18
  backbones, GroupNorm, spatial **mean pooling**, and preprocessing as DP.
- `RobotImageDataset`: replay buffer, sequence sampling/padding, batch loading,
  validation split, and normalizer. The subclass adds only absolute frame IDs.
- `RobotWorkspace.run`: training loop, optimizer, LR schedule, EMA, validation,
  train-batch sampling, JSON logging and epoch checkpoint schedule.
- Native workspace `.ckpt` format and DP's highest-numbered checkpoint resolver.

`config.yaml` is a small overlay on
`policy/DP/diffusion_policy/config/robot_dp.yaml`; it contains only LTE settings
and subclass targets. Changes to native DP defaults are picked up automatically.
The current inherited defaults are:

| Setting | Native DP and LTE_DP |
|---|---|
| Horizon / observation steps / executed actions | 8 / 3 / 6 |
| U-Net widths | 256, 512, 1024 |
| Diffusion training / inference steps | 100 / 100 |
| Training | 600 epochs, batch size 128 |
| Optimizer | AdamW, LR 1e-4, betas 0.95 / 0.999, weight decay 1e-6 |
| Scheduler | Native cosine schedule, 500 warmup steps |
| EMA | Native power 0.75, max decay 0.9999 |
| Validation / checkpoint saving | Every epoch / every 600 epochs |
| Checkpoint selection | Highest numbered epoch, EMA weights |

The three-camera and robot-state observation metadata is defined explicitly
in **LTE_DP/config.yaml only**. The original DP configuration and implementation
are unchanged. LTE inherits DP's model/training machinery while keeping these
observation choices local to its own adapter.

## Temporal additions

The encoder sees a detached native ResNet feature and its previous latent:

```text
z_t = tanh(z_(t-1) + MLP(concat(image_features_t.detach(), z_(t-1))))
loss = native_DP_diffusion_loss + lambda_history * history_reconstruction_MSE
```

The 32-value latent is appended to each native DP observation feature vector.
The temporal MLP and decoder use SiLU with hidden width 128. There are no idle,
velocity, timestep, proprioception, task-ID or language inputs to LTE. Robot
state remains ordinary native DP conditioning.

LTE unrolls the full causal prefix from the episode boundary. Its training-only
decoder reconstructs 16 stratified historical embedding queries for every
conditioned observation state, using linear interpolation and fixed dataset lag
normalization: `2 * lag / max(max_episode_length - 1, 1)`. There is no raw or
elapsed normalization mode. Multiple selected history views are concatenated.
Reconstruction gradients do not reach the visual encoder; native DP gradients do.

`temporal_num_images` selects the first N native RGB features in their stable
configured order (default 1; `null` selects all three). Camera selection never
assumes physical source camera names. The converter discovers source names,
sorts them, and records their mapping onto DP's existing three-camera schema.

The native replay buffer already holds decoded RGB. The LTE embedding table is
eagerly built at optimizer update 0 and refreshed every 5,000 **optimizer updates**,
without a per-step or middle phase. Validation and EMA sampling use separate
evaluation tables and do not change training's cache clock. The clock follows
actual `optimizer.step()` calls, rather than native logging's global_step, and
is included in workspace checkpoints. Caches are runtime-only. The LTE dataset
carries its runtime reference through the existing `get_normalizer` /
`set_normalizer` interface, so no native workspace hook or monkeypatch is needed.

## Installation

Already installed on this RTX 5090 machine:

```bash
source /home/sydney1/Repos/RMBench/XPolicyLab/policy/LTE_DP/.venv-cuda/bin/activate
```

This virtualenv reuses the existing `temporaldiffpolicy` Torch 2.9.1+cu128 stack
through system-site-packages; its native DP installation and additional
dependencies are local to the virtualenv. It depends on the source environment
remaining present. For a standalone environment:

```bash
conda create -n xpolicylab-lte python=3.10 pip -y
conda activate xpolicylab-lte
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
cd /home/sydney1/Repos/RMBench/XPolicyLab/policy/LTE_DP
bash install.sh
```

The install script installs `policy/DP` itself editable. Hydra 1.3.2 is used with
OmegaConf 2.3.0 instead of the old package metadata's Hydra 1.2 pin; the real
native training path is tested with this combination. No copied U-Net/vendor
implementation or second `diffusion_policy` source tree is needed.

## Data processing

```bash
cd /home/sydney1/Repos/RMBench/XPolicyLab/policy/LTE_DP
bash process_data.sh RMBench cover_blocks aloha_agilex joint \
  --source /home/sydney1/Repos/RMBench/data/demo_clean/cover_blocks/aloha_agilex/data
```

Arguments are `bench_name ckpt_name env_cfg_type action_type [expert_data_num]`.
`--source`, `--output`, and `--image-size HEIGHT WIDTH` are optional. Existing
outputs are preserved. `--max-frames` is for smoke tests only.

The converter produces the **native DP zarr schema**: `data/head_camera`,
`left_camera`, `right_camera`, `state`, `action`, and `meta/episode_ends`, with
NCHW uint8 RGB. These are native schema labels, not assumed source camera names.
It adds a `lte_metadata` attribute holding the discovered source-to-schema
mapping and robot/action metadata, which are persisted in checkpoints. Native DP
can read this dataset unchanged. Unannotated native zarr datasets should be
reconverted through this wrapper to record the mapping for LTE serving.

Physical input names are discovered from each demonstration and sorted; the
first three RGB views map to DP's three schema slots. Extra source views remain
in the original data and are not used by this native three-camera adapter.
Offline decoding uses only `decode_image_bit`; RGB is preserved end to end.
This path consumes XPolicyLab HDF5 directly, not a LeRobot dataset export.

## Training and checkpoints

```bash
bash train.sh RMBench cover_blocks aloha_agilex joint 0 1
```

Arguments match DP: `bench_name ckpt_name env_cfg_type action_type seed gpu_id`.
The launcher composes the actual native DP config, merges the LTE overlay,
fills dataset shapes/metadata, and calls `LTEWorkspace`, which delegates to
`RobotWorkspace.run`. Extra arguments include `--dataset`, `--output`,
`--config <partial-native-YAML>`, and native Hydra-style overrides:

```bash
bash train.sh RMBench cover_blocks aloha_agilex joint 0 1 \
  dataloader.batch_size=32 val_dataloader.batch_size=32
# Use all 50 demonstrations for a paper-budget run:
bash train.sh RMBench cover_blocks aloha_agilex joint 0 1 \
  task.dataset.val_ratio=0
```

Both examples use the normal run directory, so use a different `--output` for
multiple runs. The launcher refuses to overwrite an existing run. All settings
remain native DP defaults unless explicitly overridden. Native validation
currently reserves the final episode for any positive `val_ratio`: with the
default 0.02 and 50 demonstrations, training uses 49 and validation one.
Setting it to zero trains on all 50 and still uses final-epoch checkpointing.
Native normalization statistics use the complete replay buffer, as before.

Checkpoints land in
`checkpoints/<bench>-<ckpt>-<env_cfg>-<action>-<seed>/600.ckpt` by default.
Changing `training.checkpoint_every` produces additional numbered epochs.
The payload contains native online/EMA model and optimizer states, configuration,
and workspace pickles, including the LTE optimizer-update clock. There is no
`best.pt` or validation-loss checkpoint selection. `logs.json.txt` records native
metrics unchanged. `lte_metrics.jsonl` records the separate diffusion/history
losses and LTE cache clock at the first update and every 100 updates.

Native checkpoint load/round-trip is supported. The native automatic-resume
lookup still expects `output_dir/checkpoints/latest.ckpt`, which its numbered-only
saving loop does not ordinarily produce; this integration does not replace
native DP's resume logic with another training loop. The clock is restored when
loading a native workspace payload, and missing runtime caches rebuild once.

## Battery Try on GPU 1

The full-size run overlay is explicit in `configs/battery_try_gpu1.yaml`:
600 epochs, batch/validation batch 32 for the 32 GiB GPU, seed 0, one LTE
history image, and a 5,000-update cache refresh interval. With the converted
Battery Try cache present:

```bash
bash train.sh RMBench battery_try aloha_agilex joint 0 1 \
  --config configs/battery_try_gpu1.yaml
```

The launched run uses an informative separate output directory. Its exact
screen, command, log path and output directory are recorded in
`logs/battery_try_latest_launch.json`. Original DP files are unchanged.

## Evaluation

```bash
bash eval.sh RMBench cover_blocks cover_blocks aloha_agilex joint 0 \
  1 1 "$PWD/.venv-cuda" RMBench
```

The ten standard arguments are unchanged. The policy environment can be a
Conda name or virtualenv path. Checkpoints resolve through XPolicyLab's shared
resolver. `checkpoint_num: latest` calls DP's original highest-numeric-epoch
resolver; an explicit epoch selects `<epoch>.ckpt`. EMA weights are the default.

Streaming LTE state advances in `update_obs` / `update_obs_batch` on every
observation, including intermediate observations within an action chunk.
Per-environment state supports arbitrary IDs and partial resets. `get_action`
does not advance recurrence and calls native DP's `predict_action`/sampler.
The saved source camera mapping, shapes and observation horizon are used at
runtime. `model.py` receives already-decoded RGB arrays and never decodes them.

`deploy.yml` extra keys: `checkpoint_num`, `use_ema`, and `device` (null uses
`LTE_DP_DEVICE`, otherwise `cuda:0`). Both joint/EE conversion is implemented
with shared robot helpers, with seven-value poses in EE mode. Joint training
and serving on RMBench `aloha_agilex` are tested. Other robots/EE serving need
their own runtime validation.

## Smoke tests and compatibility

```bash
python -m pytest -q tests/test_lte.py
bash process_data.sh demo_clean cover_blocks aloha_agilex joint 2 \
  --image-size 64 64 --max-frames 32 --output smoke_artifacts/native_dp/cover_blocks.zarr
bash train.sh RMBench native_lte_smoke aloha_agilex joint 0 1 \
  --config smoke_config.yaml --dataset smoke_artifacts/native_dp/cover_blocks.zarr \
  --output smoke_artifacts/native_dp/run
EVAL_ENV_TYPE=debug bash eval.sh RMBench cover_blocks "$PWD/smoke_artifacts/native_dp/run" \
  aloha_agilex joint 0 1 1 "$PWD/.venv-cuda" RMBench
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RMBench cover_blocks \
  "$PWD/smoke_artifacts/native_dp/run" aloha_agilex joint 0 1 1 "$PWD/.venv-cuda" RMBench
```

Use fresh output paths when repeating. The smoke configuration keeps the native
8/3/6 horizons and training loop but reduces U-Net widths, image size, batch size,
epoch count and inference steps. Old standalone LTE HDF5 caches and `.pt`
checkpoints are preserved in `smoke_artifacts` but are incompatible with this
refactored native `.zarr` / `.ckpt` implementation. Full simulator task success
and long training have not been established by these smoke checks.

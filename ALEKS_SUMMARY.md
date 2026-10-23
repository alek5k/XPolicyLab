# ALEKS summary — native XPolicyLab DP with LTE

Current implementation, 16 October 2026:
`/home/sydney1/Repos/RMBench/XPolicyLab`, branch `lte-dp`.

## Repository setup

RMBench `origin` points to `https://github.com/alek5k/RMBench.git`; its original
repository is retained as `upstream`. RMBench `.gitmodules` points to
`https://github.com/alek5k/XPolicyLab.git`. Inside this submodule, `origin` is
your XPolicyLab fork and `upstream` is the original XPolicyLab repository.
Both XPolicyLab checkouts initially matched revision
`34d6135d609d46a10ccb81da24d5ddad0b4c1f7a`.

Changes remain local and uncommitted. Commit XPolicyLab changes inside the
submodule, then record that revision in RMBench when publishing. Downloaded
data, assets, earlier smoke datasets and checkpoints have been retained.

## Refactor to preserve native DP

The earlier standalone LTE implementation has been replaced with a thin
extension of **XPolicyLab's actual `policy/DP` code**:

| Component | Implementation now used |
|---|---|
| Policy | `LTEDiffusionPolicy(DiffusionUnetImagePolicy)` |
| Visual encoder | Native `MultiImageObsEncoder` + `get_resnet` |
| U-Net | Native `ConditionalUnet1D`, constructed by the parent policy |
| Diffusion loss / DDPM sampling | Inherited parent implementation |
| Dataset | `LTEImageDataset(RobotImageDataset)` |
| Replay buffer / batching / padding / split | Native DP implementation |
| Normalizer / image transforms | Native DP implementation |
| Training loop | Delegated to `RobotWorkspace.run` |
| Optimizer / scheduling / EMA | Native workspace configuration and code |
| Checkpoints | Native workspace `.ckpt` payload, numbered by epoch |
| Evaluation checkpoint choice | DP's highest-numbered checkpoint resolver |

There is no separately implemented U-Net, diffusion objective, EMA algorithm,
optimizer-update training loop, or best-validation checkpoint policy anymore.
The unused copied `vendor` U-Net source is retained only under ignored legacy
smoke artifacts. `config.yaml` is a small overlay on native `robot_dp.yaml`.

`train.py` now only composes configs, fills dataset metadata/shapes, protects
existing outputs, and launches the workspace. `workspace.py` supplies the LTE
dataset binding, actual optimizer-update counter, and checkpoint-root redirect;
the training and validation loops remain native, without changes to DP files.

## DP restored; all LTE additions are local

The two previously edited DP files have been restored byte-for-byte to HEAD:
`policy/DP/diffusion_policy/workspace/robotworkspace.py` and
`policy/DP/diffusion_policy/config/task/default_task.yaml`.
`git diff --exit-code -- policy/DP` passes: the entire DP directory is unchanged.

LTE dataset binding now uses the native workspace's existing normalizer handoff:
`LTEImageDataset.get_normalizer` carries a Python-only dataset reference, and
`LTEDiffusionPolicy.set_normalizer` copies the native numerical normalizer
state and binds that source. Neither normalization statistics nor its state_dict
are changed. Both online and EMA policies receive the same dataset this way.
There is no modification or monkeypatch of native DP.

Three-camera and robot-state shape metadata is explicit in LTE_DP's own overlay.
The original DP observation configuration is preserved. LTE diagnostics are
written separately to `lte_metrics.jsonl` by its optimizer post-hook; the
original DP JSON logger and training loop remain intact.

## Defaults now match native DP

| Setting | Inherited value |
|---|---|
| Horizon / observation window / executed actions | 8 / 3 / 6 |
| ResNet pooling | Spatial mean (`AdaptiveAvgPool2d(1)`), not SpatialSoftmax |
| Backbone | Independent ResNet18 per camera, GroupNorm, no pretrained weights |
| U-Net widths | 256 / 512 / 1024 |
| DDPM training / inference steps | 100 / 100 |
| Epochs / batch size | 600 / 128 |
| Optimizer | AdamW, LR 1e-4, betas 0.95/0.999, weight decay 1e-6 |
| Scheduler | Native cosine with 500 warmup steps |
| EMA | Native power 0.75, max decay 0.9999 |
| Validation | Native `val_ratio=0.02` and native splitter |
| Saving | Every 600 epochs; normally `600.ckpt` |
| Selection | Highest epoch number, using EMA weights |

The previous independent defaults (200K optimizer updates, batch 64, 16/2/8
horizons, 10% holdout, separate normalizer/EMA and best-validation `.pt` files)
are no longer the current recipe.

Native validation currently reserves the last episode for any positive
val_ratio. With 50 demos and default 0.02 this means 49 train / one validation,
matching native DP. Set `task.dataset.val_ratio=0` for an all-50-demo paper-budget
experiment; final-epoch checkpointing still works without validation loss.

## What LTE adds

The encoder retains the original image-only residual recurrence:

```text
z_t = tanh(z_(t-1) + MLP([detached_image_embedding_t, z_(t-1)]))
```

Latent width 32, hidden widths 128, SiLU; no timestep, idle, velocity, state,
task-ID or language input to LTE. Normalized robot state remains native DP
conditioning. The latent is appended to each original DP observation feature.

`temporal_num_images=1` selects one leading configured RGB feature; null selects
all three. Multiple selected embeddings concatenate. Physical source names are
discovered from demonstrations and persisted in conversion/checkpoint metadata;
LTE selection is by count rather than camera-name assumptions.

The complete causal prefix is reconstructed from native sampler frame IDs and
the existing replay buffer. Sixteen stratified historical embedding queries are
reconstructed for each conditioned observation state, with linear interpolation
and fixed dataset lag scale `2*lag/max(max_episode_length-1,1)`. No elapsed/raw
normalization option exists. Total loss is native diffusion MSE plus history MSE.
History inputs/targets are detached; only the native DP visual branch supplies
visual-encoder gradients.

The full detached embedding table is rebuilt at optimizer update 0, then every
5,000 optimizer updates. There is no per-step or middle phase. An optimizer
post-hook counts real updates independently of native global_step logging.
Validation and native EMA train-batch sampling use independent evaluation tables;
they do not refresh training's table. The clock is persisted; caches are excluded.

At runtime LTE advances on each observation update, including observations
between actions in a chunk. Separate states/windows are kept for arbitrary
environment IDs. Action calls do not advance recurrence. Single/batch calls,
full reset and partial reset are implemented; action generation itself calls
the original DP `predict_action` and sampler.

## Data and checkpoint compatibility

Conversion writes native DP zarr arrays: `data/head_camera`, `left_camera`,
`right_camera`, `state`, `action`, and `meta/episode_ends`. RGB is NCHW uint8.
An additional `lte_metadata` root attribute records source camera names/order,
robot/action type and image size. Native DP can consume these zarrs unchanged.
LTE requires the metadata so it can serve the correct physical views without
guessing names. The converter sorts discovered RGB names and takes the first
three, mapping them to the native schema slots. Extra original views are retained
in the downloaded data. All decoding uses the shared `decode_image_bit` helper.

Saved `.ckpt` files are native workspace payloads (`cfg`, `state_dicts`, `pickles`),
including online/EMA weights and optimizer state. The LTE clock is an extra
workspace pickle. Evaluation calls the original DP numeric checkpoint resolver.
There is no validation-based `best.pt`. Native payload load/round-trip restores
the counter; missing caches rebuild. Native automatic resume still looks for
`output_dir/checkpoints/latest.ckpt`, which its numbered-only save path does not
normally create. This limitation was not replaced with custom resume machinery.

Old standalone `.hdf5` caches and `.pt` smoke checkpoints remain on disk, but
are incompatible with this native implementation. The earlier report/README
were retained under `policy/LTE_DP/smoke_artifacts/legacy_standalone/`.

## Installation and tests

Tested environment: `policy/LTE_DP/.venv-cuda`, Python 3.10,
Torch 2.9.1+cu128 / TorchVision 0.24.1+cu128, Diffusers 0.35.2,
Hydra 1.3.2 / OmegaConf 2.3.0, NumPy 1.26.4, native DP installed editable.
It reuses the existing temporaldiffpolicy environment's installed Torch through
system-site-packages; extra dependencies are local to the virtualenv. The source
Conda environment and original LTE reference repositories are unchanged.
Hydra 1.3.2 is a tested runtime override of native DP package metadata's 1.2 pin.

**16 focused tests passed after restoring DP.** They cover actual native class reuse, identical
native sampling/normalization, exact parent diffusion objective with fixed
conditioning/noise, mean pooling, detached visual history, full-prefix causality
and padding, stratified interpolation and normalization, camera counts,
5K cache timing, optimizer clock, native checkpoints/EMA, streaming state and reset.

**GPU training smoke test:** two real cover_blocks demonstrations, 32 frames
each at 64×64, through the actual native sequence sampler and RobotWorkspace.
Reduced U-Net widths `[32,64,128]`; native 8/3/6 horizons. Two epochs with two
optimizer updates each completed on physical GPU 1 (RTX 5090).

| Result | Value |
|---|---|
| First batch total / diffusion / history loss | 2.22448 / 1.15946 / 1.06502 |
| Fourth batch total / diffusion / history loss | 1.99922 / 0.96618 / 1.03304 |
| Validation loss, epochs 1 / 2 | 2.41198 / 2.79654 |
| Training cache builds | One, at update 0 |
| Actual optimizer updates saved | Four |
| Native checkpoints | `1.ckpt`, `2.ckpt` |
| Default selected checkpoint | `2.ckpt`, despite its worse validation loss |

This specifically verifies the requested latest-epoch selection behavior.
Native train-batch action sampling and EMA evaluation also completed.

**Websocket debug evaluation:** both plain-RGB and encoded-RGB runs completed
ten synthetic episodes with `[MAIN] eval finished` and no traceback, loading
`native_dp/run/2.ckpt` through the real policy server. The single/batched state
and partial-reset contract is also covered by the focused tests.

Logs/checkpoints are under `policy/LTE_DP/smoke_artifacts/native_dp/`; all smoke
artifacts and virtualenvs are ignored by Git. The prior standalone smoke
artifacts are retained separately. Full simulator success rates and long training
remain unverified; smoke checkpoints do not establish task learning.

## Commands

```bash
cd /home/sydney1/Repos/RMBench/XPolicyLab/policy/LTE_DP
source .venv-cuda/bin/activate
# Convert the downloaded task without modifying its original files.
bash process_data.sh RMBench cover_blocks aloha_agilex joint \
  --source /home/sydney1/Repos/RMBench/data/demo_clean/cover_blocks/aloha_agilex/data
# Native DP defaults + LTE:
bash train.sh RMBench cover_blocks aloha_agilex joint 0 1
# Or use all 50 demonstrations; choose a distinct output for another run:
bash train.sh RMBench cover_blocks aloha_agilex joint 0 1 \
  --output checkpoints/cover_blocks_all50 task.dataset.val_ratio=0
# Simulator evaluation after training:
bash eval.sh RMBench cover_blocks cover_blocks aloha_agilex joint 0 \
  1 1 "$PWD/.venv-cuda" RMBench
```

The README documents the native smoke commands, config overrides, environment
setup and checkpoint layout. The nine-task paper comparison still needs to
exclude `place_block_mat` from Table I averages; this refactor does not change
RMBench's task lists or downloaded data.

## Battery Try training request

All 50 downloaded Battery Try demonstrations (33,653 frames) were converted into
`policy/LTE_DP/data/RMBench-battery_try-aloha_agilex-joint.zarr`, preserving
all original data. The explicit run overlay is `configs/battery_try_gpu1.yaml`:
full native DP architecture at 240x320, 600 epochs, batch/validation batch 32,
seed 0, one LTE history image, and refresh every 5,000 optimizer updates.
Batch size is reduced from native 128 for the 32 GiB RTX 5090. Native validation
uses 49 training demonstrations and one held-out demonstration out of the 50.
The run is detached on physical GPU 1. Launch metadata is saved in
`policy/LTE_DP/logs/battery_try_latest_launch.json`; startup verification follows below.

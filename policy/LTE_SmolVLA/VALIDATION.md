# Integration validation

Validated against XPolicyLab's SmolVLA source at LeRobot `v0.4.4`, commit
`8fff0fde7c79f23a93d845d1a50e985de01f8b8a`. The temporary environment uses
Transformers 4.57.6, draccus 0.10.0 and CPU PyTorch 2.11.0 inherited from an
existing environment. Thus these checks do not certify a clean install of every
version allowed by the baseline installer's requirements (its torch range is <2.11).
No existing conda environment was modified.

## Adapter audit

- Pass: required adapter files and all standard deploy keys; `policy_name=LTE_SmolVLA`.
- Pass: inherited XPolicyLab model contract, shared robot packing/dimensions and checkpoint resolver.
- Pass: shell syntax, Python compilation, whitespace checks and prohibited decode/channel-swap scans.
- Pass: ten CPU regression tests, including actual SmolVLA flow-matching forward/backward, LTE gradients, frozen VLM gradients, action-padding masking, camera selection/order, causal episode boundaries, per-environment memory, normalization/checkpoint round trip, base-config inheritance, and history length validation.
- Pass: two optimizer updates on a six-frame, two-episode synthetic v3 dataset; saved checkpoint; resumed to optimizer update three.
- Pass: full-size pretrained SmolVLA weights loaded with only LTE tensors missing; one CPU optimizer update with 450,337,280 total parameters, reduced 64×64 image input and three-action chunks. Final loss 2.665. This is a compatibility check, not a training result.
- Pass: plain-image and encoded-image websocket debug loops with `arx_x5`, joint actions, ten environment slots, five episodes each. Both reached `[MAIN] eval finished` without tracebacks.
- Not run: real simulator evaluation, task success measurement, production-scale training, distributed training, GPU inference, and leaderboard checkpoint publication.

Verdict: adapter checks pass; ready for code review. Real-task validation is still needed before performance claims or deployment.

## Commands and local evidence

Regression command:

```bash
PYTHONPATH=/tmp/xpl-lte-smolvla.3lYE8O/debug-workspace \
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
/tmp/xpl-lte-smolvla.3lYE8O/venv/bin/python -m pytest \
  /tmp/xpl-lte-smolvla.3lYE8O/debug-workspace/XPolicyLab/policy/LTE_SmolVLA/tests \
  -q -p no:cacheprovider
```

Training fixture generation and the two-step training command are in
`tests/make_smoke_fixture.py` and `tests/smoke_train.sh`. Resume used:

```bash
PYTHONPATH=/home/sydney1/Repos HF_HUB_OFFLINE=1 \
HF_DATASETS_CACHE=/tmp/xpl-lte-smolvla.3lYE8O/smoke/hf-datasets-cache OMP_NUM_THREADS=2 \
/tmp/xpl-lte-smolvla.3lYE8O/venv/bin/python -m XPolicyLab.policy.LTE_SmolVLA.train \
  --config_path=/tmp/xpl-lte-smolvla.3lYE8O/smoke/run/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=3
```

Exact successful debug commands, run from
`/tmp/xpl-lte-smolvla.3lYE8O/debug-workspace/XPolicyLab/policy/LTE_SmolVLA`:

```bash
PATH=/home/sydney1/miniconda3/condabin:$PATH HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls \
  /tmp/xpl-lte-smolvla.3lYE8O/smoke/run arx_x5 joint 0 0 0 \
  /tmp/xpl-lte-smolvla.3lYE8O/venv RMBench

PATH=/home/sydney1/miniconda3/condabin:$PATH HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug PYTHONUNBUFFERED=1 \
bash eval.sh RoboDojo stack_bowls \
  /tmp/xpl-lte-smolvla.3lYE8O/smoke/run arx_x5 joint 0 0 0 \
  /tmp/xpl-lte-smolvla.3lYE8O/venv RMBench
```

The temporary workspace contains the adapter, baseline dependency and shared
XPolicyLab server/helpers, with `env_cfg` linked to RMBench's existing definitions.
No changes to RMBench's worktree or its robot registrations were required.

Logs retained locally in `/tmp/xpl-lte-smolvla.3lYE8O/`: `train.log`, `resume.log`,
`full-base-train.log`, `debug-plain.log`, `debug-encoded.log`. The temporary
checkpoint and VLM assets are synthetic wiring fixtures and are not committed.

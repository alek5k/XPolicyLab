# LTE_SmolVLA training performance roadmap

A matched RMBench battery run measured approximately 0.167 seconds per baseline SmolVLA optimizer update and 0.457 seconds per LTE_SmolVLA update (about 2.74× slower). LTE component measurements attributed about 152 ms/update to the action forward and 87 ms/update to the temporal encoder forward; history decoding, reconstruction loss, and cached embedding lookup were comparatively small. The detached embedding cache had a 100% hit rate, so the main LTE-specific opportunity is reducing repeated trainable recurrence over the complete causal history without discarding valid history.

The optimization order is:

1. **Trim batch-wide causal-history padding (in progress).** Histories are left-padded to the configured maximum length. Remove only leading positions for which every batch element is invalid before running the temporal recurrence. This is exact: it preserves every valid frame, the final temporal latent, and the reconstruction objective. For the measured dataset and batch size, roughly 21% of recurrent positions are expected to be removable.
2. **Disable component-level step timing (in progress).** Per-component CUDA synchronization serializes the training stream and perturbs the workload being measured. Disable this instrumentation for normal training while retaining the common trainer's aggregate `train/update_s` metric. Startup cache-build timing can remain because it does not add synchronization to every update.
3. **Increase embedding-cache build batch size.** Benchmark larger `temporal_embedding_cache_build_batch_size` values within GPU memory limits to reduce cache-construction overhead. This affects startup or refresh cost, not steady-state recurrence.
4. **Release the raw image cache after embedding-cache construction.** When the detached embedding table is fixed, retaining both representations wastes memory. Reclaiming the raw image cache could make room for larger batches or other training state. Preserve it when periodic embedding refreshes still require the source frames.
5. **Investigate BF16 for the temporal path.** Measure numerical behavior and throughput before changing precision. Any change must preserve stable recurrence and reconstruction loss, and should be compared with the matched FP32 behavior.
6. **Encode sampled history rather than every frame (deferred).** Sparse or learned temporal sampling could reduce recurrence substantially, but changes the information available to LTE and therefore requires a separate modeling investigation rather than being treated as an exact optimization.
7. **Truncate to a recent causal window (rejected).** A fixed recent-history window would discard valid long-range history and is not an acceptable optimization for this policy.
8. **Compare action chunk lengths.** Longer action chunks may amortize policy computation during deployment, but alter action replanning frequency and do not directly remove training-time temporal recurrence. Treat this as a separate quality/latency experiment.
9. **Cache temporal latents (deferred).** This is distinct from caching image embeddings: cache the causal temporal encoder state itself so repeated prefixes can be looked up or incrementally continued. Exact training-time reuse is difficult because temporal encoder weights change after every optimizer update. A correct cache would need parameter-version invalidation or rebuilding; a stale detached cache would change gradients and training semantics. Reconsider this only after the exact padding trim and timing cleanup have been measured.




## Recommended, ranked

### 1. Trim universally padded history steps per batch

**Best semantics-preserving steady-state optimization.**

The LTE encoder currently always loops over the dataset-wide maximum episode length: **889 recurrent iterations per batch**. Shorter histories are left-padded, but the MLP is still executed for every padded position before `torch.where` discards its result in [core.py:21-32](../../../XPolicyLab/policy/LTE_SmolVLA/temporal/core.py#L21-L32).

For this dataset:

- Episode range: 456–889 frames
- Mean episode length: 673
- Estimated mean maximum active history in a batch of eight: **699**
- Therefore approximately **21% of recurrent iterations are unnecessary**

We can remove columns that are invalid for *every* sample in the batch before calling the temporal encoder. This preserves:

- every valid frame;
- every sample’s full causal history;
- final temporal latents;
- reconstruction target positions and lags.

Estimated saving:

- temporal forward: about **15–20 ms/step**
- temporal backward: likely another **20–40 ms/step**
- total: approximately **35–60 ms/step**
- roughly **3–5 minutes over 5,000 steps**
- around **8–13% faster overall**

This should be implemented first.

---

### 2. Remove production CUDA synchronization instrumentation

The diagnostic instrumentation currently performs around a dozen `torch.cuda.synchronize()` calls per step around:

- cache lookup;
- temporal encoder;
- action forward;
- action loss;
- history decoder;
- entire update.

Examples are in [modeling_lte_smolvla.py:377-422](../../../XPolicyLab/policy/LTE_SmolVLA/modeling_lte_smolvla.py#L377-L422) and [train.py:163-188](../../../XPolicyLab/policy/LTE_SmolVLA/train.py#L163-L188).

These synchronizations were useful to establish the timing breakdown, but they:

- repeatedly block the CPU on the GPU;
- prevent normal asynchronous overlap;
- amplify the cost of LTE’s many small recurrent kernels.

We can replace them with CUDA events and only synchronize once at the end of a logged step—or disable component timing outside profiling runs.

Likely saving: **small but essentially free**, perhaps 1–5%. It needs an A/B measurement because the amount of available overlap is workload-dependent.

---

### 3. Increase embedding-cache construction batch size

The current 5k LTE run uses:

```yaml
policy.temporal_embedding_cache_build_batch_size: 32
```

The full RMBench recipe already uses 256. The L40S has considerable headroom:

- 46 GB total VRAM;
- approximately 12 GB during LTE training;
- 7.22 GB of that is the raw-image cache.

The embedding-table build took **179 seconds**. Raising the build batch to 128 or 256 should improve vision-encoder utilization and could remove roughly **one or two minutes of startup**, subject to a quick memory benchmark.

This changes only cache construction throughput, not training semantics.

---

### 4. Release the raw-image cache after embedding construction

With the current configuration:

- raw-image GPU cache: 7.22 GiB;
- persistent embedding table: subsequently used on every step;
- cache hit rate: 100%;
- temporal vision encoding after startup: zero.

Because `train_expert_only: true` prevents embedding-cache refresh, the 7.22 GiB raw-image cache is no longer needed after the initial embedding table has been created.

Releasing it would not directly make this run much faster, but it would:

- reclaim 7.22 GiB;
- permit larger cache-build batches;
- permit larger training batches;
- provide headroom for BF16 experiments or compilation;
- reduce the chance of memory pressure on larger datasets.

This is safe specifically when the embedding table cannot refresh.

---

### 5. Add genuine BF16 mixed precision

Both runs currently log:

```text
policy.use_amp: false
```

The common trainer enters `accelerator.autocast()`, but its `Accelerator` is constructed without selecting mixed precision in [lerobot_train.py:175-187](../../../XPolicyLab/policy/SmolVLA/smovla/src/lerobot/scripts/lerobot_train.py#L175-L187). Therefore simply setting `policy.use_amp: true` may not be sufficient; we should wire it explicitly to BF16.

An L40S handles BF16 well. This could accelerate:

- action forward;
- action backward;
- temporal MLP recurrence;
- temporal backward.

Potential saving could be substantial—perhaps **15–30%**—but it is not numerically identical and needs a stability/loss comparison. It should be tested on both baseline and LTE for fairness.

---

## Larger savings that change the LTE algorithm

### 6. Encode 16 sampled history positions rather than every frame

Despite:

```yaml
policy.num_history_queries: 16
```

the current cached path still sends **all 889 positions** through the temporal recurrence. `num_history_queries` only controls reconstruction targets. The 16-position encoder branch is used only when there is no embedding cache, as seen in [modeling_lte_smolvla.py:355-376](../../../XPolicyLab/policy/LTE_SmolVLA/modeling_lte_smolvla.py#L355-L376).

A cache-backed query-only path could:

1. select 16 stratified causal frames;
2. gather their cached embeddings;
3. run only 16 recurrence steps instead of up to 889.

That would remove about **98% of temporal recurrence iterations**. Since nearly the entire 0.29-second baseline-to-LTE gap appears connected to temporal forward/backward, this could potentially bring LTE much closer to baseline speed.

However, it changes the temporal representation from full-resolution history to sparse history. It should be treated as an ablation, not a transparent optimization.

---

### 7. Limit causal history to a recent window — rejected

A fixed recent-history window would discard valid long-range history and is not an acceptable optimization for this policy. The startup validation must continue to reject truncated causal history.

---

### 8. Reduce action chunk length from 50 to 30

The 5k recipes inherit:

```yaml
policy.chunk_size: 50
policy.n_action_steps: 50
```

The full RMBench recipes use 30. The normal action-model path is **152 ms**, or about one third of the LTE step. Reducing the action sequence length could save meaningful time in both policies, but:

- it is not LTE-specific;
- both baseline and LTE must use the same value;
- it changes the action-prediction horizon.

---

### 9. Cache temporal latents — deferred

This is distinct from the existing image-embedding cache: cache the causal temporal encoder state itself so repeated prefixes can be looked up or incrementally continued. Exact training-time reuse is difficult because temporal encoder weights change after every optimizer update. A correct cache would require parameter-version invalidation or rebuilding; a stale detached cache would change gradients and training semantics. Reconsider this only after measuring the exact padding trim and timing cleanup.

---

## Things not worth optimizing first

- `num_history_queries`: history decoder forward is only **0.64 ms**.
- History loss calculation: less than **0.2%** of the step.
- Dataloader: only **2.9 ms**.
- Temporal vision work after startup: **zero**.
- Cache lookup itself: not a significant measured bottleneck.
- Smaller temporal latent/hidden dimensions: likely limited benefit because the recurrence is dominated by hundreds of sequential GPU launches.

## Suggested order

1. Implement exact batch-level padding trim.
2. Disable component timing and its repeated synchronizations; retain aggregate `train/update_s`.
3. Raise embedding build batch size to 256.
4. Free the raw-image GPU cache after the fixed embedding cache is built.
5. Benchmark BF16.
6. Separately test cache-backed 16-query history as a speed/quality ablation.
7. Reconsider temporal-latent caching after Items 1 and 2 have been measured.

The first four preserve model behavior and should be safe to implement without changing the currently running experiment. Recent-window truncation is not an option.
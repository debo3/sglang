# Fixing the rl9b300 "detokenizer health-check" pod deaths

**Branch:** `fix/rl9-gc-stall-and-evict-sync` (3 commits, individually cherry-pickable / upstreamable)

This branch fixes the production incident in which `staging-rl9b300` pods (sglang
`v0.5.12-cu130`, 8×B300, TP=8, DeepSeek-V3-family bf16 model, `attention_backend=trtllm_mla`,
`moe_runner_backend=flashinfer_trtllm`) were recycled by Kubernetes ~8 times per week with the
signature:

```
Health check failed. Server couldn't get a response from detokenizer for last 20 seconds.
tic start time: 19:12:20. last_heartbeat time: 19:12:17        (then SIGTERM ~88s later)
```

No traceback, no OOM, no NCCL error. The full investigation (reproduction harness, py-spy/gdb
forensics, live A/B verification on a warmed 8×B300 server) lives in
`/scratch/hai/debo/debug-rl9/README-rl9-stall.md`; this document covers the code changes and
their measured impact.

> A second, much rarer failure mode (one `torch.AcceleratorError: CUDA error: unknown error`
> surfacing in `GenerationBatchResult.copy_to_cpu` in 7 days) is a deferred async-CUDA error
> with a different cause and is **not** addressed by this branch; see §6 of the investigation
> README for its triage plan.

---

## 1. Root cause (Mode A, the dominant failure)

A chain of three mechanisms, each verified independently on a live reproduction:

1. **A memory leak in flashinfer's fused-MoE autotuner path.**
   `AutoTuner._find_nearest_profile` is `@lru_cache(maxsize=None)` keyed by
   `(shapes, tuning_config)`, and the fused-MoE entry points
   (`flashinfer/fused_moe/core.py`, e.g. `trtllm_bf16_moe_op`) construct a **new**
   `TuningConfig` — containing freshly created lambdas, hence identity-based hashes — on
   **every MoE call**. Every invocation therefore permanently inserts a never-hit cache entry
   retaining the config, its closures, and the `torch.Size` key tuple.
   This is the [flashinfer#2139](https://github.com/flashinfer-ai/flashinfer/issues/2139)
   mechanism; the upstream fix covered the `mm_fp4`/`fp8_gemm` GEMM entry points, but the
   fused-MoE entry points still leak as of flashinfer **0.6.12** (the version this repo pins).
   At 61 MoE layers/forward this leaked ~46 object-sets/s under load.

2. **Growing stop-the-world GC pauses.** CPython gen-2 collection traverses the whole live
   heap. As the leak accumulated (74.8M tracked objects after 6.5 h: 22.4M `torch.Size`,
   19.4M `function`, 16.2M `tuple`, 6.5M `cell`, 3.2M each `TuningConfig`/`DynamicTensorSpec`),
   pause duration grew linearly with uptime: **7 s @1h → 23 s @3.5h → 46 s @6.5h**. A manual
   `gc.collect()` on the live rank measured **41.17 s and freed 0 objects** — pure traversal.
   Pauses land at arbitrary allocation sites (we caught them inside `RadixCache.evict`'s heap
   comprehension, flashinfer's `_get_input_sizes`, and triton launch wrappers).

3. **TP convoy amplification.** The paused rank stops launching kernels; its GPU drains to 0 %
   while the other seven spin at 100 % inside a collective that needs its next launch. The
   victim rotates (it's whichever rank's gen-2 threshold fires). Once a pause exceeds the 20 s
   `/health_generate` window, probes fail; k8s SIGTERMs the pod ~88 s later — *before* the
   pause ends. Every stall we observed self-recovered; production pods just got killed first.

A secondary defect compounds tail latency (and was where GC pauses most often landed): the
paged allocator's `free()` does `torch.unique` on a CUDA tensor — an implicit device
synchronization — once per node evicted by `RadixCache.evict()`, which runs on the scheduler
critical path before every allocation once the KV pool is full (steady state after ~30–60 min).
Measured: ~94 % of eviction wall time inside `free()`.

## 2. The fixes in this branch

### Commit 1 — `[scheduler] Keep stop-the-world GC pauses off the batch critical path`
*Files:* `srt/managers/scheduler_gc_manager.py` (new), `srt/managers/scheduler.py`,
`srt/environ.py`

The **defense-in-depth** fix: makes the scheduler robust to *any* slow heap growth in *any*
dependency, current or future.
- Raises the gen-2 GC threshold (automatic full collections become rare; kept as a fallback
  under never-idle load).
- First fully-idle tick after warmup: one full collect, then `gc.freeze()` of the startup heap
  (millions of module/weight-metadata objects leave the traversal set permanently).
- Thereafter, periodic `gc.collect()` **only while idle**, when a pause cannot stall in-flight
  batches. A growing idle-GC duration in the logs is itself a leak early-warning.
- Deliberately does **not** re-freeze periodically: objects frozen alive are never collected if
  they later become cyclic garbage (e.g. evicted radix-tree nodes hold parent↔child cycles),
  so repeated freezing would convert churn into a true leak.

Knobs: `SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT` (default on),
`SGLANG_SCHEDULER_IDLE_GC_INTERVAL` (default 300 s),
`SGLANG_SCHEDULER_GC_GEN2_THRESHOLD` (default 10000).

### Commit 2 — `[moe] Work around flashinfer fused-MoE TuningConfig leak (flashinfer#2139)`
*Files:* `srt/utils/patch_flashinfer.py` (new),
`srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `srt/environ.py`

The **root-cause** fix for this incident: memoizes `MoERunner._make_tuning_config` per
configuration (runner attributes + which optional inputs are present + kwargs — the config
does not depend on tensor shapes or values, which are a separate part of the autotuner key).
A deployment ends up with a handful of memo entries (5 in our repro), and the stable
`TuningConfig` object makes flashinfer's `lru_cache` *hit*, eliminating both the leak and the
per-call config construction. Applied lazily by wrapping flashinfer's cached module factories
(first MoE use, after CUDA init); no-ops gracefully if flashinfer internals change, and should
be dropped once flashinfer fixes the MoE entry points upstream.
Opt-out: `SGLANG_DISABLE_FLASHINFER_TUNING_CONFIG_MEMO=1`.

### Commit 3 — `[mem_cache] Make RadixCache.evict free pages without device syncs`
*File:* `srt/mem_cache/radix_cache.py`

Collects evicted node values during the pop loop and frees them with a **fixed-shape strided
slice** (`cat[::page_size] // page_size`, equivalent to `unique(cat // page_size)` for
page-aligned page runs, which radix-tree node values always are) — pure async kernels, zero
synchronization. Non-aligned values and `page_size==1` allocators keep the original path.

## 3. Measured impact (live A/B on a warmed 8×B300 server, prod image + args + weights)

Methodology: the exact prod container served waves of 6 concurrent multi-turn conversations
(~17 k-token contexts) + sparse singles + mid-stream client aborts + a 4 s `/health_generate`
probe; fixes were hot-injected via gdb into the running, fully-warmed ranks so before/after
share identical state. Forensics: continuous 8-rank py-spy sampling, GPU-utilization capture at
every failed probe, gdb-injected heap censuses, server-log silence-gap analysis.

| Metric | Without fixes | With fixes |
|---|---|---|
| `Health check failed` (Mode A) | 47 events in ~9 h soak; in prod ~366 lines, ~8 pod recreations/week | **0** in verification soak |
| All-rank stall duration | 7 s @1h → 23 s @3.5h → **46 s @6.5h** (grows with uptime; >20 s ⇒ pod killed) | **none ≥ 20 s**; worst single gap 10 s (benign convoy, self-recovered) |
| Stall clusters | every ~25–35 min after ~2 h uptime | none observed |
| Scheduler heap | 74.8 M objects @6.5 h, +~46 object-sets/s, unbounded | leak rate **0** (23,420 → 23,420 configs over 4 min under load); memo holds 5 entries |
| Gen-2 GC cost | 41.17 s (freed 0) | startup heap frozen once; idle collects, ms-scale |
| `RadixCache.evict` | p50 ~6 ms, ~94 % in `free()` syncs; 50–80 ms spikes; GC-contaminated calls 3–4.5 s | ≤ 0.18 s worst observed post-fix (no sync; remaining cost is heap build) |
| Pod lifecycle | killed mid-pause by liveness probe; full 20-min weight reload each time | stalls absent; any future transient self-recovers |

Causality cross-checks worth noting:
- A natural control: one rank was accidentally missed by the first `gc.freeze()` pass — the 7
  frozen ranks ran 63 min with zero stalls while the lone unfrozen rank produced exactly one
  56 s stall (caught GIL-held mid-GC by the sampler).
- Negative result that sharpened the fix: making eviction's syncs fewer (batched frees) and
  then zero did **not** stop the stalls — only removing the GC pauses did. Hence commit 1+2
  are the fix and commit 3 is a (worthwhile) latency improvement.

## 4. Deployment notes

- These fixes remove the need for the interim mitigations (periodic `/freeze_gc` CronJob,
  `sitecustomize` patch). Still recommended operationally:
  - liveness probe: keep readiness tight but give liveness a few minutes of failures before
    killing — transient stalls self-recover, and a kill costs a full weight reload;
  - alert on growing idle-GC durations (logged by commit 1) and on
    `sglang:eviction_duration_seconds` p99 — both are hours-early leak indicators.
- Upstream plan: commit 1 and 3 are proposed as sglang PRs as-is; commit 2 is a workaround to
  carry until flashinfer applies the #2139 fix to the fused-MoE entry points (issue to file,
  referencing the measurements above), after which it can be dropped (it no-ops safely in the
  meantime).

# Fixing TP-wide scheduler stalls ("couldn't get a response from detokenizer") for DeepSeek-V3-family MoE serving

**Branch:** `fix/scheduler-gc-stall-and-evict-sync` (3 code commits, individually cherry-pickable /
upstreamable; backport for the v0.5.12 release line in
`fix/scheduler-gc-stall-and-evict-sync-v0512`)

**Affected setup:** any DeepSeek-V3-family MoE checkpoint — e.g. the public
[`deepseek-ai/DeepSeek-V3.1-Terminus`](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus)
— served with `moe_runner_backend=flashinfer_trtllm` (the default on Blackwell), tensor
parallelism, and the radix cache enabled. Observed in production as Kubernetes recycling pods
several times per week with this signature and **no traceback, no OOM, no NCCL error**:

```
Health check failed. Server couldn't get a response from detokenizer for last 20 seconds.
tic start time: 19:12:20. last_heartbeat time: 19:12:17        (SIGTERM ~88 s later)
```

---

## 1. Root cause

A chain of three mechanisms, each verified independently on a live reproduction (8×B300, TP=8):

1. **A memory leak in flashinfer's fused-MoE autotuner path.**
   `AutoTuner._find_nearest_profile` is `@lru_cache(maxsize=None)` keyed by
   `(shapes, tuning_config)`, and the fused-MoE entry points in
   `flashinfer/fused_moe/core.py` (`trtllm_bf16_moe_op`, the fp8 block-scale variants, …)
   construct a **new** `TuningConfig` — containing freshly created lambdas, hence
   identity-based hashes — on **every MoE call**. Every invocation therefore permanently
   inserts a never-hit cache entry retaining the config, its closures, and the `torch.Size`
   key tuple. This is the
   [flashinfer#2139](https://github.com/flashinfer-ai/flashinfer/issues/2139) mechanism; the
   upstream fix covered the `mm_fp4`/`fp8_gemm` GEMM entry points, but the fused-MoE entry
   points still leak as of flashinfer **0.6.12**. With 61 MoE layers per forward this leaks
   ~46 object-sets/s under moderate load.

2. **Growing stop-the-world GC pauses.** CPython gen-2 collection traverses the whole live
   heap. As the leak accumulates (74.8 M tracked objects after 6.5 h: 22.4 M `torch.Size`,
   19.4 M `function`, 16.2 M `tuple`, 6.5 M `cell`, 3.2 M each
   `TuningConfig`/`DynamicTensorSpec`), pause duration grows linearly with uptime:
   **7 s @1 h → 23 s @3.5 h → 46 s @6.5 h**. A manual `gc.collect()` on a live rank measured
   **41.17 s and freed 0 objects** — pure traversal. Pauses land at arbitrary allocation
   sites (caught inside `RadixCache.evict`'s heap comprehension, flashinfer's
   `_get_input_sizes`, and triton launch wrappers).

3. **TP convoy amplification.** The paused rank stops launching kernels; its GPU drains to
   0 % while the other ranks spin at 100 % inside a collective that needs its next launch.
   The victim rotates (whichever rank's gen-2 threshold fires). Once a pause exceeds the 20 s
   `/health_generate` window, probes fail and the liveness probe kills the pod *before* the
   pause ends — every stall observed in reproduction self-recovered; production pods just get
   killed first, which makes a transient GC pause look like a permanent silent hang.

A secondary defect compounds tail latency (and is where GC pauses most often landed): the
paged allocator's `free()` does `torch.unique` on a CUDA tensor — an implicit device
synchronization — once per node evicted by `RadixCache.evict()`, which runs on the scheduler
critical path before every allocation once the KV pool is full (steady state after ~30–60 min
of traffic). Measured: ~94 % of eviction wall time inside `free()`.

## 2. The fixes in this branch

### Commit 1 — `[scheduler] Keep stop-the-world GC pauses off the batch critical path`
*Files:* `srt/managers/scheduler_gc_manager.py` (new), `srt/managers/scheduler.py`,
`srt/environ.py`

The **defense-in-depth** fix: makes the scheduler robust to *any* slow heap growth in *any*
dependency, current or future.
- Raises the gen-2 GC threshold (automatic full collections become rare; kept as a fallback
  under never-idle load).
- First fully-idle tick after warmup: one full collect, then `gc.freeze()` of the startup
  heap (~1 M module/weight-metadata objects leave the traversal set permanently).
- Thereafter, periodic `gc.collect()` **only while idle**, when a pause cannot stall
  in-flight batches. A growing idle-GC duration in the logs is itself a leak early-warning.
- Deliberately does **not** re-freeze periodically: objects frozen alive are never collected
  if they later become cyclic garbage (e.g. evicted radix-tree nodes hold parent↔child
  cycles), so repeated freezing would convert churn into a true leak.

Knobs: `SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT` (default on),
`SGLANG_SCHEDULER_IDLE_GC_INTERVAL` (default 300 s),
`SGLANG_SCHEDULER_GC_GEN2_THRESHOLD` (default 10000).

### Commit 2 — `[moe] Work around flashinfer fused-MoE TuningConfig leak (flashinfer#2139)`
*Files:* `srt/utils/patch_flashinfer.py` (new),
`srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `srt/environ.py`

The **root-cause** fix for this incident: memoizes `MoERunner._make_tuning_config` per
configuration (runner attributes + which optional inputs are present + kwargs — the config
does not depend on tensor shapes or values, which are a separate part of the autotuner key).
A deployment ends up with a handful of memo entries, and the stable `TuningConfig` object
makes flashinfer's `lru_cache` *hit*, eliminating both the leak and the per-call config
construction. Applied lazily by wrapping flashinfer's cached module factories (first MoE use,
after CUDA init); covers every `MoERunner` class the factories define (bf16 and fp8 paths);
no-ops gracefully if flashinfer internals change, and should be dropped once flashinfer fixes
the MoE entry points upstream. Opt-out: `SGLANG_DISABLE_FLASHINFER_TUNING_CONFIG_MEMO=1`.

### Commit 3 — `[mem_cache] Make RadixCache.evict free pages without device syncs`
*File:* `srt/mem_cache/radix_cache.py`

Collects evicted node values during the pop loop and frees them with a **fixed-shape strided
slice** (`cat[::page_size] // page_size`, equivalent to `unique(cat // page_size)` for
page-aligned page runs, which radix-tree node values always are) — pure async kernels, zero
synchronization. Non-aligned values and `page_size==1` allocators keep the original path.

## 3. Reproducing on the public checkpoint

```bash
# 8x Blackwell (B200/B300), TP=8. flashinfer <= 0.6.12 exhibits the leak.
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3.1-Terminus \
  --tp 8 --trust-remote-code --context-length 32768 \
  --mem-fraction-static 0.85 --enable-metrics

# Drive sustained multi-turn load (concurrent growing conversations) and watch:
#  * the leak: count live TuningConfig objects in a scheduler rank (grows ~#MoE-layers per forward)
#  * gen-2 GC pause growth: gc.get_stats() / py-spy a rank during a stall
#  * the stall: /health_generate flips to 503 once a pause exceeds 20 s (after hours of uptime)
```

The stall reproduces with the *exact* production signature after a few hours (pause duration
must outgrow the 20 s health window, and the KV pool must be full for the eviction path to be
hot). On fixed builds the same harness runs stall-free.

## 4. Measured impact

### 4.1 Incident reproduction (production-identical deployment, DeepSeek-V3.1-Terminus-based bf16 checkpoint, sglang v0.5.12-cu130, 8×B300, TP=8)

Methodology: the production container served waves of 6 concurrent multi-turn conversations
(~17 k-token contexts) + sparse singles + mid-stream client aborts + a 4 s `/health_generate`
probe; fixes were hot-injected via gdb into the running, fully-warmed ranks so before/after
share identical state. Forensics: continuous 8-rank py-spy sampling, GPU-utilization capture
at every failed probe, gdb-injected heap censuses, server-log silence-gap analysis.

| Metric | Without fixes | With fixes |
|---|---|---|
| `Health check failed` events | 47 in ~9 h soak (production: ~8 pod recreations/week) | **0** in verification soak |
| All-rank stall duration | 7 s @1 h → 23 s @3.5 h → **46 s @6.5 h** (>20 s ⇒ pod killed) | **none ≥ 20 s**; worst single gap 10 s (benign convoy, self-recovered) |
| Stall clusters | every ~25–35 min after ~2 h uptime | none observed |
| Scheduler heap | 74.8 M objects @6.5 h, +~46 object-sets/s, unbounded | leak rate **0**; memo holds a handful of entries |
| Gen-2 GC cost | 41.17 s (freed 0) | startup heap frozen once; idle collects, ms-scale |
| `RadixCache.evict` | p50 ~6 ms, ~94 % in `free()` syncs; 50–80 ms spikes; GC-contaminated calls 3–4.5 s | ≤ 0.18 s worst observed |

Causality cross-checks:
- A natural control: one rank was accidentally missed by the first `gc.freeze()` pass — the 7
  frozen ranks ran 63 min with zero stalls while the lone unfrozen rank produced exactly one
  56 s stall (caught GIL-held mid-GC by the stack sampler).
- Negative result that sharpened the fix: making eviction's syncs fewer (batched frees) and
  then zero did **not** stop the stalls — only removing the GC pauses did. Hence commits 1+2
  are the fix and commit 3 is a (worthwhile) latency improvement.

### 4.2 Branch verification on hardware (this branch's code, 2-hour soak)

The branch code ran end-to-end on 8×B300, TP=8, production CLI args, mounted over the
published image via `PYTHONPATH` (the v0.5.12 backport branch, since `main` requires
`sgl-kernel >= 0.4.3` with no published image yet). Same load harness as §4.1:

| Check | Evidence |
|---|---|
| Fix 1 armed | `Scheduler GC management enabled: gen-2 threshold 10 -> 10000` on all 8 ranks; in-process `gc.get_threshold() == (700, 10, 10000)` |
| Fix 1 initial freeze | `Initial post-warmup GC freeze: collected 0, froze 954,058 objects (1.069s)` on first idle tick |
| Fix 1 idle GC | gen-2 collection counter advanced only at idle; unfrozen heap steady at ~58 k objects |
| Fix 2 armed | `Applied flashinfer TuningConfig memoization to 1 MoERunner class(es)` on all 8 ranks (lazy factory wrap) |
| Fix 2 leak dead | five censuses over 2 h under load: live `TuningConfig` count = **1** throughout; memo bounded at 12 entries |
| Fix 3 at full pool | 25,280 evictions: 25,273 ≤ 10 ms, **all** ≤ 50 ms, mean 2.7 ms |
| End to end | **0** log-silence gaps ≥ 6 s, **1,458/1,458** health probes OK, zero `Health check failed` |

### 4.3 Public checkpoint verification (`deepseek-ai/DeepSeek-V3.1-Terminus`, fp8)

[PENDING — stock-image leak reproduction + fixed-branch verification on the public weights]

## 5. Deployment notes

- Liveness probes: keep readiness tight but give liveness several minutes of failures before
  killing — these stalls self-recover, and a kill costs a full weight reload.
- Alert on growing idle-GC durations (logged by commit 1) and on
  `sglang:eviction_duration_seconds` p99 — both are hours-early leak indicators.
- Upstream plan: commits 1 and 3 are proposed as sglang PRs as-is; commit 2 is a workaround
  to carry until flashinfer applies the #2139 fix to the fused-MoE entry points, after which
  it can be dropped (it no-ops safely in the meantime).

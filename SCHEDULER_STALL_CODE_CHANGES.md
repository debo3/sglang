# Exact code changes: TP scheduler GC-stall fixes — annotated diffs

Companion to `SCHEDULER_STALL_FIXES.md` (problem statement, root-cause evidence, and measured
results). This document contains the **complete diffs** of the three code commits on
`fix/scheduler-gc-stall-and-evict-sync` with line-by-line rationale, the alternatives that were
tried and rejected (with the measurements that rejected them), and the safety analysis of each
change.

| Commit | Files | Role |
|---|---|---|
| `[scheduler] Keep stop-the-world GC pauses off the batch critical path` | `scheduler_gc_manager.py` (new), `scheduler.py`, `environ.py` | Defense in depth: no heap growth in any dependency can stall TP serving again |
| `[moe] Work around flashinfer fused-MoE TuningConfig leak (flashinfer#2139)` | `patch_flashinfer.py` (new), `moe_runner/flashinfer_trtllm.py`, `environ.py` | Root cause: stops the leak that grew GC pauses to 40+ s |
| `[mem_cache] Make RadixCache.evict free pages without device syncs` | `radix_cache.py` | Secondary: removes a per-evicted-node device sync from the scheduler hot path |

Backport note: the `fix/scheduler-gc-stall-and-evict-sync-v0512` branch carries the same three
commits onto the `v0.5.12.post1` release line. The only adaptation is in commit 1: that line's
`on_idle()` lives in `srt/managers/scheduler_runtime_checker_mixin.py` rather than
`scheduler.py`, so the `self.gc_manager.on_idle()` hook is placed there.

---

## Commit 1 — `[scheduler]` GC management

### The exact problem

CPython's generation-2 garbage collection is a stop-the-world pause that (a) costs O(live heap)
because it traverses every tracked container object, and (b) triggers at **arbitrary allocation
sites** mid-request. In a TP server every rank runs the same batches in lockstep through
collectives, so a pause in one rank's scheduler freezes all ranks: the paused rank stops
launching kernels, the peers' GPUs spin at 100 % inside a collective waiting for its next
launch, the detokenizer heartbeat freezes, and `/health_generate` returns 503 after 20 s.

Measured on the incident reproduction (8×B300, TP=8, DeepSeek-V3.1-Terminus-class model): with a
leaking dependency the heap reached 74.8 M tracked objects in 6.5 h and a single gen-2 collection
took **41.17 s while freeing 0 objects**. Pause duration grew linearly with uptime
(7 s @1 h → 46 s @6.5 h); the pod was killed by its liveness probe mid-pause every time the
pause crossed the health window.

The fix has three coordinated parts: defer automatic gen-2 collections, freeze the startup heap
once, and collect at idle.

### The diff

```diff
diff --git a/python/sglang/srt/environ.py b/python/sglang/srt/environ.py
index b71a9e149..d7873a408 100644
--- a/python/sglang/srt/environ.py
+++ b/python/sglang/srt/environ.py
@@ -208,6 +208,14 @@ class Envs:
     SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)
     SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)
 
+    # Scheduler GC management (see managers/scheduler_gc_manager.py):
+    # defer automatic gen-2 collections and run full GC at idle instead, so a
+    # stop-the-world pause in one TP rank cannot stall the whole TP group
+    # mid-batch.
+    SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT = EnvBool(True)
+    SGLANG_SCHEDULER_IDLE_GC_INTERVAL = EnvFloat(300.0)
+    SGLANG_SCHEDULER_GC_GEN2_THRESHOLD = EnvInt(10000)
+
     # Logging Options
     SGLANG_LOG_GC = EnvBool(False)
     SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
diff --git a/python/sglang/srt/managers/scheduler.py b/python/sglang/srt/managers/scheduler.py
index 31fa59a19..c8492f201 100644
--- a/python/sglang/srt/managers/scheduler.py
+++ b/python/sglang/srt/managers/scheduler.py
@@ -208,6 +208,7 @@ from sglang.srt.managers.scheduler_components.request_receiver import (
 from sglang.srt.managers.scheduler_components.weight_updater import (
     SchedulerWeightUpdaterManager,
 )
+from sglang.srt.managers.scheduler_gc_manager import SchedulerGCManager
 from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
 from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
 from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
@@ -1027,6 +1028,10 @@ class Scheduler(
         if envs.SGLANG_LOG_GC.get():
             configure_gc_logger()
 
+        # Keep stop-the-world GC pauses away from the batch critical path
+        # (a gen-2 pause in one TP rank stalls every rank).
+        self.gc_manager = SchedulerGCManager()
+
     def init_disaggregation(self):
         self.disaggregation_mode = DisaggregationMode(
             self.server_args.disaggregation_mode
@@ -3326,6 +3331,9 @@ class Scheduler(
         # Publish the idle state so /get_loads and DP balancing do not see stale load.
         self.publish_load_snapshot(force=True)
 
+        # reclaim cyclic garbage while a pause cannot stall in-flight batches
+        self.gc_manager.on_idle()
+
         # sleep until next event
         self.maybe_sleep_on_idle()
 
diff --git a/python/sglang/srt/managers/scheduler_gc_manager.py b/python/sglang/srt/managers/scheduler_gc_manager.py
new file mode 100644
index 000000000..279d8d891
--- /dev/null
+++ b/python/sglang/srt/managers/scheduler_gc_manager.py
@@ -0,0 +1,112 @@
+"""GC management for the scheduler process.
+
+CPython's automatic generation-2 garbage collection is a stop-the-world pause
+whose cost is O(live heap), and it triggers at arbitrary allocation sites. In
+a tensor-parallel server this is a correctness-adjacent problem, not just a
+latency one: a multi-second GC pause in ONE scheduler rank stalls ALL ranks,
+because the paused rank stops launching kernels while the peer GPUs spin
+inside a TP collective that needs its next launch. If anything in the process
+slowly accumulates live objects (e.g. a leaking dependency), pause duration
+grows with uptime until it exceeds the /health_generate window and the pod is
+killed by its liveness probe mid-pause — which looks like a permanent silent
+hang ("Server couldn't get a response from detokenizer") even though the
+process would have recovered.
+
+This was observed in production with sglang v0.5.12 on 8xB300 serving a
+DeepSeek-V3-family model: a flashinfer autotuner leak (TuningConfig objects
+retained by an unbounded lru_cache, see
+https://github.com/flashinfer-ai/flashinfer/issues/2139) grew the scheduler
+heap to ~75M tracked objects in ~6.5 hours, at which point a single gen-2
+collection measured 41 s while freeing nothing. Pauses crossed the 20 s
+health-check threshold after ~3.5 h and the pod was recycled several times a
+day.
+
+Strategy (all knobs env-gated):
+  * Raise the gen-2 threshold so automatic full collections become rare
+    (they remain as a fallback under sustained load that never goes idle).
+  * On the first fully-idle tick — i.e. right after warmup — run one full
+    collection and ``gc.freeze()`` the startup heap (model/module metadata,
+    several million objects) into the permanent generation so subsequent
+    collections never traverse it again.
+  * While idle, periodically run ``gc.collect()`` so cyclic garbage is
+    reclaimed at a moment when a pause cannot stall in-flight batches.
+
+We deliberately do NOT re-freeze periodically: objects frozen while alive are
+never collected if they later become cyclic garbage (e.g. evicted radix-tree
+nodes, which hold parent<->child cycles), so repeated freezing would convert
+ordinary churn into a permanent leak. The one-time startup freeze is safe
+because the startup heap is stable for the process lifetime.
+"""
+
+from __future__ import annotations
+
+import gc
+import logging
+import time
+
+from sglang.srt.environ import envs
+
+logger = logging.getLogger(__name__)
+
+
+class SchedulerGCManager:
+    def __init__(self):
+        self.enabled = envs.SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT.get()
+        self.idle_gc_interval_s = envs.SGLANG_SCHEDULER_IDLE_GC_INTERVAL.get()
+        self._did_initial_freeze = False
+        self._last_idle_gc_time = 0.0
+
+        if not self.enabled:
+            return
+
+        gen0, gen1, gen2 = gc.get_threshold()
+        new_gen2 = max(gen2, envs.SGLANG_SCHEDULER_GC_GEN2_THRESHOLD.get())
+        gc.set_threshold(gen0, gen1, new_gen2)
+        logger.info(
+            "Scheduler GC management enabled: gen-2 threshold %d -> %d, "
+            "idle GC interval %.0fs (disable with SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT=0)",
+            gen2,
+            new_gen2,
+            self.idle_gc_interval_s,
+        )
+
+    def on_idle(self):
+        """Run from the scheduler's idle housekeeping path.
+
+        First call (right after warmup): full collect + freeze of the startup
+        heap. Later calls: throttled full collect so cyclic garbage is
+        reclaimed while a pause is harmless.
+        """
+        if not self.enabled:
+            return
+
+        now = time.perf_counter()
+        if self._did_initial_freeze and (
+            now < self._last_idle_gc_time + self.idle_gc_interval_s
+        ):
+            return
+
+        tic = now
+        collected = gc.collect()
+        if not self._did_initial_freeze:
+            frozen_before = gc.get_freeze_count()
+            gc.freeze()
+            self._did_initial_freeze = True
+            logger.info(
+                "Initial post-warmup GC freeze: collected %d, froze %d objects "
+                "(%.3fs)",
+                collected,
+                gc.get_freeze_count() - frozen_before,
+                time.perf_counter() - tic,
+            )
+        else:
+            duration = time.perf_counter() - tic
+            if duration > 0.1:
+                logger.info(
+                    "Idle GC: collected %d objects in %.3fs "
+                    "(a growing duration here indicates a heap leak; "
+                    "see scheduler_gc_manager.py)",
+                    collected,
+                    duration,
+                )
+        self._last_idle_gc_time = time.perf_counter()
```

### Why each piece is the way it is

- **`gc.set_threshold(gen0, gen1, max(gen2, 10000))`** — with CPython defaults `(700, 10, 10)`,
  a full collection runs after every ~70 k allocations, i.e. every ~30 s under serving load —
  each one traversing the entire live heap mid-batch. Raising only the *gen-2* threshold keeps
  cheap gen-0/gen-1 collections untouched (they handle short-lived cycles) and keeps a very
  large heap from being walked on the hot path. The threshold is `max(...)`-ed so an operator
  who set an even larger value via Python isn't silently lowered, and automatic gen-2 is
  **deferred, not disabled** — under a pathological never-idle workload it still eventually
  runs, so cyclic garbage cannot accumulate without bound.

- **One-time `gc.freeze()` on the first fully-idle tick** — the first time `on_idle()` runs is
  right after warmup (model weights loaded, CUDA graphs captured, autotuning done). At that
  point the process holds ~1 M long-lived objects (measured: 954,058 frozen in 1.07 s) that
  will never become garbage while the process lives. `gc.freeze()` moves them to the permanent
  generation so **no future collection ever traverses them again**. Doing it at first idle
  rather than at a fixed init point means it naturally lands after whatever warmup the
  deployment performs.

- **Periodic `gc.collect()` only while idle** — `on_idle()` is reached only when
  `is_fully_idle()` is true (no running batch, no waiting queue), which is exactly when a pause
  cannot stall in-flight work, and — because schedulers run the same batches — when the *other*
  TP ranks are idle too. Throttled to `SGLANG_SCHEDULER_IDLE_GC_INTERVAL` (300 s default).
  Collections at this point reclaim reference cycles (e.g. evicted radix-tree nodes, whose
  parent↔child links are cycles that refcounting alone cannot free).

- **Deliberately NO periodic re-freeze.** This is the subtle design point. `gc.freeze()` on a
  *live* object is irreversible garbage-wise: if that object later becomes part of an
  unreachable cycle, the collector will never reclaim it. Radix-tree nodes are exactly such
  objects (alive now, cyclic garbage after eviction), so a naive "freeze every N minutes"
  policy would convert ordinary cache churn into a true, permanent leak. Freezing only the
  startup heap is safe because that heap is alive for the process lifetime by construction.

- **The log line as a leak canary** — an idle collect that takes >0.1 s logs its duration.
  A growing duration here is the earliest observable signal that something in the process is
  accumulating tracked objects, hours before any health impact.

- **Env knobs, per sglang conventions** (`environ.py`):
  `SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT` (bool, default **on**),
  `SGLANG_SCHEDULER_IDLE_GC_INTERVAL` (float seconds, 300),
  `SGLANG_SCHEDULER_GC_GEN2_THRESHOLD` (int, 10000). Disabling restores stock behavior exactly
  (thresholds untouched, no freeze, no idle collects).

### Alternative rejected, with the measurement that rejected it

Periodic `gc.freeze()` *was* trialled live (as the emergency mitigation, via the existing
`/freeze_gc` endpoint semantics): it works immediately — 7 frozen ranks ran 63 min stall-free
while the one accidentally-unfrozen rank produced exactly one 56 s stall — but the leak
re-accumulated fast enough that gen-2 cost was back to 3–4.5 s within ~35 min of each freeze.
Freeze cadence is a bridge, not a fix, and repeated freezing has the cyclic-garbage hazard
above. Hence: fix the leak (commit 2) and manage GC structurally (this commit).

### Safety

- All state is per-process and touched only from the scheduler thread.
- If the env kill-switch is set, the constructor returns before touching `gc` at all.
- `handle_freeze_gc` (the existing `/freeze_gc` RPC) is untouched and composes fine with this.

---

## Commit 2 — `[moe]` flashinfer TuningConfig leak workaround

### The exact problem

`flashinfer.autotuner.AutoTuner._find_nearest_profile` is declared:

```python
@classmethod
@lru_cache(maxsize=None)          # flashinfer/autotuner.py (0.6.11/0.6.12: line ~1469/1539)
def _find_nearest_profile(cls, shapes, tuning_config): ...
```

The cache key includes the `tuning_config` **object**. The fused-MoE entry points in
`flashinfer/fused_moe/core.py` (bf16 and fp8 block-scale alike) call
`moe_runner._make_tuning_config(...)` **on every MoE invocation**; that builds a new
`TuningConfig` whose `DynamicTensorSpec`s contain freshly created lambdas. Fresh lambdas ⇒
identity-based hash ⇒ the lru_cache key never repeats ⇒ every MoE call inserts a permanent,
never-hit entry retaining the config, ~6 closures, and a tuple of `torch.Size`s.

Measured: with 61 MoE layers per forward this leaks one config-set per layer per forward —
~46/s on a bf16 checkpoint, ~74/s on the public fp8
`deepseek-ai/DeepSeek-V3.1-Terminus` (9,244 configs already at warmup; 46,289 after 8 min of
load). After 6.5 h: 3.2 M `TuningConfig` + 3.2 M `DynamicTensorSpec` + 19.4 M `function` +
6.5 M `cell` + 22.4 M `torch.Size` = the 74.8 M-object heap from commit 1's problem statement.
The retainer was identified by walking `gc.get_referrers` from a leaked config: key-tuple →
`functools.lru_cache`'s internal linked list. Same mechanism as
[flashinfer#2139](https://github.com/flashinfer-ai/flashinfer/issues/2139), whose fix covered
the `mm_fp4`/`fp8_gemm` GEMM entry points but not the fused-MoE ones (still leaking in 0.6.12).

### The diff

```diff
diff --git a/python/sglang/srt/environ.py b/python/sglang/srt/environ.py
index d7873a408..8bc3c5d80 100644
--- a/python/sglang/srt/environ.py
+++ b/python/sglang/srt/environ.py
@@ -506,6 +506,9 @@ class Envs:
 
     # Flashinfer
     SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
+    # Opt-out for the flashinfer fused-MoE TuningConfig leak workaround
+    # (see utils/patch_flashinfer.py, flashinfer#2139).
+    SGLANG_DISABLE_FLASHINFER_TUNING_CONFIG_MEMO = EnvBool(False)
     SGLANG_FLASHINFER_USE_PAGED = EnvBool(False)
     # Default to the pick from flashinfer
     SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
diff --git a/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py b/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
index d374f5cd8..324c4ca7c 100644
--- a/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
+++ b/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
@@ -37,6 +37,16 @@ from sglang.srt.utils.common import (
     is_flashinfer_available,
     next_power_of_2,
 )
+from sglang.srt.utils.patch_flashinfer import (
+    monkey_patch_flashinfer_tuning_config_memo,
+)
+
+# Workaround for the flashinfer fused-MoE TuningConfig leak (flashinfer#2139):
+# without it, every MoE call permanently leaks a TuningConfig (+closures) into
+# the autotuner's unbounded lru_cache, growing gen-2 GC pauses with uptime
+# until one TP rank's pause stalls the whole TP group past the health-check
+# window. Lazy: takes effect on first MoE module build.
+monkey_patch_flashinfer_tuning_config_memo()
 
 _SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()
 
diff --git a/python/sglang/srt/utils/patch_flashinfer.py b/python/sglang/srt/utils/patch_flashinfer.py
new file mode 100644
index 000000000..2d4ee0048
--- /dev/null
+++ b/python/sglang/srt/utils/patch_flashinfer.py
@@ -0,0 +1,147 @@
+"""Workaround for a TuningConfig memory leak in flashinfer's fused MoE paths.
+
+flashinfer's ``AutoTuner._find_nearest_profile`` is decorated with
+``@lru_cache(maxsize=None)`` and keyed by ``(shapes, tuning_config)``. The
+fused-MoE entry points in ``flashinfer/fused_moe/core.py`` (e.g.
+``trtllm_bf16_moe_op``) build a brand-new ``TuningConfig`` — whose
+``DynamicTensorSpec``s contain freshly created lambdas, hence identity-based
+hashes — on EVERY call. Each MoE invocation therefore inserts a new,
+never-hit entry into the unbounded cache, permanently retaining the config,
+its closures, and the ``torch.Size`` key tuple.
+
+This is https://github.com/flashinfer-ai/flashinfer/issues/2139; the upstream
+fix covered the ``mm_fp4`` / ``fp8_gemm`` GEMM entry points, but the fused-MoE
+entry points still leak as of flashinfer 0.6.12.
+
+Measured impact before this workaround (sglang v0.5.12, 8xB300 TP=8,
+DeepSeek-V3-arch bf16, ``moe_runner_backend=flashinfer_trtllm``): the
+scheduler leaked one config per MoE layer per forward (~46 object-sets/s),
+reaching 74.8M tracked Python objects in 6.5h; gen-2 GC pauses grew from 7s
+to 46s with uptime, each pause stalling all 8 TP ranks and ultimately failing
+the 20s /health_generate watchdog (pod killed by k8s).
+
+The fix: memoize ``MoERunner._make_tuning_config`` per configuration. The
+config depends only on runner attributes, which optional inputs are present,
+and kwargs — not on tensor shapes or values (shapes are a separate part of
+the autotuner cache key) — so a handful of memo entries cover a deployment.
+A stable ``TuningConfig`` object makes ``_find_nearest_profile``'s lru_cache
+hit, eliminating both the leak and the per-call config construction.
+
+The patch wraps the (``functools.cache``-d) module factories so it applies
+lazily on first MoE use, after CUDA is initialized, and no-ops gracefully if
+flashinfer internals change.
+"""
+
+from __future__ import annotations
+
+import logging
+
+from sglang.srt.environ import envs
+
+logger = logging.getLogger(__name__)
+
+_PRIMITIVES = (int, float, str, bool, type(None))
+_FACTORY_NAMES = (
+    "get_trtllm_moe_sm100_module",
+    "get_cutlass_fused_moe_module",
+)
+_applied = False
+
+
+def _memo_key(runner, moe_inputs, tune_max_num_tokens, kwargs):
+    attrs = tuple(
+        sorted(
+            (k, v if isinstance(v, _PRIMITIVES) else str(v))
+            for k, v in runner.__dict__.items()
+        )
+    )
+    presence = tuple(
+        (f, getattr(moe_inputs, f, None) is not None)
+        for f in (
+            "routing_logits",
+            "topk_ids",
+            "expert_weights",
+            "hidden_states_scale",
+            "per_token_scale",
+        )
+    )
+    kw = tuple(
+        sorted(
+            (k, v if isinstance(v, _PRIMITIVES) else str(v)) for k, v in kwargs.items()
+        )
+    )
+    return (attrs, presence, tune_max_num_tokens, kw)
+
+
+def _patch_runner_cls(cls) -> bool:
+    if "_make_tuning_config" not in cls.__dict__ or getattr(
+        cls, "_sglang_tuning_config_memo_patched", False
+    ):
+        return False
+
+    orig = cls._make_tuning_config
+
+    def memoized_make_tuning_config(
+        self, moe_inputs, tune_max_num_tokens=8192, **kwargs
+    ):
+        memo = type(self)._sglang_tuning_config_memo
+        key = _memo_key(self, moe_inputs, tune_max_num_tokens, kwargs)
+        config = memo.get(key)
+        if config is None:
+            config = orig(self, moe_inputs, tune_max_num_tokens, **kwargs)
+            memo[key] = config
+        return config
+
+    cls._sglang_tuning_config_memo = {}
+    cls._make_tuning_config = memoized_make_tuning_config
+    cls._sglang_tuning_config_memo_patched = True
+    return True
+
+
+def _patch_factory_result(module_obj) -> int:
+    patched = 0
+    for name in dir(module_obj):
+        fn = getattr(module_obj, name, None)
+        fn = getattr(fn, "__wrapped__", fn)
+        for cell in getattr(fn, "__closure__", None) or ():
+            try:
+                value = cell.cell_contents
+            except ValueError:
+                continue
+            if isinstance(value, type) and value.__name__ == "MoERunner":
+                if _patch_runner_cls(value):
+                    patched += 1
+    return patched
+
+
+def monkey_patch_flashinfer_tuning_config_memo():
+    """Arm the TuningConfig memoization. Idempotent; lazy (first MoE use)."""
+    global _applied
+    if _applied or envs.SGLANG_DISABLE_FLASHINFER_TUNING_CONFIG_MEMO.get():
+        return
+    _applied = True
+
+    try:
+        import flashinfer.fused_moe.core as fi_moe_core
+    except ImportError:
+        return
+
+    for factory_name in _FACTORY_NAMES:
+        orig_factory = getattr(fi_moe_core, factory_name, None)
+        if orig_factory is None:
+            continue
+
+        def wrapped_factory(*args, _orig=orig_factory, _name=factory_name, **kw):
+            module_obj = _orig(*args, **kw)
+            n = _patch_factory_result(module_obj)
+            if n:
+                logger.info(
+                    "Applied flashinfer TuningConfig memoization to %d MoERunner "
+                    "class(es) from %s (workaround for flashinfer#2139; disable "
+                    "with SGLANG_DISABLE_FLASHINFER_TUNING_CONFIG_MEMO=1)",
+                    n,
+                    _name,
+                )
+            return module_obj
+
+        setattr(fi_moe_core, factory_name, wrapped_factory)
```

### Why each piece is the way it is

- **Memoize `_make_tuning_config` per configuration.** The `TuningConfig` is a function of
  (runner attributes, which optional inputs are present, kwargs) — *not* of tensor shapes or
  values: shapes enter the autotuner key separately as the `shapes` argument. So one config
  object per configuration is semantically identical, and a deployment ends up with a handful
  of memo entries (measured: 5 on the bf16 checkpoint, 12–13 on fp8 Terminus). With a stable
  object, flashinfer's lru_cache finally *hits*, which simultaneously kills the leak and the
  per-call config construction cost.

- **The memo key** stringifies non-primitive attribute values (enums like `DtypeTrtllmGen`)
  rather than hashing them directly — enum reprs are stable and this avoids assuming
  hashability of arbitrary future attributes. Input *presence* (not content) is part of the
  key because `_make_tuning_config` branches on which optional tensors exist.

- **Patching is deferred to first MoE use** by wrapping flashinfer's `@functools.cache`'d
  module factories (`get_trtllm_moe_sm100_module`, `get_cutlass_fused_moe_module`) instead of
  calling them at import: building those modules initializes CUDA / loads cubins, which must
  not happen at import time (wrong device context, startup cost). The wrapper scans the
  factory result's function closures for every nested `MoERunner` class and patches each —
  this is what makes the workaround cover the bf16 *and* fp8 paths without naming them.

- **Defensive by construction**: if flashinfer is absent, the factories are renamed, or the
  class no longer defines `_make_tuning_config`, the patch silently no-ops and stock behavior
  is preserved. Idempotent via a class-level flag. Kill-switch:
  `SGLANG_DISABLE_FLASHINFER_TUNING_CONFIG_MEMO=1`.

- **Why in sglang at all?** This is a flashinfer bug, and the right durable fix is upstream
  (build the config once per runner, or key/bound the lru_cache). sglang already carries
  flashinfer-behavior workarounds (e.g. the GB200/GB300 transport override in
  `flashinfer_comm_fusion.py`); this one should be dropped once flashinfer fixes the MoE entry
  points — it no-ops safely against a fixed flashinfer.

### Verification specific to this commit

Hot-injected into a live, fully-warmed leaking server: creation rate went from ~11 k configs
per 4 minutes to **0** (count pinned at 23,420 across consecutive censuses). On the public
fp8 checkpoint, stock vs. patched from boot: 9,244→46,289 vs. 1→1 over the same load window.

---

## Commit 3 — `[mem_cache]` sync-free eviction

### The exact problem

The paged allocator's `free()` (`srt/mem_cache/allocator/paged.py`) computes
`torch.unique(free_index // page_size)` on a **CUDA** tensor. `unique` has a data-dependent
output shape, so the CPU must synchronize with the GPU stream to learn it — an implicit
`cudaStreamSynchronize` per call. `RadixCache.evict()` called `free()` **once per evicted
node**, and eviction runs on the scheduler critical path inside `alloc_for_decode` /
`alloc_paged_token_slots_extend`, *before* the next batch is launched. Once the KV pool is
full of cached pages (steady state after ~30–60 min of traffic; the radix cache keeps it
full), eviction precedes nearly every allocation.

Measured: ~94 % of eviction wall time inside `free()` (50 ms of a 53 ms eviction; ~7–17 ms
per node). Worse, a scheduler blocked in a device sync while the peer ranks spin in a
collective that needs its *next* launch is a circular wait: batching the frees into **one**
sync per eviction (the first fix attempt, using the existing `free_group` API) did **not**
help — single batched evictions still measured 35–40 s inside a convoy. The sync must not
exist at all on this path.

### The diff

```diff
diff --git a/python/sglang/srt/mem_cache/radix_cache.py b/python/sglang/srt/mem_cache/radix_cache.py
index bd6adb6e3..53e0f6d8d 100644
--- a/python/sglang/srt/mem_cache/radix_cache.py
+++ b/python/sglang/srt/mem_cache/radix_cache.py
@@ -546,11 +546,29 @@ class RadixCache(KVCacheEventMixin, BasePrefixCache):
         ]
         heapq.heapify(eviction_heap)
 
+        # Eviction must not synchronize with the device. The paged allocator's
+        # free() runs `torch.unique` on a CUDA tensor; its data-dependent
+        # output shape forces the CPU to wait for the GPU stream. Eviction
+        # runs on the scheduler critical path *before* the next batch is
+        # launched, so during that wait the other TP ranks spin inside a
+        # collective that needs this rank's next launch — a circular wait
+        # observed to stall all ranks for tens of seconds once the KV pool is
+        # full (eviction then precedes nearly every allocation). Radix-tree
+        # node values are whole page-aligned page runs, so the page indices
+        # can be derived with a fixed-shape strided slice instead: no sync,
+        # pure async kernels.
+        allocator = self.token_to_kv_pool_allocator
+        page_size = allocator.page_size
+        freed_values = []
+
         num_evicted = 0
         while num_evicted < num_tokens and len(eviction_heap):
             _priority, x = heapq.heappop(eviction_heap)
 
-            self.token_to_kv_pool_allocator.free(x.value)
+            if page_size > 1 and len(x.value) % page_size == 0:
+                freed_values.append(x.value)
+            else:
+                allocator.free(x.value)
             num_evicted += len(x.value)
             self._delete_leaf(x)
 
@@ -560,6 +578,23 @@ class RadixCache(KVCacheEventMixin, BasePrefixCache):
 
             self._record_remove_event(x)
 
+        if freed_values:
+            cat = freed_values[0] if len(freed_values) == 1 else torch.cat(freed_values)
+            # Every page in a node value is a contiguous, page-aligned run of
+            # token slots, so sampling one slot per page yields each freed
+            # page exactly once — equivalent to unique(cat // page_size)
+            # without the device sync.
+            pages = cat[::page_size] // page_size
+            if allocator.is_not_in_free_group:
+                if allocator.need_sort:
+                    allocator.release_pages = torch.cat(
+                        (pages, allocator.release_pages)
+                    )
+                else:
+                    allocator.free_pages = torch.cat((pages, allocator.free_pages))
+            else:
+                allocator.free_group.append(cat)
+
         self.update_eviction_metrics(num_evicted, start_time)
         return EvictResult(num_tokens_evicted=num_evicted)
```

### Why this is correct

- **Why `unique` was there**: token-level indices contain `page_size` duplicates per page
  (64 tokens share a page), and `// page_size` maps all of them to the same page id;
  `unique` dedups. But radix-tree node values are, by construction, **whole page-aligned
  page runs**: the tree is page-aligned (keys are multiples of `page_size`; splits happen on
  page boundaries), and within a page the token slots are consecutive. Therefore every page
  contributes exactly `page_size` consecutive elements, and the strided slice
  `cat[::page_size] // page_size` yields each freed page **exactly once** — the same set
  `unique` would return, with a **fixed output shape**, i.e. pure async kernel launches and
  zero synchronization.

- **The guard** `page_size > 1 and len(x.value) % page_size == 0` routes anything unexpected
  (a non-page-multiple value, or a `page_size==1` token allocator whose `free()` never had
  the `unique` in the first place) through the original `allocator.free()` path — behavior
  for those cases is bit-identical to before.

- **Free-group compatibility**: if a caller has an open free-group
  (`is_not_in_free_group == False`), the collected tensor is appended to the group exactly as
  `free()` would have done, preserving those semantics. Otherwise pages are appended to
  `release_pages` / `free_pages` mirroring `free()`'s `need_sort` branch — the final freelist
  contents are identical to the unpatched code.

- **What changes observably**: nothing about which pages get freed or when they become
  allocatable — only the elimination of the per-node host-device round trip. Eviction metrics
  (`update_eviction_metrics`) are computed identically.

### Verification specific to this commit

At full pool on the branch build: 25,280 evictions with 25,273 ≤ 10 ms and all ≤ 50 ms
(mean 2.7 ms) on the bf16 checkpoint; 1,568 evictions all ≤ 10 ms (mean 0.38 ms) on public
fp8 Terminus with `--max-total-tokens 262144`. Unpatched baseline at the same state:
p50 ~6 ms with 94 % sync time, 50–80 ms spikes, and multi-second outliers inside convoys.

---

## How the three commits compose

Commit 2 removes the heap growth (this incident's root cause). Commit 1 guarantees that *any
future* heap growth — in any dependency — produces at worst a slow drift in idle-GC duration
(logged, alertable) instead of mid-batch stop-the-world stalls that wedge the TP group and get
pods killed. Commit 3 removes the one device sync that sat on the scheduler's per-iteration
critical path and amplified every other delay into a convoy. They are independent and each is
safe to cherry-pick alone, but the incident requires 1+2 to be considered fixed; 3 is a
latency improvement that also reduces convoy sensitivity.

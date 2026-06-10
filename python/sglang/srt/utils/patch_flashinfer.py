"""Workaround for a TuningConfig memory leak in flashinfer's fused MoE paths.

flashinfer's ``AutoTuner._find_nearest_profile`` is decorated with
``@lru_cache(maxsize=None)`` and keyed by ``(shapes, tuning_config)``. The
fused-MoE entry points in ``flashinfer/fused_moe/core.py`` (e.g.
``trtllm_bf16_moe_op``) build a brand-new ``TuningConfig`` — whose
``DynamicTensorSpec``s contain freshly created lambdas, hence identity-based
hashes — on EVERY call. Each MoE invocation therefore inserts a new,
never-hit entry into the unbounded cache, permanently retaining the config,
its closures, and the ``torch.Size`` key tuple.

This is https://github.com/flashinfer-ai/flashinfer/issues/2139; the upstream
fix covered the ``mm_fp4`` / ``fp8_gemm`` GEMM entry points, but the fused-MoE
entry points still leak as of flashinfer 0.6.12.

Measured impact before this workaround (sglang v0.5.12, 8xB300 TP=8,
DeepSeek-V3-arch bf16, ``moe_runner_backend=flashinfer_trtllm``): the
scheduler leaked one config per MoE layer per forward (~46 object-sets/s),
reaching 74.8M tracked Python objects in 6.5h; gen-2 GC pauses grew from 7s
to 46s with uptime, each pause stalling all 8 TP ranks and ultimately failing
the 20s /health_generate watchdog (pod killed by k8s).

The fix: memoize ``MoERunner._make_tuning_config`` per configuration. The
config depends only on runner attributes, which optional inputs are present,
and kwargs — not on tensor shapes or values (shapes are a separate part of
the autotuner cache key) — so a handful of memo entries cover a deployment.
A stable ``TuningConfig`` object makes ``_find_nearest_profile``'s lru_cache
hit, eliminating both the leak and the per-call config construction.

The patch wraps the (``functools.cache``-d) module factories so it applies
lazily on first MoE use, after CUDA is initialized, and no-ops gracefully if
flashinfer internals change.
"""

from __future__ import annotations

import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_PRIMITIVES = (int, float, str, bool, type(None))
_FACTORY_NAMES = (
    "get_trtllm_moe_sm100_module",
    "get_cutlass_fused_moe_module",
)
_applied = False


def _memo_key(runner, moe_inputs, tune_max_num_tokens, kwargs):
    attrs = tuple(
        sorted(
            (k, v if isinstance(v, _PRIMITIVES) else str(v))
            for k, v in runner.__dict__.items()
        )
    )
    presence = tuple(
        (f, getattr(moe_inputs, f, None) is not None)
        for f in (
            "routing_logits",
            "topk_ids",
            "expert_weights",
            "hidden_states_scale",
            "per_token_scale",
        )
    )
    kw = tuple(
        sorted(
            (k, v if isinstance(v, _PRIMITIVES) else str(v)) for k, v in kwargs.items()
        )
    )
    return (attrs, presence, tune_max_num_tokens, kw)


def _patch_runner_cls(cls) -> bool:
    if "_make_tuning_config" not in cls.__dict__ or getattr(
        cls, "_sglang_tuning_config_memo_patched", False
    ):
        return False

    orig = cls._make_tuning_config

    def memoized_make_tuning_config(
        self, moe_inputs, tune_max_num_tokens=8192, **kwargs
    ):
        memo = type(self)._sglang_tuning_config_memo
        key = _memo_key(self, moe_inputs, tune_max_num_tokens, kwargs)
        config = memo.get(key)
        if config is None:
            config = orig(self, moe_inputs, tune_max_num_tokens, **kwargs)
            memo[key] = config
        return config

    cls._sglang_tuning_config_memo = {}
    cls._make_tuning_config = memoized_make_tuning_config
    cls._sglang_tuning_config_memo_patched = True
    return True


def _patch_factory_result(module_obj) -> int:
    patched = 0
    for name in dir(module_obj):
        fn = getattr(module_obj, name, None)
        fn = getattr(fn, "__wrapped__", fn)
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, type) and value.__name__ == "MoERunner":
                if _patch_runner_cls(value):
                    patched += 1
    return patched


def monkey_patch_flashinfer_tuning_config_memo():
    """Arm the TuningConfig memoization. Idempotent; lazy (first MoE use)."""
    global _applied
    if _applied or envs.SGLANG_DISABLE_FLASHINFER_TUNING_CONFIG_MEMO.get():
        return
    _applied = True

    try:
        import flashinfer.fused_moe.core as fi_moe_core
    except ImportError:
        return

    for factory_name in _FACTORY_NAMES:
        orig_factory = getattr(fi_moe_core, factory_name, None)
        if orig_factory is None:
            continue

        def wrapped_factory(*args, _orig=orig_factory, _name=factory_name, **kw):
            module_obj = _orig(*args, **kw)
            n = _patch_factory_result(module_obj)
            if n:
                logger.info(
                    "Applied flashinfer TuningConfig memoization to %d MoERunner "
                    "class(es) from %s (workaround for flashinfer#2139; disable "
                    "with SGLANG_DISABLE_FLASHINFER_TUNING_CONFIG_MEMO=1)",
                    n,
                    _name,
                )
            return module_obj

        setattr(fi_moe_core, factory_name, wrapped_factory)

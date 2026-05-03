"""Unit tests for the GPU-memory-tier defaults of `max_prefill_tokens`.

These tests exercise `ServerArgs._handle_gpu_memory_settings` directly to verify
that the historical 16 384-token default is preserved on every existing GPU
memory tier, while big-memory Blackwell-class GPUs (B200 / B300 / MI300, ≥160 GB
HBM) get a 4× chunked_prefill_size default and H200-class (≥90 GB HBM) gets a
2× default. Without this, 8 simultaneous long-prompt requests serialize through
the chunked-prefill scheduler because at most one full chunk fits per step.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _stub_model_config():
    """Minimal stand-in for ModelConfig that satisfies the attributes
    `_handle_gpu_memory_settings` reads on the codepath we exercise."""
    return SimpleNamespace(
        context_len=8192,
        is_multimodal=False,
        is_multimodal_gen=False,
        is_audio_model=False,
        is_image_gen=False,
        is_encoder_decoder=False,
        is_hybrid=False,
        hf_config=SimpleNamespace(
            model_type="stub",
            num_hidden_layers=4,
            architectures=["StubForCausalLM"],
        ),
    )


def _make_args(**overrides):
    """Construct a partially-initialized ServerArgs that exposes only the fields
    required by `_handle_gpu_memory_settings`. Avoids `__post_init__`'s heavy
    side effects (HF model fetch, distributed bootstrap, etc.)."""
    args = ServerArgs.__new__(ServerArgs)
    args.chunked_prefill_size = None
    args.cuda_graph_max_bs = None
    args.cuda_graph_bs = None
    args.max_prefill_tokens = None
    args.tp_size = 8
    args.dp_size = 1
    args.device = "cuda"
    args.piecewise_cuda_graph_max_tokens = None
    args.disable_cuda_graph = True
    args.spec_algorithm = None
    args.use_mla_backend = lambda: False
    args.mem_fraction_static = None
    args.model_path = "stub-model"
    args.attention_backend = None
    args.disaggregation_mode = "null"
    args.enable_dp_attention = False
    args.enable_torch_compile = False
    args.enable_piecewise_cuda_graph = False
    args.torch_compile_max_bs = 32
    args.kv_cache_dtype = "auto"
    args.language_only = False
    args.encoder_only = False
    args.is_embedding = False
    args.enable_multimodal = False
    args.disable_radix_cache = False
    stub = _stub_model_config()
    args.get_model_config = lambda: stub
    args.model_config = stub
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestMaxPrefillTokensDefaults(unittest.TestCase):
    """Tier-aware default resolution for `max_prefill_tokens`."""

    def test_low_memory_tiers_preserve_legacy_default(self):
        # T4/4080 (<20GB), 4090/A10 (<35GB), A100-40 (<60GB), H100/A100-80 (<90GB)
        # all keep the historical 16384 default.
        for label, gpu_mem_mb, want_chunked in [
            ("T4_class", 16 * 1024, 2048),
            ("4090_class", 24 * 1024, 2048),
            ("A100_40", 40 * 1024, 4096),
            ("H100", 80 * 1024, 8192),
        ]:
            with self.subTest(label):
                args = _make_args()
                args._handle_gpu_memory_settings(gpu_mem_mb)
                self.assertEqual(args.chunked_prefill_size, want_chunked)
                self.assertEqual(args.max_prefill_tokens, 16384)

    def test_h200_tier_uses_2x_chunked_or_legacy_floor(self):
        # H200 default (chunked=8192) → 2*8192 = 16384, equal to legacy floor.
        args = _make_args()
        args._handle_gpu_memory_settings(141 * 1024)
        self.assertEqual(args.chunked_prefill_size, 8192)
        self.assertEqual(args.max_prefill_tokens, 16384)

        # H200 with user-set chunked_prefill_size=16384 should bump to 32768.
        args = _make_args(chunked_prefill_size=16384)
        args._handle_gpu_memory_settings(141 * 1024)
        self.assertEqual(args.max_prefill_tokens, 32768)

    def test_blackwell_class_bumps_to_4x_chunked(self):
        # B200 / B300 / MI300 (>=160GB HBM): chunked=16384 → max_prefill=65536.
        for label, gpu_mem_mb in [
            ("B200", 192 * 1024),
            ("B300", 275 * 1024),
            ("MI300", 192 * 1024),
        ]:
            with self.subTest(label):
                args = _make_args()
                args._handle_gpu_memory_settings(gpu_mem_mb)
                self.assertEqual(args.chunked_prefill_size, 16384)
                self.assertEqual(args.max_prefill_tokens, 65536)

    def test_unknown_gpu_mem_keeps_legacy_default(self):
        args = _make_args()
        args._handle_gpu_memory_settings(None)
        self.assertEqual(args.chunked_prefill_size, 4096)
        self.assertEqual(args.max_prefill_tokens, 16384)

    def test_user_override_is_respected(self):
        # An explicit --max-prefill-tokens must NOT be clobbered by the
        # tier-aware default, on any tier.
        for gpu_mem_mb in [16 * 1024, 80 * 1024, 141 * 1024, 275 * 1024, None]:
            with self.subTest(gpu_mem_mb=gpu_mem_mb):
                args = _make_args(max_prefill_tokens=8192)
                args._handle_gpu_memory_settings(gpu_mem_mb)
                self.assertEqual(args.max_prefill_tokens, 8192)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import Mock, patch

from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod


def _original_forward(*args, **kwargs):
    raise AssertionError("not called")


class TestTorchCompileBackendSelection(unittest.TestCase):
    def _make_method(self, *, use_flashinfer_trtllm_moe: bool):
        method = object.__new__(UnquantizedFusedMoEMethod)
        method._forward_method = _original_forward
        method._original_forward_method = None
        method.is_torch_compile = False
        method.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe
        return method

    def test_canonical_weights_use_native_moe_at_batch_one(self):
        method = self._make_method(use_flashinfer_trtllm_moe=False)

        method.enter_torch_compile(num_tokens=1)

        self.assertIs(method._forward_method, fused_moe_forward_native)
        method.leave_torch_compile()
        self.assertIs(method._forward_method, _original_forward)

    def test_flashinfer_block_weights_keep_cuda_moe_at_batch_one(self):
        method = self._make_method(use_flashinfer_trtllm_moe=True)

        method.enter_torch_compile(num_tokens=1)

        self.assertIs(method._forward_method, _original_forward)
        method.leave_torch_compile()
        self.assertIs(method._forward_method, _original_forward)

    def test_flashinfer_bypassed_topk_keeps_cuda_path_at_batch_one(self):
        topk = object.__new__(TopK)
        topk._forward_method = _original_forward
        topk._original_forward_method = None
        topk.is_torch_compile = False

        backend = Mock()
        backend.is_flashinfer_trtllm.return_value = True
        with patch(
            "sglang.srt.layers.moe.get_moe_runner_backend", return_value=backend
        ):
            topk.enter_torch_compile(num_tokens=1)

        self.assertIs(topk._forward_method, _original_forward)
        topk.leave_torch_compile()
        self.assertIs(topk._forward_method, _original_forward)

    def test_non_flashinfer_topk_uses_native_path_at_batch_one(self):
        topk = object.__new__(TopK)
        topk._forward_method = _original_forward
        topk._original_forward_method = None
        topk.is_torch_compile = False

        backend = Mock()
        backend.is_flashinfer_trtllm.return_value = False
        with patch(
            "sglang.srt.layers.moe.get_moe_runner_backend", return_value=backend
        ):
            topk.enter_torch_compile(num_tokens=1)

        self.assertIs(topk._forward_method.__func__, TopK.forward_native)
        topk.leave_torch_compile()
        self.assertIs(topk._forward_method, _original_forward)


if __name__ == "__main__":
    unittest.main()

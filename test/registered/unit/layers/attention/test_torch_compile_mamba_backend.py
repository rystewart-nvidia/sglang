from sglang.srt.layers.attention.hybrid_linear_attn_backend import Mamba2AttnBackend


def test_mamba_forward_is_excluded_from_torch_compile():
    assert getattr(Mamba2AttnBackend.forward, "_torchdynamo_disable", False)

"""CPU regressions for DFlash draft checkpoint configuration."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from sglang.srt.models import dflash
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dflash_worker_v2 import _get_dflash_embedding_module


class _FakeDraftLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


class _FakeNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


class _MinimalDraftModel(DFlashDraftModel):
    decoder_layer_cls = _FakeDraftLayer


def _draft_config(*, has_embed_tokens):
    return SimpleNamespace(
        hidden_size=4,
        num_hidden_layers=6,
        rms_norm_eps=1e-6,
        vocab_size=128,
        has_embed_tokens=has_embed_tokens,
        dflash_config={
            "block_size": 8,
            "target_layer_ids": [1, 5, 19, 29, 41, 51],
        },
    )


class TestDFlashDraftModel(unittest.TestCase):
    @patch.object(dflash, "VocabParallelEmbedding", _FakeEmbedding)
    @patch.object(dflash, "RMSNorm", _FakeNorm)
    def test_explicit_target_ids_are_not_validated_against_draft_depth(self):
        model = _MinimalDraftModel(_draft_config(has_embed_tokens=True))

        self.assertEqual(model.num_context_features, 6)
        self.assertEqual(model.fc.in_features, 24)
        self.assertEqual(model.block_size, 8)
        self.assertIs(model.get_input_embeddings(), model.embed_tokens)
        self.assertEqual(model.embed_tokens.kwargs["prefix"], "embed_tokens")

    @patch.object(dflash, "RMSNorm", _FakeNorm)
    def test_checkpoint_without_embedding_uses_fallback_contract(self):
        model = _MinimalDraftModel(_draft_config(has_embed_tokens=False))

        self.assertIsNone(model.get_input_embeddings())


class TestDFlashEmbeddingSelection(unittest.TestCase):
    def test_draft_embedding_is_preferred(self):
        draft_embedding = object()
        target_embedding = object()
        draft_model = SimpleNamespace(get_input_embeddings=lambda: draft_embedding)
        target_model = SimpleNamespace(get_input_embeddings=lambda: target_embedding)

        self.assertIs(
            _get_dflash_embedding_module(draft_model, target_model), draft_embedding
        )

    def test_target_embedding_is_the_fallback(self):
        target_embedding = object()
        draft_model = SimpleNamespace(get_input_embeddings=lambda: None)
        target_model = SimpleNamespace(get_input_embeddings=lambda: target_embedding)

        self.assertIs(
            _get_dflash_embedding_module(draft_model, target_model), target_embedding
        )


if __name__ == "__main__":
    unittest.main()

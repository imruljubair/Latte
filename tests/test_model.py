from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from latte.checkpoint import save_checkpoint
from latte.config import ModelConfig
from latte.model import TinyTransformer
from latte.tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parents[1]


class TinyTransformerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ModelConfig(
            model_dim=4,
            qk_dim=3,
            value_dim=4,
            num_heads=1,
            num_layers=1,
            ffn_dim=16,
            context_length=8,
            dropout=0.0,
        )
        self.tokenizer = CharTokenizer.from_text("abc ")
        self.model = TinyTransformer(self.config, self.tokenizer.vocab_size)

    def test_exact_projection_shapes(self) -> None:
        attention = self.model.blocks[0].attention
        self.assertEqual(tuple(attention.wq.weight.shape), (3, 4))
        self.assertEqual(tuple(attention.wk.weight.shape), (3, 4))
        self.assertEqual(tuple(attention.wv.weight.shape), (4, 4))

    def test_causal_attention_and_trace_shapes(self) -> None:
        inputs = torch.tensor([[0, 1, 2, 3]])
        logits, trace = self.model(inputs, return_trace=True)
        self.assertEqual(tuple(logits.shape), (1, 4, 4))
        assert trace is not None
        attention = trace["layers"][0]["attention"]
        self.assertEqual(tuple(attention["q"].shape), (1, 1, 4, 3))
        self.assertEqual(tuple(attention["v"].shape), (1, 1, 4, 4))
        probabilities = attention["attention"][0, 0]
        self.assertTrue(torch.allclose(probabilities[0], torch.tensor([1, 0, 0, 0.])))
        self.assertEqual(probabilities.triu(diagonal=1).count_nonzero().item(), 0)

    def test_export_trace_has_real_3x4_weights_and_growing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            checkpoint_path = Path(temp_directory) / "model.pt"
            trace_path = Path(temp_directory) / "trace.json"
            save_checkpoint(
                checkpoint_path,
                self.model,
                self.tokenizer,
                {},
                step=0,
                validation_loss=0.0,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "export_trace.py"),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--text",
                    "abc",
                    "--output",
                    str(trace_path),
                ],
                cwd=ROOT,
                check=True,
            )
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            wq = trace["layers"][0]["weights"]["wq_by_head"][0]
            self.assertEqual((len(wq), len(wq[0])), (3, 4))
            cache = trace["layers"][0]["kv_cache_by_position"]
            self.assertEqual(len(cache[0]["keys"][0]), 1)
            self.assertEqual(len(cache[2]["keys"][0]), 3)


if __name__ == "__main__":
    unittest.main()


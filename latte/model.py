from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.qk_dim = config.qk_dim
        self.value_dim = config.value_dim
        self.wq = nn.Linear(
            config.model_dim, config.num_heads * config.qk_dim, bias=False
        )
        self.wk = nn.Linear(
            config.model_dim, config.num_heads * config.qk_dim, bias=False
        )
        self.wv = nn.Linear(
            config.model_dim, config.num_heads * config.value_dim, bias=False
        )
        self.wo = nn.Linear(
            config.num_heads * config.value_dim, config.model_dim, bias=False
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, return_trace: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        batch_size, sequence_length, _ = x.shape
        q = self.wq(x).view(
            batch_size, sequence_length, self.num_heads, self.qk_dim
        )
        k = self.wk(x).view(
            batch_size, sequence_length, self.num_heads, self.qk_dim
        )
        v = self.wv(x).view(
            batch_size, sequence_length, self.num_heads, self.value_dim
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.qk_dim)
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        ).triu(diagonal=1)
        masked_scores = scores.masked_fill(causal_mask, float("-inf"))
        attention = F.softmax(masked_scores, dim=-1)
        dropped_attention = self.attention_dropout(attention)
        values = dropped_attention @ v
        merged = values.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.num_heads * self.value_dim
        )
        output = self.residual_dropout(self.wo(merged))

        trace = None
        if return_trace:
            trace = {
                "q": q,
                "k": k,
                "v": v,
                "scores": masked_scores,
                "attention": attention,
                "attention_values": values,
                "projected_output": output,
            }
        return output, trace


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.model_dim)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = nn.RMSNorm(config.model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, config.ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.model_dim, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, x: torch.Tensor, return_trace: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        attention_input = self.attention_norm(x)
        attention_output, attention_trace = self.attention(
            attention_input, return_trace=return_trace
        )
        after_attention = x + attention_output
        ffn_input = self.ffn_norm(after_attention)
        ffn_output = self.ffn(ffn_input)
        output = after_attention + ffn_output

        trace = None
        if return_trace:
            trace = {
                "input": x,
                "attention_input": attention_input,
                "attention": attention_trace,
                "after_attention": after_attention,
                "ffn_input": ffn_input,
                "ffn_output": ffn_output,
                "output": output,
            }
        return output, trace


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(
            config.context_length, config.model_dim
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.RMSNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, return_trace: bool = False
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        _, sequence_length = token_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length "
                f"{self.config.context_length}"
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(token_embeddings + position_embeddings)

        layer_traces: list[dict[str, Any]] = []
        for block in self.blocks:
            x, layer_trace = block(x, return_trace=return_trace)
            if layer_trace is not None:
                layer_traces.append(layer_trace)

        final_hidden = self.final_norm(x)
        logits = self.lm_head(final_hidden)

        trace = None
        if return_trace:
            trace = {
                "token_embeddings": token_embeddings,
                "position_embeddings": position_embeddings,
                "embedding_output": token_embeddings + position_embeddings,
                "layers": layer_traces,
                "final_hidden": final_hidden,
                "logits": logits,
                "probabilities": F.softmax(logits, dim=-1),
            }
        return logits, trace

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 12,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    count = min(top_k, next_logits.shape[-1])
                    threshold = torch.topk(next_logits, count).values[:, [-1]]
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, float("-inf")
                    )
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids

    def architecture_dict(self) -> dict[str, Any]:
        return {**asdict(self.config), "vocab_size": self.vocab_size}


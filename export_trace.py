from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from latte.checkpoint import load_checkpoint
from latte.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real weights and inference values as JSON"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def values(tensor: torch.Tensor) -> Any:
    return tensor.detach().float().cpu().tolist()


def finite_values(tensor: torch.Tensor) -> Any:
    # JSON has no portable infinity representation. Future-token scores are null.
    detached = tensor.detach().float().cpu()
    return [
        [
            [None if not torch.isfinite(value) else value.item() for value in row]
            for row in head
        ]
        for head in detached
    ]


def split_projection(weight: torch.Tensor, heads: int, width: int) -> list[Any]:
    reshaped = weight.detach().float().cpu().view(heads, width, -1)
    return reshaped.tolist()


def main() -> None:
    args = parse_args()
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if args.top_k < 0:
        raise ValueError("--top-k cannot be negative")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)

    token_ids = tokenizer.encode(args.text)
    if not token_ids:
        raise ValueError("--text cannot be empty")
    if len(token_ids) > model.config.context_length:
        raise ValueError(
            f"Text encodes to {len(token_ids)} tokens, exceeding context length "
            f"{model.config.context_length}. Use a shorter string."
        )

    inputs = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, trace = model(inputs, return_trace=True)
    assert trace is not None

    final_logits = logits[0, -1]
    if args.temperature == 0:
        sampling_probabilities = torch.zeros_like(final_logits)
        sampled_token_id = int(torch.argmax(final_logits).item())
        sampling_probabilities[sampled_token_id] = 1.0
        sampling_noise = torch.ones_like(final_logits)
    else:
        sampling_logits = final_logits / args.temperature
        if args.top_k > 0:
            count = min(args.top_k, sampling_logits.shape[-1])
            threshold = torch.topk(sampling_logits, count).values[-1]
            sampling_logits = sampling_logits.masked_fill(
                sampling_logits < threshold, float("-inf")
            )
        sampling_probabilities = torch.softmax(sampling_logits, dim=-1)
        rng_state = torch.random.get_rng_state()
        sampled_token_id = int(
            torch.multinomial(sampling_probabilities, num_samples=1).item()
        )
        torch.random.set_rng_state(rng_state)
        sampling_noise = torch.empty_like(sampling_probabilities).exponential_(1)
        reproduced_id = int(
            torch.argmax(sampling_probabilities / sampling_noise).item()
        )
        if reproduced_id != sampled_token_id:
            raise RuntimeError("Could not reproduce PyTorch multinomial sampling")

    layers = []
    for layer_index, (block, layer_trace) in enumerate(
        zip(model.blocks, trace["layers"])
    ):
        attention_trace = layer_trace["attention"]
        keys = attention_trace["k"][0]
        vals = attention_trace["v"][0]
        cache_by_position = []
        for position in range(len(token_ids)):
            cache_by_position.append(
                {
                    "position": position,
                    "keys": values(keys[:, : position + 1, :]),
                    "values": values(vals[:, : position + 1, :]),
                }
            )

        layers.append(
            {
                "layer": layer_index,
                "weights": {
                    "wq_by_head": split_projection(
                        block.attention.wq.weight,
                        model.config.num_heads,
                        model.config.qk_dim,
                    ),
                    "wk_by_head": split_projection(
                        block.attention.wk.weight,
                        model.config.num_heads,
                        model.config.qk_dim,
                    ),
                    "wv_by_head": split_projection(
                        block.attention.wv.weight,
                        model.config.num_heads,
                        model.config.value_dim,
                    ),
                    "wo": values(block.attention.wo.weight),
                    "attention_norm": values(block.attention_norm.weight),
                    "ffn_norm": values(block.ffn_norm.weight),
                    "ffn_in": values(block.ffn[0].weight),
                    "ffn_out": values(block.ffn[2].weight),
                },
                "activations": {
                    "input": values(layer_trace["input"][0]),
                    "attention_input": values(layer_trace["attention_input"][0]),
                    "q_by_head": values(attention_trace["q"][0]),
                    "k_by_head": values(keys),
                    "v_by_head": values(vals),
                    "attention_scores_by_head": finite_values(
                        attention_trace["scores"][0]
                    ),
                    "attention_probabilities_by_head": values(
                        attention_trace["attention"][0]
                    ),
                    "attention_values_by_head": values(
                        attention_trace["attention_values"][0]
                    ),
                    "output": values(layer_trace["output"][0]),
                },
                "kv_cache_by_position": cache_by_position,
            }
        )

    document = {
        "format_version": 1,
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_step": checkpoint["step"],
        "validation_loss": checkpoint["validation_loss"],
        "architecture": model.architecture_dict(),
        "global_weights": {
            "token_embedding": values(model.token_embedding.weight),
            "position_embedding": values(model.position_embedding.weight),
            "final_norm": values(model.final_norm.weight),
            "lm_head": values(model.lm_head.weight),
        },
        "input": {
            "text": args.text,
            "token_ids": token_ids,
            "tokens": [tokenizer.chars[token_id] for token_id in token_ids],
        },
        "generation": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "sampled_token_id": sampled_token_id,
            "sampled_token": tokenizer.chars[sampled_token_id],
            "generated_text": args.text + tokenizer.chars[sampled_token_id],
            "sampling_probabilities": values(sampling_probabilities),
            "sampling_noise": values(sampling_noise),
        },
        "embeddings": {
            "complete_token_embedding_table": values(model.token_embedding.weight),
            "selected_token_embeddings": values(trace["token_embeddings"][0]),
            "selected_position_embeddings": values(
                trace["position_embeddings"][0]
            ),
            "combined_embeddings": values(trace["embedding_output"][0]),
        },
        "layers": layers,
        "final_hidden": values(trace["final_hidden"][0]),
        "logits": values(trace["logits"][0]),
        "probabilities": values(trace["probabilities"][0]),
        "vocabulary": tokenizer.chars,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()

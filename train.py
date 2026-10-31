from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from latte.checkpoint import save_checkpoint
from latte.config import load_config
from latte.model import TinyTransformer
from latte.runtime import resolve_device, seed_everything
from latte.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an inspectable tiny LM")
    parser.add_argument("--config", required=True, help="TOML configuration file")
    parser.add_argument("--steps", type=int, help="Override training.steps")
    parser.add_argument("--device", help="Override training.device")
    return parser.parse_args()


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum_start = len(data) - context_length - 1
    if maximum_start < 1:
        raise ValueError(
            f"Dataset split needs at least {context_length + 2} tokens, "
            f"but has {len(data)}"
        )
    starts = torch.randint(maximum_start, (batch_size,))
    inputs = torch.stack([data[i : i + context_length] for i in starts])
    targets = torch.stack([data[i + 1 : i + context_length + 1] for i in starts])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyTransformer,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for name, data in (("train", train_data), ("validation", validation_data)):
        losses = []
        for _ in range(eval_batches):
            inputs, targets = get_batch(data, batch_size, context_length, device)
            logits, _ = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            losses.append(loss.item())
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config.training
    steps = args.steps if args.steps is not None else training.steps
    if steps <= 0:
        raise ValueError("--steps must be positive")

    seed_everything(training.seed)
    device = resolve_device(args.device or training.device)

    corpus_path = Path(config.data.path)
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(len(encoded) * config.data.train_split)
    train_data = encoded[:split_index]
    validation_data = encoded[split_index:]

    model = TinyTransformer(config.model, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    output_path = Path(config.output.directory) / config.output.checkpoint_name

    print(f"device: {device}")
    print(f"vocabulary: {tokenizer.vocab_size} characters")
    print(f"parameters: {parameter_count:,}")
    print(f"checkpoint: {output_path}")

    best_validation = float("inf")
    model.train()
    for step in range(1, steps + 1):
        inputs, targets = get_batch(
            train_data,
            training.batch_size,
            config.model.context_length,
            device,
        )
        logits, _ = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_evaluate = (
            step == 1
            or step % training.eval_interval == 0
            or step == steps
        )
        if should_evaluate:
            losses = estimate_loss(
                model,
                train_data,
                validation_data,
                training.batch_size,
                config.model.context_length,
                training.eval_batches,
                device,
            )
            print(
                f"step {step:5d} | train {losses['train']:.4f} | "
                f"validation {losses['validation']:.4f}"
            )
            if losses["validation"] < best_validation or step == steps:
                best_validation = min(best_validation, losses["validation"])
                save_checkpoint(
                    output_path,
                    model,
                    tokenizer,
                    config.to_dict(),
                    step,
                    losses["validation"],
                )

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()


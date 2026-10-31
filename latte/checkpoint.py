from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .model import TinyTransformer
from .tokenizer import CharTokenizer


def save_checkpoint(
    path: str | Path,
    model: TinyTransformer,
    tokenizer: CharTokenizer,
    project_config: dict[str, Any],
    step: int,
    validation_loss: float,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_config": model.architecture_dict(),
            "tokenizer": tokenizer.to_dict(),
            "project_config": project_config,
            "step": step,
            "validation_loss": validation_loss,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[TinyTransformer, CharTokenizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_model_config = dict(checkpoint["model_config"])
    vocab_size = raw_model_config.pop("vocab_size")
    model = TinyTransformer(ModelConfig(**raw_model_config), vocab_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    return model, tokenizer, checkpoint


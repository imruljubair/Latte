from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any
import json


def write_architecture_config(source: Path, destination: Path, code: str) -> None:
    """Apply the compact architecture notation to an otherwise unchanged config."""
    if len(code) not in (5, 6) or any(char not in '123456789' for char in code):
        raise ValueError('Architecture code must contain five or six non-zero digits')
    digits = list(map(int, code))
    if len(digits) == 5:
        dim, head, heads, layers, ffn = digits
        digits = [dim, head, head, heads, layers, ffn]
    with source.open('rb') as handle:
        raw = tomllib.load(handle)
    keys = ('model_dim', 'qk_dim', 'value_dim', 'num_heads', 'num_layers', 'ffn_dim')
    raw['model'].update(zip(keys, digits))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n\n'.join(
        f'[{section}]\n' + '\n'.join(f'{key} = {json.dumps(value)}' for key, value in values.items())
        for section, values in raw.items()
    ) + '\n', encoding='utf-8')


@dataclass(frozen=True)
class ModelConfig:
    model_dim: int
    qk_dim: int
    value_dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    context_length: int
    dropout: float = 0.0


@dataclass(frozen=True)
class DataConfig:
    path: str
    tokenizer: str = "char"
    train_split: float = 0.9


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    steps: int = 1500
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    eval_interval: int = 100
    eval_batches: int = 20
    seed: int = 1337
    device: str = "auto"


@dataclass(frozen=True)
class OutputConfig:
    directory: str
    checkpoint_name: str = "model.pt"


@dataclass(frozen=True)
class GenerationConfig:
    prompt: str = "Once upon a time"
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_k: int = 12


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    output: OutputConfig
    generation: GenerationConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_config(config: ProjectConfig) -> None:
    model = config.model
    for name in (
        "model_dim",
        "qk_dim",
        "value_dim",
        "num_heads",
        "num_layers",
        "ffn_dim",
        "context_length",
    ):
        _positive(f"model.{name}", getattr(model, name))
    if not 0.0 <= model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if config.data.tokenizer != "char":
        raise ValueError("Only the 'char' tokenizer is currently supported")
    if not 0.0 < config.data.train_split < 1.0:
        raise ValueError("data.train_split must be between 0 and 1")
    for name in ("batch_size", "steps", "eval_interval", "eval_batches"):
        _positive(f"training.{name}", getattr(config.training, name))
    if config.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if config.generation.max_new_tokens < 0:
        raise ValueError("generation.max_new_tokens cannot be negative")
    if config.generation.temperature < 0:
        raise ValueError("generation.temperature cannot be negative")
    if config.generation.top_k < 0:
        raise ValueError("generation.top_k cannot be negative")


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    config = ProjectConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw.get("training", {})),
        output=OutputConfig(**raw["output"]),
        generation=GenerationConfig(**raw.get("generation", {})),
    )
    validate_config(config)
    return config


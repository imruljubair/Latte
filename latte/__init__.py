"""Configurable, inspectable toy language model."""

from .config import ProjectConfig, load_config
from .model import TinyTransformer
from .tokenizer import CharTokenizer

__all__ = ["CharTokenizer", "ProjectConfig", "TinyTransformer", "load_config"]


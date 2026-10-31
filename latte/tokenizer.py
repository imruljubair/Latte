from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    chars: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Cannot build a tokenizer from empty text")
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {character: index for index, character in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        lookup = self.stoi
        unknown = sorted(set(text) - set(lookup))
        if unknown:
            rendered = ", ".join(repr(character) for character in unknown)
            raise ValueError(f"Text contains characters outside the vocabulary: {rendered}")
        return [lookup[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.chars[token_id] for token_id in token_ids)

    def to_dict(self) -> dict[str, list[str]]:
        return {"chars": self.chars}

    @classmethod
    def from_dict(cls, value: dict[str, list[str]]) -> "CharTokenizer":
        return cls(list(value["chars"]))


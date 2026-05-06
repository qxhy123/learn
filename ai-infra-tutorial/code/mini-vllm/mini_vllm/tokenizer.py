"""Thin wrapper around HF `tokenizers`. We use the GPT-2 BPE tokenizer for
toy_gpt (vocab_size 50257) and the Llama tokenizer for TinyLlama (Plan 3)."""
from __future__ import annotations
from typing import List
from tokenizers import Tokenizer


class TokenizerWrapper:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained_gpt2(cls) -> "TokenizerWrapper":
        # Loads the canonical GPT-2 tokenizer from the HF tokenizers cache.
        # Uses the lightweight `tokenizers` lib, NOT `transformers`.
        tk = Tokenizer.from_pretrained("gpt2")
        return cls(tk)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

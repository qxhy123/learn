"""Configuration dataclasses. Plain data, no logic."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model architecture parameters (filled by each model class)."""
    model_type: str               # "toy_gpt" | "llama"
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_kv_heads: int             # equals num_attention_heads for non-GQA
    head_dim: int
    max_position_embeddings: int
    intermediate_size: int        # FFN hidden dim
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    dtype: str = "float32"        # "float32" | "float16" | "bfloat16"


@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int = 256     # set by CacheEngine.profile in later plans
    num_cpu_blocks: int = 0       # Plan 6 wires this up


@dataclass
class SamplingParams:
    """Generation hyper-parameters. When `greedy=True`, the temperature/
    top_p/top_k fields are ignored. Plan 7 adds the non-greedy path."""
    max_tokens: int = 32
    greedy: bool = True
    stop_token_ids: tuple[int, ...] = ()
    # Non-greedy sampling (only used when greedy=False)
    temperature: float = 1.0       # 1.0 = no scaling; lower → sharper
    top_p: float = 1.0             # 1.0 = no nucleus filter
    top_k: int = 0                 # 0 = no top-k filter
    seed: int | None = None        # per-request seed for reproducibility


@dataclass
class EngineConfig:
    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    device: str = "cpu"           # "cpu" | "mps" | "cuda"
    seed: int = 0
    # Feature flags. Each plan flips its flag to True default upon completion.
    # Set to False to opt back into the prior baseline (for benchmarks).
    enable_continuous_batching: bool = True   # Plan 4
    enable_chunked_prefill: bool = True       # Plan 5
    enable_prefix_caching: bool = True        # Plan 5
    enable_swap: bool = True                  # Plan 6
    max_num_batched_tokens: int = 2048
    chunked_prefill_size: int = 512

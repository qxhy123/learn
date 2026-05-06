"""Owns the physical KV tensor pool. One key/value tensor pair per layer."""
from __future__ import annotations
from typing import List, Tuple
import torch

from mini_vllm.config import CacheConfig, ModelConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]  # (key_cache, value_cache) per layer


class CacheEngine:
    """Layout per layer:
        key_cache:   [num_blocks, num_kv_heads, head_dim, block_size]
        value_cache: [num_blocks, num_kv_heads, head_dim, block_size]
    The trailing block_size dim makes per-block contiguous reads efficient.
    """
    def __init__(self, model_cfg: ModelConfig, cache_cfg: CacheConfig,
                 device: str, dtype: torch.dtype):
        self.model_cfg = model_cfg
        self.cache_cfg = cache_cfg
        self.device = device
        self.dtype = dtype
        self.kv_caches: List[KVCache] = []
        for _ in range(model_cfg.num_hidden_layers):
            shape = (cache_cfg.num_gpu_blocks, model_cfg.num_kv_heads,
                     model_cfg.head_dim, cache_cfg.block_size)
            k = torch.zeros(shape, device=device, dtype=dtype)
            v = torch.zeros(shape, device=device, dtype=dtype)
            self.kv_caches.append((k, v))

    @property
    def num_layers(self) -> int:
        return len(self.kv_caches)

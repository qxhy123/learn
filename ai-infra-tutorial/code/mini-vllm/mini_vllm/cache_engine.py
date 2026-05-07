"""Owns the physical KV tensor pools.

Plan 1: GPU pool only (one (K, V) tensor pair per layer).
Plan 6: optional CPU pool for swap. Blocks can move between pools when
the GPU pool is over-subscribed; the data layout is identical.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import torch

from mini_vllm.config import CacheConfig, ModelConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]  # (key_cache, value_cache) per layer


class CacheEngine:
    """Per-layer GPU layout (and identical for the optional CPU pool):
        key_cache:   [num_blocks, num_kv_heads, head_dim, block_size]
        value_cache: [num_blocks, num_kv_heads, head_dim, block_size]
    """
    def __init__(self, model_cfg: ModelConfig, cache_cfg: CacheConfig,
                 device: str, dtype: torch.dtype):
        self.model_cfg = model_cfg
        self.cache_cfg = cache_cfg
        self.device = device
        self.dtype = dtype
        gpu_shape = (cache_cfg.num_gpu_blocks, model_cfg.num_kv_heads,
                     model_cfg.head_dim, cache_cfg.block_size)
        self.kv_caches: List[KVCache] = [
            (torch.zeros(gpu_shape, device=device, dtype=dtype),
             torch.zeros(gpu_shape, device=device, dtype=dtype))
            for _ in range(model_cfg.num_hidden_layers)
        ]
        # CPU swap pool (Plan 6). Always allocated on 'cpu' regardless of `device`.
        self.cpu_kv_caches: List[KVCache] = []
        if cache_cfg.num_cpu_blocks > 0:
            cpu_shape = (cache_cfg.num_cpu_blocks, model_cfg.num_kv_heads,
                         model_cfg.head_dim, cache_cfg.block_size)
            self.cpu_kv_caches = [
                (torch.zeros(cpu_shape, device='cpu', dtype=dtype),
                 torch.zeros(cpu_shape, device='cpu', dtype=dtype))
                for _ in range(model_cfg.num_hidden_layers)
            ]

    @property
    def num_layers(self) -> int:
        return len(self.kv_caches)

    def swap_out(self, mapping: Dict[int, int]) -> None:
        """Copy K/V tensor data from GPU pool to CPU pool.
        `mapping` is {gpu_block_id: cpu_block_id}.
        """
        if not mapping or not self.cpu_kv_caches:
            return
        for layer in range(self.num_layers):
            kc_g, vc_g = self.kv_caches[layer]
            kc_c, vc_c = self.cpu_kv_caches[layer]
            for g_id, c_id in mapping.items():
                kc_c[c_id].copy_(kc_g[g_id].to('cpu'))
                vc_c[c_id].copy_(vc_g[g_id].to('cpu'))

    def swap_in(self, mapping: Dict[int, int]) -> None:
        """Copy K/V tensor data from CPU pool back to GPU pool.
        `mapping` is {cpu_block_id: gpu_block_id}.
        """
        if not mapping or not self.cpu_kv_caches:
            return
        for layer in range(self.num_layers):
            kc_g, vc_g = self.kv_caches[layer]
            kc_c, vc_c = self.cpu_kv_caches[layer]
            for c_id, g_id in mapping.items():
                kc_g[g_id].copy_(kc_c[c_id].to(self.device))
                vc_g[g_id].copy_(vc_c[c_id].to(self.device))

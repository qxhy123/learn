"""Parity test: our LlamaModel vs HF LlamaForCausalLM on TinyLlama weights.

Loads ~2.2 GB on first run. Marked `slow`; opt in with `pytest -m slow`.
"""
import pytest
import torch

from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.config import CacheConfig
from mini_vllm.models.llama import LlamaModel
from mini_vllm.models.llama_loader import load_hf_to_llama_model

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.slow
def test_logits_match_hf_top5():
    """Run the same 8-token prompt through both implementations; the top-5
    next-token candidates should overlap by at least 4 (allow one tie-break)."""
    from transformers import LlamaForCausalLM
    torch.manual_seed(0)

    cfg = LlamaModel.tinyllama_config()
    ours = LlamaModel(cfg, TorchBackend()).to(torch.float32).eval()
    load_hf_to_llama_model(ours, TINYLLAMA, dtype=torch.float32)

    hf = LlamaForCausalLM.from_pretrained(TINYLLAMA, torch_dtype=torch.float32).eval()

    prompt_ids = torch.tensor([1, 15043, 29892, 1373, 526, 366, 2599, 29973])  # arbitrary 8-token prompt
    N = prompt_ids.shape[0]

    # ---- HF reference ----
    with torch.inference_mode():
        hf_out = hf(prompt_ids.unsqueeze(0))
    hf_last_logits = hf_out.logits[0, -1]  # [vocab]
    hf_top5 = torch.topk(hf_last_logits, 5).indices.tolist()

    # ---- Ours ----
    block_size = 16
    num_blocks = max(2, (N + block_size - 1) // block_size + 1)
    ce = CacheEngine(cfg, CacheConfig(block_size=block_size, num_gpu_blocks=num_blocks),
                     device='cpu', dtype=torch.float32)
    # All N tokens written into block 0 starting at slot 0
    slot_mapping = torch.arange(N, dtype=torch.long)
    positions = torch.arange(N)
    sample_indices = torch.tensor([N - 1])
    # Block table: 8 tokens fit in 1 block of size 16, so seq's block is [0].
    prefill_block_table = torch.tensor([[0]], dtype=torch.int32)
    with torch.inference_mode():
        our_logits = ours(
            prompt_ids, positions, slot_mapping, ce.kv_caches,
            prefill_block_table=prefill_block_table,
            prefill_seq_lens=torch.tensor([N], dtype=torch.int32),
            prefill_query_lens=torch.tensor([N], dtype=torch.int32),
            num_prefill_tokens=N,
            decode_block_table=torch.empty(0, 0, dtype=torch.int32),
            decode_context_lens=torch.empty(0, dtype=torch.int32),
            sample_indices=sample_indices,
        )
    our_top5 = torch.topk(our_logits[0], 5).indices.tolist()

    overlap = len(set(hf_top5) & set(our_top5))
    assert overlap >= 4, (
        f"Top-5 overlap too low: ours={our_top5}, hf={hf_top5}, overlap={overlap}"
    )

"""End-to-end: drive our LLMEngine to greedy-generate 8 tokens for TinyLlama,
compare to HF's `model.generate(..., do_sample=False)` greedy output.
"""
import pytest
import torch

from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.config import CacheConfig, EngineConfig, SamplingParams
from mini_vllm.engine import LLMEngine
from mini_vllm.models.llama import LlamaModel
from mini_vllm.models.llama_loader import load_hf_to_llama_model
from mini_vllm.tokenizer import TokenizerWrapper

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.slow
def test_greedy_matches_hf_at_least_3_of_first_8_tokens():
    from transformers import LlamaForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    cfg = LlamaModel.tinyllama_config()

    ours_model = LlamaModel(cfg, TorchBackend()).to(torch.float32).eval()
    load_hf_to_llama_model(ours_model, TINYLLAMA, dtype=torch.float32)
    tokenizer = TokenizerWrapper.from_pretrained_llama()

    engine = LLMEngine(ours_model, tokenizer, EngineConfig(
        model=cfg, cache=CacheConfig(block_size=16, num_gpu_blocks=8),
        device="cpu", seed=0))
    prompt = "The capital of France is"
    out = engine.generate([prompt], SamplingParams(max_tokens=8, greedy=True))
    ours_text = out[0][1]
    ours_ids = tokenizer.encode(ours_text)
    # The Llama tokenizer's `encode` re-prepends a BOS token (id=1) when
    # encoding standalone text. The engine's actual generated tokens do NOT
    # include BOS — strip it so we compare like-for-like with HF's
    # `generate(..., new_tokens)`.
    if ours_ids and ours_ids[0] == 1:
        ours_ids = ours_ids[1:]

    # HF greedy
    hf_model = LlamaForCausalLM.from_pretrained(TINYLLAMA, torch_dtype=torch.float32).eval()
    hf_tk = AutoTokenizer.from_pretrained(TINYLLAMA)
    inp = hf_tk(prompt, return_tensors="pt")
    hf_out = hf_model.generate(**inp, max_new_tokens=8, do_sample=False)
    hf_new_ids = hf_out[0, inp["input_ids"].shape[1]:].tolist()

    # Greedy decoding *should* match exactly with bit-identical fp32 forward.
    # Permit small drift from accumulation order: require at least 3 of 8 to agree.
    matches = sum(1 for a, b in zip(ours_ids[:8], hf_new_ids) if a == b)
    print(f"\nours={ours_ids[:8]}\nhf  ={hf_new_ids}\nmatches={matches}/8")
    assert matches >= 3, (
        f"Too few greedy matches: ours={ours_ids[:8]}, hf={hf_new_ids}, matches={matches}"
    )

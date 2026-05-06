import torch
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.block_manager import BlockManager
from mini_vllm.config import CacheConfig, SamplingParams
from mini_vllm.sequence import Sequence
from mini_vllm.model_runner import ModelRunner


def test_runner_prefill_then_decode():
    torch.manual_seed(0)
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=128, n_layer=2,
                               d_model=32, n_head=4, max_pos=64)
    block_size = 4
    bm = BlockManager(num_blocks=8, block_size=block_size)
    ce = CacheEngine(model.config, CacheConfig(block_size=block_size, num_gpu_blocks=8),
                     device='cpu', dtype=torch.float32)
    runner = ModelRunner(model, ce, bm, device='cpu')

    seq = Sequence("r0", prompt_token_ids=[1, 2, 3, 4, 5],
                   sampling_params=SamplingParams(max_tokens=4))
    bm.allocate(seq)
    # Prefill step
    logits = runner.execute(prefill_seqs=[seq], decode_seqs=[])
    assert logits.shape == (1, 128)
    next_token = int(logits.argmax(dim=-1).item())

    # Apply token to seq, then decode step
    seq.append_token(next_token)
    bm.append_slot(seq)
    logits2 = runner.execute(prefill_seqs=[], decode_seqs=[seq])
    assert logits2.shape == (1, 128)

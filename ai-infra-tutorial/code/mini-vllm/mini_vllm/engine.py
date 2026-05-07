"""Top-level orchestration loop."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch

from mini_vllm.block_manager import BlockManager
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.config import EngineConfig, SamplingParams
from mini_vllm.model_runner import ModelRunner
from mini_vllm.models.base import CausalLM
from mini_vllm.sampler import Sampler
from mini_vllm.scheduler import Scheduler
from mini_vllm.sequence import Sequence
from mini_vllm.tokenizer import TokenizerWrapper


@dataclass
class StepOutput:
    request_id: str
    new_token_id: int
    is_finished: bool


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class LLMEngine:
    def __init__(self, model: CausalLM, tokenizer: TokenizerWrapper, cfg: EngineConfig):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model = model
        dtype = _DTYPES[cfg.model.dtype]
        self.cache_engine = CacheEngine(cfg.model, cfg.cache, cfg.device, dtype)
        self.block_manager = BlockManager(
            cfg.cache.num_gpu_blocks, cfg.cache.block_size,
            num_cpu_blocks=cfg.cache.num_cpu_blocks,
            enable_prefix_caching=cfg.enable_prefix_caching)
        self.scheduler = Scheduler(
            self.block_manager,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            enable_continuous_batching=cfg.enable_continuous_batching,
            enable_chunked_prefill=cfg.enable_chunked_prefill,
            chunked_prefill_size=cfg.chunked_prefill_size,
            enable_swap=cfg.enable_swap,
        )
        self.runner = ModelRunner(model, self.cache_engine, self.block_manager, cfg.device)
        self.sampler = Sampler()
        self._next_request_id = 0
        torch.manual_seed(cfg.seed)

    # ---- public API ----
    def add_request(self, prompt: str, sampling: SamplingParams) -> str:
        rid = f"req-{self._next_request_id}"
        self._next_request_id += 1
        token_ids = self.tokenizer.encode(prompt)
        seq = Sequence(rid, token_ids, sampling)
        self.scheduler.add(seq)
        return rid

    def step(self) -> List[StepOutput]:
        sched = self.scheduler.schedule()
        # Apply swap copies BEFORE the forward pass so the K/V tensors are at
        # the right physical block ids by the time attention reads them.
        if sched.swap_out:
            self.cache_engine.swap_out(sched.swap_out)
        if sched.swap_in:
            self.cache_engine.swap_in(sched.swap_in)
        if not sched.prefill_seqs and not sched.decode_seqs:
            return []
        logits = self.runner.execute(sched.prefill_seqs, sched.decode_seqs)

        # Tokens are sampled only for seqs whose prefill COMPLETES this step
        # (chunk reaches end of prompt) plus all decode seqs. A seq still in
        # mid-chunk-prefill produces no logit row this step.
        sampled_seqs: List[Sequence] = [
            s for s in sched.prefill_seqs
            if s.num_prefilled + s.scheduled_chunk_len == s.num_prompt_tokens
        ]
        sampled_seqs.extend(sched.decode_seqs)

        # Advance prefill progress for ALL prefill seqs (whether they finished
        # this step or not). Then schedule sees them in the right state next step.
        for seq in sched.prefill_seqs:
            seq.num_prefilled += seq.scheduled_chunk_len
            # Register newly-filled blocks for prefix-cache reuse.
            self.block_manager.register_filled_blocks(seq)
        # Decode steps may also fill a block (when seq_len % block_size == 0).
        for seq in sched.decode_seqs:
            self.block_manager.register_filled_blocks(seq)

        outputs: List[StepOutput] = []
        if sampled_seqs:
            params = [s.sampling_params for s in sampled_seqs]
            tokens = self.sampler.sample(logits, params)
            for seq, tok in zip(sampled_seqs, tokens):
                seq.append_token(tok)
                outputs.append(StepOutput(seq.request_id, tok, seq.is_finished()))
        self.scheduler.free_finished()
        return outputs

    def generate(self, prompts: List[str], sampling: SamplingParams) -> List[Tuple[str, str]]:
        rids = [self.add_request(p, sampling) for p in prompts]
        outputs: dict[str, List[int]] = {rid: [] for rid in rids}
        while self.scheduler.has_unfinished():
            for so in self.step():
                outputs[so.request_id].append(so.new_token_id)
        return [(rid, self.tokenizer.decode(outputs[rid])) for rid in rids]

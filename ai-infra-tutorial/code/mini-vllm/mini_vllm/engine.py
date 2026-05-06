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
        self.block_manager = BlockManager(cfg.cache.num_gpu_blocks, cfg.cache.block_size)
        self.scheduler = Scheduler(self.block_manager)
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
        if not sched.prefill_seqs and not sched.decode_seqs:
            return []
        logits = self.runner.execute(sched.prefill_seqs, sched.decode_seqs)
        # Order in logits: prefill_seqs (one row each, last-position) then decode_seqs
        all_seqs = sched.prefill_seqs + sched.decode_seqs
        params = [s.sampling_params for s in all_seqs]
        tokens = self.sampler.sample(logits, params)

        outputs: List[StepOutput] = []
        for seq in sched.prefill_seqs:
            self.scheduler.mark_prefilled(seq)
        for seq, tok in zip(all_seqs, tokens):
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

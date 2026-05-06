"""Per-request mutable state."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

from mini_vllm.config import SamplingParams

if TYPE_CHECKING:
    from mini_vllm.block_manager import BlockTable


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"     # used by Plan 6
    FINISHED = "finished"


@dataclass
class Sequence:
    request_id: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: List[int] = field(default_factory=list)
    block_table: Optional["BlockTable"] = None
    # Number of prompt tokens already prefilled. In Plan 1-4 (no chunking)
    # it jumps from 0 to num_prompt_tokens in one step. In Plan 5 chunked
    # prefill it advances by `scheduled_chunk_len` per step.
    num_prefilled: int = 0
    # Set by the Scheduler each step: how many prompt tokens to prefill in
    # the current step. Equal to (num_prompt_tokens - num_prefilled) when
    # chunked prefill is OFF; capped to chunk_size when ON.
    scheduled_chunk_len: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def seq_len(self) -> int:
        return self.num_prompt_tokens + self.num_generated_tokens

    @property
    def token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        if self._should_finish(token_id):
            self.status = SequenceStatus.FINISHED

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def _should_finish(self, last_token: int) -> bool:
        if self.num_generated_tokens >= self.sampling_params.max_tokens:
            return True
        if last_token in self.sampling_params.stop_token_ids:
            return True
        return False

# Mini-vLLM Plan 1: Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a minimal end-to-end mini-vLLM engine that runs a toy GPT-2-style model on CPU/MPS via a Torch paged-attention backend, supporting batched prefill+decode under a naive FCFS scheduler. This is the foundation that all later phases (Triton kernel, TinyLlama, continuous batching, prefix caching, swap, streaming) build on.

**Architecture:** Layered (engine → scheduler → block_manager + model_runner → backend) with thin interfaces. Plan 1 implements the simplest correct version of every layer, deferring all advanced behavior to flags that are off (or absent) by default.

**Tech Stack:** Python 3.10+, PyTorch 2.x, `tokenizers` (HF), `pytest`. No Triton, no `transformers`, no GPU dependency in Plan 1.

**Out of scope (deferred):**
- Continuous batching, chunked prefill (Plan 4)
- Prefix caching / CoW (Plan 5)
- Swap to CPU / preemption (Plan 6)
- Triton backend (Plan 2)
- TinyLlama / HF weight loading (Plan 3)
- Streaming, top-p/top-k/temperature sampling (Plan 7)

**Plan 1 scheduler simplification:** at engine start, all requests added before `step()` are batched together; no admission mid-batch, no preemption. Mixed prefill+decode in one step **is** supported (this keeps the kernel/runner interfaces correct from day one — only the scheduler is "dumb"). Sampling is greedy only.

---

## File Structure

```
ai-infra-tutorial/code/mini-vllm/
├── pyproject.toml                      # Task 1
├── README.md                           # Task 1 (minimal)
├── mini_vllm/
│   ├── __init__.py                     # Task 1
│   ├── config.py                       # Task 2
│   ├── sequence.py                     # Task 3
│   ├── block_manager.py                # Task 4
│   ├── cache_engine.py                 # Task 5
│   ├── tokenizer.py                    # Task 11
│   ├── sampler.py                      # Task 12
│   ├── model_runner.py                 # Tasks 15-16
│   ├── scheduler.py                    # Task 17
│   ├── engine.py                       # Task 18
│   ├── backends/
│   │   ├── __init__.py                 # Task 6
│   │   ├── interface.py                # Task 6
│   │   ├── reference.py                # Task 7
│   │   └── torch_backend.py            # Tasks 8-10
│   └── models/
│       ├── __init__.py                 # Task 13
│       ├── base.py                     # Task 13
│       └── toy_gpt.py                  # Tasks 13-14
├── examples/
│   └── run_toy.py                      # Task 19
└── tests/
    ├── __init__.py                     # Task 1
    ├── test_block_manager.py           # Task 4
    ├── test_attention.py               # Tasks 8-10
    └── test_e2e.py                     # Task 20
```

**Responsibility per file** (each file does one thing):
- `config.py`: dataclasses only — no logic
- `sequence.py`: per-request state container — no logic except simple status transitions
- `block_manager.py`: physical block bookkeeping (alloc/free/append_slot, ref_count), unaware of model/scheduler
- `cache_engine.py`: owns the KV tensor pool, exposes `key_cache`/`value_cache` for backends
- `backends/*`: pure-functional attention kernels (no state)
- `models/*`: PyTorch nn.Modules; call into backends but unaware of scheduler/cache_engine
- `model_runner.py`: assembles `ModelInput` from `SchedulerOutput` and runs `model.forward`
- `scheduler.py`: state machine over `Sequence` queues
- `engine.py`: top-level orchestration loop
- `sampler.py`, `tokenizer.py`: thin utilities

---

## Tasks

### Task 1: Project skeleton

**Files:**
- Create: `code/mini-vllm/pyproject.toml`
- Create: `code/mini-vllm/README.md`
- Create: `code/mini-vllm/mini_vllm/__init__.py`
- Create: `code/mini-vllm/tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p code/mini-vllm/mini_vllm/{backends,models}
mkdir -p code/mini-vllm/{examples,tests}
```

- [ ] **Step 2: Write pyproject.toml**

Create `code/mini-vllm/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mini-vllm"
version = "0.1.0"
description = "Educational mini reimplementation of vLLM with PagedAttention"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "tokenizers>=0.15",
    "safetensors>=0.4",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-xdist>=3.5"]
triton = ["triton>=2.2"]

[tool.setuptools.packages.find]
include = ["mini_vllm*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

- [ ] **Step 3: Write README.md and __init__.py stubs**

Create `code/mini-vllm/README.md`:
```markdown
# mini-vLLM

Educational reimplementation of vLLM with PagedAttention. Companion code
to `part5-serving-infra/16a-lab-mini-vllm.md`.

## Install
    pip install -e ".[dev]"

## Quickstart
    python examples/run_toy.py
```

Create `code/mini-vllm/mini_vllm/__init__.py`:
```python
"""mini-vLLM: educational reimplementation of vLLM with PagedAttention."""
__version__ = "0.1.0"
```

Create `code/mini-vllm/tests/__init__.py` (empty file).

Create `code/mini-vllm/mini_vllm/backends/__init__.py` (empty file).

Create `code/mini-vllm/mini_vllm/models/__init__.py` (empty file).

- [ ] **Step 4: Verify install works**

Run from `code/mini-vllm/`: `pip install -e ".[dev]"`
Expected: completes without error; `python -c "import mini_vllm; print(mini_vllm.__version__)"` prints `0.1.0`.

- [ ] **Step 5: Commit**

```bash
cd code/mini-vllm
git add pyproject.toml README.md mini_vllm tests
git commit -m "mini-vllm: project skeleton"
```

---

### Task 2: Configuration dataclasses

**Files:**
- Create: `code/mini-vllm/mini_vllm/config.py`

- [ ] **Step 1: Write config.py**

```python
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
    """Plan 1 supports greedy only; later plans add temp/top-p/top-k."""
    max_tokens: int = 32
    greedy: bool = True
    stop_token_ids: tuple[int, ...] = ()


@dataclass
class EngineConfig:
    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    device: str = "cpu"           # "cpu" | "mps" | "cuda"
    seed: int = 0
    # Feature flags - all False/absent in Plan 1
    enable_continuous_batching: bool = False
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    enable_swap: bool = False
    max_num_batched_tokens: int = 2048
```

- [ ] **Step 2: Smoke import**

Run: `python -c "from mini_vllm.config import EngineConfig, ModelConfig, CacheConfig, SamplingParams; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/config.py
git commit -m "mini-vllm: config dataclasses"
```

---

### Task 3: Sequence & request state

**Files:**
- Create: `code/mini-vllm/mini_vllm/sequence.py`
- Create: `code/mini-vllm/tests/test_sequence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sequence.py`:
```python
from mini_vllm.sequence import Sequence, SequenceStatus
from mini_vllm.config import SamplingParams


def test_sequence_initial_state():
    seq = Sequence(request_id="r0", prompt_token_ids=[1, 2, 3, 4],
                   sampling_params=SamplingParams(max_tokens=8))
    assert seq.status == SequenceStatus.WAITING
    assert seq.num_prompt_tokens == 4
    assert seq.num_generated_tokens == 0
    assert seq.token_ids == [1, 2, 3, 4]
    assert not seq.is_finished()


def test_sequence_append_token():
    seq = Sequence(request_id="r0", prompt_token_ids=[1, 2],
                   sampling_params=SamplingParams(max_tokens=3))
    seq.append_token(99)
    assert seq.token_ids == [1, 2, 99]
    assert seq.num_generated_tokens == 1
    assert not seq.is_finished()
    seq.append_token(100)
    seq.append_token(101)
    assert seq.is_finished()  # max_tokens reached


def test_sequence_stop_token():
    seq = Sequence(request_id="r0", prompt_token_ids=[1],
                   sampling_params=SamplingParams(max_tokens=10, stop_token_ids=(7,)))
    seq.append_token(5)
    assert not seq.is_finished()
    seq.append_token(7)
    assert seq.is_finished()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_sequence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mini_vllm.sequence'`.

- [ ] **Step 3: Implement sequence.py**

```python
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
    # In Plan 4+: number of prompt tokens already prefilled (for chunked prefill).
    # In Plan 1 it equals num_prompt_tokens after the first prefill step.
    num_prefilled: int = 0

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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_sequence.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add mini_vllm/sequence.py tests/test_sequence.py
git commit -m "mini-vllm: Sequence + SequenceStatus"
```

---

### Task 4: BlockManager (basic alloc/free/append_slot)

**Files:**
- Create: `code/mini-vllm/mini_vllm/block_manager.py`
- Create: `code/mini-vllm/tests/test_block_manager.py`

Plan 1 BlockManager has NO prefix caching, NO swap, NO CoW. Only basic block allocation. Later plans extend.

- [ ] **Step 1: Write failing tests**

Create `tests/test_block_manager.py`:
```python
import pytest
from mini_vllm.block_manager import BlockManager, AllocStatus
from mini_vllm.sequence import Sequence
from mini_vllm.config import SamplingParams


def make_seq(rid: str, prompt_len: int) -> Sequence:
    return Sequence(request_id=rid, prompt_token_ids=list(range(prompt_len)),
                    sampling_params=SamplingParams(max_tokens=4))


def test_allocate_consumes_blocks():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=10)  # needs ceil(10/8) = 2 blocks
    assert bm.can_allocate(seq) == AllocStatus.OK
    bm.allocate(seq)
    assert seq.block_table is not None
    assert len(seq.block_table.physical_blocks) == 2
    assert bm.num_free_blocks == 2


def test_allocate_when_full():
    bm = BlockManager(num_blocks=2, block_size=8)
    seq = make_seq("r0", prompt_len=10)  # needs 2 blocks → fits exactly
    bm.allocate(seq)
    seq2 = make_seq("r1", prompt_len=4)  # needs 1, none free
    assert bm.can_allocate(seq2) == AllocStatus.LATER


def test_append_slot_within_last_block():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=4)  # 1 block, 4 slots used
    bm.allocate(seq)
    # simulate 3 more tokens — still in same block
    for _ in range(3):
        seq.output_token_ids.append(0)
        bm.append_slot(seq)
    assert len(seq.block_table.physical_blocks) == 1
    assert bm.num_free_blocks == 3


def test_append_slot_extends_blocks():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=8)  # 1 full block
    bm.allocate(seq)
    # next token forces a new block
    seq.output_token_ids.append(0)
    bm.append_slot(seq)
    assert len(seq.block_table.physical_blocks) == 2
    assert bm.num_free_blocks == 2


def test_free_returns_blocks():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=10)
    bm.allocate(seq)
    bm.free(seq)
    assert bm.num_free_blocks == 4
    assert seq.block_table is None


def test_invariant_no_block_leak_after_alloc_free_cycles():
    bm = BlockManager(num_blocks=8, block_size=4)
    for i in range(20):
        seq = make_seq(f"r{i}", prompt_len=5 + (i % 3))
        bm.allocate(seq)
        bm.free(seq)
    assert bm.num_free_blocks == 8


def test_slot_mapping_for_seq():
    bm = BlockManager(num_blocks=4, block_size=8)
    seq = make_seq("r0", prompt_len=10)
    bm.allocate(seq)
    # Slot mapping for the prompt: positions 0..9 → physical slots
    mapping = bm.get_slot_mapping(seq, start=0, end=10)
    assert len(mapping) == 10
    # Positions 0..7 map to block0; 8..9 to block1
    block_ids = [pb.block_id for pb in seq.block_table.physical_blocks]
    assert mapping[0] == block_ids[0] * 8 + 0
    assert mapping[7] == block_ids[0] * 8 + 7
    assert mapping[8] == block_ids[1] * 8 + 0
    assert mapping[9] == block_ids[1] * 8 + 1
```

- [ ] **Step 2: Run, expect ImportError**

Run: `pytest tests/test_block_manager.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement block_manager.py**

```python
"""Physical KV-block bookkeeping. Plan 1: basic alloc/free/append.
Prefix caching (ref_count > 1, hash chain), swap, CoW are added in later plans
but the data model already accommodates them (ref_count, device fields).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mini_vllm.sequence import Sequence


class AllocStatus(Enum):
    OK = "ok"
    LATER = "later"      # not enough blocks now, retry later
    NEVER = "never"      # request larger than total capacity


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 1
    block_hash: Optional[int] = None     # Plan 5 fills this
    device: str = "gpu"                  # Plan 6 toggles to "cpu"


@dataclass
class BlockTable:
    physical_blocks: List[PhysicalBlock] = field(default_factory=list)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Free list: stack of available block_ids (LIFO is fine)
        self._free_block_ids: List[int] = list(range(num_blocks))
        # All allocated blocks indexed by id (so we can lookup by id later)
        self._all_blocks: dict[int, PhysicalBlock] = {}

    # ---- query ----
    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    def can_allocate(self, seq: "Sequence") -> AllocStatus:
        needed = self._num_blocks_needed(seq.num_prompt_tokens)
        if needed > self.num_blocks:
            return AllocStatus.NEVER
        if needed > self.num_free_blocks:
            return AllocStatus.LATER
        return AllocStatus.OK

    # ---- mutate ----
    def allocate(self, seq: "Sequence") -> BlockTable:
        needed = self._num_blocks_needed(seq.num_prompt_tokens)
        blocks = [self._take_free_block() for _ in range(needed)]
        seq.block_table = BlockTable(physical_blocks=blocks)
        return seq.block_table

    def append_slot(self, seq: "Sequence") -> None:
        """Called per generated token. Allocates a new block iff the last
        block is full."""
        assert seq.block_table is not None, "sequence not allocated"
        used_slots = seq.seq_len  # prompt + generated tokens already counted
        needed = self._num_blocks_needed(used_slots)
        have = len(seq.block_table.physical_blocks)
        if needed > have:
            assert needed == have + 1, "append_slot extends by exactly one block"
            seq.block_table.physical_blocks.append(self._take_free_block())

    def free(self, seq: "Sequence") -> None:
        if seq.block_table is None:
            return
        for pb in seq.block_table.physical_blocks:
            pb.ref_count -= 1
            if pb.ref_count == 0:
                self._free_block_ids.append(pb.block_id)
                del self._all_blocks[pb.block_id]
        seq.block_table = None

    def get_slot_mapping(self, seq: "Sequence", start: int, end: int) -> List[int]:
        """Return physical slot ids for token positions [start, end) within seq."""
        mapping: List[int] = []
        bs = self.block_size
        for pos in range(start, end):
            block_idx = pos // bs
            offset = pos % bs
            pb = seq.block_table.physical_blocks[block_idx]
            mapping.append(pb.block_id * bs + offset)
        return mapping

    # ---- helpers ----
    def _num_blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def _take_free_block(self) -> PhysicalBlock:
        block_id = self._free_block_ids.pop()
        pb = PhysicalBlock(block_id=block_id, ref_count=1)
        self._all_blocks[block_id] = pb
        return pb
```

- [ ] **Step 4: Run tests, verify all pass**

Run: `pytest tests/test_block_manager.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add mini_vllm/block_manager.py tests/test_block_manager.py
git commit -m "mini-vllm: BlockManager basic alloc/free/append_slot"
```

---

### Task 5: CacheEngine (KV tensor pool)

**Files:**
- Create: `code/mini-vllm/mini_vllm/cache_engine.py`

- [ ] **Step 1: Implement cache_engine.py**

```python
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
```

- [ ] **Step 2: Smoke import**

Run:
```python
python -c "
import torch
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.config import CacheConfig, ModelConfig
mc = ModelConfig(model_type='toy_gpt', vocab_size=100, hidden_size=32,
    num_hidden_layers=2, num_attention_heads=4, num_kv_heads=4, head_dim=8,
    max_position_embeddings=64, intermediate_size=64)
ce = CacheEngine(mc, CacheConfig(block_size=4, num_gpu_blocks=8), 'cpu', torch.float32)
assert ce.kv_caches[0][0].shape == (8, 4, 8, 4)
print('ok')
"
```
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/cache_engine.py
git commit -m "mini-vllm: CacheEngine KV pool"
```

---

### Task 6: AttentionBackend interface

**Files:**
- Create: `code/mini-vllm/mini_vllm/backends/interface.py`

- [ ] **Step 1: Write interface.py**

```python
"""AttentionBackend protocol. All backends (torch, triton) implement this."""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class AttentionBackend(Protocol):
    def reshape_and_cache(
        self,
        key: torch.Tensor,         # [num_tokens, num_kv_heads, head_dim]
        value: torch.Tensor,       # same
        key_cache: torch.Tensor,   # [num_blocks, num_kv_heads, head_dim, block_size]
        value_cache: torch.Tensor, # same
        slot_mapping: torch.Tensor # [num_tokens] int64; block_id*block_size + offset
    ) -> None: ...

    def prefill(
        self,
        q: torch.Tensor,           # [num_prefill_tokens, num_heads, head_dim]
        k: torch.Tensor,           # [num_prefill_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,           # same
        seq_lens: torch.Tensor,    # [batch] full ctx len after this prefill
        query_lens: torch.Tensor,  # [batch] tokens being prefilled this step
        scale: float,
    ) -> torch.Tensor:             # [num_prefill_tokens, num_heads, head_dim]
        ...

    def decode(
        self,
        q: torch.Tensor,           # [batch, num_heads, head_dim]
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor, # [batch, max_blocks] int32
        context_lens: torch.Tensor, # [batch] int32 — kv length to attend over
        scale: float,
    ) -> torch.Tensor:             # [batch, num_heads, head_dim]
        ...
```

Plan 1 `prefill` does NOT read from KV cache (no prefix caching yet); it computes attention purely from current-step K/V. Plan 5 will add a `block_table` parameter for prefix-cache hits.

- [ ] **Step 2: Smoke import**

Run: `python -c "from mini_vllm.backends.interface import AttentionBackend; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/backends/interface.py
git commit -m "mini-vllm: AttentionBackend protocol"
```

---

### Task 7: Reference attention (golden, for tests only)

**Files:**
- Create: `code/mini-vllm/mini_vllm/backends/reference.py`

- [ ] **Step 1: Write reference.py**

```python
"""Naive 'gold-standard' attention used only by tests. Materializes the full
KV from block_table back into a contiguous tensor and calls SDPA. Slow but
trivially correct. Backends are validated against this."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def reference_decode(
    q: torch.Tensor,           # [B, H, D]
    key_cache: torch.Tensor,   # [num_blocks, H_kv, D, block_size]
    value_cache: torch.Tensor, # same
    block_table: torch.Tensor, # [B, max_blocks] int
    context_lens: torch.Tensor,# [B] int
    scale: float,
) -> torch.Tensor:
    B, H, D = q.shape
    H_kv = key_cache.shape[1]
    block_size = key_cache.shape[3]
    assert H % H_kv == 0
    group = H // H_kv

    out = torch.zeros_like(q)
    for b in range(B):
        ctx = int(context_lens[b].item())
        # Gather K/V for the seq into [ctx, H_kv, D]
        k_list, v_list = [], []
        remaining = ctx
        for blk_idx in range(block_table.shape[1]):
            if remaining <= 0:
                break
            block_id = int(block_table[b, blk_idx].item())
            take = min(block_size, remaining)
            # key_cache[block_id]: [H_kv, D, block_size] -> [block_size, H_kv, D]
            k_blk = key_cache[block_id, :, :, :take].permute(2, 0, 1)
            v_blk = value_cache[block_id, :, :, :take].permute(2, 0, 1)
            k_list.append(k_blk)
            v_list.append(v_blk)
            remaining -= take
        K = torch.cat(k_list, dim=0)  # [ctx, H_kv, D]
        V = torch.cat(v_list, dim=0)
        # Broadcast K/V across query heads in each group
        K = K.repeat_interleave(group, dim=1)  # [ctx, H, D]
        V = V.repeat_interleave(group, dim=1)
        # q[b]: [H, D], K: [ctx, H, D] -> scores [H, ctx]
        scores = torch.einsum("hd,thd->ht", q[b], K) * scale
        attn = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("ht,thd->hd", attn, V)
    return out


def reference_prefill(
    q: torch.Tensor,           # [N, H, D]
    k: torch.Tensor,           # [N, H_kv, D]
    v: torch.Tensor,           # [N, H_kv, D]
    seq_lens: torch.Tensor,    # [B]
    query_lens: torch.Tensor,  # [B]   for Plan 1 query_lens == seq_lens
    scale: float,
) -> torch.Tensor:
    """Causal attention within each sequence; sequences are independent."""
    H = q.shape[1]
    H_kv = k.shape[1]
    group = H // H_kv
    out = torch.zeros_like(q)
    cursor = 0
    for b in range(len(seq_lens)):
        n = int(query_lens[b].item())
        qb = q[cursor:cursor + n]                # [n, H, D]
        kb = k[cursor:cursor + n]                # [n, H_kv, D]
        vb = v[cursor:cursor + n]
        kb = kb.repeat_interleave(group, dim=1)  # [n, H, D]
        vb = vb.repeat_interleave(group, dim=1)
        # SDPA wants [B=1, H, T, D]
        ob = F.scaled_dot_product_attention(
            qb.transpose(0, 1).unsqueeze(0),
            kb.transpose(0, 1).unsqueeze(0),
            vb.transpose(0, 1).unsqueeze(0),
            is_causal=True, scale=scale,
        )  # [1, H, n, D]
        out[cursor:cursor + n] = ob.squeeze(0).transpose(0, 1)
        cursor += n
    return out
```

- [ ] **Step 2: Smoke run**

Run:
```python
python -c "
import torch
from mini_vllm.backends.reference import reference_prefill, reference_decode
torch.manual_seed(0)
q = torch.randn(4, 4, 8); k = torch.randn(4, 4, 8); v = torch.randn(4, 4, 8)
out = reference_prefill(q, k, v, torch.tensor([4]), torch.tensor([4]), 1/8**0.5)
assert out.shape == (4, 4, 8)
print('prefill ok')
kc = torch.randn(8, 4, 8, 4); vc = torch.randn(8, 4, 8, 4)
qd = torch.randn(2, 4, 8)
bt = torch.tensor([[0,1],[2,3]])
out = reference_decode(qd, kc, vc, bt, torch.tensor([6, 5]), 1/8**0.5)
assert out.shape == (2, 4, 8)
print('decode ok')
"
```
Expected: prints `prefill ok` then `decode ok`.

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/backends/reference.py
git commit -m "mini-vllm: reference attention (golden for tests)"
```

---

### Task 8: Torch backend — reshape_and_cache

**Files:**
- Create: `code/mini-vllm/mini_vllm/backends/torch_backend.py`
- Create: `code/mini-vllm/tests/test_attention.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_attention.py`:
```python
import torch
from mini_vllm.backends.torch_backend import TorchBackend


def test_reshape_and_cache_writes_correct_slots():
    block_size = 4
    num_blocks = 4
    H_kv = 2
    D = 8
    kc = torch.zeros(num_blocks, H_kv, D, block_size)
    vc = torch.zeros(num_blocks, H_kv, D, block_size)
    # Two tokens: token 0 -> slot 5 (block 1, offset 1); token 1 -> slot 11 (block 2, offset 3)
    key = torch.randn(2, H_kv, D)
    val = torch.randn(2, H_kv, D)
    slot_mapping = torch.tensor([5, 11], dtype=torch.long)
    backend = TorchBackend()
    backend.reshape_and_cache(key, val, kc, vc, slot_mapping)
    # block 1 offset 1
    assert torch.allclose(kc[1, :, :, 1], key[0])
    assert torch.allclose(vc[1, :, :, 1], val[0])
    # block 2 offset 3
    assert torch.allclose(kc[2, :, :, 3], key[1])
    assert torch.allclose(vc[2, :, :, 3], val[1])
    # Other slots untouched
    assert (kc[0] == 0).all()
    assert (kc[3] == 0).all()
```

- [ ] **Step 2: Run test, expect ImportError**

Run: `pytest tests/test_attention.py::test_reshape_and_cache_writes_correct_slots -v`
Expected: FAIL — `TorchBackend` not found.

- [ ] **Step 3: Implement TorchBackend.reshape_and_cache**

Create `mini_vllm/backends/torch_backend.py`:
```python
"""CPU/MPS reference implementation. Slow but correct; primary use is to
validate the Triton backend (Plan 2) and to serve as the only backend on
machines without CUDA."""
from __future__ import annotations
import torch
import torch.nn.functional as F

from mini_vllm.backends.reference import reference_decode, reference_prefill


class TorchBackend:
    def reshape_and_cache(self, key, value, key_cache, value_cache, slot_mapping):
        """Scatter `key`/`value` (shape [N, H_kv, D]) into the paged cache at
        positions given by `slot_mapping` (shape [N], values = block_id*block_size + offset).
        Cache layout: [num_blocks, H_kv, D, block_size].
        """
        block_size = key_cache.shape[3]
        block_ids = (slot_mapping // block_size).long()
        offsets = (slot_mapping % block_size).long()
        # key: [N, H_kv, D] -> we want to write key[n] into key_cache[block_ids[n], :, :, offsets[n]]
        for n in range(slot_mapping.shape[0]):
            key_cache[block_ids[n], :, :, offsets[n]] = key[n]
            value_cache[block_ids[n], :, :, offsets[n]] = value[n]

    def prefill(self, q, k, v, seq_lens, query_lens, scale):
        # In Plan 1, prefill computes attention on the fresh K/V only (no prefix cache).
        return reference_prefill(q, k, v, seq_lens, query_lens, scale)

    def decode(self, q, key_cache, value_cache, block_table, context_lens, scale):
        # Plan 1 uses the reference implementation as the production decode kernel.
        # Plan 2 will replace this with a Triton kernel; the Torch backend will keep
        # using `reference_decode` as its (slow) fallback path.
        return reference_decode(q, key_cache, value_cache, block_table, context_lens, scale)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_attention.py::test_reshape_and_cache_writes_correct_slots -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_vllm/backends/torch_backend.py tests/test_attention.py
git commit -m "mini-vllm: TorchBackend.reshape_and_cache"
```

---

### Task 9: Torch backend — decode (validate against reference)

**Files:**
- Modify: `code/mini-vllm/tests/test_attention.py`

- [ ] **Step 1: Add failing test for decode**

Append to `tests/test_attention.py`:
```python
from mini_vllm.backends.reference import reference_decode


def test_torch_decode_matches_reference():
    torch.manual_seed(0)
    B, H, D, H_kv = 3, 8, 16, 2  # GQA: 4 query heads per kv head
    block_size = 4
    num_blocks = 16
    max_blocks_per_seq = 4
    kc = torch.randn(num_blocks, H_kv, D, block_size)
    vc = torch.randn(num_blocks, H_kv, D, block_size)
    q = torch.randn(B, H, D)
    block_table = torch.tensor([
        [0, 1, 2, 3],
        [4, 5, 6, 0],
        [7, 8, 0, 0],
    ], dtype=torch.int32)
    context_lens = torch.tensor([13, 10, 6], dtype=torch.int32)
    scale = D ** -0.5
    backend = TorchBackend()
    out = backend.decode(q, kc, vc, block_table, context_lens, scale)
    ref = reference_decode(q, kc, vc, block_table, context_lens, scale)
    assert torch.allclose(out, ref, atol=1e-5)
```

- [ ] **Step 2: Run, verify pass (TorchBackend.decode IS reference_decode)**

Run: `pytest tests/test_attention.py::test_torch_decode_matches_reference -v`
Expected: PASS.

(This test exists to lock the contract: when Plan 2 swaps in a Triton kernel, this same test pattern is reused with `atol=1e-2` for fp16.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_attention.py
git commit -m "mini-vllm: torch decode contract test"
```

---

### Task 10: Torch backend — prefill (validate against reference)

**Files:**
- Modify: `code/mini-vllm/tests/test_attention.py`

- [ ] **Step 1: Add failing test for prefill**

Append to `tests/test_attention.py`:
```python
from mini_vllm.backends.reference import reference_prefill


def test_torch_prefill_matches_reference_and_is_causal():
    torch.manual_seed(0)
    H, D, H_kv = 8, 16, 2
    # Two seqs, lengths 5 and 3
    seq_lens = torch.tensor([5, 3])
    query_lens = torch.tensor([5, 3])
    N = 8
    q = torch.randn(N, H, D)
    k = torch.randn(N, H_kv, D)
    v = torch.randn(N, H_kv, D)
    scale = D ** -0.5
    out = TorchBackend().prefill(q, k, v, seq_lens, query_lens, scale)
    ref = reference_prefill(q, k, v, seq_lens, query_lens, scale)
    assert torch.allclose(out, ref, atol=1e-5)
    # Causal sanity: position 0 of seq 0 attends only to itself
    # Easy check: re-run with k/v of pos>0 zeroed out and result for pos 0 unchanged
    k2 = k.clone(); v2 = v.clone()
    k2[1:5] = 0; v2[1:5] = 0  # zero out future positions of seq 0
    out2 = TorchBackend().prefill(q, k2, v2, seq_lens, query_lens, scale)
    assert torch.allclose(out[0], out2[0], atol=1e-5)
```

- [ ] **Step 2: Run, verify pass**

Run: `pytest tests/test_attention.py::test_torch_prefill_matches_reference_and_is_causal -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_attention.py
git commit -m "mini-vllm: torch prefill causal test"
```

---

### Task 11: Tokenizer wrapper

**Files:**
- Create: `code/mini-vllm/mini_vllm/tokenizer.py`

- [ ] **Step 1: Write tokenizer.py**

```python
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
```

- [ ] **Step 2: Smoke run**

Run:
```python
python -c "
from mini_vllm.tokenizer import TokenizerWrapper
tk = TokenizerWrapper.from_pretrained_gpt2()
ids = tk.encode('hello world')
assert tk.decode(ids).strip() == 'hello world'
assert tk.vocab_size == 50257
print('ok')
"
```
Expected: prints `ok`. (First run downloads tokenizer.)

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/tokenizer.py
git commit -m "mini-vllm: GPT-2 tokenizer wrapper"
```

---

### Task 12: Sampler (greedy only, for Plan 1)

**Files:**
- Create: `code/mini-vllm/mini_vllm/sampler.py`
- Create: `code/mini-vllm/tests/test_sampler.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sampler.py`:
```python
import torch
from mini_vllm.sampler import Sampler
from mini_vllm.config import SamplingParams


def test_greedy_argmax():
    sampler = Sampler()
    logits = torch.tensor([[0.1, 5.0, 0.3], [2.0, 0.5, 1.0]])
    params = [SamplingParams(greedy=True), SamplingParams(greedy=True)]
    out = sampler.sample(logits, params)
    assert out == [1, 0]
```

- [ ] **Step 2: Run, expect ImportError**

Run: `pytest tests/test_sampler.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement sampler.py (greedy only — Plan 7 extends)**

```python
"""Token sampler. Plan 1 implements greedy only.
Plan 7 will add temperature, top-p, top-k."""
from __future__ import annotations
from typing import List
import torch

from mini_vllm.config import SamplingParams


class Sampler:
    def sample(self, logits: torch.Tensor, params: List[SamplingParams]) -> List[int]:
        """logits: [B, vocab]. Returns one token id per row."""
        assert logits.dim() == 2
        assert all(p.greedy for p in params), "Plan 1 supports greedy only"
        return logits.argmax(dim=-1).tolist()
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_sampler.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_vllm/sampler.py tests/test_sampler.py
git commit -m "mini-vllm: greedy sampler"
```

---

### Task 13: Model base + toy GPT layers

**Files:**
- Create: `code/mini-vllm/mini_vllm/models/base.py`
- Create: `code/mini-vllm/mini_vllm/models/toy_gpt.py`

- [ ] **Step 1: Write models/base.py**

```python
"""Model interface. Engine/runner only see this."""
from __future__ import annotations
from typing import Protocol, List, Tuple, runtime_checkable
import torch

from mini_vllm.config import ModelConfig

KVTensorPair = Tuple[torch.Tensor, torch.Tensor]


@runtime_checkable
class CausalLM(Protocol):
    config: ModelConfig
    def forward(
        self,
        input_ids: torch.Tensor,           # [N_total]
        positions: torch.Tensor,           # [N_total]
        slot_mapping: torch.Tensor,        # [N_total]
        kv_caches: List[KVTensorPair],
        # prefill block
        prefill_seq_lens: torch.Tensor,    # [B_pre]
        prefill_query_lens: torch.Tensor,  # [B_pre]
        num_prefill_tokens: int,
        # decode block
        decode_block_table: torch.Tensor,  # [B_dec, max_blocks]
        decode_context_lens: torch.Tensor, # [B_dec]
    ) -> torch.Tensor:                     # [B_pre + B_dec, vocab]   (one logit row per seq sampled position)
        ...
```

- [ ] **Step 2: Write models/toy_gpt.py (GPT-2 style: LayerNorm, MHA, GELU MLP, learned positional emb)**

```python
"""Toy GPT-2-style decoder-only LM used for tests, examples, and the lab
chapter's first walkthrough. Random-initialized weights — outputs are
gibberish but the engine plumbing is exercised end-to-end."""
from __future__ import annotations
from typing import List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_vllm.config import ModelConfig
from mini_vllm.backends.interface import AttentionBackend


class ToyAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.cfg = cfg
        self.backend = backend
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        # Fused QKV
        self.qkv_proj = nn.Linear(cfg.hidden_size,
            (cfg.num_attention_heads + 2 * cfg.num_kv_heads) * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim,
                                cfg.hidden_size, bias=True)

    def forward(self, x, slot_mapping, kv_cache,
                prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens):
        N = x.shape[0]
        qkv = self.qkv_proj(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_kv_heads, self.head_dim)
        v = v.view(N, self.num_kv_heads, self.head_dim)

        # Write all current K/V into the paged cache.
        kc, vc = kv_cache
        self.backend.reshape_and_cache(k, v, kc, vc, slot_mapping)

        out_pre = None
        out_dec = None
        if num_prefill_tokens > 0:
            out_pre = self.backend.prefill(
                q[:num_prefill_tokens], k[:num_prefill_tokens], v[:num_prefill_tokens],
                prefill_seq_lens, prefill_query_lens, self.scale)
        if N - num_prefill_tokens > 0:
            qd = q[num_prefill_tokens:]   # [B_dec, H, D]
            out_dec = self.backend.decode(
                qd, kc, vc, decode_block_table, decode_context_lens, self.scale)

        if out_pre is not None and out_dec is not None:
            out = torch.cat([out_pre, out_dec], dim=0)
        else:
            out = out_pre if out_pre is not None else out_dec
        out = out.reshape(N, self.num_heads * self.head_dim)
        return self.o_proj(out)


class ToyMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class ToyBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = ToyAttention(cfg, backend)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = ToyMLP(cfg)

    def forward(self, x, slot_mapping, kv_cache,
                prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens):
        h = self.attn(self.ln1(x), slot_mapping, kv_cache,
                      prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                      decode_block_table, decode_context_lens)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x


class ToyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend):
        super().__init__()
        self.config = cfg
        self.backend = backend
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.blocks = nn.ModuleList([ToyBlock(cfg, backend) for _ in range(cfg.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids, positions, slot_mapping, kv_caches,
                prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                decode_block_table, decode_context_lens,
                sample_indices: torch.Tensor):
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for i, blk in enumerate(self.blocks):
            x = blk(x, slot_mapping, kv_caches[i],
                    prefill_seq_lens, prefill_query_lens, num_prefill_tokens,
                    decode_block_table, decode_context_lens)
        x = self.ln_f(x)
        # Only compute logits at the positions we actually need to sample from
        x_sample = x[sample_indices]
        if self.lm_head is None:
            logits = x_sample @ self.tok_emb.weight.T
        else:
            logits = self.lm_head(x_sample)
        return logits

    @classmethod
    def random_init(cls, backend: AttentionBackend,
                    vocab_size: int = 50257,
                    n_layer: int = 6, d_model: int = 384, n_head: int = 6,
                    max_pos: int = 1024, dtype: torch.dtype = torch.float32,
                    device: str = "cpu", seed: int = 0) -> "ToyGPT":
        torch.manual_seed(seed)
        cfg = ModelConfig(
            model_type="toy_gpt", vocab_size=vocab_size, hidden_size=d_model,
            num_hidden_layers=n_layer, num_attention_heads=n_head, num_kv_heads=n_head,
            head_dim=d_model // n_head, max_position_embeddings=max_pos,
            intermediate_size=4 * d_model, dtype=str(dtype).split('.')[-1],
        )
        m = cls(cfg, backend).to(device=device, dtype=dtype)
        return m
```

- [ ] **Step 3: Smoke instantiate**

Run:
```python
python -c "
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.models.toy_gpt import ToyGPT
m = ToyGPT.random_init(TorchBackend(), vocab_size=128, n_layer=2, d_model=32, n_head=4, max_pos=64)
n = sum(p.numel() for p in m.parameters())
print(f'ok params={n}')
"
```
Expected: prints something like `ok params=29952`.

- [ ] **Step 4: Commit**

```bash
git add mini_vllm/models/base.py mini_vllm/models/toy_gpt.py
git commit -m "mini-vllm: ToyGPT model + CausalLM protocol"
```

---

### Task 14: ToyGPT one-shot forward smoke test

**Files:**
- Create: `code/mini-vllm/tests/test_toy_gpt.py`

- [ ] **Step 1: Write smoke test**

Create `tests/test_toy_gpt.py`:
```python
import torch
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.config import CacheConfig


def test_toy_gpt_prefill_only_forward():
    torch.manual_seed(0)
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=128, n_layer=2,
                               d_model=32, n_head=4, max_pos=64)
    ce = CacheEngine(model.config, CacheConfig(block_size=4, num_gpu_blocks=8),
                     device='cpu', dtype=torch.float32)

    # One sequence, prefill of length 5
    N = 5
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    positions = torch.arange(N)
    # All five tokens go to the first 5 slots of block 0
    slot_mapping = torch.arange(N, dtype=torch.long)
    sample_indices = torch.tensor([N - 1])  # only sample the last position

    logits = model(
        input_ids, positions, slot_mapping, ce.kv_caches,
        prefill_seq_lens=torch.tensor([N]),
        prefill_query_lens=torch.tensor([N]),
        num_prefill_tokens=N,
        decode_block_table=torch.empty(0, 0, dtype=torch.int32),
        decode_context_lens=torch.empty(0, dtype=torch.int32),
        sample_indices=sample_indices,
    )
    assert logits.shape == (1, 128)
    # KV cache should be populated at slots 0..4
    assert (ce.kv_caches[0][0][0, :, :, :5] != 0).any()
    assert (ce.kv_caches[0][0][0, :, :, 5:] == 0).all()
```

- [ ] **Step 2: Run, verify pass**

Run: `pytest tests/test_toy_gpt.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_toy_gpt.py
git commit -m "mini-vllm: ToyGPT prefill-only smoke test"
```

---

### Task 15: ModelRunner — input batching

**Files:**
- Create: `code/mini-vllm/mini_vllm/model_runner.py`

- [ ] **Step 1: Implement model_runner.py — ModelInput build only (forward in next task)**

```python
"""Builds tensors from scheduler output and runs the model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch

from mini_vllm.sequence import Sequence
from mini_vllm.block_manager import BlockManager
from mini_vllm.cache_engine import CacheEngine
from mini_vllm.models.base import CausalLM


@dataclass
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    prefill_seq_lens: torch.Tensor
    prefill_query_lens: torch.Tensor
    num_prefill_tokens: int
    decode_block_table: torch.Tensor
    decode_context_lens: torch.Tensor
    sample_indices: torch.Tensor


class ModelRunner:
    def __init__(self, model: CausalLM, cache_engine: CacheEngine,
                 block_manager: BlockManager, device: str):
        self.model = model
        self.cache_engine = cache_engine
        self.block_manager = block_manager
        self.device = device

    def build_input(self, prefill_seqs: List[Sequence],
                    decode_seqs: List[Sequence]) -> ModelInput:
        bs = self.block_manager.block_size
        input_ids: List[int] = []
        positions: List[int] = []
        slot_mapping: List[int] = []
        prefill_seq_lens: List[int] = []
        prefill_query_lens: List[int] = []
        sample_indices: List[int] = []
        cursor = 0

        # Prefill region first
        for seq in prefill_seqs:
            # Plan 1: prefill the full prompt in one shot.
            n = seq.num_prompt_tokens
            input_ids.extend(seq.prompt_token_ids)
            positions.extend(range(n))
            slot_mapping.extend(self.block_manager.get_slot_mapping(seq, 0, n))
            prefill_seq_lens.append(n)
            prefill_query_lens.append(n)
            sample_indices.append(cursor + n - 1)  # last position of this seq
            cursor += n
        num_prefill_tokens = cursor

        # Decode region
        max_blocks = max((len(s.block_table.physical_blocks) for s in decode_seqs),
                        default=0)
        decode_block_table_list: List[List[int]] = []
        decode_context_lens: List[int] = []
        for seq in decode_seqs:
            pos = seq.seq_len - 1   # the new (just-written) token's position
            input_ids.append(seq.token_ids[-1])
            positions.append(pos)
            slot_mapping.extend(self.block_manager.get_slot_mapping(seq, pos, pos + 1))
            ids = [pb.block_id for pb in seq.block_table.physical_blocks]
            ids = ids + [0] * (max_blocks - len(ids))   # pad with 0 (won't be read)
            decode_block_table_list.append(ids)
            decode_context_lens.append(seq.seq_len)     # K/V len = full ctx incl. just-written
            sample_indices.append(cursor)
            cursor += 1

        dev = self.device
        return ModelInput(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=dev),
            positions=torch.tensor(positions, dtype=torch.long, device=dev),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=dev),
            prefill_seq_lens=torch.tensor(prefill_seq_lens, dtype=torch.int32, device=dev),
            prefill_query_lens=torch.tensor(prefill_query_lens, dtype=torch.int32, device=dev),
            num_prefill_tokens=num_prefill_tokens,
            decode_block_table=torch.tensor(decode_block_table_list, dtype=torch.int32, device=dev)
                if decode_block_table_list else torch.empty(0, 0, dtype=torch.int32, device=dev),
            decode_context_lens=torch.tensor(decode_context_lens, dtype=torch.int32, device=dev),
            sample_indices=torch.tensor(sample_indices, dtype=torch.long, device=dev),
        )

    def execute(self, prefill_seqs: List[Sequence],
                decode_seqs: List[Sequence]) -> torch.Tensor:
        mi = self.build_input(prefill_seqs, decode_seqs)
        with torch.inference_mode():
            logits = self.model(
                mi.input_ids, mi.positions, mi.slot_mapping,
                self.cache_engine.kv_caches,
                mi.prefill_seq_lens, mi.prefill_query_lens, mi.num_prefill_tokens,
                mi.decode_block_table, mi.decode_context_lens,
                sample_indices=mi.sample_indices,
            )
        return logits
```

- [ ] **Step 2: Smoke import**

Run: `python -c "from mini_vllm.model_runner import ModelRunner, ModelInput; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/model_runner.py
git commit -m "mini-vllm: ModelRunner.build_input + execute"
```

---

### Task 16: ModelRunner integration test (one prefill + one decode)

**Files:**
- Create: `code/mini-vllm/tests/test_model_runner.py`

- [ ] **Step 1: Write the test**

Create `tests/test_model_runner.py`:
```python
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
```

- [ ] **Step 2: Run, verify pass**

Run: `pytest tests/test_model_runner.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_runner.py
git commit -m "mini-vllm: ModelRunner integration test"
```

---

### Task 17: Naive Scheduler

**Files:**
- Create: `code/mini-vllm/mini_vllm/scheduler.py`
- Create: `code/mini-vllm/tests/test_scheduler.py`

Plan 1 scheduler: the moment `step()` is called, drain `waiting` (those that fit) into `running` once. From then on, only run/decode existing seqs until they finish; no further admission until `running` is empty. (This is `enable_continuous_batching=False` behavior; Plan 4 makes it dynamic.)

- [ ] **Step 1: Write the failing test**

Create `tests/test_scheduler.py`:
```python
from mini_vllm.config import SamplingParams
from mini_vllm.sequence import Sequence, SequenceStatus
from mini_vllm.block_manager import BlockManager
from mini_vllm.scheduler import Scheduler


def make_seq(rid, prompt_len, max_tokens=4):
    return Sequence(rid, list(range(prompt_len)), SamplingParams(max_tokens=max_tokens))


def test_initial_admit_runs_prefill():
    bm = BlockManager(num_blocks=8, block_size=4)
    sched = Scheduler(bm)
    s1 = make_seq("a", 5); s2 = make_seq("b", 3)
    sched.add(s1); sched.add(s2)
    out = sched.schedule()
    assert {s.request_id for s in out.prefill_seqs} == {"a", "b"}
    assert out.decode_seqs == []
    assert s1.status == SequenceStatus.RUNNING
    assert s2.status == SequenceStatus.RUNNING


def test_after_prefill_marked_steps_become_decode():
    bm = BlockManager(num_blocks=8, block_size=4)
    sched = Scheduler(bm)
    s = make_seq("a", 4)
    sched.add(s)
    sched.schedule()
    sched.mark_prefilled(s)
    out2 = sched.schedule()
    assert out2.prefill_seqs == []
    assert [x.request_id for x in out2.decode_seqs] == ["a"]


def test_no_admission_until_running_drained():
    bm = BlockManager(num_blocks=8, block_size=4)
    sched = Scheduler(bm)
    s1 = make_seq("a", 4)
    sched.add(s1)
    sched.schedule()
    sched.mark_prefilled(s1)
    # Add s2 while s1 is decoding — should NOT be admitted in Plan 1
    s2 = make_seq("b", 4)
    sched.add(s2)
    out = sched.schedule()
    assert [x.request_id for x in out.prefill_seqs] == []
    assert [x.request_id for x in out.decode_seqs] == ["a"]
    assert s2.status == SequenceStatus.WAITING
```

- [ ] **Step 2: Run, expect ImportError**

Run: `pytest tests/test_scheduler.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement scheduler.py**

```python
"""Plan 1 scheduler: FCFS, no continuous batching, no preemption.

State machine:
    waiting -> running: admitted in a `schedule()` call when running is empty
    running -> finished: when seq.is_finished()

Plan 4 will turn `_can_admit_more()` from "running is empty" into
continuous-batching-with-token-budget; Plan 5 adds prefix cache lookup;
Plan 6 adds swap_in/out and the `swapped` queue.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from mini_vllm.block_manager import BlockManager, AllocStatus
from mini_vllm.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    prefill_seqs: List[Sequence] = field(default_factory=list)
    decode_seqs: List[Sequence] = field(default_factory=list)
    swap_in: Dict[int, int] = field(default_factory=dict)    # Plan 6
    swap_out: Dict[int, int] = field(default_factory=dict)   # Plan 6
    blocks_to_copy: List[Tuple[int, int]] = field(default_factory=list)  # Plan 5


class Scheduler:
    def __init__(self, block_manager: BlockManager):
        self.bm = block_manager
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def mark_prefilled(self, seq: Sequence) -> None:
        seq.num_prefilled = seq.num_prompt_tokens

    def free_finished(self) -> List[Sequence]:
        """Return finished seqs and remove them from running. Caller frees blocks."""
        still_running, finished = [], []
        for s in self.running:
            (finished if s.is_finished() else still_running).append(s)
        self.running = still_running
        for s in finished:
            self.bm.free(s)
        return finished

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        # Decode existing running seqs first.
        for seq in self.running:
            if seq.num_prefilled < seq.num_prompt_tokens:
                # First step after admission: still need prefill.
                out.prefill_seqs.append(seq)
            else:
                # Need to ensure a slot exists for the upcoming token.
                # Plan 1: just append; out-of-blocks raises (no preemption).
                self.bm.append_slot(seq)
                out.decode_seqs.append(seq)

        # Admit new requests only when no running seqs exist (no continuous batching).
        if not self.running:
            while self.waiting:
                seq = self.waiting[0]
                status = self.bm.can_allocate(seq)
                if status == AllocStatus.OK:
                    self.bm.allocate(seq)
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                    out.prefill_seqs.append(seq)
                    self.waiting.popleft()
                elif status == AllocStatus.LATER:
                    break
                else:  # NEVER
                    raise RuntimeError(
                        f"Request {seq.request_id} too large for cache "
                        f"({seq.num_prompt_tokens} tokens)")
        return out
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_scheduler.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add mini_vllm/scheduler.py tests/test_scheduler.py
git commit -m "mini-vllm: Plan 1 naive Scheduler (no continuous batching)"
```

---

### Task 18: LLMEngine

**Files:**
- Create: `code/mini-vllm/mini_vllm/engine.py`

- [ ] **Step 1: Implement engine.py**

```python
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
```

- [ ] **Step 2: Smoke import**

Run: `python -c "from mini_vllm.engine import LLMEngine; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add mini_vllm/engine.py
git commit -m "mini-vllm: LLMEngine orchestration loop"
```

---

### Task 19: examples/run_toy.py

**Files:**
- Create: `code/mini-vllm/examples/run_toy.py`

- [ ] **Step 1: Write run_toy.py**

```python
"""End-to-end smoke run with a randomly-initialized toy GPT.
Output text is gibberish (random weights) but proves the engine plumbing works.
"""
import torch
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.cache_engine import CacheEngine  # noqa: F401  (used via engine)
from mini_vllm.config import CacheConfig, EngineConfig, ModelConfig, SamplingParams
from mini_vllm.engine import LLMEngine
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.tokenizer import TokenizerWrapper


def main():
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=4,
                               d_model=128, n_head=4, max_pos=512, seed=42)
    tokenizer = TokenizerWrapper.from_pretrained_gpt2()
    engine = LLMEngine(
        model, tokenizer,
        EngineConfig(
            model=model.config,
            cache=CacheConfig(block_size=16, num_gpu_blocks=64),
            device="cpu", seed=42,
        ),
    )
    prompts = ["Hello world,", "Once upon a time"]
    sp = SamplingParams(max_tokens=16, greedy=True)
    for rid, text in engine.generate(prompts, sp):
        print(f"[{rid}] {text!r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end**

Run from `code/mini-vllm/`: `python examples/run_toy.py`
Expected: two lines, each like `[req-0] ' some gibberish tokens'`. No tracebacks.

- [ ] **Step 3: Commit**

```bash
git add examples/run_toy.py
git commit -m "mini-vllm: end-to-end run_toy example"
```

---

### Task 20: End-to-end smoke test + block-leak invariant

**Files:**
- Create: `code/mini-vllm/tests/test_e2e.py`

- [ ] **Step 1: Write the test**

Create `tests/test_e2e.py`:
```python
import torch
from mini_vllm.backends.torch_backend import TorchBackend
from mini_vllm.config import CacheConfig, EngineConfig, SamplingParams
from mini_vllm.engine import LLMEngine
from mini_vllm.models.toy_gpt import ToyGPT
from mini_vllm.tokenizer import TokenizerWrapper


def _build_engine():
    backend = TorchBackend()
    model = ToyGPT.random_init(backend, vocab_size=50257, n_layer=2,
                               d_model=64, n_head=4, max_pos=128, seed=0)
    tokenizer = TokenizerWrapper.from_pretrained_gpt2()
    return LLMEngine(model, tokenizer, EngineConfig(
        model=model.config,
        cache=CacheConfig(block_size=8, num_gpu_blocks=32),
        device="cpu", seed=0))


def test_e2e_single_request():
    eng = _build_engine()
    out = eng.generate(["Hello"], SamplingParams(max_tokens=4, greedy=True))
    assert len(out) == 1 and len(out[0][1]) > 0
    # No block leak: all blocks back in free pool
    assert eng.block_manager.num_free_blocks == eng.cfg.cache.num_gpu_blocks


def test_e2e_two_sequential_batches():
    eng = _build_engine()
    eng.generate(["Hello", "World"], SamplingParams(max_tokens=4, greedy=True))
    eng.generate(["foo", "bar baz"],  SamplingParams(max_tokens=3, greedy=True))
    assert eng.block_manager.num_free_blocks == eng.cfg.cache.num_gpu_blocks


def test_e2e_determinism():
    eng1 = _build_engine()
    a = eng1.generate(["Hello there"], SamplingParams(max_tokens=8, greedy=True))
    eng2 = _build_engine()
    b = eng2.generate(["Hello there"], SamplingParams(max_tokens=8, greedy=True))
    assert a[0][1] == b[0][1]
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: all tests pass (block_manager, sequence, sampler, attention, toy_gpt, model_runner, scheduler, e2e).

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "mini-vllm: end-to-end smoke + no-leak invariant"
```

---

### Task 21: Plan 1 wrap-up — README quickstart + final test sweep

**Files:**
- Modify: `code/mini-vllm/README.md`

- [ ] **Step 1: Expand README**

Replace `code/mini-vllm/README.md` content with:
```markdown
# mini-vLLM

Educational reimplementation of vLLM with PagedAttention. Companion code
to `part5-serving-infra/16a-lab-mini-vllm.md`.

## Status

**Plan 1 (skeleton):** end-to-end engine running toy GPT on CPU/MPS via Torch
paged-attention backend; naive FCFS scheduler; greedy sampling. Foundation
for later plans.

## Install

    cd code/mini-vllm
    pip install -e ".[dev]"

## Quickstart

    python examples/run_toy.py

Expected output: two lines of (gibberish) text generated by a randomly-initialized
toy GPT.

## Run tests

    pytest tests/ -v

## Roadmap

- [x] Plan 1: skeleton — Torch backend, naive scheduler, toy GPT
- [ ] Plan 2: Triton paged-attention kernel
- [ ] Plan 3: TinyLlama-1.1B + HF safetensors loader
- [ ] Plan 4: continuous batching + chunked prefill
- [ ] Plan 5: prefix caching + CoW
- [ ] Plan 6: swap to CPU + preemption
- [ ] Plan 7: streaming + full sampler (temperature/top-p/top-k) + bench
- [ ] Plan 8: tutorial chapter `16a-lab-mini-vllm.md`
```

- [ ] **Step 2: Final test sweep**

Run: `pytest tests/ -v && python examples/run_toy.py`
Expected: all tests pass + run_toy.py produces two output lines without error.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "mini-vllm: Plan 1 README + roadmap"
```

---

## Summary

Plan 1 delivers a working mini-vLLM that:
- Runs a toy GPT-2-style model on CPU via the Torch paged-attention backend
- Allocates/frees KV blocks correctly (no leaks across runs)
- Supports batched prefill+decode in a single step (kernel-side ready for continuous batching, even though the scheduler doesn't use it yet)
- Has 100% test coverage of the BlockManager state machine, the attention contract, and the engine smoke path

Plan 2 will replace `TorchBackend.decode` with a Triton kernel, validated against the same `reference_decode` golden. Plan 3 will add Llama. Plans 4–6 will progressively unlock the scheduler's advanced features. Plan 7 finalizes sampler+streaming+bench. Plan 8 writes the tutorial chapter.

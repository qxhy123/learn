# AI Infra Tutorial Wave 2 Systems Evidence And Capacity Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. User constraint for subagent execution: use `gpt-5.5` with `xhigh` reasoning for all subagents, with at most 5 subagents running concurrently.

**Goal:** Turn Part 0/2 into evidence-driven systems chapters with explicit capacity models, profiling workflows, troubleshooting paths, and retest criteria.

**Architecture:** Work in disjoint chapter families so each batch can be reviewed and committed independently. Preserve existing chapter IDs and Markdown source paths, treat the current working tree as source of truth, and do not touch `html/` in this wave. Use fresh `gpt-5.5 xhigh` subagents per family, keep concurrent subagents at or below 5, and run a spec-compliance review plus a quality review before moving to the next batch.

**Tech Stack:** Markdown source, shell, `rg`, `wc`, `git diff --check`, `git status --short`, and subagent-driven review.

---

## Source References

- Spec: `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`
- Wave 2 scope: Part 0/2 mechanism depth, profiling labs, evidence bundles, capacity worksheets, and `appendix/tooling-map.md`
- Current tutorial source root: `part1-foundations/`, `part0-foundations-of-systems/`, `part2-systems-stack/`, `appendix/`
- Generated HTML is not source of truth. Do not edit `html/` in this wave.

## File Structure

### Read

- `docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md`
- `part1-foundations/02-compute-storage-network.md`
- `part0-foundations-of-systems/0a-cpu-microarchitecture.md`
- `part0-foundations-of-systems/0a1-pipeline.md`
- `part0-foundations-of-systems/0a2-out-of-order-execution.md`
- `part0-foundations-of-systems/0a3-branch-prediction.md`
- `part0-foundations-of-systems/0a4-simd.md`
- `part0-foundations-of-systems/0a5-cache-hierarchy.md`
- `part0-foundations-of-systems/0a6-mesi-coherence.md`
- `part0-foundations-of-systems/0a7-false-sharing.md`
- `part0-foundations-of-systems/0a8-cpu-worked-example.md`
- `part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md`
- `part0-foundations-of-systems/0b1-virtual-memory-page-tables-and-tlb.md`
- `part0-foundations-of-systems/0b2-page-cache-writeback-and-huge-pages.md`
- `part0-foundations-of-systems/0b3-numa-pcie-dma-and-pinned-memory.md`
- `part0-foundations-of-systems/0b4-syscall-epoll-io-uring-and-service-io.md`
- `part2-systems-stack/04-gpu-and-accelerators.md`
- `part2-systems-stack/04a-gpu-execution-model-and-tensor-cores.md`
- `part2-systems-stack/04b-hbm-memory-and-roofline.md`
- `part2-systems-stack/04c-gpu-interconnect-and-systems.md`
- `part2-systems-stack/04d-gpu-selection-virtualization-and-heterogeneous-accelerators.md`
- `part2-systems-stack/05-memory-interconnect-io.md`
- `part2-systems-stack/05a-memory-storage-hierarchy-and-data-residency.md`
- `part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md`
- `part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md`
- `part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md`
- `part2-systems-stack/06-cuda-runtime-and-kernels.md`
- `part2-systems-stack/06a-framework-dispatch-runtime-and-kernel-launch.md`
- `part2-systems-stack/06b-streams-synchronization-and-cuda-graphs.md`
- `part2-systems-stack/06c-kernel-libraries-fusion-and-sm-resource-limits.md`
- `part2-systems-stack/06d-profiling-debugging-and-performance-sop.md`
- `appendix/tooling-map.md`

### Modify

- `part1-foundations/02-compute-storage-network.md`
- `part0-foundations-of-systems/0a-cpu-microarchitecture.md`
- `part0-foundations-of-systems/0a1-pipeline.md`
- `part0-foundations-of-systems/0a2-out-of-order-execution.md`
- `part0-foundations-of-systems/0a3-branch-prediction.md`
- `part0-foundations-of-systems/0a4-simd.md`
- `part0-foundations-of-systems/0a5-cache-hierarchy.md`
- `part0-foundations-of-systems/0a6-mesi-coherence.md`
- `part0-foundations-of-systems/0a7-false-sharing.md`
- `part0-foundations-of-systems/0a8-cpu-worked-example.md`
- `part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md`
- `part0-foundations-of-systems/0b1-virtual-memory-page-tables-and-tlb.md`
- `part0-foundations-of-systems/0b2-page-cache-writeback-and-huge-pages.md`
- `part0-foundations-of-systems/0b3-numa-pcie-dma-and-pinned-memory.md`
- `part0-foundations-of-systems/0b4-syscall-epoll-io-uring-and-service-io.md`
- `part2-systems-stack/04-gpu-and-accelerators.md`
- `part2-systems-stack/04a-gpu-execution-model-and-tensor-cores.md`
- `part2-systems-stack/04b-hbm-memory-and-roofline.md`
- `part2-systems-stack/04c-gpu-interconnect-and-systems.md`
- `part2-systems-stack/04d-gpu-selection-virtualization-and-heterogeneous-accelerators.md`
- `part2-systems-stack/05-memory-interconnect-io.md`
- `part2-systems-stack/05a-memory-storage-hierarchy-and-data-residency.md`
- `part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md`
- `part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md`
- `part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md`
- `part2-systems-stack/06-cuda-runtime-and-kernels.md`
- `part2-systems-stack/06a-framework-dispatch-runtime-and-kernel-launch.md`
- `part2-systems-stack/06b-streams-synchronization-and-cuda-graphs.md`
- `part2-systems-stack/06c-kernel-libraries-fusion-and-sm-resource-limits.md`
- `part2-systems-stack/06d-profiling-debugging-and-performance-sop.md`
- `appendix/tooling-map.md`

Do not modify:

- `html/`
- Part 3, Part 4, Part 5, Part 6, Part 7, Part 8 source chapters in this wave
- `appendix/answers.md`

---

## Task 1: Tighten The Part 1 Bridge Chapter

**Files:**

- Modify: `part1-foundations/02-compute-storage-network.md`

- [ ] **Step 1: Inspect the chapter and wave scope**

Run:

```bash
sed -n '1,220p' docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
sed -n '1,220p' part1-foundations/02-compute-storage-network.md
```

Expected: understand how the bridge chapter connects the later Part 0/2 mechanism chapters.

- [ ] **Step 2: Add evidence-first framing and a resource-chain model**

Add or tighten the opening so the chapter explicitly carries the Wave 2 contract:

- name the problem as a resource chain, not a model tutorial
- add an `EvidenceBundle`-style diagnostic field set for resource bottlenecks
- add a `CapacityLedger` or capacity worksheet that maps compute, storage, memory, and network
- include at least one mermaid diagram for the end-to-end resource chain
- include at least one formula or decision rule for throughput / utilization / bottleneck analysis
- keep the chapter as a bridge chapter; do not expand it into a duplicate of 0a/0b/0d

- [ ] **Step 3: Verify the bridge chapter**

Run:

```bash
wc -l part1-foundations/02-compute-storage-network.md
rg -n "EvidenceBundle|CapacityLedger|BenchmarkProtocol|retest|threshold|perf|fio|H2D|NCCL|资源链|关键路径|木桶效应" part1-foundations/02-compute-storage-network.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part1-foundations/02-compute-storage-network.md
git diff --check -- part1-foundations/02-compute-storage-network.md
```

Expected: evidence/capacity terms are present, no placeholders remain, diff check passes.

- [ ] **Step 4: Commit Task 1**

```bash
git add part1-foundations/02-compute-storage-network.md
git commit -m "Wave 2: tighten compute-storage-network bridge"
```

---

## Task 2: Rewrite The Part 0 CPU Family

**Files:**

- Modify: `part0-foundations-of-systems/0a-cpu-microarchitecture.md`
- Modify: `part0-foundations-of-systems/0a1-pipeline.md`
- Modify: `part0-foundations-of-systems/0a2-out-of-order-execution.md`
- Modify: `part0-foundations-of-systems/0a3-branch-prediction.md`
- Modify: `part0-foundations-of-systems/0a4-simd.md`
- Modify: `part0-foundations-of-systems/0a5-cache-hierarchy.md`
- Modify: `part0-foundations-of-systems/0a6-mesi-coherence.md`
- Modify: `part0-foundations-of-systems/0a7-false-sharing.md`
- Modify: `part0-foundations-of-systems/0a8-cpu-worked-example.md`

- [ ] **Step 1: Inspect the 0a family and the wave spec**

Run:

```bash
sed -n '2165,2215p' docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
sed -n '1,220p' part0-foundations-of-systems/0a-cpu-microarchitecture.md
```

Expected: understand the CPU mechanism ladder and the wave boundary.

- [ ] **Step 2: Rewrite the 0a family with evidence and decision rules**

For the overview and all eight detail chapters:

- add or strengthen the `第一性原理拆解 + 学习大纲` opener
- add boundary statements, control paths, and failure paths
- add concrete `perf` / `perf stat` / `perf c2c` / `topdown` evidence references
- add at least one numerical model or decision rule in each chapter family
- add troubleshooting tables with symptom, evidence, root cause, action, and retest
- keep `0a8` as the worked example chapter, not a generic conclusion

- [ ] **Step 3: Verify the 0a family**

Run:

```bash
wc -l part0-foundations-of-systems/0a-cpu-microarchitecture.md part0-foundations-of-systems/0a*.md
rg -n "第一性原理拆解|EvidenceBundle|CapacityLedger|perf stat|perf c2c|topdown|branch-misses|LLC|NUMA|false sharing|MFU|HFU|worked example|故障排除" part0-foundations-of-systems/0a-cpu-microarchitecture.md part0-foundations-of-systems/0a*.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part0-foundations-of-systems/0a-cpu-microarchitecture.md part0-foundations-of-systems/0a*.md
git diff --check -- part0-foundations-of-systems/0a-cpu-microarchitecture.md part0-foundations-of-systems/0a*.md
```

Expected: all nine 0a files carry the evidence-first opener, required CPU/perf keywords are present, no placeholders remain, diff check passes.

- [ ] **Step 4: Commit Task 2**

```bash
git add part0-foundations-of-systems/0a-cpu-microarchitecture.md part0-foundations-of-systems/0a*.md
git commit -m "Wave 2: deepen CPU microarchitecture family"
```

---

## Task 3: Rewrite The Part 0 Memory / IO Family

**Files:**

- Modify: `part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md`
- Modify: `part0-foundations-of-systems/0b1-virtual-memory-page-tables-and-tlb.md`
- Modify: `part0-foundations-of-systems/0b2-page-cache-writeback-and-huge-pages.md`
- Modify: `part0-foundations-of-systems/0b3-numa-pcie-dma-and-pinned-memory.md`
- Modify: `part0-foundations-of-systems/0b4-syscall-epoll-io-uring-and-service-io.md`

- [ ] **Step 1: Inspect the 0b family and the wave spec**

Run:

```bash
sed -n '2165,2215p' docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
sed -n '1,220p' part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md
```

Expected: understand how the memory / IO chapters should expose evidence and failure boundaries.

- [ ] **Step 2: Rewrite the 0b family with capacity and IO evidence**

For the overview and all four detail chapters:

- add the first-principles opener and make the chapter boundaries explicit
- turn Page Cache, pinned memory, H2D, NUMA, io_uring, RDMA, and checkpoint topics into path-and-evidence driven prose
- add at least one capacity formula or decision rule per chapter family
- add troubleshooting tables and retest criteria for bottlenecks such as page-fault storms, H2D stalls, dirty writeback, and service IO latency
- keep `0b2` focused on page cache / huge pages / writeback, not a generic storage chapter

- [ ] **Step 3: Verify the 0b family**

Run:

```bash
wc -l part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md part0-foundations-of-systems/0b*.md
rg -n "第一性原理拆解|EvidenceBundle|CapacityLedger|page cache|pinned memory|H2D|NUMA|io_uring|RDMA|checkpoint|Lustre|object storage|retest|threshold" part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md part0-foundations-of-systems/0b*.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md part0-foundations-of-systems/0b*.md
git diff --check -- part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md part0-foundations-of-systems/0b*.md
```

Expected: all five 0b files carry the evidence-first opener, required memory/IO keywords are present, no placeholders remain, diff check passes.

- [ ] **Step 4: Commit Task 3**

```bash
git add part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md part0-foundations-of-systems/0b*.md
git commit -m "Wave 2: deepen memory and IO family"
```

---

## Task 4: Rewrite The Part 2 GPU Family

**Files:**

- Modify: `part2-systems-stack/04-gpu-and-accelerators.md`
- Modify: `part2-systems-stack/04a-gpu-execution-model-and-tensor-cores.md`
- Modify: `part2-systems-stack/04b-hbm-memory-and-roofline.md`
- Modify: `part2-systems-stack/04c-gpu-interconnect-and-systems.md`
- Modify: `part2-systems-stack/04d-gpu-selection-virtualization-and-heterogeneous-accelerators.md`

- [ ] **Step 1: Inspect the 04 family and the wave spec**

Run:

```bash
sed -n '2165,2215p' docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
sed -n '1,200p' part2-systems-stack/04-gpu-and-accelerators.md
```

Expected: understand how the GPU family is split across execution, capacity, interconnect, and selection.

- [ ] **Step 2: Rewrite the 04 family with evidence, formulas, and decision rules**

For the overview and all four detail chapters:

- add the first-principles opener and explicit ownership boundaries
- add GPU execution evidence paths using `nsys`, `ncu`, `torch.profiler`, `DCGM`, and topology commands
- add HBM / Roofline capacity models and decision rules
- add interconnect and topology troubleshooting with concrete signals and retest criteria
- add a selection / virtualization chapter that names MIG, MPS, heterogeneous accelerators, and datasheet caveats

- [ ] **Step 3: Verify the 04 family**

Run:

```bash
wc -l part2-systems-stack/04-gpu-and-accelerators.md part2-systems-stack/04*.md
rg -n "第一性原理拆解|EvidenceBundle|CapacityLedger|BenchmarkProtocol|nsys|ncu|torch.profiler|DCGM|Roofline|HBM|MIG|MPS|NVLink|NVSwitch|GPU selection|retest|threshold" part2-systems-stack/04-gpu-and-accelerators.md part2-systems-stack/04*.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part2-systems-stack/04-gpu-and-accelerators.md part2-systems-stack/04*.md
git diff --check -- part2-systems-stack/04-gpu-and-accelerators.md part2-systems-stack/04*.md
```

Expected: all five 04 files carry the evidence-first opener, required GPU keywords are present, no placeholders remain, diff check passes.

- [ ] **Step 4: Commit Task 4**

```bash
git add part2-systems-stack/04-gpu-and-accelerators.md part2-systems-stack/04*.md
git commit -m "Wave 2: deepen GPU and accelerator family"
```

---

## Task 5: Rewrite The Part 2 Memory / Interconnect / IO Family

**Files:**

- Modify: `part2-systems-stack/05-memory-interconnect-io.md`
- Modify: `part2-systems-stack/05a-memory-storage-hierarchy-and-data-residency.md`
- Modify: `part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md`
- Modify: `part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md`
- Modify: `part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md`

- [ ] **Step 1: Inspect the 05 family and the wave spec**

Run:

```bash
sed -n '2165,2215p' docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
sed -n '1,200p' part2-systems-stack/05-memory-interconnect-io.md
```

Expected: understand the memory / interconnect / IO decision tree and how it supports checkpoint and training diagnostics.

- [ ] **Step 2: Rewrite the 05 family with capacity worksheets and troubleshooting paths**

For the overview and all four detail chapters:

- add the first-principles opener and explicit path / boundary statements
- add capacity worksheets for data residency, H2D overlap, RDMA collective bottlenecks, and checkpoint IO
- add formulas and thresholds that can be checked against `fio`, `ib_write_bw`, `nccl-tests`, `iostat`, and topology commands
- include troubleshooting tables that separate symptoms, evidence, root cause, action, and retest
- keep `05d` focused on training storage / checkpoint / IO diagnosis rather than general storage theory

- [ ] **Step 3: Verify the 05 family**

Run:

```bash
wc -l part2-systems-stack/05-memory-interconnect-io.md part2-systems-stack/05*.md
rg -n "第一性原理拆解|EvidenceBundle|CapacityLedger|BenchmarkProtocol|fio|ib_write_bw|nccl-tests|iostat|page cache|pinned memory|RDMA|checkpoint|Lustre|retest|threshold" part2-systems-stack/05-memory-interconnect-io.md part2-systems-stack/05*.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part2-systems-stack/05-memory-interconnect-io.md part2-systems-stack/05*.md
git diff --check -- part2-systems-stack/05-memory-interconnect-io.md part2-systems-stack/05*.md
```

Expected: all five 05 files carry the evidence-first opener, required memory/interconnect/IO keywords are present, no placeholders remain, diff check passes.

- [ ] **Step 4: Commit Task 5**

```bash
git add part2-systems-stack/05-memory-interconnect-io.md part2-systems-stack/05*.md
git commit -m "Wave 2: deepen memory interconnect and IO family"
```

---

## Task 6: Rewrite The Part 2 CUDA / Runtime Family And Tooling Map

**Files:**

- Modify: `part2-systems-stack/06-cuda-runtime-and-kernels.md`
- Modify: `part2-systems-stack/06a-framework-dispatch-runtime-and-kernel-launch.md`
- Modify: `part2-systems-stack/06b-streams-synchronization-and-cuda-graphs.md`
- Modify: `part2-systems-stack/06c-kernel-libraries-fusion-and-sm-resource-limits.md`
- Modify: `part2-systems-stack/06d-profiling-debugging-and-performance-sop.md`
- Modify: `appendix/tooling-map.md`

- [ ] **Step 1: Inspect the 06 family and the tooling appendix**

Run:

```bash
sed -n '2165,2215p' docs/superpowers/specs/2026-05-04-ai-infra-tutorial-improvement-spec.md
sed -n '1,220p' part2-systems-stack/06-cuda-runtime-and-kernels.md
sed -n '1,220p' appendix/tooling-map.md
```

Expected: understand how launch, stream, graph, fusion, and profiling should be expressed as an evidence-backed execution stack.

- [ ] **Step 2: Rewrite the 06 family and tighten the tooling map**

For the overview and all four detail chapters:

- add the first-principles opener and explicit control / data / failure paths
- add profiler-driven evidence paths using `nsys`, `ncu`, `torch.profiler`, `perf`, and `DCGM`
- add formulas or decision rules for launch overhead, overlap, occupancy, and fusion trade-offs
- add troubleshooting tables for implicit sync, bad overlap, launch overhead, graph regressions, and kernel resource pressure
- keep `06d` focused on a practical profiling SOP with retest criteria and escalation boundary

For `appendix/tooling-map.md`:

- tighten the descriptions so the tool categories map to the Wave 2 evidence gate
- add or refine entries for profiling, timing, bandwidth, topology, and benchmark tools used in Part 0/2
- make the appendix a lookup table for later chapter-owned evidence and retest commands

- [ ] **Step 3: Verify the 06 family and tooling map**

Run:

```bash
wc -l part2-systems-stack/06-cuda-runtime-and-kernels.md part2-systems-stack/06*.md appendix/tooling-map.md
rg -n "第一性原理拆解|EvidenceBundle|CapacityLedger|BenchmarkProtocol|nsys|ncu|torch.profiler|perf stat|CUDA Graph|stream|kernel launch|occupancy|fusion|retest|threshold" part2-systems-stack/06-cuda-runtime-and-kernels.md part2-systems-stack/06*.md appendix/tooling-map.md
rg -n "TODO|FIXME|TBD|附加推演|补充推演|边界问题" part2-systems-stack/06-cuda-runtime-and-kernels.md part2-systems-stack/06*.md appendix/tooling-map.md
git diff --check -- part2-systems-stack/06-cuda-runtime-and-kernels.md part2-systems-stack/06*.md appendix/tooling-map.md
```

Expected: all five 06 files plus the tooling appendix carry the evidence-first opener or tighter evidence mapping, required runtime/profiling keywords are present, no placeholders remain, diff check passes.

- [ ] **Step 4: Commit Task 6**

```bash
git add part2-systems-stack/06-cuda-runtime-and-kernels.md part2-systems-stack/06*.md appendix/tooling-map.md
git commit -m "Wave 2: deepen CUDA runtime family and tooling map"
```

---

## Task 7: Final Wave 2 Verification

**Files:**

- Read: all modified files from Tasks 1-6

- [ ] **Step 1: Run final verification in one shell block**

Run:

```bash
WAVE2_FILES=(
  part1-foundations/02-compute-storage-network.md
  part0-foundations-of-systems/0a-cpu-microarchitecture.md
  part0-foundations-of-systems/0a1-pipeline.md
  part0-foundations-of-systems/0a2-out-of-order-execution.md
  part0-foundations-of-systems/0a3-branch-prediction.md
  part0-foundations-of-systems/0a4-simd.md
  part0-foundations-of-systems/0a5-cache-hierarchy.md
  part0-foundations-of-systems/0a6-mesi-coherence.md
  part0-foundations-of-systems/0a7-false-sharing.md
  part0-foundations-of-systems/0a8-cpu-worked-example.md
  part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md
  part0-foundations-of-systems/0b1-virtual-memory-page-tables-and-tlb.md
  part0-foundations-of-systems/0b2-page-cache-writeback-and-huge-pages.md
  part0-foundations-of-systems/0b3-numa-pcie-dma-and-pinned-memory.md
  part0-foundations-of-systems/0b4-syscall-epoll-io-uring-and-service-io.md
  part2-systems-stack/04-gpu-and-accelerators.md
  part2-systems-stack/04a-gpu-execution-model-and-tensor-cores.md
  part2-systems-stack/04b-hbm-memory-and-roofline.md
  part2-systems-stack/04c-gpu-interconnect-and-systems.md
  part2-systems-stack/04d-gpu-selection-virtualization-and-heterogeneous-accelerators.md
  part2-systems-stack/05-memory-interconnect-io.md
  part2-systems-stack/05a-memory-storage-hierarchy-and-data-residency.md
  part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md
  part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md
  part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md
  part2-systems-stack/06-cuda-runtime-and-kernels.md
  part2-systems-stack/06a-framework-dispatch-runtime-and-kernel-launch.md
  part2-systems-stack/06b-streams-synchronization-and-cuda-graphs.md
  part2-systems-stack/06c-kernel-libraries-fusion-and-sm-resource-limits.md
  part2-systems-stack/06d-profiling-debugging-and-performance-sop.md
  appendix/tooling-map.md
)

rg -n "\]\([^)]*\.html[)#]?" "${WAVE2_FILES[@]}" | rg -v "https?://" || true
rg -n "EvidenceBundle|CapacityLedger|BenchmarkProtocol|retest|threshold|perf stat|perf c2c|topdown|fio|ib_write_bw|nccl-tests|nsys|ncu|torch.profiler|CUDA Graph|H2D|Roofline|MIG|FSDP|RDMA|checkpoint" "${WAVE2_FILES[@]}" || true
rg -n "T[O]DO|T[B]D|F[I]XME|待[定]|待[补]|后续[补]|这里不[展]开|PLACE[H]OLDER|\\.\\.\\." "${WAVE2_FILES[@]}" || true
git diff --check -- "${WAVE2_FILES[@]}"
git diff --stat fc06bb4..HEAD -- "${WAVE2_FILES[@]}"
git status --short -- "${WAVE2_FILES[@]}"
```

Expected:

- generated HTML links do not appear in the Wave 2 source files
- evidence / capacity / profiling terms are present across the family
- placeholder scan has no unresolved placeholders
- `git diff --check` exits 0
- `git diff --stat` shows only the exact Wave 2 source files
- scoped `git status` is clean after commits

- [ ] **Step 2: Review staged and unstaged changes**

Run:

```bash
git status --short
git diff --stat fc06bb4..HEAD -- "${WAVE2_FILES[@]}"
```

Expected: modified files are limited to Wave 2 Markdown scope plus any pre-existing unrelated dirty files. Do not stage `html/` files in this wave.

- [ ] **Step 3: Commit final verification note if needed**

If Tasks 1-6 each committed cleanly and Task 7 made no file changes, do not create an empty commit.

If Task 7 required small fixes, commit only those fixes:

```bash
git add "${WAVE2_FILES[@]}"
git commit -m "Verify Wave 2 AI infra systems evidence cleanup"
```

---

## Execution Notes

- The worktree already contains unrelated uncommitted Markdown and HTML changes. Do not revert or stage unrelated changes.
- Prefer one commit per family so review can isolate Part 1 bridge, Part 0 CPU, Part 0 memory/IO, Part 2 GPU, Part 2 memory/interconnect, and Part 2 runtime/tooling changes.
- If a file has pre-existing user edits outside the planned section, work with them and do not rewrite unrelated content.
- If generated HTML needs rebuilding after the Markdown changes, make that a separate follow-up plan or task outside Wave 2.

## Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-ai-infra-tutorial-wave-2-systems-evidence-and-capacity-depth.md`. Execution options:

1. **Subagent-Driven** - dispatch fresh `gpt-5.5 xhigh` subagents per task, with at most 5 concurrent agents, and review between tasks.
2. **Inline Execution** - execute tasks in this session with `executing-plans`, batch execution with checkpoints for review.


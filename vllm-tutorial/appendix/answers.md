# 附录C：练习答案汇总

本附录提供各章练习题的要点提示。建议先独立思考，再查看答案。

---

## 第1章：LLM 推理的挑战

**基础题**

1. **为什么每次只能生成 1 个 token？** 因为每个 token 的生成依赖前一个 token 的结果（自回归），形成串行依赖。不能一次性生成所有 token，因为后续 token 的概率分布取决于前面已生成的 token。

2. **Prefill vs Decode 的瓶颈？** Prefill 是计算密集的（一次处理大量 token，矩阵乘法利用率高）；Decode 是内存带宽密集的（每次只处理 1 个 token，但需要读取全部权重）。

3. **TTFT vs TPOT？** TTFT = 首 token 延迟，受 prompt 长度和 prefill 速度影响。TPOT = token 间延迟，受模型大小、batch size 和 KV Cache 大小影响。

**实践题**

4. 70B FP16 = 140 GB；INT4 = 35 GB，节省 105 GB（75%）。

5. 公式：2 × 32 × 32 × 128 × seq_len × 2 bytes。seq=512 → 0.5 GB；seq=2048 → 2.0 GB；seq=8192 → 8.0 GB。

**思考题**

6. 更多请求共享一次权重读取，提高了算术强度。上限在于显存（KV Cache 空间有限）和计算（batch 太大后变成 compute-bound）。

7. 实时聊天应减小 max_num_seqs，优先保证低 TPOT 和低 TTFT，而非最大吞吐。

---

## 第2章：KV Cache 原理

1. **为什么不缓存 Q？** 因为 Query 只与当前 token 相关，每步都是新的，无法复用。而 K 和 V 对应的是历史 token，不会改变。

2. 2 × 40 × 8 × 128 × 4096 × 2 = 约 2.5 GB。

3. GQA 将多个 Query 头共享一组 KV 头，减少 KV 头数量，直接缩小 KV Cache。

---

## 第3章：vLLM 概览

1. "v" 代表 virtual（虚拟内存），对应 PagedAttention 借鉴操作系统虚拟内存分页的思想。

2. 连续批处理在每个 iteration 动态添加/移除请求，保持 batch 饱满，而静态批处理必须等最慢的请求。

---

## 第7章：PagedAttention

1. 借鉴了操作系统的虚拟内存分页（Paging）机制。

2. 逻辑块是请求看到的连续编号（0,1,2...），物理块是显存中实际位置（可以不连续）。间接层消除了碎片。

3. 当多个序列共享同一个物理块，且其中一个需要向该块写入新 token 时，触发 Copy-on-Write。

---

## 第8章：连续批处理

1. 4 个请求生成 10/50/200/500 token，静态 batch 时间 = 500 步。有效计算 = 10+50+200+500 = 760。利用率 = 760/(500×4) = 38%。

2. 通过在每个 iteration 动态管理请求：完成的立刻移除，新的立刻加入，保持 GPU 始终有工作可做。

---

## 第9章：调度器

1. 当前仓库 V1 的主状态是 Waiting、Running、Preempted；如果请求依赖 grammar、远端 KV 或流式输入，还会先进入对应的阻塞等待态。

2. 当前仓库主路径是 Recompute / Preempted：释放块后重新调度并重算；旧资料里的本地 CPU swap 不再是 V1 主线语义。

---

## 第12章：量化

1. 量化减少了权重的数据量，在 memory-bound 的 decode 阶段减少了内存传输时间。

5. 复杂推理需要模型在细微的概率差异上做正确选择，量化的舍入误差可能改变这些关键位置的概率排序。

---

## 第15章：投机解码

1. 投机解码使用拒绝采样，数学上保证接受的 token 分布与直接用大模型采样完全一致。

4. 如果所有投机 token 都被拒绝，仍然能产出 1 个 token（大模型在验证时采样的修正 token），所以最坏情况等于标准 decode + 小模型的少量开销，略慢但不会差很多。

---

## 第19章：张量并行

1. `tensor_parallel_size` 会同时影响两类对象：一类是模型内部并行层与 attention head / KV head 的切分方式；另一类是 `ParallelConfig`、executor world size 和 worker 拓扑。

2. 因为当前模型实现里，当 `total_num_kv_heads < tp_size` 时，KV heads 可能在多个 TP rank 上复制，而不是被完全均分，所以 KV cache 不一定严格缩成原来的 `1 / tp_size`。

3. 因为在无 NVLink 或切分不整齐的机器上，PP 的 stage 间传递可能比 TP 的重通信更便宜，官方文档明确建议把 `TP=1, PP=<GPU数>` 作为 benchmark 候选。

4. 先看 `GPU KV cache size` 和 `Maximum concurrency for ... tokens per request` 这两条日志，它们比经验公式更能说明 TP 调整后是否真的换来了更大缓存和更高并发。

---

## 第20章：流水线并行

1. 当前仓库里一个模型想支持 PP，至少要满足三点：`supports_pp=True`、实现 `make_empty_intermediate_tensors(...)`、并让 `forward(...)` 接受 `intermediate_tensors`。

2. 因为 PP 不是 executor 单方面就能“替模型打开”的能力，模型类必须显式实现 stage 间中间态传递逻辑，否则 registry / runtime 无法安全支持它。

3. 因为在 PCIe-only 机器上，TP 需要更频繁的集合通信，而 PP 主要是 stage 间传 hidden states，通信模式可能更适合这种硬件。

4. `TP=4, PP=2` 在当前 executor 里意味着总共 8 个并行 rank。部署时你要据此规划 GPU 数、节点数和 world size，而不是只盯着单一维度的“4 卡 TP”。

---

## 第21章：多节点分布式部署

1. 当前至少有两条主路径：Ray 集群路线，以及 `--nnodes/--node-rank/--headless` 的 multiprocessing 路线。

2. `--headless` 让非主节点只运行 engine / worker，而不启动 API server，因此它是 multiprocessing 多节点部署里的执行节点形态，不是单纯的调试开关。

3. 优先检查 `VLLM_HOST_IP`、Ray 看到的节点 IP，以及 `ray status` / `ray list nodes` 的输出是否和你的机器实际 IP 对得上。

4. 用 `NCCL_DEBUG=TRACE` 跑起来看日志；出现 `NET/IB/GDRDMA` 说明走到了高性能 RDMA 路线，出现 `NET/Socket` 则说明只是普通 socket。

---

## 第27章：自定义模型接入

1. 当前推荐的 out-of-tree 路径是 `vllm.general_plugins + ModelRegistry.register_model(...)`。直改 `models/__init__.py` 属于 in-tree 老路径，难维护，也不适合插件化扩展。

2. `trust_remote_code=True` 能帮助 vLLM 读取 HF 侧自定义 config / class；但它不能自动替你完成 vLLM-native 模型实现、权重映射、attention backend 或 worker/platform 扩展。

3. 当你的问题已经变成“新设备 / 新 worker / 新 attention backend / 新通信后端”时，只写 model plugin 不够，需要继续写 platform plugin，以及对应的 `WorkerBase`、`AttentionBackend`、`DeviceCommunicatorBase` 实现。

4. 如果一个新模型要支持 `pipeline_parallel_size > 1`，还必须补齐 `supports_pp` 相关接口，而不是只让单卡 forward 能跑通。

---

## 第28章：前沿进展与生态

1. `Disaggregated Prefill` 的主要目标是分开调 TTFT 和 ITL，并控制 tail ITL；官方文档明确说它**不**提升吞吐。

2. FlashInfer 在当前仓库里至少出现在三类位置：能力探测与兼容包装（`utils/flashinfer.py`）、V1 attention backend（`v1/attention/backends/flashinfer.py`）、启动期 warmup / autotune（`model_executor/warmup/kernel_warmup.py`）。

3. 因为当前 vLLM 的生态已经包含 model / platform plugin、connector、disaggregated serving、Ray/KubeRay、Dynamo 等集成面，OpenAI 兼容 API 只是其中一个入口。

4. 持续跟踪 frontier 时，优先看 `docs/usage/v1_guide.md`、`docs/features/`、`docs/serving/`、`distributed/`、`examples/online_serving/` 和 plugin / integration 文档，而不是只看版本标题。

---

*其他章节的练习答案原则：先确认对核心概念的理解，再动手验证。性能相关的题目必须以实际测量数据为准，不要凭直觉。*

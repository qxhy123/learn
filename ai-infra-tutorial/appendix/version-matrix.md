# 附录F：版本矩阵

> 本附录记录本文写作时采用的版本假设、核对日期、必须复测项和易漂移默认值。它不是安装指南，也不是上游最新版本声明。生产落地时应把精确版本、镜像 digest、chart / manifest revision、驱动和固件版本写入 ReleaseUnit 或 EvidenceBundle。

## 1. 使用口径

| 字段 | 口径 |
|------|------|
| 本文假设版本 | 以 2026-05-07 前后主流生产栈能力为基线；精确 minor / patch 必须由目标集群重新锁定 |
| 核对日期 | 2026-05-07 |
| 必须复测项 | 任一组件升级、GPU 代际变化、镜像重建、Kubernetes 升级、驱动或固件变更后都要复测 |
| 易漂移默认值 | 上游默认开关、调度策略、缓存策略、序列化安全默认、metric label、chart values 和 admission policy |

版本矩阵的核心原则是：正文讲机制和判断，附录记录“哪些默认值容易漂移”。如果实际版本与本文假设不同，应优先相信目标环境的 release note、compatibility matrix 和回归测试结果。

## 2. GPU 运行时与通信

| 组件 | 教程涉及能力 | 核对口径 | 必须复测项 | 相关章节 |
|------|--------------|----------|------------|----------|
| CUDA | kernel launch、stream、CUDA Graph、library dispatch、镜像用户态 CUDA | 核对 CUDA toolkit / runtime、driver API 兼容、镜像内 cuDNN / cuBLAS / NCCL 版本 | `nvidia-smi`、最小 torch CUDA、CUDA Graph capture / fallback、关键 kernel 性能 | [06](../part2-systems-stack/06-cuda-runtime-and-kernels.md)、[06b](../part2-systems-stack/06b-streams-synchronization-and-cuda-graphs.md)、[18a](../part6-platform-and-orchestration/18a-ai-images-and-cuda-compatibility.md) |
| NVIDIA driver | GPU 可见性、MIG、DCGM、container device injection、GDRDMA 前置条件 | 核对 driver branch、GPU 代际、kernel module、firmware、GPU Operator 驱动管理模式 | 节点重启后 GPU ready、ECC / XID、DCGM 指标、MIG 分片可见性、容器内最小 workload | [04d](../part2-systems-stack/04d-gpu-selection-virtualization-and-heterogeneous-accelerators.md)、[18b](../part6-platform-and-orchestration/18b-container-runtime-and-device-injection.md)、[19b](../part6-platform-and-orchestration/19b-gpu-scheduling-and-topology.md) |
| NCCL | AllReduce、ReduceScatter、AllGather、AllToAll、拓扑选择、NET/IB 与 socket fallback | 核对 NCCL release、net plugin、OFED / rdma-core、GID / MTU、GPU/NIC/NUMA 拓扑 | `nccl-tests`、`NCCL_DEBUG=INFO`、topology dump、真实训练 step time、rank stall 与 socket fallback | [0d4](../part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md)、[05c](../part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md)、[08](../part3-training-infra/08-data-parallel.md) |

易漂移默认值：`CUDA_VISIBLE_DEVICES` 注入方式、CUDA Graph capture 约束、NCCL algorithm / protocol / channel 自动选择、`NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX`、driver-container compatibility。

## 3. 训练框架与数据读取

| 组件 | 教程涉及能力 | 核对口径 | 必须复测项 | 相关章节 |
|------|--------------|----------|------------|----------|
| PyTorch | DDP / FSDP、AMP、`torch.compile`、profiler、checkpoint、serialization | 核对 Python ABI、CUDA wheel、distributed backend、Inductor / Triton 版本、序列化默认行为 | 单机训练、DDP smoke、FSDP checkpoint 保存恢复、`torch.profiler` 映射、`torch.load` 安全路径 | [07](../part3-training-infra/07-single-node-training.md)、[08](../part3-training-infra/08-data-parallel.md)、[10](../part3-training-infra/10-memory-checkpointing-and-recovery.md)、[23](../part7-reliability-security/23-security-isolation-and-governance.md) |
| TorchData | streaming dataset、DataLoader2 / DataPipes、远程对象存储读取、resume | 核对 TorchData 与 PyTorch minor 兼容、DataPipe API 状态、shuffle / sharding 语义 | 多 worker resume、分布式 shard 不重复不漏读、对象存储 retry、prefetch 与 pin memory 对 GPU wait 的影响 | [11d](../part4-data-and-storage/11d-streaming-and-dataloader-engineering.md)、[05d](../part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md) |

易漂移默认值：DataLoader `num_workers` / `prefetch_factor` / `persistent_workers`、`pin_memory`、`torch.compile` backend、FSDP state dict 类型、`torch.load` 安全默认。

## 4. 推理运行时与编译栈

| 组件 | 教程涉及能力 | 核对口径 | 必须复测项 | 相关章节 |
|------|--------------|----------|------------|----------|
| vLLM | PagedAttention、continuous batching、prefix cache、chunked prefill、speculative decoding、Multi-LoRA、OpenAI-compatible API | 核对 vLLM release、attention backend、模型支持列表、CUDA / PyTorch / driver 组合、engine 参数默认值 | TTFT / TPOT、tokens/s、KV cache 峰值、长上下文、prefix cache hit、LoRA 热加载、OOM / eviction | [15](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md)、[16a](../part5-serving-infra/16a-vllm-internals.md)、[17](../part5-serving-infra/17-multitenancy-and-cost.md) |
| SGLang | RadixAttention、structured output、tool / agent serving、cache-aware scheduling、speculative decoding | 核对 SGLang release、frontend API、backend runtime、grammar / constrained decoding 支持 | 结构化输出正确率、prefix 复用、长对话缓存、并发混部、agent 工具调用超时和隔离 | [13d](../part4-data-and-storage/13d-rag-engineering.md)、[16b](../part5-serving-infra/16b-sglang-internals.md)、[25](../part8-advanced-and-capstone/25-agent-and-inference-time-compute.md) |
| TensorRT-LLM | engine build、plugin、inflight batching、KV cache config、FP8 / FP4 / AWQ / GPTQ、Triton 集成 | 核对 TensorRT-LLM、TensorRT、CUDA、driver、GPU 架构、Triton container 和 build flags | engine rebuild、校准集、精度回归、workspace 峰值、吞吐延迟、跨 GPU 代际兼容、Triton reload | [16](../part5-serving-infra/16-quantization-compilation-and-engines.md)、[16c](../part5-serving-infra/16c-trt-llm-internals.md) |
| ModelOpt | 量化、剪枝 / 蒸馏、导出到 TensorRT-LLM 的模型优化路径 | 核对 ModelOpt 与目标模型结构、TensorRT-LLM exporter、校准数据和精度指标 | 校准后 perplexity / eval、逐层误差、serving 精度、fallback 算子、重新导出后的 engine build | [16c](../part5-serving-infra/16c-trt-llm-internals.md)、[12c](../part4-data-and-storage/12c-release-governance.md) |

易漂移默认值：attention backend、chunked prefill、prefix cache、CUDA Graph、scheduler policy、KV cache block size、quantization format、Triton model config。

## 5. 平台编排与 GPU Operator

| 组件 | 教程涉及能力 | 核对口径 | 必须复测项 | 相关章节 |
|------|--------------|----------|------------|----------|
| Kubernetes | Pod / Job、GPU resource、affinity、topology、quota、NetworkPolicy、Secret | 核对 Kubernetes minor、CRI、CNI、CSI、scheduler profile、API deprecation | GPU Pod admission、Pod Pending reason、DNS / Service、NetworkPolicy、PVC、滚动升级与驱逐 | [19](../part6-platform-and-orchestration/19-kubernetes-for-ai.md)、[19a](../part6-platform-and-orchestration/19a-kubernetes-ai-workloads.md)、[19d](../part6-platform-and-orchestration/19d-kubernetes-ai-troubleshooting.md) |
| Kueue | ClusterQueue、ResourceFlavor、admission、borrow / lend、preemption | 核对 Kueue release、CRD version、cohort / flavor 配置、namespace quota 对齐 | 多租户排队、admission 状态、抢占、配额归还、job suspend / resume | [20](../part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md)、[20a](../part6-platform-and-orchestration/20a-queues-quotas-priority-and-fairness.md) |
| Volcano | PodGroup、gang scheduling、queue priority、分布式训练同时启动 | 核对 Volcano release、scheduler plugin、PodGroup API、队列和优先级配置 | `minAvailable` 不满足时不半启动、满足后同时启动、抢占和排队时长 | [19b](../part6-platform-and-orchestration/19b-gpu-scheduling-and-topology.md)、[20a](../part6-platform-and-orchestration/20a-queues-quotas-priority-and-fairness.md) |
| NVIDIA GPU Operator | driver、container toolkit、device plugin、DCGM exporter、GPU Feature Discovery | 核对 operator chart、operand version、driver 管理模式、MIG strategy、DCGM exporter labels | operator 组件 ready、节点新增 / 重启、最小 GPU workload、DCGM 指标、MIG / time-slicing | [18b](../part6-platform-and-orchestration/18b-container-runtime-and-device-injection.md)、[19c](../part6-platform-and-orchestration/19c-ai-crd-and-operators.md)、[20b](../part6-platform-and-orchestration/20b-gpu-partitioning-and-sharing.md) |

易漂移默认值：CRD schema、scheduler plugin 开关、GPU resource name、MIG strategy、time-slicing / MPS 配置、DCGM metric label、chart values。

## 6. 可观测性

| 组件 | 教程涉及能力 | 核对口径 | 必须复测项 | 相关章节 |
|------|--------------|----------|------------|----------|
| Prometheus | metrics scrape、recording rule、alert、GPU / service 指标 | 核对 scrape interval、retention、remote write、label cardinality、DCGM exporter metric 名 | 告警触发、cardinality 增长、GPU 指标归属、P95 / P99 recording rule | [21](../part7-reliability-security/21-observability-and-capacity.md)、[appendix B](./tooling-map.md) |
| Grafana | dashboard、容量趋势、发布和事故视图 | 核对 datasource、dashboard JSON、变量、权限、timezone | dashboard 变量、时间窗口、发布标记、tenant / model / queue 过滤 | [21](../part7-reliability-security/21-observability-and-capacity.md) |
| OTel | trace、metric、log 语义约定、采样、跨服务关联 | 核对 SDK / Collector version、semantic convention、exporter、采样策略 | trace 端到端串联、采样率、属性脱敏、collector backpressure、日志关联 ID | [21](../part7-reliability-security/21-observability-and-capacity.md)、[22](../part7-reliability-security/22-evaluation-release-and-incident.md) |

易漂移默认值：metric label、histogram bucket、采样策略、OTel semantic convention、dashboard timezone、alert for / window。

## 7. 序列化、安全与供应链

| 组件 | 教程涉及能力 | 核对口径 | 必须复测项 | 相关章节 |
|------|--------------|----------|------------|----------|
| PyTorch serialization | checkpoint、`torch.save` / `torch.load`、pickle 风险、权重加载边界 | 核对 `weights_only`、allowlist、安全加载策略、checkpoint manifest 和来源 | 恶意 pickle 拒绝、旧 checkpoint 兼容、FSDP / sharded checkpoint 恢复、回滚加载 | [10](../part3-training-infra/10-memory-checkpointing-and-recovery.md)、[12b](../part4-data-and-storage/12b-checkpoint-engineering.md)、[23](../part7-reliability-security/23-security-isolation-and-governance.md) |
| SafeTensors | 权重安全分发、metadata、零拷贝读取、避免任意代码执行 | 核对格式版本、metadata 约定、framework loader、模型仓库支持 | 权重完整性、shape / dtype、加载性能、与 LoRA / sharded 权重兼容 | [12d](../part4-data-and-storage/12d-supply-chain-and-signing.md)、[23](../part7-reliability-security/23-security-isolation-and-governance.md) |
| 安全供应链工具 | Trivy、Grype、Syft、cosign、SLSA、SBOM、attestation、admission policy | 核对扫描数据库日期、签名身份、证书 / Rekor 策略、SLSA 级别、准入控制规则 | 漏洞扫描、SBOM 生成、签名验证、篡改镜像拒绝、例外到期、CI / admission 双路径 | [12d](../part4-data-and-storage/12d-supply-chain-and-signing.md)、[18c](../part6-platform-and-orchestration/18c-artifact-supply-chain-and-image-governance.md)、[23](../part7-reliability-security/23-security-isolation-and-governance.md) |

易漂移默认值：`torch.load` 安全参数、模型仓库 loader 行为、scan DB freshness、cosign identity policy、SBOM 格式、admission webhook fail-open / fail-close。

## 8. 版本变更复测清单

- [ ] 记录核对日期、owner、变更范围、旧版本、新版本、镜像 digest、chart / manifest revision。
- [ ] 跑最小 GPU workload：`nvidia-smi`、torch CUDA、单 GPU 推理或训练 smoke。
- [ ] 跑通信验收：单机和跨机 `nccl-tests`，保存 `NCCL_DEBUG=INFO` 与 topology dump。
- [ ] 跑真实 workload 小窗口：训练 step time 或 serving TTFT / TPOT / tokens/s，至少包含 P50/P95/P99。
- [ ] 核对平台对象：Pod event、queue admission、GPU allocation、DCGM 指标、Prometheus / Grafana / OTel 链路。
- [ ] 核对安全门禁：SafeTensors / PyTorch serialization 加载策略、SBOM、扫描、签名、attestation、admission policy。
- [ ] 对所有易漂移默认值给出“沿用 / 修改 / 禁用 / 待观察”结论，并写入 ReleaseUnit。

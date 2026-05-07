# 附录F：全书知识地图

> 本附录不按章节顺序阅读，而按现场问题反查。先定位你遇到的症状，再看首读章节、深入章节、证据和误区。

## 资源瓶颈

| 你遇到的问题 / 症状 | 先读章节 | 深入章节 | 关键指标 / 证据 | 常见误区 |
|---|---|---|---|---|
| GPU 很忙但训练吞吐低，`nvidia-smi` 看不出根因 | [第2章](../part1-foundations/02-compute-storage-network.md)、[第7章](../part3-training-infra/07-single-node-training.md) | [第4b章](../part2-systems-stack/04b-hbm-memory-and-roofline.md)、[第6d章](../part2-systems-stack/06d-profiling-debugging-and-performance-sop.md) | MFU/HFU、SM/HBM util、kernel timeline、CPU self time | 把 GPU utilization 当成有效 FLOPs |
| 加 worker、加卡或加 batch 后反而变慢 | [第2章](../part1-foundations/02-compute-storage-network.md)、[第8章](../part3-training-infra/08-data-parallel.md) | [第0a-7章](../part0-foundations-of-systems/0a7-false-sharing.md)、[第0b章](../part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md)、[第5b章](../part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md) | CPU profile、NUMA locality、H2D overlap、rank skew | 只扩资源，不验证瓶颈是否迁移 |
| checkpoint、dataset 或对象存储拖慢 step | [第11章](../part4-data-and-storage/11-data-pipeline.md)、[第12b章](../part4-data-and-storage/12b-checkpoint-engineering.md) | [第0c章](../part0-foundations-of-systems/0c-filesystems-and-storage-internals.md)、[第5d章](../part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md) | IOPS/吞吐、p95 write latency、Page Cache/dirty、checkpoint wall time | 只看平均吞吐，忽略尾延迟和持久化语义 |

## 训练链路

| 你遇到的问题 / 症状 | 先读章节 | 深入章节 | 关键指标 / 证据 | 常见误区 |
|---|---|---|---|---|
| 单机训练不达标，不知道从数据、显存还是 kernel 查起 | [第7章](../part3-training-infra/07-single-node-training.md) | [第6a章](../part2-systems-stack/06a-framework-dispatch-runtime-and-kernel-launch.md)、[第6d章](../part2-systems-stack/06d-profiling-debugging-and-performance-sop.md) | step breakdown、dataloader wait、CUDA gaps、op/module profile | 先调超参，后补系统基线 |
| DDP/FSDP 多机 step time 抖动或 AllReduce 尾巴长 | [第8章](../part3-training-infra/08-data-parallel.md)、[第10章](../part3-training-infra/10-memory-checkpointing-and-recovery.md) | [第5c章](../part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md)、[第0d4章](../part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md) | NCCL_DEBUG、nccl-tests、bucket timeline、最慢 rank 证据 | 把所有慢步都归因于网络 |
| MoE、TP/PP/EP 配好后仍不稳定 | [第9章](../part3-training-infra/09-model-pipeline-parallel.md)、[第09e章](../part3-training-infra/09e-moe-training-infrastructure.md) | [第10章](../part3-training-infra/10-memory-checkpointing-and-recovery.md)、[第21章](../part7-reliability-security/21-observability-and-capacity.md) | token routing skew、bubble、activation/optimizer memory、straggler timeline | 只追求并行度，不看负载均衡和恢复成本 |

## 推理链路

| 你遇到的问题 / 症状 | 先读章节 | 深入章节 | 关键指标 / 证据 | 常见误区 |
|---|---|---|---|---|
| P99 TTFT 高，但 P50 正常 | [第14章](../part5-serving-infra/14-online-inference-architecture.md)、[第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md) | [第16a章](../part5-serving-infra/16a-vllm-internals.md)、[第20c章](../part6-platform-and-orchestration/20c-inference-autoscaling.md) | queue time、prefill time、KV headroom、cold start ratio | 只盯模型 forward latency |
| TPOT、ITL 或吞吐不稳，batching 一调就互相伤害 | [第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md) | [第16b章](../part5-serving-infra/16b-sglang-internals.md)、[第16c章](../part5-serving-infra/16c-trt-llm-internals.md) | batch occupancy、decode step time、KV block usage、goodput | 用离线吞吐替代线上 SLO |
| 量化、编译或 engine 切换后质量和延迟都变了 | [第16章](../part5-serving-infra/16-quantization-compilation-and-engines.md)、[第22章](../part7-reliability-security/22-evaluation-release-and-incident.md) | [第12c章](../part4-data-and-storage/12c-release-governance.md)、[第16c章](../part5-serving-infra/16c-trt-llm-internals.md) | eval gate、A/B 指标、model/engine digest、回滚目标 | 只比较延迟，不绑定质量和版本证据 |

## 平台治理

| 你遇到的问题 / 症状 | 先读章节 | 深入章节 | 关键指标 / 证据 | 常见误区 |
|---|---|---|---|---|
| 多租户抢 GPU，排队、公平性和成本说不清 | [第17章](../part5-serving-infra/17-multitenancy-and-cost.md)、[第20章](../part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md) | [第20a章](../part6-platform-and-orchestration/20a-queues-quotas-priority-and-fairness.md)、[第20b章](../part6-platform-and-orchestration/20b-gpu-partitioning-and-sharing.md) | queue wait、quota usage、preemption log、CapacityLedger | 只做资源上限，不定义优先级和借还规则 |
| GPU Operator、device plugin、MIG/MPS 后资源口径混乱 | [第19章](../part6-platform-and-orchestration/19-kubernetes-for-ai.md)、[第19b章](../part6-platform-and-orchestration/19b-gpu-scheduling-and-topology.md) | [第18b章](../part6-platform-and-orchestration/18b-container-runtime-and-device-injection.md)、[第20d章](../part6-platform-and-orchestration/20d-capacity-and-troubleshooting-sop.md) | pod allocation、MIG UUID、DCGM 标签、node topology | 把 Kubernetes request 当成真实可用容量 |
| 发布、评测、回滚和事故复盘各自为政 | [第22章](../part7-reliability-security/22-evaluation-release-and-incident.md)、[第24章](../part8-advanced-and-capstone/24-build-an-ai-platform.md) | [第12c章](../part4-data-and-storage/12c-release-governance.md)、[附录E](./end-to-end-case.md) | ReleaseUnit、EvalRun、rollback target、postmortem action | 没有把发布门禁和事故证据连成闭环 |

## 故障排查

| 你遇到的问题 / 症状 | 先读章节 | 深入章节 | 关键指标 / 证据 | 常见误区 |
|---|---|---|---|---|
| 线上变慢、错误升高或用户报障，但无法定界 | [第21章](../part7-reliability-security/21-observability-and-capacity.md)、[第22章](../part7-reliability-security/22-evaluation-release-and-incident.md) | [第18d章](../part6-platform-and-orchestration/18d-runtime-troubleshooting.md)、[第19d章](../part6-platform-and-orchestration/19d-kubernetes-ai-troubleshooting.md) | symptom、scope、timeline、change list、rollback/retest | 一上来找根因，忘了先止血和缩小范围 |
| NCCL hang、节点掉卡、XID/ECC 或容器看不到 GPU | [第18章](../part6-platform-and-orchestration/18-containers-and-runtime.md)、[第19d章](../part6-platform-and-orchestration/19d-kubernetes-ai-troubleshooting.md) | [第0d4章](../part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md)、[第6d章](../part2-systems-stack/06d-profiling-debugging-and-performance-sop.md) | XID/ECC、driver/runtime version、NCCL log、pod event | 混淆驱动、runtime、镜像和调度层问题 |
| 修了一个点，复测无法证明真的好了 | [附录B](./tooling-map.md)、[附录C](./checklists.md) | [第21章](../part7-reliability-security/21-observability-and-capacity.md)、[第20d章](../part6-platform-and-orchestration/20d-capacity-and-troubleshooting-sop.md) | EvidenceBundle、固定 workload、阈值、修复前后对比 | 只保存截图，不保存可复现命令和阈值 |

## 数据 / 制品 / 安全

| 你遇到的问题 / 症状 | 先读章节 | 深入章节 | 关键指标 / 证据 | 常见误区 |
|---|---|---|---|---|
| 数据质量、版本或血缘变化导致训练结果不可复现 | [第11章](../part4-data-and-storage/11-data-pipeline.md)、[第11e章](../part4-data-and-storage/11e-data-versioning-and-lineage.md) | [第11b章](../part4-data-and-storage/11b-data-cleaning-dedup-quality.md)、[第11c章](../part4-data-and-storage/11c-tokenization-and-dataset-formats.md) | dataset manifest、lineage、dedup ratio、质量分布 | 只记录代码版本，不记录数据版本 |
| 模型、checkpoint、adapter、tokenizer 对不上 | [第12章](../part4-data-and-storage/12-artifacts-and-checkpoints.md)、[第12a章](../part4-data-and-storage/12a-model-registry.md) | [第12b章](../part4-data-and-storage/12b-checkpoint-engineering.md)、[第10c章](../part3-training-infra/10c-finetuning-and-multi-adapter.md) | artifact digest、manifest、compatibility matrix、restore test | 把文件路径当成制品身份 |
| 镜像、依赖、secret 或模型供应链无法过审 | [第18c章](../part6-platform-and-orchestration/18c-artifact-supply-chain-and-image-governance.md)、[第23章](../part7-reliability-security/23-security-isolation-and-governance.md) | [第12d章](../part4-data-and-storage/12d-supply-chain-and-signing.md)、[附录F](./version-matrix.md) | SBOM、signature、attestation、policy report、audit log | 扫描通过就上线，不做签名、准入和例外到期 |

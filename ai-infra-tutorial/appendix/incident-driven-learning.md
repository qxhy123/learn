# 附录G：事故驱动学习路径

> 本附录是 runbook 索引，不替代完整排障章节。使用方式是：先按事故类型收集第一批证据，建立假设树，再跳到对应章节深读；任何缓解动作都必须有复测和 rollback 目标。

## 事故索引

| 事故 | 先读章节 | 深入章节 |
|------|----------|----------|
| GPU 利用率低 | [第2章](../part1-foundations/02-compute-storage-network.md)、[第7章](../part3-training-infra/07-single-node-training.md) | [第0b章](../part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md)、[第05b章](../part2-systems-stack/05b-host-device-io-pcie-numa-and-overlap.md)、[第06d章](../part2-systems-stack/06d-profiling-debugging-and-performance-sop.md)、[第11d章](../part4-data-and-storage/11d-streaming-and-dataloader-engineering.md) |
| NCCL timeout | [第8章](../part3-training-infra/08-data-parallel.md)、[第10章](../part3-training-infra/10-memory-checkpointing-and-recovery.md) | [第0d章](../part0-foundations-of-systems/0d-network-stack-fundamentals.md)、[第0d4章](../part0-foundations-of-systems/0d4-nccl-collectives-and-network-diagnostics.md)、[第05c章](../part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md)、[第19b章](../part6-platform-and-orchestration/19b-gpu-scheduling-and-topology.md) |
| checkpoint 卡住 | [第10章](../part3-training-infra/10-memory-checkpointing-and-recovery.md)、[第12b章](../part4-data-and-storage/12b-checkpoint-engineering.md) | [第0b2章](../part0-foundations-of-systems/0b2-page-cache-writeback-and-huge-pages.md)、[第0c3章](../part0-foundations-of-systems/0c3-storage-semantics-fsync-direct-io-and-checkpoints.md)、[第05d章](../part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md)、[第12章](../part4-data-and-storage/12-artifacts-and-checkpoints.md) |
| TTFT 飙升 | [第14章](../part5-serving-infra/14-online-inference-architecture.md)、[第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md) | [第16a章](../part5-serving-infra/16a-vllm-internals.md)、[第16c章](../part5-serving-infra/16c-trt-llm-internals.md)、[第20c章](../part6-platform-and-orchestration/20c-inference-autoscaling.md)、[第21章](../part7-reliability-security/21-observability-and-capacity.md) |
| KV OOM | [第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md)、[第16章](../part5-serving-infra/16-quantization-compilation-and-engines.md) | [第16a章](../part5-serving-infra/16a-vllm-internals.md)、[第17章](../part5-serving-infra/17-multitenancy-and-cost.md)、[第20b章](../part6-platform-and-orchestration/20b-gpu-partitioning-and-sharing.md)、[第21章](../part7-reliability-security/21-observability-and-capacity.md) |
| RAG 召回下降 | [第13章](../part4-data-and-storage/13-feature-vector-and-cache.md)、[第13d章](../part4-data-and-storage/13d-rag-engineering.md) | [第11e章](../part4-data-and-storage/11e-data-versioning-and-lineage.md)、[第13b章](../part4-data-and-storage/13b-vector-index-algorithms.md)、[第13c章](../part4-data-and-storage/13c-vector-db-selection-and-operations.md)、[第13e章](../part4-data-and-storage/13e-embedding-and-cache-layer.md) |
| 发布后质量回退 | [第12c章](../part4-data-and-storage/12c-release-governance.md)、[第22章](../part7-reliability-security/22-evaluation-release-and-incident.md) | [第11f章](../part4-data-and-storage/11f-data-flywheel-online-learning.md)、[第12a章](../part4-data-and-storage/12a-model-registry.md)、[第12d章](../part4-data-and-storage/12d-supply-chain-and-signing.md)、[第23章](../part7-reliability-security/23-security-isolation-and-governance.md) |

## GPU 利用率低

- 症状：训练 step time 变长，SM util 长期低于预期，GPU 空洞和 batch wait 增加。
- 第一批证据：固定 workload 的 `nvidia-smi dmon` / DCGM、`torch.profiler`、`nsys` 时间线、DataLoader wait、H2D 带宽、CPU `perf stat`、数据源吞吐。
- 假设树：数据加载或 tokenizer 慢；H2D/NUMA/pinned memory 不匹配；kernel launch / 小算子过多；存储或网络喂数不足；调度把 GPU、CPU、NIC 放错拓扑。
- 先读章节：第2章、第7章。
- 深入章节：第0b章、第05b章、第06d章、第11d章。
- 常见错误处理：只加 DataLoader worker；只看平均 GPU util；把 profiler warmup、首次编译和真实 steady state 混在一起。
- 复测/rollback：固定 batch、seq_len、数据版本和节点后复测 step time、tokens/s、SM util、batch wait；若改动导致吞吐或正确性回退，rollback 到上一版数据加载、runtime 参数或节点 placement。

## NCCL timeout

- 症状：部分 rank 卡在 collective，训练日志出现 `NCCL timeout`，step p99 抖动或作业整体 abort。
- 第一批证据：rank 日志、`NCCL_DEBUG=INFO`、`nccl-tests`、`NCCL_TOPO_DUMP_FILE`、GPU/NIC/CPU 拓扑、IB/RoCE 端口错误、PFC/ECN、调度 placement、同时间 checkpoint/数据流量。
- 假设树：rank placement 或 GPU-NIC rail 错；socket fallback；慢 rank 或节点健康异常；RoCE 拥塞和 PFC pause；checkpoint/对象存储归档抢网络；NCCL/env 版本组合变化。
- 先读章节：第8章、第10章。
- 深入章节：第0d章、第0d4章、第05c章、第19b章。
- 常见错误处理：先调大 timeout；只在单节点跑 `nccl-tests`；只看应用日志不对齐交换机和调度时间线。
- 复测/rollback：用同节点集合复测 `nccl-tests` 和真实训练窗口；高峰期重复 smoke test；若修复无效，rollback 到上一版 NCCL/env、placement policy 或流量隔离配置。

## checkpoint 卡住

- 症状：checkpoint 阶段 GPU util 掉到 0，写入前快后慢，`fsync` 或 manifest 提交长尾，恢复演练失败。
- 第一批证据：checkpoint size/shard 数、写入耗时分布、`iostat -xz 1`、`pidstat -d 1`、`Dirty/Writeback`、对象存储 put/list 延迟、并行文件系统 MDS/OST 指标、manifest 和 checksum。
- 假设树：dirty page 债务集中偿还；rank 并发写打满存储；小文件或 manifest 元数据热点；checkpoint storm；对象存储语义被当 POSIX 使用；原子提交协议缺失。
- 先读章节：第10章、第12b章。
- 深入章节：第0b2章、第0c3章、第05d章、第12章。
- 常见错误处理：只减少 checkpoint 频率；直接跳过 `fsync`；只保存权重不保存 optimizer、RNG 和数据游标；恢复路径接受半成品。
- 复测/rollback：复测 checkpoint p95/p99、restore drill、kill -9 崩溃矩阵和 loss resume delta；若失败，rollback 到上一个完整 manifest 和已验证的 checkpoint writer。

## TTFT 飙升

- 症状：Time To First Token 或 P95/P99 突然升高，TPOT 可能正常，请求队列和 prefill 等待变长。
- 第一批证据：按 prompt length 分桶的 TTFT/TPOT、queue wait、batch size、prefill/decode 分解、cold start、路由命中、prefix cache hit、engine 日志、发布 diff。
- 假设树：prefill 队列堆积；长 prompt 分布变化；continuous batching 参数变化；冷启动或扩容滞后；prefix cache 被路由打散；网关/下游依赖增加首包等待；新模型或量化 engine 改变 prefill 性能。
- 先读章节：第14章、第15章。
- 深入章节：第16a章、第16c章、第20c章、第21章。
- 常见错误处理：只扩副本不看 warm pool；把 TTFT 和 TPOT 混在一个延迟指标里；只按 QPS 扩容而忽略输入 token 分布。
- 复测/rollback：用固定 trace 回放，分桶复测 TTFT、queue wait、prefill tokens/s、P99；若 SLO burn 继续，rollback 到上一版 engine、batching 参数、router 或模型 ReleaseUnit。

## KV OOM

- 症状：服务出现 CUDA OOM、KV cache eviction 激增、worker 重启、长上下文请求失败或多租户互相挤占显存。
- 第一批证据：模型尺寸、max context、并发、active sequence、KV block 使用率、evict/recompute、GPU memory summary、OOM 日志、tenant 和路由维度。
- 假设树：max_num_seqs/max_model_len 超预算；长上下文流量突增；prefix cache 或 speculative decoding 额外占用；LoRA/adapter 或 CUDA Graph buffer 占用未入账；多租户 headroom 不足；量化策略未覆盖 KV。
- 先读章节：第15章、第16章。
- 深入章节：第16a章、第17章、第20b章、第21章。
- 常见错误处理：只降低 batch size；只看权重显存不算 KV；允许 OOM 后靠重启恢复；没有按租户和 prompt length 做配额。
- 复测/rollback：用长上下文和峰值并发压测复测 OOM 次数、KV block watermarks、吞吐和 P99；必要时 rollback 到较小 context、上一版 engine 参数、限流策略或模型副本规格。

## RAG 召回下降

- 症状：Recall@K、MRR、命中文档覆盖率下降，答案引用变少或引用不相关，线上用户反馈质量下降。
- 第一批证据：golden query 集、Recall@K/MRR、embedding 模型版本、index manifest、chunking 配置、过滤条件、hybrid search 权重、reranker 版本、向量库 build/compact 日志。
- 假设树：数据源或 chunking 变化；embedding 模型/tokenizer 变更未全量重建；index alias 指错；ANN 参数或压缩降低召回；metadata filter 误过滤；reranker/prompt 掩盖检索回退；增量索引漏写。
- 先读章节：第13章、第13d章。
- 深入章节：第11e章、第13b章、第13c章、第13e章。
- 常见错误处理：只调 prompt；只看生成质量不看检索指标；线上索引和离线评测索引不是同一个 immutable id；重建索引缺少 shadow 对比。
- 复测/rollback：固定 golden query 和线上抽样回放，复测 Recall@K、MRR、引用正确率、延迟；若未达阈值，rollback 到上一版 index manifest、embedding 模型、chunking 或 reranker。

## 发布后质量回退

- 症状：新模型、prompt、router、adapter、RAG index 或推理引擎发布后，离线或线上质量指标下降，bad output rate、投诉或人工抽检失败升高。
- 第一批证据：ReleaseUnit diff、模型/tokenizer/prompt/adapter/index/image digest、离线 eval、A/B 和 canary 指标、golden prompt diff、日志采样、回滚目标、审批记录。
- 假设树：模型权重与 tokenizer/config 不匹配；prompt 或 guardrail 变更影响分布；RAG index/embedding 版本错配；量化或 engine 变更带来质量损失；路由灰度污染实验；数据飞轮引入反馈污染；评测门禁覆盖不足。
- 先读章节：第12c章、第22章。
- 深入章节：第11f章、第12a章、第12d章、第23章。
- 常见错误处理：只回滚模型不回滚 tokenizer、prompt、adapter 或 index；只看系统 SLI 不看质量 SLI；没有冻结实验分桶；事后无法还原发布单元。
- 复测/rollback：回放 golden set、线上 shadow 样本和 canary 阈值；若质量未恢复，rollback 整个 ReleaseUnit 到明确 rollback target，并记录复盘和防复发门禁。


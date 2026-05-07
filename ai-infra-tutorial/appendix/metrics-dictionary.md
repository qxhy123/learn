# 附录O：指标字典与归因地图

本附录把常见指标翻译成可验证的工程判断。每个指标都只是一条证据，不能单独证明根因；归因时要继续检查相邻指标、版本、拓扑、输入分布和发布变更。

## 1. 使用规则

| 规则 | 含义 |
|------|------|
| 先分层 | 先判断指标属于训练、推理、平台、数据/RAG 哪条链路，避免把平台排队误判为模型慢。 |
| 看分位数 | 平均值只能看趋势，SLO 和事故归因必须看 P50/P95/P99、最慢 rank、最坏租户或最长 prompt。 |
| 绑定 workload | 指标必须绑定模型、数据版本、输入/输出长度、batch、并发、硬件、镜像和引擎版本。 |
| 记录反证 | 每次归因都写“它不能证明什么”，防止把相关性当因果。 |
| 下一跳检查 | 任何单点异常都要进入下一指标，而不是直接扩容、重启或换框架。 |

## 2. 训练指标

| 指标 | 定义 | 上升/下降通常意味着 | 不能证明什么 | 下一指标 | 相关章节 |
|------|------|----------------------|--------------|----------|----------|
| MFU | Model FLOPs Utilization，实际模型 FLOPs / 理论硬件峰值 FLOPs，通常按有效训练 step 估算。 | 上升通常表示算子形状、batch、并行策略或通信 overlap 更好；下降可能是 GPU 空等、通信尾巴、数据等待或小算子增多。 | 不能单独证明训练更快或成本更低；FLOPs 口径错误会让 MFU 失真。 | step time、GPU SM util、NCCL tail、data wait。 | [第7章](../part3-training-infra/07-single-node-training.md)、[第8章](../part3-training-infra/08-data-parallel.md)、[第21章](../part7-reliability-security/21-observability-and-capacity.md) |
| HFU | Hardware FLOPs Utilization，硬件实际执行 FLOPs / 理论峰值，包含重算、padding、无效计算。 | 上升可能表示硬件更忙；若 MFU 不升，可能是无效计算或重算增加。下降可能是 kernel 空洞、同步或等待。 | 不能证明有效训练效率提升；高 HFU 可能只是做了更多无用 work。 | MFU、tokens/s、activation recompute ratio、padding ratio。 | [第7章](../part3-training-infra/07-single-node-training.md)、[第10章](../part3-training-infra/10-memory-checkpointing-and-recovery.md) |
| step time | 一个 optimizer step 的端到端 wall time，含 load、H2D、forward、backward、sync、update 和日志/checkpoint。 | 上升表示训练吞吐下降；下降表示端到端更快，但要确认 batch、accumulation 和数据没变。 | 不能说明瓶颈在哪一段，也不能证明收敛速度变好。 | step breakdown、data wait、AllReduce time、checkpoint time。 | [第7章](../part3-training-infra/07-single-node-training.md)、[第8章](../part3-training-infra/08-data-parallel.md) |
| data wait | GPU 或训练循环等待 DataLoader、解码、采样、远端读取或 H2D 的时间。 | 上升通常表示数据读取、CPU preprocessing、Page Cache、对象存储、worker 数或 NUMA/H2D 路径有问题。下降说明数据供给更接近计算消费。 | 不能证明存储本身慢；也可能是 CPU、网络、batch 拼接或主进程同步造成。 | DataLoader worker util、I/O await、H2D copy time、CPU IPC、page fault。 | [第11d章](../part4-data-and-storage/11d-streaming-and-dataloader-engineering.md)、[第5d章](../part2-systems-stack/05d-training-storage-checkpoint-and-io-diagnostics.md) |
| AllReduce time | 梯度 AllReduce 在 step timeline 中占用或暴露的时间。 | 上升通常表示通信量变大、bucket/overlap 变差、拓扑退化、慢 rank 或网络拥塞。下降表示同步成本降低或 overlap 更好。 | 不能单独证明网络故障；bucket 变大、计算变短或 profiler 口径变化也会改变它。 | NCCL busbw、rank skew、bucket timeline、NIC/RDMA 错误。 | [第8章](../part3-training-infra/08-data-parallel.md)、[第5c章](../part2-systems-stack/05c-rdma-collectives-and-cluster-topology.md) |
| rank skew | 同一 step 内不同 rank 的计算、数据、通信或总耗时差异。 | 上升通常表示数据倾斜、慢节点、热文件、拓扑不一致、降频或邻居干扰；下降表示同步组更均衡。 | 不能证明最慢 rank 的根因；只说明同步组被不均衡拖住。 | per-rank data wait、GPU clock/throttle、NCCL trace、node placement。 | [第8章](../part3-training-infra/08-data-parallel.md)、[第19b章](../part6-platform-and-orchestration/19b-gpu-scheduling-and-topology.md) |
| samples/s 或 tokens/s | 单位时间完成的训练样本数或 token 数，通常按全局吞吐统计。 | 上升通常表示训练产出更高；下降可能来自 step time 上升、有效 batch 改变、数据过滤或并行效率下降。 | 不能证明成本效率、收敛质量或模型质量提升。 | step time、global batch、loss curve、GPU hours/token。 | [第7章](../part3-training-infra/07-single-node-training.md)、[第21章](../part7-reliability-security/21-observability-and-capacity.md) |

## 3. 推理指标

| 指标 | 定义 | 上升/下降通常意味着 | 不能证明什么 | 下一指标 | 相关章节 |
|------|------|----------------------|--------------|----------|----------|
| TTFT | Time To First Token，从请求进入服务到首个 token 返回的时间。 | 上升通常表示排队、路由、tokenize、prefill、冷启动或长 prompt 变差；下降表示首包体验改善。 | 不能证明流式生成稳定，也不能证明总耗时低。 | queue wait、prefill time、prompt tokens、cold start、KV hit。 | [第14章](../part5-serving-infra/14-online-inference-architecture.md)、[第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md) |
| TPOT | Time Per Output Token，输出 token 平均生成耗时。 | 上升通常表示 decode 慢、batch 形态差、KV 访问压力、显存带宽或引擎路径退化。下降表示输出吞吐改善。 | 不能反映首 token 体验，也可能被输出长度分布掩盖。 | ITL、decode time、output tokens/s、KV usage、batch occupancy。 | [第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md)、[第16章](../part5-serving-infra/16-quantization-compilation-and-engines.md) |
| ITL | Inter-Token Latency，流式输出中相邻 token 的间隔。 | 上升通常表示 decode iteration、调度、公平性或 KV 读取变差；抖动增大说明用户感知不稳定。 | 不能证明 prefill 或排队没问题；首 token 之前的问题不体现在 ITL。 | decode batch size、active sequences、KV bandwidth、scheduler policy。 | [第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md) |
| goodput | 满足 SLO 和质量约束的有效吞吐，例如达标 token/s 或请求/s。 | 上升表示可售卖容量提升；下降可能是尾延迟、错误率、重试或质量门禁失败。 | 不能替代 raw throughput；也不能说明所有租户公平。 | error rate、P99、queue wait、cost/request、tenant split。 | [第17章](../part5-serving-infra/17-multitenancy-and-cost.md)、[第21章](../part7-reliability-security/21-observability-and-capacity.md) |
| queue wait | 请求进入副本、引擎或调度器前等待的时间。 | 上升通常表示并发超过 admission、batching 过度、队列策略不公平或副本不足。下降表示入口拥塞缓解。 | 不能证明模型 runtime 慢；也可能是网关、限流或上游重试造成。 | TTFT、active requests、replica load、autoscaling lag、429/503。 | [第14章](../part5-serving-infra/14-online-inference-architecture.md)、[第20c章](../part6-platform-and-orchestration/20c-inference-autoscaling.md) |
| prefill time | 处理 prompt 并生成初始 KV 的时间。 | 上升通常表示 prompt 变长、chunked prefill 配置不当、计算资源不足或 prefix cache 未命中。下降表示首 token 前计算压力降低。 | 不能说明 decode 阶段表现，也不能证明总成本下降。 | prompt length、prefix cache hit、TTFT、chunk size、GPU util。 | [第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md)、[第16a章](../part5-serving-infra/16a-vllm-internals.md) |
| decode time | decode iteration 或输出阶段累计耗时。 | 上升通常表示活跃序列多、KV 读写压力大、batch occupancy 差或 speculative decoding 接受率低。 | 不能证明 prefill 池不足，也不能证明用户首包体验差。 | ITL、TPOT、active sequences、KV cache bytes、acceptance rate。 | [第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md)、[第16b章](../part5-serving-infra/16b-sglang-internals.md) |
| KV usage | KV Cache 已用字节、block 数、碎片率或 headroom。 | 上升通常表示长上下文、高并发、prefix 复用不足或 admission 放得太宽；下降可能表示请求减少、回收改善或上下文变短。 | 不能证明显存 OOM 一定来自 KV；权重、workspace、CUDA Graph buffer 也可能占用显存。 | active sequences、context length、block fragmentation、GPU memory headroom。 | [第15章](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md)、[第16a章](../part5-serving-infra/16a-vllm-internals.md) |

## 4. 平台指标

| 指标 | 定义 | 上升/下降通常意味着 | 不能证明什么 | 下一指标 | 相关章节 |
|------|------|----------------------|--------------|----------|----------|
| queue wait | 作业或请求在平台队列中等待资源、配额、优先级或 gang 条件满足的时间。 | 上升通常表示容量不足、配额耗尽、碎片化、优先级挤压或 gang scheduling 等待。下降表示 admission 更快。 | 不能证明训练 step 或推理 runtime 慢。 | allocation latency、quota usage、pending reason、fragmentation。 | [第20章](../part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md)、[第20a章](../part6-platform-and-orchestration/20a-queues-quotas-priority-and-fairness.md) |
| allocation latency | 从调度通过到 Pod/Job 获得可用 GPU、CPU、存储、网络和镜像就绪的时间。 | 上升通常表示镜像拉取、设备插件、节点初始化、PVC attach、拓扑约束或 runtime 注入慢。 | 不能证明队列策略不公平；也不能说明模型执行慢。 | image pull time、device plugin events、node readiness、PVC attach time。 | [第18章](../part6-platform-and-orchestration/18-containers-and-runtime.md)、[第19章](../part6-platform-and-orchestration/19-kubernetes-for-ai.md) |
| preemption count/rate | 作业、Pod 或副本被更高优先级任务抢占的次数或比例。 | 上升通常表示容量紧张、优先级策略激进或低优任务被当作弹性池。下降表示资源稳定性更好。 | 不能证明被抢占任务本身有故障；也不能证明平台成本更低。 | restart time、checkpoint interval、wasted GPU hours、priority mix。 | [第20a章](../part6-platform-and-orchestration/20a-queues-quotas-priority-and-fairness.md)、[第10章](../part3-training-infra/10-memory-checkpointing-and-recovery.md) |
| cost | 按租户、队列、模型、任务或请求归因的 GPU/CPU/存储/网络成本。 | 上升通常表示用量、单价、空闲、重试、长上下文或低 goodput 增加；下降表示资源效率或采购单价改善。 | 不能证明业务价值提高或降低；没有标签治理时归因可能错误。 | GPU hours、goodput、idle ratio、egress/storage cost、标签覆盖率。 | [第17章](../part5-serving-infra/17-multitenancy-and-cost.md)、[第21章](../part7-reliability-security/21-observability-and-capacity.md) |
| cardinality | metrics/log labels 的唯一组合数量。 | 上升通常表示租户、模型、请求 ID、prompt、pod 或动态标签进入指标，可能推高观测成本和查询延迟。下降表示标签治理或聚合改善。 | 不能证明系统业务流量变大；也不能证明可观测性更好。 | top labels、series count、query latency、retention cost。 | [第21章](../part7-reliability-security/21-observability-and-capacity.md) |
| GPU allocation ratio | 已分配 GPU / 可用 GPU，或按 flavor/MIG 分片统计的分配率。 | 上升表示资源被占用更多；下降可能是需求减少、碎片化、故障隔离或调度约束过严。 | 不能证明 GPU 被有效使用；已分配不等于在产出。 | GPU util、queue wait、fragmentation、idle allocated GPU。 | [第19b章](../part6-platform-and-orchestration/19b-gpu-scheduling-and-topology.md)、[第20b章](../part6-platform-and-orchestration/20b-gpu-partitioning-and-sharing.md) |

## 5. 数据与 RAG 指标

| 指标 | 定义 | 上升/下降通常意味着 | 不能证明什么 | 下一指标 | 相关章节 |
|------|------|----------------------|--------------|----------|----------|
| ingestion lag | 数据从源系统产生到进入训练、索引或特征管道的延迟。 | 上升通常表示采集、队列、schema、清洗或下游写入变慢；下降表示数据新鲜度改善。 | 不能证明数据质量好，也不能证明模型效果变好。 | DLQ rate、schema error、throughput、freshness by source。 | [第11a章](../part4-data-and-storage/11a-data-ingestion.md)、[第11e章](../part4-data-and-storage/11e-data-versioning-and-lineage.md) |
| dedup ratio | 去重后移除样本或 token 的比例。 | 上升通常表示源重复、爬虫污染或清洗规则更激进；下降可能表示源更多样或去重覆盖不足。 | 不能证明数据质量一定提升；过度去重可能损失有用分布。 | quality score、domain mix、near-dup threshold、eval delta。 | [第11b章](../part4-data-and-storage/11b-data-cleaning-dedup-quality.md) |
| tokenization throughput | tokenizer 或 dataset packing 每秒处理的文本、样本或 token 数。 | 上升表示 CPU/SIMD/并行/格式路径改善；下降可能让 GPU data wait 增加。 | 不能证明训练吞吐一定提升；瓶颈可能在存储、H2D 或训练同步。 | data wait、CPU IPC、I/O throughput、packed sequence utilization。 | [第11c章](../part4-data-and-storage/11c-tokenization-and-dataset-formats.md)、[第0a-4章](../part0-foundations-of-systems/0a4-simd.md) |
| retrieval latency | RAG 检索阶段从 query 到候选文档返回的耗时。 | 上升通常表示向量库负载、过滤条件、索引参数、网络或 rerank 前置逻辑变慢。下降表示检索链路更快。 | 不能证明答案质量更好，也不能证明生成端没问题。 | recall@k、index QPS、filter selectivity、rerank latency。 | [第13c章](../part4-data-and-storage/13c-vector-db-selection-and-operations.md)、[第13d章](../part4-data-and-storage/13d-rag-engineering.md) |
| recall@k | 标注相关文档在前 k 个检索结果中出现的比例。 | 上升通常表示 embedding、chunking、索引参数或 hybrid retrieval 改善；下降可能导致回答缺证据。 | 不能证明最终答案正确；reranker、prompt 和模型仍可能失败。 | MRR/NDCG、citation accuracy、answer correctness、latency。 | [第13b章](../part4-data-and-storage/13b-vector-index-algorithms.md)、[第13d章](../part4-data-and-storage/13d-rag-engineering.md) |
| rerank latency | reranker 对候选文档重排的耗时。 | 上升通常表示候选数、文档长度、模型变大或批处理变差；下降表示重排成本降低。 | 不能证明召回或答案质量更好；也不能证明端到端 TTFT 达标。 | candidate count、rerank quality delta、TTFT、GPU/CPU util。 | [第13d章](../part4-data-and-storage/13d-rag-engineering.md)、[第14章](../part5-serving-infra/14-online-inference-architecture.md) |
| citation accuracy | 生成答案中的引用是否支持对应断言。 | 上升表示 RAG 证据链更可靠；下降可能是检索、rerank、prompt 或模型忠实性问题。 | 不能证明答案完整，也不能证明所有事实都正确。 | recall@k、answer correctness、hallucination rate、judge agreement。 | [第13d章](../part4-data-and-storage/13d-rag-engineering.md)、[第22章](../part7-reliability-security/22-evaluation-release-and-incident.md) |
| embedding cache hit rate | embedding、semantic cache 或 query cache 命中的比例。 | 上升通常表示重复查询多或缓存策略有效；下降可能表示流量分布变化、版本切换或 TTL 太短。 | 不能证明回答质量更好；缓存命中还可能返回过期语义。 | cache staleness、embedding version、retrieval latency、quality eval。 | [第13e章](../part4-data-and-storage/13e-embedding-and-cache-layer.md) |

## 6. 归因地图

| 现象 | 首先看 | 然后看 | 常见错误归因 |
|------|--------|--------|--------------|
| 训练 GPU 利用率低 | data wait、step breakdown、H2D copy time | CPU IPC、I/O await、rank skew、NCCL tail | 直接认为模型代码差或 GPU 不够。 |
| 扩卡后吞吐不线性 | scaling efficiency、AllReduce time、rank skew | NCCL busbw、bucket overlap、node topology、data skew | 只看平均 GPU utilization。 |
| TTFT P99 突然升高 | queue wait、prefill time、prompt length | cold start、prefix cache hit、autoscaling lag、路由变更 | 直接归因于模型变慢。 |
| 流式输出卡顿 | ITL、decode time、active sequences | KV usage、batch occupancy、scheduler policy、显存带宽 | 只看 TPOT 平均值。 |
| 平台排队变长 | queue wait、quota usage、pending reason | allocation latency、GPU fragmentation、preemption、队列优先级 | 误判为训练或推理 runtime 性能下降。 |
| 观测系统成本暴涨 | cardinality、series count、top labels | retention、query latency、tenant/model label 设计 | 误判为业务流量暴涨。 |
| RAG 答案不可信 | citation accuracy、recall@k | rerank quality、chunking、prompt、judge agreement | 只调生成模型温度或扩大上下文。 |

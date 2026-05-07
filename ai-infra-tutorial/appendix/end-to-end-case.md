# 附录E：端到端主线案例

## LLaMA-7B/70B 从数据到事故复盘的工程路线图

这个案例把 LLaMA-7B/70B 的生命周期串成一条可执行主线：数据准备 -> 单机 baseline -> 分布式训练 -> checkpoint -> registry -> vLLM serving -> autoscaling -> incident。它不是替代各章细节，而是帮助你在读完单章后，把产物、指标和故障串回同一条证据链。

### 0. 规模假设

| 项目 | 7B 路线 | 70B 路线 |
|------|---------|----------|
| 目标 | 验证数据、训练脚本、发布链路和服务 SLO | 验证多机并行、checkpoint 语义、容量治理和事故响应 |
| 训练形态 | 单机 8 GPU baseline 后扩到小规模 DP/FSDP | 多机 TP/PP/DP/FSDP 组合，强依赖拓扑与 checkpoint |
| Serving 形态 | 单副本或少量副本，重点看延迟和冷启动 | 多副本、TP serving、KV Cache 容量、扩缩容和限流 |
| 核心风险 | 数据管道喂不满 GPU、tokenizer/config 不一致 | 通信尾延迟、checkpoint 风暴、发布回滚和容量误判 |

## 阶段路线图

| 阶段 | 目标 | 关键产物 | 关键指标 | 常见故障 | 回链章节 |
|------|------|----------|----------|----------|----------|
| 1. 数据准备 | 把原始语料变成可复现、可恢复、可审计的训练样本。 | 数据快照 ID、清洗规则、tokenizer 版本、packing 配置、dataset manifest、resume offset 规则。 | 有效 token 数、重复率、过滤率、平均 sequence utilization、DataLoader 读吞吐、样本坏块率。 | tokenizer 与模型配置不匹配；packing 后 padding 过高；远程 shard 小文件过多；resume 后 shuffle 顺序变化。 | [11c](../part4-data-and-storage/11c-tokenization-and-dataset-formats.md)、[11d](../part4-data-and-storage/11d-streaming-and-dataloader-engineering.md)、[11e](../part4-data-and-storage/11e-data-versioning-and-lineage.md) |
| 2. 单机 baseline | 先在单机建立可解释的 step time、显存和吞吐基线，再决定是否扩到多机。 | baseline 配置、固定 batch/seq_len、profiler trace、显存预算表、失败样本清单、成本粗算。 | tokens/s、MFU/HFU、GPU utilization、HBM peak、DataLoader wait、H2D 时间、step time breakdown。 | GPU 利用率低但原因在 CPU tokenizer 或 DataLoader；OOM 来自 activation 或 optimizer state 估算错误；AMP/FP8 配置导致 loss spike。 | [07](../part3-training-infra/07-single-node-training.md)、[04](../part2-systems-stack/04-gpu-and-accelerators.md)、[05](../part2-systems-stack/05-memory-interconnect-io.md)、[10](../part3-training-infra/10-memory-checkpointing-and-recovery.md) |
| 3. 分布式训练 | 把单机 baseline 扩展到目标 GPU 数，同时保持吞吐、收敛和恢复语义可解释。 | 并行策略表、拓扑 placement、NCCL/env 配置、global batch 计划、扩展性 benchmark、训练 run spec。 | scaling efficiency、all-reduce/all-to-all 时间、通信占比、straggler gap、step variance、loss continuity。 | NCCL fallback 到 socket；跨机 TP 放错拓扑；global batch 调整后收敛变化；慢节点拖垮全局 step。 | [08](../part3-training-infra/08-data-parallel.md)、[09](../part3-training-infra/09-model-pipeline-parallel.md)、[10](../part3-training-infra/10-memory-checkpointing-and-recovery.md)、[20](../part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md) |
| 4. Checkpoint | 让训练能在节点故障、抢占、版本升级和发布转换中恢复到明确状态。 | sharded checkpoint、manifest、checksum、RNG/optimizer/scheduler/dataset state、retention 策略、restore drill 记录。 | checkpoint 写入时间、RPO、RTO、对象数量、写入带宽、restore 成功率、loss resume delta。 | manifest 未原子提交；只保存权重未保存数据游标；checkpoint storm 打爆存储；并行策略变更后无法转换。 | [10](../part3-training-infra/10-memory-checkpointing-and-recovery.md)、[12](../part4-data-and-storage/12-artifacts-and-checkpoints.md)、[12b](../part4-data-and-storage/12b-checkpoint-engineering.md) |
| 5. Registry | 把训练 checkpoint 转成可发布、可回滚、可审计的模型版本。 | model package、tokenizer/config、eval report、license/security metadata、stage alias、release gate 记录。 | 离线 eval 分数、golden prompt diff、artifact size、下载时间、签名/扫描状态、兼容性检查结果。 | checkpoint 与 tokenizer 版本错配；registry alias 被覆盖；量化后质量未过门禁；SafeTensors/签名缺失。 | [12a](../part4-data-and-storage/12a-model-registry.md)、[12c](../part4-data-and-storage/12c-release-governance.md)、[12d](../part4-data-and-storage/12d-supply-chain-and-signing.md)、[22](../part7-reliability-security/22-evaluation-release-and-incident.md) |
| 6. vLLM serving | 用 vLLM 建立线上副本，明确 prefill/decode、KV Cache、批处理和路由边界。 | serving spec、engine 参数、TP 配置、routing policy、warmup 脚本、load test report、rollback plan。 | TTFT、ITL、P95/P99、tokens/s、request/s、KV Cache hit/evict、batch size、OOM 次数。 | 长上下文挤爆 KV Cache；prefix cache 命中率被随机路由拉低；70B TP 副本冷启动过慢；量化策略降低质量。 | [14](../part5-serving-infra/14-online-inference-architecture.md)、[15](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md)、[16](../part5-serving-infra/16-quantization-compilation-and-engines.md)、[16a](../part5-serving-infra/16a-vllm-internals.md) |
| 7. Autoscaling | 让副本数、队列、配额和降级策略随负载变化，而不是只靠手工扩容。 | HPA/KEDA 或自研 scaler 配置、队列指标、SLO policy、tenant quota、冷启动预算、降级规则。 | queue wait、goodput、SLO burn rate、replica ready time、GPU utilization、cost/request、限流率。 | 只按 GPU 利用率扩容导致队列已爆；扩容速度慢于流量斜率；多租户互相抢 KV Cache；scale down 杀掉热副本。 | [17](../part5-serving-infra/17-multitenancy-and-cost.md)、[20](../part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md)、[21](../part7-reliability-security/21-observability-and-capacity.md) |
| 8. Incident | 把线上异常收敛为证据、缓解动作、根因和防复发项。 | incident timeline、dashboard snapshot、release diff、mitigation record、rollback/degrade 记录、postmortem。 | error rate、P99、SLO burn、OOM/restart count、queue depth、bad output rate、MTTA/MTTR。 | 新模型 alias 切换后质量下降；KV Cache OOM 触发重启风暴；上游流量突增但 autoscaler 滞后；回滚只回模型未回 tokenizer/prompt。 | [21](../part7-reliability-security/21-observability-and-capacity.md)、[22](../part7-reliability-security/22-evaluation-release-and-incident.md)、[23](../part7-reliability-security/23-security-isolation-and-governance.md)、[14](../part5-serving-infra/14-online-inference-architecture.md) |

## 读法

- 先沿表格横向读一遍，只确认每阶段是否有可交付的关键产物。
- 再纵向挑一个指标做追踪，例如 `tokens/s` 从数据、训练、serving 到 autoscaling 是否口径一致。
- 最后用“常见故障”反推证据：每个故障都应该能落到日志、指标、manifest、release record 或 incident timeline。

# 附录M：事故复盘案例库

> 本附录按 AI Infra 值班现场写法组织：先看症状和错误直觉，再沿证据链定位、修复、复测和预防。每个案例都是可复用的 postmortem 骨架。

## 1. GPU util 高但 MFU 低

- **症状**：`nvidia-smi` 显示 GPU utilization 90%+，但 tokens/s、samples/s 和 MFU 明显低于基线。
- **错误第一猜测**：GPU 算力不够，先加卡或换更贵实例。
- **证据链**：训练日志显示 step time 上升；profiler 中小 kernel、同步点、CPU launch gap 增多；SM util 高但 Tensor Core 利用不足；dataloader wait 或 H2D copy 与 compute 没有重叠。
- **定位路径**：先固定 workload 和 batch，拆 step timeline；再按 CPU 数据管道、framework dispatch、kernel shape、H2D overlap、NCCL tail 逐层排除。
- **修复**：合并小 op、启用 fused kernel / CUDA Graph、修正 batch shape、提高 dataloader 预取和 pinned memory 配置，必要时调整 bucket 与 overlap。
- **复测**：同一 commit、同一数据 shard 跑 200 step，对比 MFU、tokens/s、CUDA gap、CPU self time 和 p95 step time。
- **预防**：上线训练模板必须保存 profiler 摘要和 MFU 基线；容量评审禁止只用 GPU utilization 做结论。

## 2. NCCL timeout

- **症状**：多机训练随机 hang，日志出现 `NCCL timeout`、rank 退出不一致或 watchdog 触发。
- **错误第一猜测**：直接调大 timeout，或者重启后继续跑。
- **证据链**：`NCCL_DEBUG=INFO` 显示某些 rank 走 `NET/Socket` fallback；`nccl-tests` 跨节点带宽低于同规格基线；交换机端口错误计数、PFC pause 或 RoCE ECN 标记异常；慢 rank 与特定机架或 NIC 相关。
- **定位路径**：先确认是否所有 rank 卡在同一 collective；再查 GPU-NIC 拓扑、GID/HCA 选择、MTU、PFC/ECN、NCCL topo dump 和 pair-wise bandwidth。
- **修复**：修正 NCCL 网卡选择和拓扑文件；恢复 MTU/PFC/ECN 一致性；隔离异常 NIC/交换机端口；对训练作业启用 gang fail-fast 和自动重试。
- **复测**：跑 `nccl-tests` 覆盖全部节点组合，再跑业务小规模训练，要求 all-reduce bus bandwidth、p99 step time 和 timeout 计数回到基线。
- **预防**：新节点入池必须通过 NCCL 验收；交换机错误计数和 socket fallback 进入告警；拓扑漂移后自动阻止调度。

## 3. Checkpoint 卡住

- **症状**：训练 step 周期性停顿，checkpoint 阶段几十分钟无进展，恢复点落后。
- **错误第一猜测**：模型太大，只能降低保存频率。
- **证据链**：日志停在 shard upload 或 manifest commit；对象存储 PUT p99 升高；节点 dirty page、Page Cache 回写和本地盘队列堆积；部分 rank 保存完成但 barrier 等最慢 rank。
- **定位路径**：拆分 serialize、local write、remote upload、manifest commit、barrier 五段；按 rank 对比 shard 大小和写入耗时；检查并行度、multipart 配置、fsync 语义和对象存储限流。
- **修复**：启用 async checkpoint；调小单 shard 尾部风险；增加 multipart 并发但限制总 inflight；manifest 原子提交；必要时先写本地 NVMe 再后台归档。
- **复测**：连续保存 3 次 checkpoint，对比每段耗时、最慢 rank、对象存储 p99、恢复演练耗时和训练有效吞吐。
- **预防**：checkpoint SLA 与训练 SLA 分开监控；每个 checkpoint 必须有 manifest、校验和、恢复演练和过期策略。

## 4. P99 TTFT 突刺

- **症状**：线上 P99 TTFT 从 1-2s 升到 10s+，P50 和 TPOT 基本正常。
- **错误第一猜测**：模型 forward 变慢，先回滚权重。
- **证据链**：gateway 排队时间上升；prefill 队列被长 prompt 占住；冷启动比例升高；KV 可用块接近阈值；新流量中长上下文请求占比增加。
- **定位路径**：把 TTFT 拆成 gateway、scheduler queue、prefill、first decode；按 prompt length、tenant、model revision、engine replica 分组；检查 autoscaling 与 warm pool 时间线。
- **修复**：拆长短请求队列；限制超长 prompt 并启用 chunked prefill；增加 warm replicas；调整 max batched tokens 和 admission control；必要时回滚流量入口策略。
- **复测**：固定流量回放，比较 P50/P95/P99 TTFT、queue time、prefill occupancy、冷启动比例和拒绝率。
- **预防**：TTFT dashboard 必须按队列和 prompt length 分桶；上线新租户前做长上下文容量演练。

## 5. KV Cache OOM

- **症状**：推理实例间歇 OOM 或频繁驱逐，错误集中在高并发长上下文时段。
- **错误第一猜测**：显存不够，直接降低并发或换更大 GPU。
- **证据链**：权重显存稳定但 KV block 使用率冲顶；`max_num_seqs`、`max_model_len` 和 batch token 上限组合超过 engine shape；长输出请求占比上升；prefix cache 命中下降。
- **定位路径**：核算权重、workspace、KV、fragmentation 和 safety headroom；按 tenant 统计 prompt/output token 分布；查看 OOM 前 block allocator 和 scheduler 日志。
- **修复**：降低 max concurrency 或 max model len；启用 KV FP8 / prefix cache；设置 token budget admission；拆分长上下文专用池；调整 `gpu_memory_utilization` 留足 headroom。
- **复测**：用峰值 token 分布压测，确认 OOM 为零、KV 使用率低于阈值、TTFT/TPOT 未超 SLO。
- **预防**：容量表必须用 token 分布而不是 QPS；新增模型或租户前更新 KV headroom 预算。

## 6. Tokenizer / 版本导致质量回归

- **症状**：发布后离线评测或线上反馈质量下降，但延迟、错误率和 GPU 指标正常。
- **错误第一猜测**：模型权重训练坏了。
- **证据链**：model digest 未变，但 tokenizer 文件、chat template 或 special tokens 版本变更；相同输入 token ids 不一致；A/B 中短文本影响小，工具调用或多轮对话下降明显。
- **定位路径**：固定一批 golden prompts，对比 tokenization、prompt rendering、stop tokens、tool schema 和 generation config；再核对 registry 中 model、tokenizer、adapter、engine 的兼容矩阵。
- **修复**：回滚 tokenizer / template / generation config；补齐 artifact bundle 绑定；对不兼容 adapter 阻止上线。
- **复测**：golden set 逐条 diff token ids 和输出；跑质量 eval gate、工具调用成功率和线上 shadow 指标。
- **预防**：tokenizer、chat template、generation config 与权重同属 ReleaseUnit；发布门禁必须包含 token-level diff。

## 7. CUDA / Driver / Runtime 不匹配

- **症状**：新镜像上线后部分节点启动失败、CUDA kernel 报错、容器内看不到 GPU 或性能异常下降。
- **错误第一猜测**：应用代码改坏了。
- **证据链**：失败只集中在特定节点池；容器内 `nvidia-smi`、driver version、CUDA runtime、PyTorch CUDA build 不一致；device plugin 事件显示设备注入失败；XID 或 runtime error 与镜像 rollout 同时发生。
- **定位路径**：按节点池、镜像 digest、driver、container toolkit、GPU Operator 版本交叉分组；运行最小 CUDA workload；核对版本矩阵和兼容范围。
- **修复**：回滚镜像或节点池；统一 driver / runtime 基线；修复 GPU Operator 和 device plugin；为不兼容节点加 taint，阻止继续调度。
- **复测**：每个节点池运行 `nvidia-smi`、最小 CUDA 程序和一条业务 smoke test，对比启动成功率、kernel error 和性能基线。
- **预防**：镜像发布前做矩阵测试；节点入池绑定版本标签；准入策略校验镜像 CUDA 需求和节点能力。

## 8. 数据血缘回归

- **症状**：训练 loss、评测指标或下游召回突然变化，代码和模型配置看似未变。
- **错误第一猜测**：随机种子或训练不稳定。
- **证据链**：dataset manifest 指向新快照；过滤规则、去重版本或样本权重变化；同一样本 ID 的内容 hash 变化；OpenLineage / DVC / lakeFS 显示上游任务重跑。
- **定位路径**：锁定训练 run 的 dataset version、tokenizer version、sample count、语言/域分布和质量分桶；对比上一个健康 run 的 manifest diff。
- **修复**：回滚到上一版数据快照；修正清洗或 join 逻辑；重新生成 manifest 并补 lineage；必要时废弃受污染 checkpoint。
- **复测**：跑小规模复现实验，对比 loss curve、eval score、样本分布、hash 校验和数据质量指标。
- **预防**：训练任务只接受 immutable manifest；数据管道变更必须经过分布 diff 和 golden sample audit。

## 9. 对象存储尾延迟

- **症状**：训练数据读取、checkpoint 上传或 RAG 文档加载偶发超时，平均吞吐正常但 p99 很差。
- **错误第一猜测**：网络带宽不足。
- **证据链**：对象存储 GET/PUT p99 和 5xx/429 上升；热点 prefix 或小对象请求激增；客户端重试放大流量；节点本地网络无拥塞证据。
- **定位路径**：按 bucket、prefix、对象大小、operation、region/zone 和 client 版本切分；检查 multipart、connection pool、retry backoff 和并发上限。
- **修复**：打散热点 prefix；合并小文件或使用 shard 格式；调整连接池和指数退避；为训练热数据加本地 NVMe / 并行文件系统缓存。
- **复测**：回放同一对象访问序列，比较 GET/PUT p95/p99、重试次数、端到端 step time 或召回链路耗时。
- **预防**：对象存储 SLI 使用尾延迟和错误率；数据格式评审必须包含对象大小分布和 prefix 设计。

## 10. 多租户配额 / 容量事故

- **症状**：一个大租户上线后，其他租户训练排队暴涨或推理 SLO 被挤占。
- **错误第一猜测**：集群容量整体不足，马上扩容。
- **证据链**：queue wait 和 pending reason 指向单租户占用；quota borrow 未归还；GPU flavor 碎片化；抢占策略没有保护线上服务；CapacityLedger 中 warm pool 被训练任务吃掉。
- **定位路径**：按 tenant、queue、priority、GPU flavor、MIG/MPS 形态和 workload 类型查看 admission timeline；核对实际占用与声明配额。
- **修复**：冻结低优先级提交；恢复线上 serving reserve；调整 borrow/lend 和 preemption；拆分训练、批推理、在线推理节点池。
- **复测**：用多租户混合流量演练，确认 queue wait、抢占结果、SLO、配额归还和 GPU 碎片率符合预期。
- **预防**：容量变更必须更新 CapacityLedger；大租户上线前做配额演练；公平性和保留容量进入发布门禁。

## 11. RAG 召回下降

- **症状**：RAG 答案相关性下降，LLM 本身评测正常，用户反馈“找不到已存在的文档”。
- **错误第一猜测**：生成模型 hallucination，需要换更强模型。
- **证据链**：Recall@k 下降；embedding 模型或 chunking 参数变更；索引 build 未覆盖最新文档；过滤条件误杀；向量库 p99 正常但命中文档版本过旧。
- **定位路径**：从 query 到 retrieved docs 做链路 replay；检查 embedding version、chunk id、metadata filter、index timestamp、top-k、reranker 和文档权限。
- **修复**：回滚 embedding / chunking；重建索引；修正 metadata filter；补 reranker 或提高 top-k；对缺失文档做 backfill。
- **复测**：golden query set 对比 Recall@k、MRR、命中文档版本、端到端延迟和人工抽检结果。
- **预防**：RAG 发布必须绑定 corpus snapshot、embedding version 和 index build id；索引构建后跑召回门禁。

## 12. Autoscaling 冷启动

- **症状**：流量阶跃后错误率和 TTFT 飙升，几分钟后自动恢复。
- **错误第一猜测**：HPA/KEDA 没生效，直接调大最大副本数。
- **证据链**：scale-out 已触发但 pod ready 慢；镜像拉取、权重下载、engine build、KV warmup 和健康检查耗时叠加；冷实例接流量过早；节点池无空闲 GPU。
- **定位路径**：拆 pending、image pull、model load、engine init、readiness、first request；按模型大小和节点池查看 warmup 时间；核对 autoscaling 指标是否滞后。
- **修复**：保留 warm pool；预拉镜像和权重；延迟 readiness 到 engine warmup 后；扩容指标加入 queue depth / TTFT；为大模型准备专用 buffer 容量。
- **复测**：做固定流量阶跃压测，比较 scale-out time、冷启动比例、P99 TTFT、错误率和成本。
- **预防**：每个模型维护 cold-start budget；发布和容量评审必须包含从 0 到 N 副本的演练数据。

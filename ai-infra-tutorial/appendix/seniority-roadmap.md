# 附录N：AI Infra 职级能力路线图

> 本附录面向学习者、转岗者和面试准备者。职级不是年限标签，而是你能独立处理的系统范围、证据质量和跨团队影响力。

## 使用方式

- 先按当前最弱项定位级别，不要只按最强项自评。
- 每一级都要留下可复查证据：设计图、压测记录、profile、事故复盘、发布单、面试答案。
- 面试准备时，把每个项目都改写成同一条线：问题背景 -> 资源约束 -> 链路拆解 -> 证据 -> 方案取舍 -> 复测或 rollback。
- 横向补课先读 [附录F：全书知识地图](./learning-map.md)；按事故补课读 [附录H：事故驱动学习路径](./incident-driven-learning.md)；校准面试表达读 [附录I：评分 Rubric](./interview-rubric.md)。

## 总览矩阵

| 级别 | 能独立负责的范围 | 面试中应展示的信号 |
|------|------------------|--------------------|
| Junior | 单节点、单服务、单组件的部署、观测和基础排障 | 能说清 GPU/CPU/内存/网络/存储各自影响什么，并能拿到基础证据 |
| Mid | 一个训练或推理链路的端到端交付和性能问题定位 | 能把数据、模型、制品、服务、观测串起来，给出可复测优化 |
| Senior | 多团队共享平台能力、容量规划、发布治理和复杂事故闭环 | 能在不完整信息下建立假设树，推动跨层修复并量化风险 |
| Staff | 组织级 AI Infra 架构、技术路线、成本/可靠性策略和标准化机制 | 能定义平台边界、统一证据标准，影响多个团队的长期工程效率 |

## Junior：能看懂链路，能拿到第一批证据

**预期能力**

- 理解 AI Infra 和传统后端、SRE、大数据系统的差异。
- 能区分计算、显存、CPU、网络、存储、调度分别会造成什么症状。
- 能部署或运行一个基础训练/推理 workload，并接入日志、指标和简单告警。
- 能使用 `nvidia-smi`、DCGM、`iostat`、`perf stat`、服务延迟指标做初步定位。

**章节 to read**

- [第1章：什么是 AI Infra](../part1-foundations/01-what-is-ai-infra.md)
- [第2章：算力、存储与网络](../part1-foundations/02-compute-storage-network.md)
- [第3章：从模型实验到生产系统](../part1-foundations/03-from-model-to-production.md)
- [第7章：单机训练系统](../part3-training-infra/07-single-node-training.md)
- [第14章：在线推理架构](../part5-serving-infra/14-online-inference-architecture.md)
- [附录A：术语表](./glossary.md)

**Labs / cases to practice**

- 跑通单机训练 baseline，记录 step time、GPU utilization、显存峰值和 DataLoader wait。
- 跑通一个推理服务，记录 TTFT、TPOT、P95/P99、错误率和冷启动时间。
- 用 [附录C：上线与排障检查清单](./checklists.md) 补齐 owner、phase、evidence、threshold、action、retest。

**自测 evidence**

- 一页资源链路图，能解释请求或 batch 从输入到 GPU 执行再到输出的路径。
- 一份最小 EvidenceBundle：症状、scope、workload、version、evidence、hypothesis、action、retest。
- 至少 3 个命令输出或 dashboard 截图，能支撑“瓶颈可能在哪一层”。

**常见 gaps**

- 只会说“GPU 没跑满”，说不出 CPU、数据加载、H2D、kernel launch、调度各自怎么验证。
- 只看平均值，不看 P95/P99 和分桶。
- 只会部署组件，不会说明组件解决的工程问题。

## Mid：能交付一条端到端链路，能定位主要瓶颈

**预期能力**

- 能独立负责一个训练、微调、RAG 或在线推理链路的生产化。
- 能把模型、tokenizer、数据版本、checkpoint、registry、engine、router、index 作为发布单元管理。
- 能做基本容量估算：显存、KV Cache、batch、并发、GPU 数、存储吞吐、网络带宽。
- 能用 profiler 或 trace 把性能问题归因到数据、CPU、GPU kernel、NCCL、存储或队列。

**章节 to read**

- [第8章：数据并行](../part3-training-infra/08-data-parallel.md)
- [第10章：内存优化、检查点与恢复](../part3-training-infra/10-memory-checkpointing-and-recovery.md)
- [第12章：制品、模型与检查点管理总览](../part4-data-and-storage/12-artifacts-and-checkpoints.md)
- [第13d章：RAG 工程化](../part4-data-and-storage/13d-rag-engineering.md)
- [第15章：批处理、调度与 KV Cache](../part5-serving-infra/15-batching-scheduling-and-kv-cache.md)
- [第16a-lab章：Mini-vLLM 实战](../part5-serving-infra/16a-lab-mini-vllm.md)
- [第21章：可观测性与容量规划](../part7-reliability-security/21-observability-and-capacity.md)

**Labs / cases to practice**

- 完成 [Mini-vLLM 实战](../part5-serving-infra/16a-lab-mini-vllm.md)，解释 scheduler、block manager、KV cache 和 sampler 的边界。
- 用 [附录E：端到端主线案例](./end-to-end-case.md) 写一版 LLaMA-7B 小规模训练到 serving 的 ReleaseUnit。
- 选一个事故：GPU 利用率低、TTFT 飙升、KV OOM 或 RAG 召回下降，按 [附录H](./incident-driven-learning.md) 写假设树和复测计划。

**自测 evidence**

- 一份容量账本：workload shape、GPU 型号、batch、context、KV 预算、吞吐、P99、headroom。
- 一份 profiler/trace 记录，能把热点映射回代码、配置或系统层。
- 一次灰度或压测报告，包含基线、变更、阈值、结果和 rollback target。

**常见 gaps**

- 只优化单点 benchmark，无法证明端到端收益。
- 忘记 tokenizer、prompt、adapter、index 也是发布风险。
- 容量估算只算权重，不算 KV、activation、workspace、CUDA Graph buffer 或缓存。

## Senior：能处理复杂事故，能建设可复用平台能力

**预期能力**

- 能主导多租户训练或推理平台的一块核心能力：调度、制品、发布、观测、容量、成本或安全。
- 能把复杂事故拆成时间线、假设树、证据包、缓解动作、根因和防复发项。
- 能设计发布门禁、回滚机制、恢复演练和错误预算策略。
- 能在性能、成本、可靠性、交付速度之间做明确取舍，并让其他团队接受边界。

**章节 to read**

- [第5章：内存、互联与 IO](../part2-systems-stack/05-memory-interconnect-io.md)
- [第6d章：Profiling、Debugging 与性能 SOP](../part2-systems-stack/06d-profiling-debugging-and-performance-sop.md)
- [第12c章：制品版本治理与发布门禁](../part4-data-and-storage/12c-release-governance.md)
- [第17章：多租户与成本治理](../part5-serving-infra/17-multitenancy-and-cost.md)
- [第19章：Kubernetes for AI](../part6-platform-and-orchestration/19-kubernetes-for-ai.md)
- [第20章：队列、配额与自动扩缩容](../part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md)
- [第22章：评测、发布与故障处理](../part7-reliability-security/22-evaluation-release-and-incident.md)

**Labs / cases to practice**

- 设计一次 70B serving 容量评审：副本规格、KV 预算、warm pool、autoscaling、限流和成本。
- 设计一次 NCCL timeout 或 checkpoint 卡住事故演练，要求证据覆盖应用、GPU、网络、存储和调度。
- 用 [附录I：评分 Rubric](./interview-rubric.md) 给自己的系统设计答案打分，补齐“证据”和“rollback”。

**自测 evidence**

- 一份跨团队设计文档，明确非目标、容量假设、SLO、风险、迁移计划和 rollback。
- 一次事故复盘，包含 MTTA/MTTR、用户影响、根因、修复证据和防复发 owner。
- 一套 dashboard 或告警规范，能区分资源 SLI、服务 SLI、质量 SLI 和成本指标。

**常见 gaps**

- 技术判断很强，但无法把风险转化成发布门禁、配额、审批或平台默认值。
- 只做救火，不沉淀 runbook、自动化检测和复测协议。
- 讲系统设计时没有 rollback、降级和渐进迁移路径。

## Staff：能定义方向，能让组织少重复踩坑

**预期能力**

- 能定义组织级 AI Infra 技术路线：训练平台、推理平台、数据/制品治理、评测发布、安全和成本体系如何协同。
- 能识别哪些能力应平台化，哪些应保持团队自助或组件化。
- 能建立统一的 EvidenceBundle、CapacityLedger、ReleaseUnit、StateManifest 和 postmortem 标准。
- 能把事故、成本、可靠性和研发效率转化成路线图优先级。

**章节 to read**

- [第23章：安全、隔离与治理](../part7-reliability-security/23-security-isolation-and-governance.md)
- [第24章：构建一个 AI 平台](../part8-advanced-and-capstone/24-build-an-ai-platform.md)
- [第25章：AI Agent 与推理时计算基础设施](../part8-advanced-and-capstone/25-agent-and-inference-time-compute.md)
- [第26章：AI Infra 面试题、自测与面试官题库](../part8-advanced-and-capstone/26-ai-infra-interview-questions.md)
- [第27章：AI Infra 模拟面试与评分校准手册](../part8-advanced-and-capstone/27-ai-infra-interview-questions.md)
- [附录B：工具图谱](./tooling-map.md)
- [附录G：版本矩阵](./version-matrix.md)

**Labs / cases to practice**

- 写一个 6-12 个月 AI Infra 路线图：目标用户、非目标、能力分层、里程碑、风险和度量。
- 做一次 build vs buy 评审：vLLM/SGLang/TRT-LLM、KServe/自研服务、向量库、队列调度或观测栈。
- 组织一次面试校准会，用第26章题库和第27章流程统一评分标准。

**自测 evidence**

- 一份平台蓝图，能把训练、制品、评测、发布、推理、观测、安全和成本放在同一张图里。
- 一份年度或季度技术路线图，包含取舍依据、退出条件和成功指标。
- 一套组织级标准：发布单元、容量账本、事故证据、版本矩阵、复测协议。

**常见 gaps**

- 把 Staff 理解成“更会写底层代码”，忽略组织级默认值和决策质量。
- 过度平台化，把所有差异都塞进统一系统。
- 路线图只列组件，不说明用户、约束、迁移成本和失败退出条件。

## 面试准备建议

| 准备项 | 目标 | 产物 |
|--------|------|------|
| 项目压缩 | 90 秒讲清项目层级、约束和结果 | 项目介绍稿 |
| 深挖案例 | 证明你不是只参与过，而是真能决策 | 事故或设计复盘 |
| 证据链 | 把口头判断落到指标、命令、trace 或发布记录 | EvidenceBundle |
| 取舍题 | 展示 seniority，而不是背方案 | trade-off 表 |
| rollback | 证明你能安全交付 | rollback target 和复测阈值 |

准备面试时，至少打磨 3 个故事：

1. 一个性能或稳定性事故：例如 GPU 利用率低、NCCL timeout、TTFT 飙升、KV OOM。
2. 一个端到端交付：例如从 checkpoint 到 registry，再到灰度 serving。
3. 一个平台取舍：例如为什么选择某个调度、推理引擎、向量库或观测方案。

## 升级判断

- 从 Junior 到 Mid：不再只执行 checklist，而能交付一条可观测、可回滚的链路。
- 从 Mid 到 Senior：不再只优化自己负责的服务，而能处理跨层事故和平台默认值。
- 从 Senior 到 Staff：不再只解决一类问题，而能改变组织处理 AI Infra 问题的方式。

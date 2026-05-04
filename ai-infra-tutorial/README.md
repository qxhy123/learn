# 从零到高阶的 AI Infra 教程

## 项目简介

本教程的目标，不是简单罗列 “GPU / Kubernetes / vLLM / 向量数据库” 这些名词，而是建立一套完整的 **AI 基础设施（AI Infrastructure）** 认知框架：为什么一个模型从实验脚本走到稳定生产服务，会自然演化出算力管理、数据管道、训练编排、模型制品、推理调度、可观测性、安全治理和成本控制这些系统问题。

如果说模型教程回答的是：

- 模型为什么有效？
- 训练为什么能收敛？
- 结构为什么能提升效果？

那么 AI Infra 教程回答的是：

- 模型为什么训不动？
- GPU 为什么很贵却总跑不满？
- 为什么单机能跑、多机就开始掉速？
- 为什么离线效果不错，上线后时延、成本和质量一起失控？
- 为什么 demo 能工作，但平台一做大就开始互相抢资源、抢队列、抢预算？

本教程试图解决的，不是某个单点组件怎么配，而是更底层的工程判断：

1. **问题到底出在哪一层？**
2. **这一层的约束是计算、内存、网络、存储还是调度？**
3. **该优先优化吞吐、延迟、稳定性还是成本？**
4. **哪些能力值得平台化，哪些能力更适合保持简单？**

---

## 本教程的定位

本教程在仓库中的定位介于几类内容之间：

- 它不像 `transformer-tutorial/` 那样重点讲模型结构和训练算法
- 它不像 `cuda-tutorial/` 那样深入到底层 kernel 编程与 GPU 微观执行
- 它也不像传统运维手册那样只讲部署步骤和工具命令

它更像一张 **“把模型、系统和平台连起来的总装图”**：

- 往下，能连接 GPU、内存、网络、容器和调度
- 往上，能连接实验、评测、发布、推理和业务指标
- 横向，能连接团队协作、多租户治理、权限、安全与成本

因此，本教程特别强调三种能力：

1. **系统直觉**：知道组件为什么存在，以及它在整条链路中的位置
2. **瓶颈直觉**：看到问题时，能先判断属于哪一类资源或哪一段链路
3. **平台直觉**：知道什么时候该建设平台能力，什么时候不该过度设计

---

## 目标受众

- 希望从训练脚本走向平台化与生产化的算法工程师
- 想系统理解 GPU 集群、训练系统、推理系统的平台工程师
- 对 AI 系统、MLOps、模型服务化感兴趣的学生与研究者
- 需要和算法、后端、运维协同设计 AI 平台的技术负责人

如果你已经会训练模型、会写服务接口、会用一些开源组件，但总觉得它们彼此之间像一堆散点，而不是一张地图，那么这套教程就是为你写的。

---

## 章节导航目录

### 开始之前

- [前言：如何使用本教程](./00-preface.md)

### 第零部分：体系结构基础

| 章节 | 标题 | 主要内容 | 工程重点 |
|------|------|----------|----------|
| 第0a章 | [CPU 微架构总览](./part0-foundations-of-systems/0a-cpu-microarchitecture.md) | 第一性原理推导链 + 8 章导览 + 角色阅读路径 | 入口章 / 学习路径选择 |
| 第0a-1章 | [流水线（Pipeline）](./part0-foundations-of-systems/0a1-pipeline.md) | 5 段经典流水、深流水、冒险与 forwarding、CPI/IPC 推算、host-side 真实 IPC | 深挖 ILP 第一层 |
| 第0a-2章 | [乱序执行、Register Renaming 与 ROB](./part0-foundations-of-systems/0a2-out-of-order-execution.md) | OoO 引擎结构、ROB 容量、LSQ、退役吞吐、指针追逐为何让 OoO 失效 | 理解 backend bound |
| 第0a-3章 | [分支预测](./part0-foundations-of-systems/0a3-branch-prediction.md) | BTB / RAS、2-bit 饱和、GShare、TAGE、误预测代价、cold path 治理 | P99 抖动诊断 |
| 第0a-4章 | [SIMD：SSE / AVX / AVX-512](./part0-foundations-of-systems/0a4-simd.md) | ISA 演进、AVX-512 频率降级、自动向量化、intrinsics、对齐惩罚 | tokenizer 加速决策 |
| 第0a-5章 | [Cache 层级](./part0-foundations-of-systems/0a5-cache-hierarchy.md) | L1/L2/L3 延迟带宽、cache line 64B、关联度、替换策略、LLC slice、prefetcher | 数组 stride / worker 数选型 |
| 第0a-6章 | [MESI 一致性协议](./part0-foundations-of-systems/0a6-mesi-coherence.md) | 四状态机、snoop vs directory、MOESI/MESIF、跨 socket UPI 流量 | 多线程 atomic 崩盘排查 |
| 第0a-7章 | [伪共享（False Sharing）](./part0-foundations-of-systems/0a7-false-sharing.md) | 物理粒度 vs 语义粒度、检测与修复、padding/alignas、per-thread + reduce | 加 worker 反而变慢诊断 |
| 第0a-8章 | [CPU 综合排障 Worked Example](./part0-foundations-of-systems/0a8-cpu-worked-example.md) | Top-Down 方法论、三个完整剧本、工具栈对照、SOP、反模式速查 | on-call runbook 模板 |
| 第0b章 | [内存、虚拟内存与 IO](./part0-foundations-of-systems/0b-memory-virtual-memory-and-io.md) | 页表/TLB、Page Cache、Huge Pages、NUMA、syscall/io_uring、PCIe、DMA 与 H2D Worked Example | 理解内存与 IO 路径 |
| 第0c章 | [文件系统与存储内核](./part0-foundations-of-systems/0c-filesystems-and-storage-internals.md) | VFS、inode/dentry、ext4/XFS/ZFS、fsync/O_DIRECT、对象存储、并行文件系统与 checkpoint Worked Example | 判断存储语义和吞吐边界 |
| 第0d章 | [网络协议栈基础](./part0-foundations-of-systems/0d-network-stack-fundamentals.md) | TCP/IP、MTU、socket/epoll/io_uring、offload、RDMA verbs、GPUDirect RDMA 与 AllReduce Worked Example | 拆分 control plane 与 data plane |

### 第一部分：AI Infra 基础认知

| 章节 | 标题 | 主要内容 | 工程重点 |
|------|------|----------|----------|
| 第1章 | [什么是 AI Infra](./part1-foundations/01-what-is-ai-infra.md) | AI 系统全景图、角色分工、核心对象、第一性原理学习地图 | 建立全局认知 |
| 第2章 | [算力、存储与网络](./part1-foundations/02-compute-storage-network.md) | CPU/GPU/内存/磁盘/网络的职责与瓶颈，Page Cache/NUMA 浅引用并指向 Part 0 | 识别资源短板 |
| 第3章 | [从模型实验到生产系统](./part1-foundations/03-from-model-to-production.md) | 从 notebook 到线上服务的演进路径、平台化边界与生产链路推导 | 理解系统链路 |

### 第二部分：硬件与系统栈

| 章节 | 标题 | 主要内容 | 工程重点 |
|------|------|----------|----------|
| 第4章 | [GPU 与加速器](./part2-systems-stack/04-gpu-and-accelerators.md) | GPU 架构、吞吐思维、NVSwitch、HGX H100/H200、GB200/NVL72 与推理阶段瓶颈 | 理解并行计算基础 |
| 第5章 | [内存、互联与 IO](./part2-systems-stack/05-memory-interconnect-io.md) | HBM、PCIe、NVLink、RDMA、对象存储 IO、集群网络拓扑与 Job Placement | 理解“为什么慢” |
| 第6章 | [CUDA、运行时与算子执行](./part2-systems-stack/06-cuda-runtime-and-kernels.md) | CUDA 栈、Kernel、库与编译链路、SM 调度、warp 与 register spill | 软件栈分层认知 |

### 第三部分：训练基础设施

| 章节 | 标题 | 主要内容 | 工程重点 |
|------|------|----------|----------|
| 第7章 | [单机训练系统](./part3-training-infra/07-single-node-training.md) | 训练循环、Profiler、数据管道与显存、LLaMA-7B Worked Example、MFU/HFU 与 AMP | 单机性能基线 |
| 第8章 | [数据并行](./part3-training-infra/08-data-parallel.md) | AllReduce、同步点、NCCL、吞吐扩展、梯度压缩与 PowerSGD | 规模化第一步 |
| 第9章 | [模型并行与流水并行](./part3-training-infra/09-model-pipeline-parallel.md) | TP/PP/EP、SP/CP、Interleaved/Zero Bubble、并行策略决策树与配置实例 | 超大模型训练 |
| 第10章 | [内存优化、检查点与恢复](./part3-training-infra/10-memory-checkpointing-and-recovery.md) | 激活重计算、ZeRO、Checkpoint、NCCL Hang 排查、Straggler、Elastic Training、FP8 | 稳定完成训练 |
| 第10b章 | [对齐训练与后训练基础设施](./part3-training-infra/10b-alignment-and-post-training.md) | RLHF、DPO、PPO/GRPO、RM 部署、PPO Worked Example 与多模型 checkpoint 一致性 | 理解后训练的独特资源模式 |
| 第10c章 | [Fine-Tuning 基础设施与多 Adapter 服务](./part3-training-infra/10c-finetuning-and-multi-adapter.md) | LoRA、QLoRA、Multi-LoRA 显存预算、Adapter/Base 兼容与 FTaaS pipeline | 微调与 adapter 的平台化 |

### 第四部分：数据与存储基础设施

| 章节 | 标题 | 主要内容 | 工程重点 |
|------|------|----------|----------|
| 第11章 | [数据管道](./part4-data-and-storage/11-data-pipeline.md) | 采集、清洗、切分、分片、流式读取、dataset shard 与 Part 0 存储路径联动 | 数据吞吐与一致性 |
| 第12章 | [制品、模型与检查点管理](./part4-data-and-storage/12-artifacts-and-checkpoints.md) | Model Registry、Checkpoint、版本治理、checkpoint 文件系统选型 | 训练资产管理 |
| 第13章 | [特征、向量与缓存](./part4-data-and-storage/13-feature-vector-and-cache.md) | Feature Store、Embedding、向量索引、ANN、RAG Chunking、增量重建、Prefix Caching | 在线数据访问 |

### 第五部分：推理与服务基础设施

| 章节 | 标题 | 主要内容 | 工程重点 |
|------|------|----------|----------|
| 第14章 | [在线推理架构](./part5-serving-infra/14-online-inference-architecture.md) | 网关、路由、服务、模型副本、推理控制面与数据面拆解 | 线上推理主链路 |
| 第15章 | [批处理、调度与 KV Cache](./part5-serving-infra/15-batching-scheduling-and-kv-cache.md) | Dynamic Batching、Prefill/Decode、PagedAttention、70B 容量规划 Worked Example、PD 分离、Speculative Decoding、ITL | 提升吞吐与稳定性 |
| 第16章 | [量化、编译与推理引擎](./part5-serving-infra/16-quantization-compilation-and-engines.md) | TRT-LLM、vLLM、SGLang、ONNX Runtime、量化/引擎选型决策树与校准 | 降低延迟与成本 |
| 第16a章 | [vLLM 内部机制深入](./part5-serving-infra/16a-vllm-internals.md) | Engine/Scheduler/Worker/Block Manager、PagedAttention 实现、Continuous Batching 调度循环、Prefix Caching、Chunked Prefill、Speculative、TP/PP/EP、Multi-LoRA、量化集成、V1 重构、调优手册 | vLLM 工程师视角 |
| 第16b章 | [SGLang 内部机制深入](./part5-serving-infra/16b-sglang-internals.md) | Frontend Language、RadixAttention 实现、Constrained Decoding、Speculative、Cache-aware 调度、与 vLLM/TRT-LLM 选型决策、Agent + Tool Use Worked Example | 复杂结构化 / Agent serving |
| 第17章 | [多租户与成本治理](./part5-serving-infra/17-multitenancy-and-cost.md) | 配额、SLA、冷热分层、Cloud vs On-Prem TCO、Spot、MFU vs Utilization、Chargeback | 服务化经营能力 |

### 第六部分：平台与编排

| 章节 | 标题 | 主要内容 | 工程重点 |
|------|------|----------|----------|
| 第18章 | [容器与运行时](./part6-platform-and-orchestration/18-containers-and-runtime.md) | 镜像、Runtime、设备插件、构建发布、运行环境的不可变边界 | 可复制的运行环境 |
| 第19章 | [Kubernetes for AI](./part6-platform-and-orchestration/19-kubernetes-for-ai.md) | Pod、Job、Operator、GPU 调度、Volcano/Kueue、拓扑感知、亲和/反亲和 | 平台化基础 |
| 第20章 | [队列、配额与自动扩缩容](./part6-platform-and-orchestration/20-queues-quotas-and-autoscaling.md) | 队列系统、优先级、MIG/MPS/Time-Slicing、GPU 碎片化、DRF、公平调度与弹性伸缩 | 资源治理 |

### 第七部分：稳定性、安全与治理

| 章节 | 标题 | 主要内容 | 工程重点 |
|------|------|----------|----------|
| 第21章 | [可观测性与容量规划](./part7-reliability-security/21-observability-and-capacity.md) | Metrics、Logs、Traces、采样策略、cardinality 治理、错误预算 burn-down、成本归因 | 看清系统状态 |
| 第22章 | [评测、发布与故障处理](./part7-reliability-security/22-evaluation-release-and-incident.md) | 离线评测、A/B、灰度、质量采样、Prompt/配置变更、回滚与复盘 | 线上可靠交付 |
| 第23章 | [安全、隔离与治理](./part7-reliability-security/23-security-isolation-and-governance.md) | 权限、Secrets、租户隔离、pickle/SafeTensors、cosign/Trivy/SLSA、合规审计 | 降低平台风险 |

### 第八部分：高阶主题与完整项目

| 章节 | 标题 | 主要内容 | 工程重点 |
|------|------|----------|----------|
| 第24章 | [构建一个 AI 平台](./part8-advanced-and-capstone/24-build-an-ai-platform.md) | 训练、评测、制品、部署、推理、观测的端到端蓝图与平台边界推导 | 形成系统设计能力 |
| 第25章 | [AI Agent 与推理时计算基础设施](./part8-advanced-and-capstone/25-agent-and-inference-time-compute.md) | Agent 状态管理、thinking tokens 四模式、推理预算工程与推理服务集成 | 新范式下的推理系统设计 |

### 附录

| 附录 | 标题 | 内容说明 |
|------|------|----------|
| 附录A | [术语表](./appendix/glossary.md) | GPU、KV Cache、RDMA、Checkpoint 等核心术语 |
| 附录B | [工具图谱](./appendix/tooling-map.md) | 常见 AI Infra 组件与典型职责 |
| 附录C | [上线与排障检查清单](./appendix/checklists.md) | 训练、推理、RAG、成本治理的常用检查项 |
| 附录D | [练习题详细参考解答](./appendix/answers.md) | 各章练习题的完整思路、结果与架构示意 |

---

## 本教程会怎么讲

为了避免教程再次变成浅层概览，本教程尽量按以下方式组织内容：

### 1. 先讲问题，再讲组件

例如，在讲 KV Cache 前，我们先问：为什么 decode 阶段会变慢？为什么长上下文会让显存吃满？
在讲 Kubernetes 前，我们先问：AI 工作负载为什么需要统一运行底座？它不能解决什么？

### 2. 先建立因果关系，再记忆术语

你会在教程里反复看到这样的提问：

- 这个瓶颈来自计算、内存、网络还是存储？
- 这个优化是在减少计算、减少数据搬运，还是减少调度等待？
- 这个平台能力是为了单租户最优，还是为了多租户可治理？

### 3. 同时保留三层深度

- **概念直觉**：为什么存在
- **机制细节**：如何工作
- **工程边界**：什么时候有效、什么时候会失败

这三层缺任何一层，读者都容易停留在“记住了名词，却不会用”。

### 4. 在关键章节加入决策工程

在适合的章节中提供选型决策树，帮助读者把知识转化为判断。
不是只知道有哪些方案，而是能根据目标、约束和代价，判断当前更适合哪一类设计。

---

## 学习路径建议

### 路径一：工程师快速入门（2-3 周）

1. 可先选读 Part 0 的 0b、0d，再学习第 1-3 章，建立 AI 系统全景图
2. 学习第 14-17 章与第 25 章，理解线上推理链路与 agent 推理新范式
3. 学习第 21-23 章，理解稳定性、安全与成本

### 路径二：训练平台路线（4-6 周）

1. 先完成 Part 0，再学习第 1-6 章，建立硬件与系统栈认知
2. 重点学习第 7-10 章，并继续完成第 10b、10c 章，掌握训练、后训练与微调基础设施
3. 补充第 11-12 章与第 18-20 章，形成平台理解

### 路径三：推理平台路线（4-5 周）

1. 先完成 Part 0 的 0a、0b、0d，再完成第 2、4、5、6 章，理解硬件和运行时约束
2. 重点学习第 14-17 章，再补第 25 章，理解服务链路、调度、成本与 agent 运行形态
3. 再学习第 21-23 章，补齐监控、评测和治理

### 路径四：完整 AI 平台路线（6-8 周）

1. 按章节顺序完整学习 Part 0 与第 1-25 章，包括新增/扩充的第 10b、10c 和第 25 章
2. 每章完成练习题与设计清单
3. 最后以第 24-25 章为蓝图，尝试画出自己的 AI 平台架构图

### 路径五：体系结构深度路径（Part 0 + Part 2 + Part 3）

1. 完整学习 Part 0，建立 CPU、内存、文件系统与网络协议栈的机制地图
2. 接着学习第 4-6 章，把 GPU、互联、IO、CUDA runtime 与底层系统约束连起来
3. 最后学习第 7-10c 章，用单机训练、数据并行、模型并行、Checkpoint 与后训练案例检验体系结构判断

---

## 前置要求

- **必需**：基础 Linux 命令行
- **必需**：Python 基础
- **推荐**：了解深度学习训练/推理的基本概念
- **推荐**：知道容器、HTTP、数据库的基础概念

如果你希望补齐相关背景，可以配合本仓库的以下教程一起学习：

- [CUDA 教程](../cuda-tutorial/README.md)
- [Transformer 教程](../transformer-tutorial/README.md)
- [计算机网络教程](../computer-network-tutorial/README.md)
- [Python 教程](../python-tutorial/README.md)

---

## 如何判断自己真的学会了

读完本教程后，你应该能够做到四件事：

1. **能诊断**：面对“训练慢 / 推理贵 / 服务抖 / 队列堵”这类问题，先缩小到正确层级
2. **能权衡**：知道某个优化是在拿什么换什么，例如吞吐换时延、显存换计算、平台复杂度换多租户治理
3. **能设计**：能够给出一套不是只堆组件名的 AI 平台方案
4. **能估算**：给定一个模型和集群，能在 10 分钟内估算出训练/推理的资源需求
5. **能推导**：能从第一性原理推导每个机制为什么存在、解决什么不可化简的问题、边界在哪里

如果你只会说出很多工具名字，但说不清为什么这里需要它、它的边界是什么，那说明还没真正进入 AI Infra 的思维方式。

---

## 教程特色

- **系统视角**：从 GPU 到网关，从训练到推理，覆盖一条完整链路
- **第一性原理思维框架**：每章用“拆、推、绘、导”把机制从不可化简的问题中推出来
- **工程导向**：每章都强调瓶颈、故障、权衡与治理
- **尽量定量**：在适合的地方加入容量估算、通信体量、显存预算、吞吐关系
- **多文件 HTML 版本**：除 Markdown 源文档外，提供适合浏览和分发的多文件静态 HTML 版本
- **适合中文学习者**：尽量把术语翻译清楚，把链路讲完整
- **与仓库教程互补**：不重复讲模型算法，重点讲基础设施和系统设计

---

## HTML 版本

本教程同时提供静态 HTML 版本（位于 `html/` 目录），适合离线浏览与分发：

```bash
cd html && python3 -m http.server 8000
# 浏览器访问 http://localhost:8000/index.html
```

HTML 版本特点：

- 浅色 paper 风格，每章独立文件
- 左侧 sidebar 可视化全 31 章导航
- mermaid 图表 + 手工 SVG 流程图
- 所有内容与 Markdown 版本同步

---

## 许可证

本项目采用 MIT 许可证开源。

---

*如有建议或发现错误，欢迎提交 Issue 或 Pull Request。*

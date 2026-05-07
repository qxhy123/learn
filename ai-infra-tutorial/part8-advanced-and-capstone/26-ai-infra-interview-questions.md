# 第26章：AI Infra 面试题、自测与面试官题库

> AI Infra 面试考的不是记住多少组件名，而是能不能在真实约束下判断资源、链路、故障和治理。

> **关联章节**：本章与 [第24章：构建一个 AI 平台](24-build-an-ai-platform.md) 和 [第25章：AI Agent 与推理时计算基础设施](25-agent-and-inference-time-compute.md) 配合阅读：前者提供平台全局蓝图，后者提供新型推理负载的系统约束。

## 1. 第一性原理拆解 + 学习大纲

把 Kubernetes、GPU、NCCL、vLLM、registry 这些名字拿掉，AI Infra 面试真正想确认的是：候选人是否能把模型系统看成一条受资源约束、会排队、会失败、需要治理的工程链路。组件名只是语言，系统判断才是核心能力。

一场有效的 AI Infra 面试通常沿四条线展开：资源线看 GPU、HBM、CPU、网络、存储、队列和预算；链路线看数据、训练、checkpoint、registry、评测、发布、推理和观测；故障线看如何从吞吐下降、p99 升高、训练挂起、恢复失败走到根因；治理线看多租户、公平性、成本、安全、发布门禁和审计如何进入设计。

本章按系统层次组织题库：26.1 建立基础认知，26.2 回到底层资源，26.3 和 26.4 覆盖训练与制品，26.5 和 26.6 覆盖推理与平台调度，26.7 覆盖治理，26.8 用综合 case 把前面内容串起来。核心结论很简单：AI Infra interviews test system judgment, not memorized component names。

## 2. 学习目标

完成本章后，你应该能够：

1. 用资源、链路、故障、治理四条线组织 AI Infra 面试回答。
2. 区分训练、推理、平台、可靠性岗位的考察重点。
3. 针对 GPU、内存、网络、存储、调度、制品、发布、安全和成本给出可解释的系统取舍。
4. 把一个故障现象拆成指标假设、验证顺序、止血动作和长期修复。
5. 作为面试官，写出可复盘、可校准的评分理由。

## 3. 使用方式

- **候选人准备**：每题先闭卷讲 2-5 分钟，再看回答框架补洞。不要背原句，要练习先给判断、再分层、最后落到例子。
- **自测复盘**：每节统计不会的题，把薄弱点映射回正文章节。例如 26.5 卡住，就回读第14-16章；26.6 卡住，就回读第18-20章。
- **面试官出题**：从岗位画像出发选题。推理岗多问 KV Cache、batching、SLO；训练岗多问并行、checkpoint、恢复；平台岗多问队列、配额、发布和治理。
- **团队校准**：面试后只讨论证据，不讨论感觉。用评分要点描述候选人到底讲清了哪条链路、漏掉了哪个约束。

## 4. 题目格式约定

每道题统一 5 段结构：`问题` 是题面，`考察点` 是面试官要验证的能力，`回答框架` 是候选人组织语言的骨架，`追问` 用来压深，`评分要点` 用于校准。下面先给出完整题块形状，后续所有问题都沿用同一结构。


## 26.1 AI Infra 基础认知与系统分层

### 26.1.1 AI Infra 面试到底在考什么

**问题**
说明 AI Infra 面试的核心考点，不要只列组件名，要区分资源、链路、故障和治理。

**考察点**
- 是否能从系统而不是名词出发
- 是否能讲清资源和约束
- 是否能把诊断、设计和治理串起来

**回答框架**
- 先定义 AI Infra 面试考什么
- 再给出四条判断线
- 最后举一个训练或推理的例子

**追问**
- 为什么“会用组件”不等于“会做系统”？
- 训练岗和推理岗的回答侧重点有什么不同？

**评分要点**
- 及格：能讲出资源和系统边界
- 良好：能给出一条完整的判断路径
- 优秀：能结合真实故障或设计案例展开

### 26.1.2 AI Infra 与传统后端基础设施的核心差异

**问题**
一个资深后端工程师说自己懂 K8s、负载均衡、缓存和监控，所以 AI Infra 只是“后端加 GPU”。你如何反驳？

**考察点**
- 是否能区分可复用经验和需要重建的直觉
- 是否理解 GPU、显存、token、模型制品对系统形态的影响
- 是否能把训练和推理的生命周期差异说清

**回答框架**
- 先承认服务化、发布、观测、容量规划仍可复用
- 指出 GPU/HBM/互联成为一等稀缺资源
- 说明训练是长任务和恢复语义，推理是在线 SLO 与动态 batching
- 补充模型版本、数据血缘、成本归因和安全审计

**追问**
- 为什么水平扩容在大模型推理里不是万能解？
- 传统无状态服务直觉在哪里会失效？

**评分要点**
- 及格：能说清基本概念和主要边界
- 良好：能给出可执行的判断路径和关键取舍
- 优秀：能结合真实指标、故障或设计案例展开

### 26.1.3 用一张图给新人讲 AI Infra 系统分层

**问题**
请给新人画一张 AI Infra 系统分层图，说明从硬件资源到平台治理至少应该分成哪些层，每层解决什么问题。

**考察点**
- 是否具备清晰层次模型
- 是否能把组件放到正确责任边界
- 是否能解释层与层之间的依赖

**回答框架**
- 资源层：GPU、CPU、内存、网络、存储决定物理上限
- 运行层：驱动、CUDA、通信库、容器和 Kubernetes 保证任务能跑
- 工作流层：数据、训练、checkpoint、评测、发布形成交付闭环
- 服务层：网关、推理引擎、batching、KV Cache 处理在线请求
- 治理层：配额、成本、观测、安全、审计保证长期运营

**追问**
- 为什么不直接按训练系统和推理系统两大块分？
- model registry 更像工作流层还是治理层？

**评分要点**
- 及格：能说清基本概念和主要边界
- 良好：能给出可执行的判断路径和关键取舍
- 优秀：能结合真实指标、故障或设计案例展开

### 26.1.4 训练、推理与 Agent 负载的资源画像差异

**问题**
同一台 8 卡节点分别跑 70B 训练、70B 推理和复杂 Agent 任务。请比较它们在显存、计算、通信、I/O、生命周期上的差异。

**考察点**
- 是否理解不同 AI 负载的物理形态
- 是否能把资源画像映射到调度策略
- 是否知道 Agent 不是普通长请求

**回答框架**
- 训练显存包含参数、梯度、优化器和激活，通信以集合通信为主
- 推理权重常驻，KV Cache 随请求增长，prefill 和 decode 形态不同
- Agent 由多次模型调用和工具等待组成，GPU 工作不连续但 state 生命周期长
- 训练适合 gang/topology-aware，推理需要优先级保护，Agent 需要预算和工具限流

**追问**
- 为什么训练任务比推理服务更依赖 checkpoint？
- 为什么 Agent session 不应该等同于一个超长推理请求？

**评分要点**
- 及格：能说清基本概念和主要边界
- 良好：能给出可执行的判断路径和关键取舍
- 优秀：能结合真实指标、故障或设计案例展开

### 26.1.5 控制面、数据面与观测面的边界

**问题**
请定义 AI 平台里的控制面、数据面和观测面，并用一次模型发布或推理故障说明三者分别负责什么。

**考察点**
- 是否能区分慢路径控制和热路径执行
- 是否理解观测不是业务热路径
- 是否能说明边界对可用性和安全的影响

**回答框架**
- 控制面负责提交任务、修改配额、发布模型、扩缩副本
- 数据面负责训练 step、推理请求、KV block 分配和网络通信
- 观测面负责指标、日志、trace、审计和质量采样
- 发布策略在控制面，请求执行在数据面，效果验证在观测面

**追问**
- 控制面短暂不可用时在线服务是否必须中断？
- 把复杂评分逻辑放进网关数据面有什么风险？

**评分要点**
- 及格：能说清基本概念和主要边界
- 良好：能给出可执行的判断路径和关键取舍
- 优秀：能结合真实指标、故障或设计案例展开

### 26.1.6 在线与离线工作负载的资源策略差异

**问题**
平台同时运行在线推理、批量推理、训练任务和夜间数据处理。为什么不能用同一种 GPU 资源策略？

**考察点**
- 是否理解不同工作负载的 SLO 和弹性差异
- 是否能提出优先级、配额、抢占和恢复策略
- 是否考虑在线业务和离线吞吐的冲突

**回答框架**
- 在线推理高优先级、容量预留、避免抢占
- 批量推理可排队可切分，关注完成时间和单位成本
- 训练任务周期长，依赖 gang scheduling、拓扑和 checkpoint
- 数据处理低优先级、可重试，适合填谷
- 原则是在线保底、离线填谷、训练按恢复成本决定是否可抢占

**追问**
- 如何决定在线推理需要预留多少 GPU？
- 训练任务可抢占的前提是什么？

**评分要点**
- 及格：能说清基本概念和主要边界
- 良好：能给出可执行的判断路径和关键取舍
- 优秀：能结合真实指标、故障或设计案例展开

### 26.1.7 一次告警驱动的故障诊断思路

**问题**
凌晨告警显示某在线大模型服务 p99 延迟升高 5 倍，错误率从 0.2% 到 3%。你前 10 分钟会按什么顺序看哪些信号？

**考察点**
- 是否有结构化排障路径
- 是否能先分类再行动
- 是否能区分止血和根因分析

**回答框架**
- 先确认影响面：全局、单租户、单模型、单机房、单实例
- 看流量特征：QPS、输入/输出 token、上下文长度、来源是否突变
- 看服务内部：队列、batch size、prefill/decode、KV evict、engine 错误
- 看资源和依赖：GPU/HBM、CPU、网络、存储、registry、向量库
- 形成初判后限流、降级、回滚、隔离或扩容

**追问**
- GPU 利用率很低但延迟很高，你怀疑什么？
- 什么情况下应该先回滚而不是继续定位？

**评分要点**
- 及格：能说清基本概念和主要边界
- 良好：能给出可执行的判断路径和关键取舍
- 优秀：能结合真实指标、故障或设计案例展开

### 26.1.8 平台工程师的最小可运营单元

**问题**
你要为一个 20 人算法团队建设第一版 AI 平台，只能优先打通一条闭环。你会选哪条链路，为什么？

**考察点**
- 是否能从最小闭环而不是组件清单出发
- 是否理解训练、制品、评测、发布之间的因果关系
- 是否能控制平台 V1 范围

**回答框架**
- 推荐闭环：训练 Job -> checkpoint -> registry -> 评测 -> staging -> 观测 -> 回滚
- 理由是覆盖可复现、可发布、可回滚和可审计
- 暂缓复杂 feature store、全自动调参、细粒度 chargeback
- 成功标准是模型来源可查、发布有门禁、异常能回滚、失败能恢复

**追问**
- 如果没有 registry，这条闭环会在哪里断？
- 老板要求先做统一门户页面，你怎么解释风险？

**评分要点**
- 及格：能说清基本概念和主要边界
- 良好：能给出可执行的判断路径和关键取舍
- 优秀：能结合真实指标、故障或设计案例展开

### 26.1.9 描述你做过的项目时如何对齐 AI Infra 视角

**问题**
请用 90 秒介绍你做过的一个 AI Infra 项目，让面试官能判断它解决了哪一层问题、受什么约束、带来了什么可量化结果。

**考察点**
- 是否能结构化表达项目经验
- 是否能从系统问题而不是工具实施讲起
- 是否能提供可信指标和取舍

**回答框架**
- 一句话定位：服务哪类负载，解决哪一层问题
- 背景约束：GPU 数量、模型规模、SLO、团队或交付时间
- 关键设计：做了什么取舍，为什么没有选另一路径
- 结果证据：吞吐、延迟、恢复时间、成本、稳定性或上线效率

**追问**
- 这个项目最大的错误判断是什么？
- 如果资源翻倍或减半，你的方案会怎么变？

**评分要点**
- 及格：能说清基本概念和主要边界
- 良好：能给出可执行的判断路径和关键取舍
- 优秀：能结合真实指标、故障或设计案例展开

### 26.1.10 AI Infra 工程师 vs 算法、SRE、DevOps 的边界

**问题**
同一个“推理延迟高”的问题，算法工程师、AI Infra 工程师、SRE、DevOps 分别应该负责什么？交接点在哪里？

**考察点**
- 是否能定义岗位边界和协作接口
- 是否能避免把 AI Infra 简化为运维或算法助理
- 是否理解复杂问题需要共同 owner

**回答框架**
- 算法负责模型结构、量化影响、输入输出长度和效果评测
- AI Infra 负责推理引擎、batching、KV Cache、GPU 容量、平台抽象
- SRE 负责 SLO、告警、容量演练、事故响应和复盘
- DevOps 负责镜像、CI/CD、网络、密钥和基础环境一致性
- 交接材料包括 checkpoint、评测报告、运行手册、发布记录和指标面板

**追问**
- 如果延迟升高来自模型输出变长，谁主导修复？
- 小团队里这些角色合并时最大的风险是什么？

**评分要点**
- 及格：能说清基本概念和主要边界
- 良好：能给出可执行的判断路径和关键取舍
- 优秀：能结合真实指标、故障或设计案例展开

## 26.2 硬件、GPU、内存、网络与存储基础

### 26.2.1 选择 GPU 时应该看哪些指标

**问题**
团队要采购 GPU 跑 70B 训练和推理。请说明你不会只看 TFLOPS，还会看哪些指标。

**考察点**
- 是否能说清 GPU 选型的关键变量：显存容量、HBM 带宽、低精度能力、NVLink/NVSwitch/IB、驱动和引擎生态
- 是否能把 GPU 选型与训练吞吐、推理并发、采购成本和后续运维风险联系起来
- 是否能通过目标模型的训练 step、TTFT/ITL、tokens/s、单位成本和故障恢复 POC 验证判断

**回答框架**
- 先界定题面场景和 GPU 选型的判断边界
- 拆关键变量：显存容量、HBM 带宽、低精度能力、NVLink/NVSwitch/IB、驱动和引擎生态
- 说明主要风险：只看 TFLOPS 导致显存或互联先到瓶颈
- 给出验证或落地方式：用目标模型做训练 step、TTFT/ITL、tokens/s、单位成本和故障恢复 POC

**追问**
- HBM 更大但互联更弱的卡更适合训练还是推理？
- 采购前最小 benchmark 应覆盖哪些模型、context 和并发？

**评分要点**
- 及格：能说出 GPU 选型的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对训练吞吐、推理并发、采购成本和后续运维风险的影响

### 26.2.2 GPU 利用率高低如何解释

**问题**
监控显示 GPU utilization 只有 35%，业务认为“GPU 很闲”。这个指标可能误导在哪里？你还看什么？

**考察点**
- 是否能说清 GPU 指标的关键变量：SM 活跃、HBM 带宽、kernel 时间、queue time、batch size、DataLoader wait、KV evict
- 是否能把 GPU 指标与资源效率、延迟误判和容量规划联系起来
- 是否能通过 step breakdown、engine metrics、GPU timeline 和固定流量 A/B 验证判断

**回答框架**
- 先界定题面场景和 GPU 指标的判断边界
- 拆关键变量：SM 活跃、HBM 带宽、kernel 时间、queue time、batch size、DataLoader wait、KV evict
- 说明主要风险：GPU util 高低都可能掩盖 CPU、内存、通信或调度瓶颈
- 给出验证或落地方式：用 step breakdown、engine metrics、GPU timeline 和固定流量 A/B 验证

**追问**
- GPU util 高但 tokens/s 低优先怀疑什么？
- GPU util 低但 p99 高可能是哪一层在排队？

**评分要点**
- 及格：能说出 GPU 指标的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对资源效率、延迟误判和容量规划的影响

### 26.2.3 显存容量如何决定模型能否运行

**问题**
请说明模型在训练和推理时分别占用哪些显存，为什么参数大小远小于实际显存需求。

**考察点**
- 是否能说清显存预算的关键变量：参数、梯度、optimizer、activation、KV Cache、workspace、通信 buffer 和碎片
- 是否能把显存预算与训练能否启动、推理并发上限和 OOM 风险联系起来
- 是否能通过 dtype、层数、context、batch、并发估算，并用 OOM 时机和显存曲线验证判断

**回答框架**
- 先界定题面场景和显存预算的判断边界
- 拆关键变量：参数、梯度、optimizer、activation、KV Cache、workspace、通信 buffer 和碎片
- 说明主要风险：只按参数大小估算会漏掉训练状态和长上下文 KV 放大
- 给出验证或落地方式：按 dtype、层数、context、batch、并发估算，并用 OOM 时机和显存曲线验证

**追问**
- 启动 OOM 和运行中 OOM 的根因通常有什么不同？
- 长 prompt 流量导致 OOM 时先改 context、batch 还是并发？

**评分要点**
- 及格：能说出显存预算的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对训练能否启动、推理并发上限和 OOM 风险的影响

### 26.2.4 HBM、CPU 内存与对象存储的层级关系

**问题**
数据从对象存储进入 GPU 参与训练，中间经过哪些存储和内存层级？每层慢在哪里？

**考察点**
- 是否能说清 内存与存储层级的关键变量：对象存储、本地 NVMe、Page Cache、CPU 内存、pinned memory、PCIe/NVLink、HBM
- 是否能把 内存与存储层级与 GPU feeding、训练吞吐和数据成本联系起来
- 是否能通过 synthetic data、本地缓存、worker profile、H2D timeline 和缓存命中率验证判断

**回答框架**
- 先界定题面场景和 内存与存储层级的判断边界
- 拆关键变量：对象存储、本地 NVMe、Page Cache、CPU 内存、pinned memory、PCIe/NVLink、HBM
- 说明主要风险：小文件、远程读、CPU 解码、跨 NUMA 和 H2D 拷贝让 GPU 空转
- 给出验证或落地方式：用 synthetic data、本地缓存、worker profile、H2D timeline 和缓存命中率验证

**追问**
- 海量小文件为什么比大 shard 更容易拖慢训练？
- 什么时候本地 NVMe 比增加 DataLoader worker 更有效？

**评分要点**
- 及格：能说出 内存与存储层级的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对 GPU feeding、训练吞吐和数据成本的影响

### 26.2.5 NUMA 与 PCIe 拓扑为什么影响性能

**问题**
同一台 8 卡服务器上，训练任务换了 GPU 编号后吞吐下降 20%。NUMA、PCIe 和 CPU 绑核可能如何导致？

**考察点**
- 是否能说清 NUMA/PCIe 拓扑的关键变量：CPU socket、PCIe root complex、GPU 拓扑、NVLink、绑核和内存亲和
- 是否能把 NUMA/PCIe 拓扑 与同机多卡吞吐稳定性和调度 placement联系起来
- 是否能通过 nvidia-smi topo、lstopo、numactl、NCCL topo log 和带宽测试验证判断

**回答框架**
- 先界定题面场景和 NUMA/PCIe 拓扑的判断边界
- 拆关键变量：CPU socket、PCIe root complex、GPU 拓扑、NVLink、绑核和内存亲和
- 说明主要风险：换 GPU 编号可能破坏 NUMA locality 或 NCCL ring
- 给出验证或落地方式：用 nvidia-smi topo、lstopo、numactl、NCCL topo log 和带宽测试验证

**追问**
- Kubernetes 如何保证 Pod 拿到同一拓扑域资源？
- 跨 socket 访问对 DataLoader 和 RDMA 各有什么影响？

**评分要点**
- 及格：能说出 NUMA/PCIe 拓扑的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对同机多卡吞吐稳定性和调度 placement的影响

### 26.2.6 PCIe、NVLink 与 NVSwitch 的取舍

**问题**
请比较 PCIe、NVLink、NVSwitch 在多 GPU 训练和推理中的作用。什么时候互联会成为首要瓶颈？

**考察点**
- 是否能说清 GPU 互联的关键变量：链路带宽、延迟、拓扑均匀性、GPU-GPU 通信频率和跨节点边界
- 是否能把 GPU 互联与 TP/DP/PP 性能、decode 同步和 scaling efficiency联系起来
- 是否能通过对比同机/跨机通信时间、NCCL tests、TP size A/B 和 step breakdown验证判断

**回答框架**
- 先界定题面场景和 GPU 互联的判断边界
- 拆关键变量：链路带宽、延迟、拓扑均匀性、GPU-GPU 通信频率和跨节点边界
- 说明主要风险：高频通信落到 PCIe 或跨节点会放大等待
- 给出验证或落地方式：对比同机/跨机通信时间、NCCL tests、TP size A/B 和 step breakdown

**追问**
- 为什么 TP 比 DP 更依赖 NVLink/NVSwitch？
- 只有 PCIe 机器时你会如何调整并行策略？

**评分要点**
- 及格：能说出 GPU 互联的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对TP/DP/PP 性能、decode 同步和 scaling efficiency的影响

### 26.2.7 RDMA、RoCE 与普通 TCP 的差异

**问题**
多节点训练中为什么引入 RDMA/RoCE？它相比普通 TCP 解决什么问题，又带来哪些运维约束？

**考察点**
- 是否能说清 高性能网络的关键变量：kernel bypass、CPU 参与、PFC/ECN、MTU、QoS、拥塞控制、网卡和交换机 counters
- 是否能把 高性能网络 与多节点训练通信延迟、带宽和稳定性联系起来
- 是否能通过 nccl-tests、NCCL_DEBUG、ibstat/ethtool、交换机 counters 和 placement 对比验证判断

**回答框架**
- 先界定题面场景和 高性能网络的判断边界
- 拆关键变量：kernel bypass、CPU 参与、PFC/ECN、MTU、QoS、拥塞控制、网卡和交换机 counters
- 说明主要风险：RoCE 配置不一致会表现为 NCCL timeout、rank 慢或训练 hang
- 给出验证或落地方式：用 nccl-tests、NCCL_DEBUG、ibstat/ethtool、交换机 counters 和 placement 对比验证

**追问**
- RoCE 问题为什么常在规模扩大后才暴露？
- 网络丢包和某个 rank straggler 如何互相印证？

**评分要点**
- 及格：能说出 高性能网络的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对多节点训练通信延迟、带宽和稳定性的影响

### 26.2.8 存储吞吐与 checkpoint 写入

**问题**
训练每 30 分钟写一次 3 TB checkpoint，step time 周期性抖动。你如何分析存储瓶颈？

**考察点**
- 是否能说清 checkpoint 存储的关键变量：checkpoint 大小、写入窗口、shard 数、manifest、元数据、小文件和对象存储限流
- 是否能把 checkpoint 存储 与训练 step 抖动、恢复时间和存储成本联系起来
- 是否能通过时间线对齐、带宽估算、写入 profile、hash 校验和 restore smoke test 验证判断

**回答框架**
- 先界定题面场景和 checkpoint 存储的判断边界
- 拆关键变量：checkpoint 大小、写入窗口、shard 数、manifest、元数据、小文件和对象存储限流
- 说明主要风险：文件写完不代表 checkpoint 可恢复，异步写也可能提交半状态
- 给出验证或落地方式：用时间线对齐、带宽估算、写入 profile、hash 校验和 restore smoke test 验证

**追问**
- checkpoint 写成功和可恢复为什么不是一回事？
- 异步 checkpoint 可能破坏哪类一致性？

**评分要点**
- 及格：能说出 checkpoint 存储的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对训练 step 抖动、恢复时间和存储成本的影响

### 26.2.9 对象存储、并行文件系统与本地 NVMe 的选择

**问题**
请比较对象存储、并行文件系统、本地 NVMe 在训练数据读取和模型制品保存中的适用场景。

**考察点**
- 是否能说清 存储选型的关键变量：持久性、吞吐、延迟、目录语义、元数据性能、节点故障和成本
- 是否能把 存储选型 与训练数据读取、checkpoint 保存和制品分发联系起来
- 是否能通过按访问模式做对象存储持久化、NVMe 缓存、并行 FS 热写的混合验证判断

**回答框架**
- 先界定题面场景和 存储选型的判断边界
- 拆关键变量：持久性、吞吐、延迟、目录语义、元数据性能、节点故障和成本
- 说明主要风险：单一存储承载所有冷热路径会在成本或恢复语义上失衡
- 给出验证或落地方式：按访问模式做对象存储持久化、NVMe 缓存、并行 FS 热写的混合验证

**追问**
- 大 shard 和海量小文件分别适合什么存储策略？
- NVMe staging 后如何保证最终 checkpoint 可恢复？

**评分要点**
- 及格：能说出 存储选型的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对训练数据读取、checkpoint 保存和制品分发的影响

### 26.2.10 网络拓扑如何影响 AllReduce

**问题**
64 卡数据并行训练中，单机 8 卡很快，跨 8 机后 scaling efficiency 很差。请从网络拓扑分析。

**考察点**
- 是否能说清 AllReduce 网络的关键变量：机架/fabric、spine、收敛比、MTU、PFC/ECN、rank 映射和 NCCL 算法
- 是否能把 AllReduce 网络 与数据并行扩展效率和同步等待联系起来
- 是否能通过 nccl-tests、NCCL_DEBUG、交换机 counters、placement A/B 和 step breakdown 验证判断

**回答框架**
- 先界定题面场景和 AllReduce 网络的判断边界
- 拆关键变量：机架/fabric、spine、收敛比、MTU、PFC/ECN、rank 映射和 NCCL 算法
- 说明主要风险：单机 benchmark 无法覆盖跨节点拥塞和拓扑不均
- 给出验证或落地方式：用 nccl-tests、NCCL_DEBUG、交换机 counters、placement A/B 和 step breakdown 验证

**追问**
- 为什么 8 卡单机快不能证明 64 卡训练快？
- 如果只能改 placement，你会怎样约束节点集合？

**评分要点**
- 及格：能说出 AllReduce 网络的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对数据并行扩展效率和同步等待的影响

### 26.2.11 CPU 在 AI Infra 中为什么仍然重要

**问题**
既然核心计算在 GPU 上，为什么 CPU 仍然可能成为训练或推理瓶颈？请给出至少四个场景。

**考察点**
- 是否能说清 CPU 角色的关键变量：tokenizer、DataLoader、解码、采样、scheduler、gateway、日志、加密、网络栈和 NUMA
- 是否能把 CPU 角色与 GPU 空转、推理 p99 和数据管道吞吐联系起来
- 是否能通过 run queue、context switch、perf、线程池队列、tokenizer latency 和 GPU idle gap 验证判断

**回答框架**
- 先界定题面场景和 CPU 角色的判断边界
- 拆关键变量：tokenizer、DataLoader、解码、采样、scheduler、gateway、日志、加密、网络栈和 NUMA
- 说明主要风险：CPU 热路径常被 GPU 指标掩盖
- 给出验证或落地方式：用 run queue、context switch、perf、线程池队列、tokenizer latency 和 GPU idle gap 验证

**追问**
- DataLoader worker 增加反而变慢可能是什么原因？
- tokenizer 如何成为高 QPS 推理瓶颈？

**评分要点**
- 及格：能说出 CPU 角色的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对GPU 空转、推理 p99 和数据管道吞吐的影响

### 26.2.12 硬件故障与软件故障的初步区分

**问题**
训练随机报 CUDA error，有时是 ECC error，有时是 NCCL timeout。你如何区分硬件、驱动、网络和代码问题？

**考察点**
- 是否能说清 硬件故障排查的关键变量：固定节点/GPU/rank/数据、ECC、Xid、温度、驱动、NCCL、RoCE counters 和镜像版本
- 是否能把 硬件故障排查 与训练稳定性、自动隔离和故障归因联系起来
- 是否能通过通过任务迁移、节点隔离、burn-in、nccl-tests、日志关联和自动 drain 验证判断

**回答框架**
- 先界定题面场景和 硬件故障排查的判断边界
- 拆关键变量：固定节点/GPU/rank/数据、ECC、Xid、温度、驱动、NCCL、RoCE counters 和镜像版本
- 说明主要风险：随机 CUDA error 容易被误判为代码 bug 或网络抖动
- 给出验证或落地方式：通过任务迁移、节点隔离、burn-in、nccl-tests、日志关联和自动 drain 验证

**追问**
- 固定 GPU 复现和固定数据 batch 复现分别指向什么？
- 疑似坏卡自动下线前后要保留哪些证据？

**评分要点**
- 及格：能说出 硬件故障排查的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对训练稳定性、自动隔离和故障归因的影响

## 26.3 训练基础设施与分布式训练

### 26.3.1 单机训练吞吐下降的拆解

**问题**
单机 8 卡训练从 120 samples/s 降到 80 samples/s。你如何拆 step time，区分数据、计算、通信和 checkpoint 瓶颈？

**考察点**
- 是否能说清 单机训练瓶颈的关键变量：data/forward/backward/optimizer/communication/checkpoint 分段时间
- 是否能把 单机训练瓶颈 与单机基线、后续分布式扩展和成本联系起来
- 是否能通过 profiler、synthetic data、关闭 checkpoint、单卡/多卡对比验证判断

**回答框架**
- 先界定题面场景和 单机训练瓶颈的判断边界
- 拆关键变量：data/forward/backward/optimizer/communication/checkpoint 分段时间
- 说明主要风险：把整体 samples/s 下降误判成 GPU 问题
- 给出验证或落地方式：profiler、synthetic data、关闭 checkpoint、单卡/多卡对比

**追问**
- synthetic data 后吞吐恢复说明什么？
- checkpoint 抖动和 DataLoader 抖动如何区分？

**评分要点**
- 及格：能说出 单机训练瓶颈的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对单机基线、后续分布式扩展和成本的影响

### 26.3.2 DataLoader 让 GPU 等数据

**问题**
GPU 利用率周期性掉到 0，CPU worker 很忙。请分析 DataLoader、预取、pin memory 和远程读取的可能问题。

**考察点**
- 是否能说清 数据管道 feeding 的关键变量：远程读、小文件、解码/tokenization、worker 数、prefetch、pin memory、NUMA
- 是否能把 数据管道 feeding与 GPU feeding 和训练吞吐联系起来
- 是否能通过本地缓存、synthetic dataset、worker profile、H2D timeline验证判断

**回答框架**
- 先界定题面场景和 数据管道 feeding 的判断边界
- 拆关键变量：远程读、小文件、解码/tokenization、worker 数、prefetch、pin memory、NUMA
- 说明主要风险：盲目加 worker 可能加剧 CPU 或存储竞争
- 给出验证或落地方式：本地缓存、synthetic dataset、worker profile、H2D timeline

**追问**
- worker 数增加后变慢可能是 CPU 还是存储问题？
- pin memory 什么时候不是关键瓶颈？

**评分要点**
- 及格：能说出 数据管道 feeding 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对 GPU feeding 和训练吞吐的影响

### 26.3.3 数据并行 scaling limit

**问题**
32 卡数据并行从 16 卡扩到 32 卡只提升 20%。请解释同步、通信、global batch 和优化器对扩展效率的影响。

**考察点**
- 是否能说清 DP 扩展边界的关键变量：global batch、同步点、AllReduce、bucket、网络拓扑、straggler、optimizer
- 是否能把 DP 扩展边界 与多卡扩展效率和收敛风险联系起来
- 是否能通过scaling efficiency、step breakdown、NCCL 日志、bucket A/B 验证判断

**回答框架**
- 先界定题面场景和 DP 扩展边界的判断边界
- 拆关键变量：global batch、同步点、AllReduce、bucket、网络拓扑、straggler、optimizer
- 说明主要风险：只扩卡不调 batch/通信会降低效率
- 给出验证或落地方式：scaling efficiency、step breakdown、NCCL 日志、bucket A/B

**追问**
- global batch 改变为什么不是纯系统问题？
- 通信占比上升时先改 bucket 还是 placement？

**评分要点**
- 及格：能说出 DP 扩展边界的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对多卡扩展效率和收敛风险的影响

### 26.3.4 AllReduce 与梯度 bucket

**问题**
训练通信时间很长。你如何解释梯度 bucket、通信计算 overlap、NCCL 算法和网络拓扑的关系？

**考察点**
- 是否能说清 梯度通信的关键变量：bucket size、backward graph、NCCL 算法、overlap、网络带宽和 launch 开销
- 是否能把 梯度通信 与通信隐藏效果和训练 step time联系起来
- 是否能通过 profiler timeline、NCCL_DEBUG、bucket size A/B 验证判断

**回答框架**
- 先界定题面场景和 梯度通信的判断边界
- 拆关键变量：bucket size、backward graph、NCCL 算法、overlap、网络带宽和 launch 开销
- 说明主要风险：bucket 过大降低 overlap，过小增加协议开销
- 给出验证或落地方式：profiler timeline、NCCL_DEBUG、bucket size A/B

**追问**
- 为什么 bucket 调大不一定更快？
- 如何判断通信真正和 backward 重叠？

**评分要点**
- 及格：能说出 梯度通信的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对通信隐藏效果和训练 step time的影响

### 26.3.5 ZeRO 分片的取舍

**问题**
模型单卡放不下，有人建议直接上 ZeRO-3。请说明 ZeRO-1/2/3 分别省什么，代价是什么。

**考察点**
- 是否能说清 ZeRO 分片的关键变量：ZeRO-1/2/3 分别切 optimizer、gradient、parameter，通信和恢复复杂度
- 是否能把 ZeRO 分片 与显存上限、训练吞吐和 checkpoint 格式联系起来
- 是否能通过显存曲线、吞吐、通信时间、restore test、推理权重导出验证判断

**回答框架**
- 先界定题面场景和 ZeRO 分片的判断边界
- 拆关键变量：ZeRO-1/2/3 分别切 optimizer、gradient、parameter，通信和恢复复杂度
- 说明主要风险：省显存可能把瓶颈转移到网络和 gather/scatter
- 给出验证或落地方式：显存曲线、吞吐、通信时间、restore test、推理权重导出

**追问**
- ZeRO-3 为什么可能让网络成为瓶颈？
- ZeRO checkpoint 转推理权重要注意什么？

**评分要点**
- 及格：能说出 ZeRO 分片的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对显存上限、训练吞吐和 checkpoint 格式的影响

### 26.3.6 Tensor Parallel 的使用条件

**问题**
70B 训练需要 TP。请说明 TP 切分什么、依赖什么互联、会引入哪些通信和实现复杂度。

**考察点**
- 是否能说清 Tensor Parallel 的关键变量：矩阵切分、attention/FFN 通信、TP size、同机互联和 kernel efficiency
- 是否能把 Tensor Parallel 与超大模型可运行性和推理 decode 性能联系起来
- 是否能通过 TP size A/B、NCCL timeline、同机拓扑、tokens/s验证判断

**回答框架**
- 先界定题面场景和 Tensor Parallel 的判断边界
- 拆关键变量：矩阵切分、attention/FFN 通信、TP size、同机互联和 kernel efficiency
- 说明主要风险：TP size 过大导致通信压过计算
- 给出验证或落地方式：TP size A/B、NCCL timeline、同机拓扑、tokens/s

**追问**
- TP size 增大为什么不一定更快？
- 训练 TP 和推理 TP 的瓶颈有什么不同？

**评分要点**
- 及格：能说出 Tensor Parallel 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对超大模型可运行性和推理 decode 性能的影响

### 26.3.7 Pipeline Parallel 的气泡问题

**问题**
使用 PP 后 GPU 利用率不稳定。请解释 pipeline bubble、microbatch、stage balance 和调度策略。

**考察点**
- 是否能说清 Pipeline bubble 的关键变量：stage 划分、microbatch 数、stage balance、调度策略和通信
- 是否能把 Pipeline bubble与 GPU 利用率、显存和端到端吞吐联系起来
- 是否能通过pipeline timeline、stage time、microbatch A/B、重平衡实验验证判断

**回答框架**
- 先界定题面场景和 Pipeline bubble 的判断边界
- 拆关键变量：stage 划分、microbatch 数、stage balance、调度策略和通信
- 说明主要风险：按层数平均切分可能造成慢 stage 和 bubble
- 给出验证或落地方式：pipeline timeline、stage time、microbatch A/B、重平衡实验

**追问**
- microbatch 增大会带来哪些显存影响？
- stage balance 为什么不等于层数平均？

**评分要点**
- 及格：能说出 Pipeline bubble 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对GPU 利用率、显存和端到端吞吐的影响

### 26.3.8 混合并行策略选择

**问题**
给定 64 张 GPU 训练 70B 模型，请说明如何在 DP/TP/PP/ZeRO 之间做第一版并行策略。

**考察点**
- 是否能说清 混合并行的关键变量：DP、TP、PP、ZeRO、节点拓扑、模型规模、sequence length 和 batch
- 是否能把 混合并行 与64 卡训练可行性、吞吐和恢复复杂度联系起来
- 是否能通过显存估算、通信估算、小规模 dry run、restore test验证判断

**回答框架**
- 先界定题面场景和 混合并行的判断边界
- 拆关键变量：DP、TP、PP、ZeRO、节点拓扑、模型规模、sequence length 和 batch
- 说明主要风险：某一维并行过度会制造通信或 bubble 瓶颈
- 给出验证或落地方式：显存估算、通信估算、小规模 dry run、restore test

**追问**
- 为什么不能只把 64 卡全做 DP？
- 网络较弱时你会减少哪一维通信？

**评分要点**
- 及格：能说出 混合并行的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对64 卡训练可行性、吞吐和恢复复杂度的影响

### 26.3.9 激活重计算的收益与代价

**问题**
显存不够时开启 activation checkpointing。请说明它省了什么，增加了什么，如何评估是否值得。

**考察点**
- 是否能说清 激活重计算的关键变量：activation 保存点、重算范围、batch/context、额外 forward 计算和吞吐下降
- 是否能把 激活重计算 与显存换计算、batch 扩大和训练成本联系起来
- 是否能通过显存峰值、step time、layer profile、batch/context A/B 验证判断

**回答框架**
- 先界定题面场景和 激活重计算的判断边界
- 拆关键变量：activation 保存点、重算范围、batch/context、额外 forward 计算和吞吐下降
- 说明主要风险：粒度不当会省很少显存却显著拖慢
- 给出验证或落地方式：显存峰值、step time、layer profile、batch/context A/B

**追问**
- 什么情况下重计算反而提升总体吞吐？
- 如何判断 checkpoint 粒度过细？

**评分要点**
- 及格：能说出 激活重计算的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对显存换计算、batch 扩大和训练成本的影响

### 26.3.10 训练中 straggler 诊断

**问题**
多节点训练偶发某个 rank 慢 2 倍导致全局等待。你如何定位 straggler？

**考察点**
- 是否能说清 straggler 排查的关键变量：固定 rank/节点、GPU 降频、DataLoader wait、网络 counters、存储和温度
- 是否能把 straggler 排查 与同步训练全局等待和资源浪费联系起来
- 是否能通过rank timeline、迁移节点、nccl-tests、关闭数据路径对比验证判断

**回答框架**
- 先界定题面场景和 straggler 排查的判断边界
- 拆关键变量：固定 rank/节点、GPU 降频、DataLoader wait、网络 counters、存储和温度
- 说明主要风险：慢 rank 会拖住所有 rank，平均指标掩盖问题
- 给出验证或落地方式：rank timeline、迁移节点、nccl-tests、关闭数据路径对比

**追问**
- 固定 rank 慢和随机 rank 慢分别说明什么？
- straggler 为什么会放大全局 step time？

**评分要点**
- 及格：能说出 straggler 排查的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对同步训练全局等待和资源浪费的影响

### 26.3.11 Elastic Training 与恢复语义

**问题**
训练集群会抢占节点。请说明 elastic training 需要哪些状态、checkpoint 和一致性保证。

**考察点**
- 是否能说清 弹性训练的关键变量：模型、optimizer、scheduler、RNG、data cursor、global step、world size 和 shard 映射
- 是否能把 弹性训练 与抢占恢复、训练正确性和资源利用率联系起来
- 是否能通过checkpoint restore test、world size 变化演练、consumed samples 对齐验证判断

**回答框架**
- 先界定题面场景和 弹性训练的判断边界
- 拆关键变量：模型、optimizer、scheduler、RNG、data cursor、global step、world size 和 shard 映射
- 说明主要风险：只保存权重会导致恢复后语义漂移
- 给出验证或落地方式：checkpoint restore test、world size 变化演练、consumed samples 对齐

**追问**
- world size 变化会影响哪些训练语义？
- 抢占宽限期和 checkpoint 间隔如何一起设计？

**评分要点**
- 及格：能说出 弹性训练的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对抢占恢复、训练正确性和资源利用率的影响

### 26.3.12 NCCL Hang 的排查顺序

**问题**
训练挂住没有报错，只看到 NCCL 超时。你如何按 rank、网络、驱动、拓扑和代码顺序排查？

**考察点**
- 是否能说清 NCCL Hang 的关键变量：rank、op、节点、NCCL_DEBUG、driver、IB/RoCE counters、topology 和进程状态
- 是否能把 NCCL Hang 与分布式训练可用性和 oncall 效率联系起来
- 是否能通过缩小 world size、替换节点、nccl-tests、交换机 counters验证判断

**回答框架**
- 先界定题面场景和 NCCL Hang 的判断边界
- 拆关键变量：rank、op、节点、NCCL_DEBUG、driver、IB/RoCE counters、topology 和进程状态
- 说明主要风险：只看应用日志可能漏掉网络或坏节点证据
- 给出验证或落地方式：缩小 world size、替换节点、nccl-tests、交换机 counters

**追问**
- NCCL timeout 和 silent hang 排查有什么不同？
- 为什么要同时看 rank 日志和交换机 counters？

**评分要点**
- 及格：能说出 NCCL Hang 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对分布式训练可用性和 oncall 效率的影响

## 26.4 数据、制品、Checkpoint 与 Registry

### 26.4.1 训练数据版本为什么重要

**问题**
一次模型效果回退，但代码和超参没变。请说明数据版本、采样规则和血缘如何影响复现。

**考察点**
- 是否能说清 数据版本的关键变量：dataset hash、manifest、过滤规则、采样权重、tokenizer、时间窗口和血缘
- 是否能把 数据版本 与效果复现、回退定位和合规审计联系起来
- 是否能通过样本 diff、分布对比、小样本重跑、实验追踪记录验证判断

**回答框架**
- 先界定题面场景和 数据版本的判断边界
- 拆关键变量：dataset hash、manifest、过滤规则、采样权重、tokenizer、时间窗口和血缘
- 说明主要风险：同一路径下数据漂移会让实验不可解释
- 给出验证或落地方式：样本 diff、分布对比、小样本重跑、实验追踪记录

**追问**
- 为什么同一目录路径不能代表同一数据集？
- 数据删除或合规要求如何影响复现？

**评分要点**
- 及格：能说出 数据版本的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对效果复现、回退定位和合规审计的影响

### 26.4.2 Dataset manifest 的设计

**问题**
为大规模训练数据设计 manifest，你会记录哪些字段来支持 resume、审计和质量排查？

**考察点**
- 是否能说清 Dataset manifest 的关键变量：shard 路径/hash、样本数、token 数、schema、来源、过滤版本、shuffle seed、cursor
- 是否能把 Dataset manifest与 resume、审计和质量排查联系起来
- 是否能通过不可变 manifest、hash 校验、resume smoke test、血缘查询验证判断

**回答框架**
- 先界定题面场景和 Dataset manifest 的判断边界
- 拆关键变量：shard 路径/hash、样本数、token 数、schema、来源、过滤版本、shuffle seed、cursor
- 说明主要风险：缺少 manifest 会让数据覆盖和 resume 错误不可见
- 给出验证或落地方式：不可变 manifest、hash 校验、resume smoke test、血缘查询

**追问**
- manifest 缺少 shuffle seed 会影响什么？
- 如何避免训练中途数据集被覆盖？

**评分要点**
- 及格：能说出 Dataset manifest 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对resume、审计和质量排查的影响

### 26.4.3 Checkpoint sharding 的基本语义

**问题**
为什么大模型 checkpoint 常做 sharding？分片保存时 manifest、rank 映射和原子提交要解决什么？

**考察点**
- 是否能说清 Checkpoint sharding 的关键变量：shard、rank、tensor name、shape、dtype、offset、hash、parallel mapping、manifest
- 是否能把 Checkpoint sharding 与大模型并行保存、恢复和 rank 重映射联系起来
- 是否能通过原子 manifest、hash 校验、rank remap、restore smoke test验证判断

**回答框架**
- 先界定题面场景和 Checkpoint sharding 的判断边界
- 拆关键变量：shard、rank、tensor name、shape、dtype、offset、hash、parallel mapping、manifest
- 说明主要风险：分片文件齐全不代表版本一致或可恢复
- 给出验证或落地方式：原子 manifest、hash 校验、rank remap、restore smoke test

**追问**
- 为什么只看 shard 文件数量不够？
- rank 数变化后恢复会遇到什么映射问题？

**评分要点**
- 及格：能说出 Checkpoint sharding 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对大模型并行保存、恢复和 rank 重映射的影响

### 26.4.4 Checkpoint restore 的一致性

**问题**
训练从 checkpoint 恢复后 loss 异常。你会检查模型权重、优化器、scheduler、random state 和数据游标哪些语义？

**考察点**
- 是否能说清 Restore 一致性的关键变量：权重、optimizer、scheduler、RNG、global step、data cursor、parallel state、tokenizer/config
- 是否能把 Restore 一致性 与恢复后 loss、收敛和复现实验联系起来
- 是否能通过loss/grad norm/LR/consumed samples 对比、状态 hash、短跑校验验证判断

**回答框架**
- 先界定题面场景和 Restore 一致性的判断边界
- 拆关键变量：权重、optimizer、scheduler、RNG、global step、data cursor、parallel state、tokenizer/config
- 说明主要风险：只恢复权重会让优化器和数据游标漂移
- 给出验证或落地方式：loss/grad norm/LR/consumed samples 对比、状态 hash、短跑校验

**追问**
- 只恢复权重不恢复 optimizer 会怎样？
- data cursor 错误为什么可能不立刻报错？

**评分要点**
- 及格：能说出 Restore 一致性的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对恢复后 loss、收敛和复现实验的影响

### 26.4.5 异步 checkpoint 的风险

**问题**
为了减少训练停顿，有人要做 async checkpoint。请说明它的收益、失败模式和校验机制。

**考察点**
- 是否能说清 异步 checkpoint 的关键变量：快照隔离、后台写队列、manifest 提交、hash、失败回调和带宽反压
- 是否能把 异步 checkpoint 与训练停顿、数据安全和恢复可靠性联系起来
- 是否能通过快照版本号、写入状态机、restore test、后台失败注入验证判断

**回答框架**
- 先界定题面场景和 异步 checkpoint 的判断边界
- 拆关键变量：快照隔离、后台写队列、manifest 提交、hash、失败回调和带宽反压
- 说明主要风险：异步写可能保存半更新状态或静默失败
- 给出验证或落地方式：快照版本号、写入状态机、restore test、后台失败注入

**追问**
- 异步写入时如何避免半更新状态？
- 后台写失败后训练是否应该继续？

**评分要点**
- 及格：能说出 异步 checkpoint 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对训练停顿、数据安全和恢复可靠性的影响

### 26.4.6 Model Registry 应该记录什么

**问题**
一个 registry 只保存模型文件路径够不够？请设计必须记录的 metadata。

**考察点**
- 是否能说清 Model Registry 的关键变量：模型 hash、base、adapter、tokenizer、config、dtype、训练数据、代码、评测、安全扫描和状态机
- 是否能把 Model Registry 与发布门禁、兼容校验、审计和回滚联系起来
- 是否能通过metadata schema、不可变版本、状态流转、发布校验验证判断

**回答框架**
- 先界定题面场景和 Model Registry 的判断边界
- 拆关键变量：模型 hash、base、adapter、tokenizer、config、dtype、训练数据、代码、评测、安全扫描和状态机
- 说明主要风险：只保存路径会导致 latest 漂移和错误加载
- 给出验证或落地方式：metadata schema、不可变版本、状态流转、发布校验

**追问**
- 缺少 tokenizer 版本会造成什么线上问题？
- 哪些 metadata 应不可变，哪些可追加？

**评分要点**
- 及格：能说出 Model Registry 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对发布门禁、兼容校验、审计和回滚的影响

### 26.4.7 Release unit 如何定义

**问题**
模型上线的 release unit 应只包含权重，还是包含 tokenizer、配置、prompt、adapter 和评测报告？为什么？

**考察点**
- 是否能说清 Release unit 的关键变量：权重、tokenizer、model/generation config、adapter、prompt/template、runtime、评测、签名
- 是否能把 Release unit 与线上行为一致性、灰度和回滚联系起来
- 是否能通过不可变 release bundle、兼容校验、灰度和回滚演练验证判断

**回答框架**
- 先界定题面场景和 Release unit 的判断边界
- 拆关键变量：权重、tokenizer、model/generation config、adapter、prompt/template、runtime、评测、签名
- 说明主要风险：只发布权重会造成依赖漂移
- 给出验证或落地方式：不可变 release bundle、兼容校验、灰度和回滚演练

**追问**
- 只回滚权重不回滚 tokenizer 会怎样？
- prompt/template 是否应该进入 release unit？

**评分要点**
- 及格：能说出 Release unit 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对线上行为一致性、灰度和回滚的影响

### 26.4.8 Adapter 与 Base Model 兼容

**问题**
多 LoRA adapter 平台中，如何防止 adapter 挂到错误 base model 或错误 tokenizer 上？

**考察点**
- 是否能说清 Adapter 兼容的关键变量：base id/hash、tokenizer、target modules、rank、dtype、架构和权限
- 是否能把 Adapter 兼容 与多 LoRA 平台正确性和安全隔离联系起来
- 是否能通过加载前兼容校验、registry 组合 release、压测切换延迟验证判断

**回答框架**
- 先界定题面场景和 Adapter 兼容的判断边界
- 拆关键变量：base id/hash、tokenizer、target modules、rank、dtype、架构和权限
- 说明主要风险：adapter 挂错 base 可能质量异常但不报错
- 给出验证或落地方式：加载前兼容校验、registry 组合 release、压测切换延迟

**追问**
- adapter 挂错 base 为什么可能不立刻报错？
- 多租户 adapter 如何防止越权加载？

**评分要点**
- 及格：能说出 Adapter 兼容的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对多 LoRA 平台正确性和安全隔离的影响

### 26.4.9 模型制品签名与供应链

**问题**
为什么模型制品需要签名、SBOM 或 attestations？它们在发布门禁中阻止什么风险？

**考察点**
- 是否能说清 供应链签名的关键变量：hash、签名者、SBOM、attestation、SafeTensors、扫描、准入策略
- 是否能把 供应链签名 与制品篡改、恶意加载和发布合规联系起来
- 是否能通过构建到 registry 到 runtime 的签名校验和审计验证判断

**回答框架**
- 先界定题面场景和 供应链签名的判断边界
- 拆关键变量：hash、签名者、SBOM、attestation、SafeTensors、扫描、准入策略
- 说明主要风险：未签名模型或 pickle checkpoint 会绕过信任链
- 给出验证或落地方式：构建到 registry 到 runtime 的签名校验和审计

**追问**
- 签名证明了什么，不能证明什么？
- 为什么 pickle checkpoint 在生产加载有额外风险？

**评分要点**
- 及格：能说出 供应链签名的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对制品篡改、恶意加载和发布合规的影响

### 26.4.10 制品回滚与废弃策略

**问题**
线上模型发现问题需要回滚。registry 和发布系统要保留哪些状态，才能安全回滚和废弃旧版本？

**考察点**
- 是否能说清 制品回滚的关键变量：release graph、production pointer、兼容 runtime、缓存、状态机、审计和清理策略
- 是否能把 制品回滚 与线上止损、旧版本治理和合规留存联系起来
- 是否能通过回滚演练、cache invalidation、deprecated/blocked/deleted 状态验证判断

**回答框架**
- 先界定题面场景和 制品回滚的判断边界
- 拆关键变量：release graph、production pointer、兼容 runtime、缓存、状态机、审计和清理策略
- 说明主要风险：删除旧版本会破坏回滚和审计
- 给出验证或落地方式：回滚演练、cache invalidation、deprecated/blocked/deleted 状态

**追问**
- 回滚时哪些缓存需要失效？
- deprecated 和 deleted 为什么不能混用？

**评分要点**
- 及格：能说出 制品回滚的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对线上止损、旧版本治理和合规留存的影响

## 26.5 推理服务、KV Cache、Batching 与推理引擎

### 26.5.1 Prefill 与 Decode 的资源差异

**问题**
请解释 prefill 和 decode 在计算、内存访问、batching 和延迟上的差异，为什么容量规划必须拆开看。

**考察点**
- 是否能说清 Prefill/Decode 的关键变量：prompt tokens、decode tokens、TTFT、ITL、batch、HBM/KV 访问
- 是否能把 Prefill/Decode 与容量规划、长尾延迟和调度公平联系起来
- 是否能通过拆 TTFT/ITL、prefill time、decode batch、长短请求压测验证判断

**回答框架**
- 先界定题面场景和 Prefill/Decode 的判断边界
- 拆关键变量：prompt tokens、decode tokens、TTFT、ITL、batch、HBM/KV 访问
- 说明主要风险：把 prefill/decode 混成一个平均延迟会误导扩容
- 给出验证或落地方式：拆 TTFT/ITL、prefill time、decode batch、长短请求压测

**追问**
- 为什么长 prompt 会拖慢短请求 TTFT？
- prefill/decode 分离部署的收益和代价是什么？

**评分要点**
- 及格：能说出 Prefill/Decode 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对容量规划、长尾延迟和调度公平的影响

### 26.5.2 KV Cache 容量规划

**问题**
给定 70B 模型、8K context 和并发请求，请说明 KV Cache 如何估算，哪些因素会导致 OOM。

**考察点**
- 是否能说清 KV Cache 的关键变量：layers、heads、head_dim、dtype、context、并发、beam/candidate、block size
- 是否能把 KV Cache 与并发上限、OOM 和单位成本联系起来
- 是否能通过 KV 分布估算、压测峰值并发、监控 allocated/free blocks 验证判断

**回答框架**
- 先界定题面场景和 KV Cache 的判断边界
- 拆关键变量：layers、heads、head_dim、dtype、context、并发、beam/candidate、block size
- 说明主要风险：平均 context 估算会低估 p99 长上下文显存
- 给出验证或落地方式：按分布估算 KV、压测峰值并发、监控 allocated/free blocks

**追问**
- KV Cache 为什么不能简单放 Redis？
- 取消请求后哪些 KV 资源必须释放？

**评分要点**
- 及格：能说出 KV Cache 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对并发上限、OOM 和单位成本的影响

### 26.5.3 PagedAttention 的价值

**问题**
为什么 KV Cache 需要分页或 block 管理？它解决连续显存分配中的什么问题？

**考察点**
- 是否能说清 PagedAttention 的关键变量：block size、logical sequence、physical block、free list、fragmentation、eviction
- 是否能把 PagedAttention 与显存利用率和长上下文稳定性联系起来
- 是否能通过观察 allocated/free blocks、OOM、evict、TTFT/ITL 验证判断

**回答框架**
- 先界定题面场景和 PagedAttention 的判断边界
- 拆关键变量：block size、logical sequence、physical block、free list、fragmentation、eviction
- 说明主要风险：连续分配导致碎片和扩容失败
- 给出验证或落地方式：观察 allocated/free blocks、OOM、evict、TTFT/ITL

**追问**
- block size 过大或过小分别有什么问题？
- PagedAttention 不能解决哪些性能瓶颈？

**评分要点**
- 及格：能说出 PagedAttention 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对显存利用率和长上下文稳定性的影响

### 26.5.4 Continuous Batching 的取舍

**问题**
推理引擎使用 continuous batching 后吞吐升高但个别请求变慢。请说明原因和调参方向。

**考察点**
- 是否能说清 Continuous Batching 的关键变量：max batch tokens、waiting policy、priority、preemption、sequence lifecycle
- 是否能把 Continuous Batching 与吞吐、p99、公平性和 GPU 利用率联系起来
- 是否能通过混合长度压测、queue time、TTFT/ITL、per-tenant p99 验证判断

**回答框架**
- 先界定题面场景和 Continuous Batching 的判断边界
- 拆关键变量：max batch tokens、waiting policy、priority、preemption、sequence lifecycle
- 说明主要风险：吞吐提升可能牺牲单请求尾延迟
- 给出验证或落地方式：混合长度压测、queue time、TTFT/ITL、per-tenant p99

**追问**
- 吞吐升高但 p99 变差先调什么？
- 如何避免单个长请求占住 batch？

**评分要点**
- 及格：能说出 Continuous Batching 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对吞吐、p99、公平性和 GPU 利用率的影响

### 26.5.5 Chunked Prefill 的适用场景

**问题**
长 prompt 请求拖慢短请求。chunked prefill 如何改善队列公平性，代价是什么？

**考察点**
- 是否能说清 Chunked Prefill 的关键变量：chunk size、长短 prompt 混合、scheduler slot、KV 增长、TTFT/ITL
- 是否能把 Chunked Prefill 与长上下文公平性和 head-of-line blocking 联系起来
- 是否能通过输入长度分布压测、TTFT 分位数、ITL 抖动验证判断

**回答框架**
- 先界定题面场景和 Chunked Prefill 的判断边界
- 拆关键变量：chunk size、长短 prompt 混合、scheduler slot、KV 增长、TTFT/ITL
- 说明主要风险：chunk 太小增加调度开销，太大仍阻塞短请求
- 给出验证或落地方式：输入长度分布压测、TTFT 分位数、ITL 抖动

**追问**
- chunk 太小会带来什么开销？
- 哪些流量最适合开启 chunked prefill？

**评分要点**
- 及格：能说出 Chunked Prefill 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对长上下文公平性和 head-of-line blocking的影响

### 26.5.6 Prefix Cache 生命周期

**问题**
系统启用 prefix cache 后命中率不稳定。你如何设计 key、TTL、eviction 和租户隔离？

**考察点**
- 是否能说清 Prefix Cache 的关键变量：cache key、model version、tokenizer、adapter、TTL、refcount、eviction、tenant boundary
- 是否能把 Prefix Cache与 prefill 成本、显存占用和数据隔离联系起来
- 是否能通过命中率、节省 tokens、evict 原因、版本失效测试验证判断

**回答框架**
- 先界定题面场景和 Prefix Cache 的判断边界
- 拆关键变量：cache key、model version、tokenizer、adapter、TTL、refcount、eviction、tenant boundary
- 说明主要风险：错误 key 或跨租户复用会造成错误输出或泄露
- 给出验证或落地方式：命中率、节省 tokens、evict 原因、版本失效测试

**追问**
- 为什么 key 不能只用 prompt 文本？
- 模型灰度时旧 prefix cache 如何处理？

**评分要点**
- 及格：能说出 Prefix Cache 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对prefill 成本、显存占用和数据隔离的影响

### 26.5.7 Speculative Decoding 的收益边界

**问题**
有人建议用 speculative decoding 降低延迟。它依赖什么条件，什么时候收益会被抵消？

**考察点**
- 是否能说清 Speculative Decoding 的关键变量：draft model、acceptance rate、verify cost、输出长度、部署共置
- 是否能把 Speculative Decoding与 decode 延迟、额外显存和质量风险联系起来
- 是否能通过A/B TTFT/ITL、acceptance rate、质量和单位成本验证判断

**回答框架**
- 先界定题面场景和 Speculative Decoding 的判断边界
- 拆关键变量：draft model、acceptance rate、verify cost、输出长度、部署共置
- 说明主要风险：低接受率或草稿模型太慢会适得其反
- 给出验证或落地方式：A/B TTFT/ITL、acceptance rate、质量和单位成本

**追问**
- 接受率低时为什么可能更慢？
- 草稿模型共置会带来什么资源竞争？

**评分要点**
- 及格：能说出 Speculative Decoding 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对decode 延迟、额外显存和质量风险的影响

### 26.5.8 Serving Engine 选型

**问题**
vLLM、SGLang、TensorRT-LLM 或通用框架如何选？请按模型类型、延迟、吞吐、功能和运维成熟度比较。

**考察点**
- 是否能说清 Serving Engine 的关键变量：模型架构、context、LoRA、结构化输出、量化、硬件、团队运维能力
- 是否能把 Serving Engine 与性能、功能覆盖和升级风险联系起来
- 是否能通过同一压测集比较 TTFT/ITL/tokens/s/OOM/稳定性/发布流程验证判断

**回答框架**
- 先界定题面场景和 Serving Engine 的判断边界
- 拆关键变量：模型架构、context、LoRA、结构化输出、量化、硬件、团队运维能力
- 说明主要风险：只看 benchmark 可能忽略调试、灰度和兼容成本
- 给出验证或落地方式：同一压测集比较 TTFT/ITL/tokens/s/OOM/稳定性/发布流程

**追问**
- 什么时候不该选性能最高的引擎？
- 引擎升级如何设计灰度和回滚？

**评分要点**
- 及格：能说出 Serving Engine 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对性能、功能覆盖和升级风险的影响

### 26.5.9 量化对推理系统的影响

**问题**
INT8/FP8/4bit 量化降低显存和带宽，但可能影响质量和兼容。你如何做上线评估？

**考察点**
- 是否能说清 量化上线的关键变量：weight/activation/KV 量化、校准集、硬件支持、质量指标、回滚版本
- 是否能把 量化上线 与显存、带宽、吞吐、质量和兼容联系起来
- 是否能通过离线评测、shadow/canary、长尾 prompt、安全任务和成本对比验证判断

**回答框架**
- 先界定题面场景和 量化上线的判断边界
- 拆关键变量：weight/activation/KV 量化、校准集、硬件支持、质量指标、回滚版本
- 说明主要风险：性能提升可能换来长尾质量退化
- 给出验证或落地方式：离线评测、shadow/canary、长尾 prompt、安全任务和成本对比

**追问**
- 量化后吞吐提升但投诉增加看哪些证据？
- 为什么校准集不等于线上分布？

**评分要点**
- 及格：能说出 量化上线的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对显存、带宽、吞吐、质量和兼容的影响

### 26.5.10 流式输出与取消语义

**问题**
用户取消请求后，推理服务需要如何释放队列、KV Cache、网络连接和计费状态？

**考察点**
- 是否能说清 取消语义的关键变量：gateway、scheduler、sequence、KV blocks、stream connection、billing、trace
- 是否能把 取消语义 与资源泄漏、用户体验和计费正确性联系起来
- 是否能通过cancel rate、释放延迟、leaked blocks、幂等取消测试验证判断

**回答框架**
- 先界定题面场景和 取消语义的判断边界
- 拆关键变量：gateway、scheduler、sequence、KV blocks、stream connection、billing、trace
- 说明主要风险：取消传播不完整会泄漏 KV 或重复计费
- 给出验证或落地方式：cancel rate、释放延迟、leaked blocks、幂等取消测试

**追问**
- 取消正在 decode 的请求会影响同 batch 吗？
- 如何证明没有 KV block 泄漏？

**评分要点**
- 及格：能说出 取消语义的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对资源泄漏、用户体验和计费正确性的影响

### 26.5.11 多模型共置的风险

**问题**
同一批 GPU 上放多个模型副本或 adapter。请说明显存碎片、cache 污染和 SLO 干扰如何控制。

**考察点**
- 是否能说清 多模型共置的关键变量：权重常驻、显存碎片、adapter 切换、cache 污染、队列隔离、SLO tier
- 是否能把 多模型共置与 GPU 利用率、p99 和租户隔离联系起来
- 是否能通过混合流量压测、cache partition、admission control、独占池对比验证判断

**回答框架**
- 先界定题面场景和 多模型共置的判断边界
- 拆关键变量：权重常驻、显存碎片、adapter 切换、cache 污染、队列隔离、SLO tier
- 说明主要风险：共置可能用平均成本换来尾延迟和故障归因困难
- 给出验证或落地方式：混合流量压测、cache partition、admission control、独占池对比

**追问**
- 什么时候多模型共置不值得？
- adapter 热切换如何影响 p99？

**评分要点**
- 及格：能说出 多模型共置的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对GPU 利用率、p99 和租户隔离的影响

### 26.5.12 推理网关与引擎边界

**问题**
哪些逻辑应放在网关，哪些应放在 engine scheduler？请以鉴权、路由、限流、batching 为例。

**考察点**
- 是否能说清 网关/引擎边界的关键变量：鉴权、路由、限流、tenant metadata、batch scheduler、KV state、GPU 执行
- 是否能把 网关/引擎边界 与热路径性能和控制面可维护性联系起来
- 是否能通过定义 request metadata 契约、降级路径和路由反馈指标验证判断

**回答框架**
- 先界定题面场景和 网关/引擎边界的判断边界
- 拆关键变量：鉴权、路由、限流、tenant metadata、batch scheduler、KV state、GPU 执行
- 说明主要风险：把业务逻辑塞进 engine 或把调度放到网关都会失衡
- 给出验证或落地方式：定义 request metadata 契约、降级路径和路由反馈指标

**追问**
- 把限流放 engine 里有什么问题？
- 网关不知道 KV 状态时如何做路由？

**评分要点**
- 及格：能说出 网关/引擎边界的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对热路径性能和控制面可维护性的影响

### 26.5.13 Agent 推理与普通 Chat Serving

**问题**
Agent session 会多次调用模型和工具。推理服务与 Agent runtime 如何分工？

**考察点**
- 是否能说清 Agent Serving 的关键变量：session、step、tool call、budget、trace、prefix reuse、model call
- 是否能把 Agent Serving 与 Agent 成本、可取消性和 GPU 利用率联系起来
- 是否能通过 step trace、budget envelope、工具限流、GPU 活跃时间/等待时间拆分验证判断

**回答框架**
- 先界定题面场景和 Agent Serving 的判断边界
- 拆关键变量：session、step、tool call、budget、trace、prefix reuse、model call
- 说明主要风险：把 Agent 当超长请求会占住资源并失去治理
- 给出验证或落地方式：step trace、budget envelope、工具限流、GPU 活跃时间/等待时间拆分

**追问**
- 工具等待时 KV Cache 是否保留？
- Agent 失败时如何回放 step trace？

**评分要点**
- 及格：能说出 Agent Serving 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对 Agent 成本、可取消性和 GPU 利用率的影响

### 26.5.14 推理压测指标设计

**问题**
为新模型上线做压测，你会报告哪些指标，如何覆盖输入长度、输出长度、并发和租户混合？

**考察点**
- 是否能说清 推理压测的关键变量：输入/输出长度分布、并发、到达过程、cache warm/cold、租户混合、取消和故障
- 是否能把 推理压测 与容量规划、autoscaling 和发布门禁联系起来
- 是否能通过报告 TTFT/ITL/E2E/tokens/s/queue/KV/OOM/error/cost验证判断

**回答框架**
- 先界定题面场景和 推理压测的判断边界
- 拆关键变量：输入/输出长度分布、并发、到达过程、cache warm/cold、租户混合、取消和故障
- 说明主要风险：只测平均 1K prompt 会严重乐观
- 给出验证或落地方式：报告 TTFT/ITL/E2E/tokens/s/queue/KV/OOM/error/cost

**追问**
- 为什么平均 prompt 压测会误导容量？
- 如何避免 cache warm 造成过度乐观？

**评分要点**
- 及格：能说出 推理压测的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对容量规划、autoscaling 和发布门禁的影响

## 26.6 Kubernetes、调度、队列、配额与平台化

### 26.6.1 Kubernetes 中 GPU 资源的特殊性

**问题**
为什么 GPU 不能像 CPU millicore 一样随意切分？设备插件、MIG、MPS、time-slicing 各解决什么？

**考察点**
- 是否能说清 GPU 资源抽象的关键变量：device plugin、MIG、MPS、time-slicing、显存隔离、故障域
- 是否能把 GPU 资源抽象 与调度可用性、SLO 和资源利用率联系起来
- 是否能通过按负载压测共享/独占/MIG，观察 p99、OOM 和故障隔离验证判断

**回答框架**
- 先界定题面场景和 GPU 资源抽象的判断边界
- 拆关键变量：device plugin、MIG、MPS、time-slicing、显存隔离、故障域
- 说明主要风险：把 GPU 当 CPU millicore 会破坏隔离和归因
- 给出验证或落地方式：按负载压测共享/独占/MIG，观察 p99、OOM 和故障隔离

**追问**
- MIG 和 time-slicing 哪个更适合在线 SLO？
- 共享 GPU 为什么让故障归因变难？

**评分要点**
- 及格：能说出 GPU 资源抽象的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对调度可用性、SLO 和资源利用率的影响

### 26.6.2 Gang Scheduling 的必要性

**问题**
分布式训练为什么需要 gang scheduling？如果只启动了一半 Pod 会发生什么？

**考察点**
- 是否能说清 Gang Scheduling 的关键变量：minAvailable、rendezvous、world size、队列准入、超时回收
- 是否能把 Gang Scheduling 与分布式训练启动成功率和 GPU 空占联系起来
- 是否能通过调度事件、启动时间线、超时回收和失败重试演练验证判断

**回答框架**
- 先界定题面场景和 Gang Scheduling 的判断边界
- 拆关键变量：minAvailable、rendezvous、world size、队列准入、超时回收
- 说明主要风险：半启动会导致 NCCL/rendezvous 超时和资源浪费
- 给出验证或落地方式：调度事件、启动时间线、超时回收和失败重试演练

**追问**
- gang scheduling 如何和抢占冲突？
- 等待资源时是否应该先启动部分 Pod？

**评分要点**
- 及格：能说出 Gang Scheduling 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对分布式训练启动成功率和 GPU 空占的影响

### 26.6.3 拓扑感知调度

**问题**
TP/PP/DP 任务提交到集群时，调度器如何考虑同机、同机架、同 fabric 和 NUMA？

**考察点**
- 是否能说清 拓扑调度的关键变量：TP/PP/DP 通信模式、节点内拓扑、机架、fabric、NUMA、存储 locality
- 是否能把 拓扑调度 与训练吞吐稳定性和推理副本性能联系起来
- 是否能通过placement A/B、NCCL tests、step time、等待时间对比验证判断

**回答框架**
- 先界定题面场景和 拓扑调度的判断边界
- 拆关键变量：TP/PP/DP 通信模式、节点内拓扑、机架、fabric、NUMA、存储 locality
- 说明主要风险：拓扑约束过松会慢，过严会排队
- 给出验证或落地方式：placement A/B、NCCL tests、step time、等待时间对比

**追问**
- 拓扑约束太严格会带来什么排队问题？
- 如何证明 placement 改善了训练效率？

**评分要点**
- 及格：能说出 拓扑调度的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对训练吞吐稳定性和推理副本性能的影响

### 26.6.4 GPU 碎片化治理

**问题**
集群有很多 1 卡、2 卡、8 卡任务，为什么会出现 GPU 碎片？如何用队列和装箱策略缓解？

**考察点**
- 是否能说清 GPU 碎片化的关键变量：GPU shape、节点完整性、MIG 切片、bin packing、reservation、backfill
- 是否能把 GPU 碎片化 与大任务等待时间和集群利用率联系起来
- 是否能通过碎片率、可满足资源 shape、队列等待和 backfill 效果验证判断

**回答框架**
- 先界定题面场景和 GPU 碎片化的判断边界
- 拆关键变量：GPU shape、节点完整性、MIG 切片、bin packing、reservation、backfill
- 说明主要风险：空闲卡数量不等于可满足大任务 shape
- 给出验证或落地方式：碎片率、可满足资源 shape、队列等待和 backfill 效果

**追问**
- 为什么 8 张分散空闲卡不等于一个 8 卡节点？
- backfill 如何避免饿死大任务？

**评分要点**
- 及格：能说出 GPU 碎片化的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对大任务等待时间和集群利用率的影响

### 26.6.5 队列隔离与优先级规则

**问题**
多租户共享 GPU 平台，如何设计队列、priority class、抢占和保底，避免互相饿死？

**考察点**
- 是否能说清 队列优先级的关键变量：queue、priority、preemption、quota、borrow/lend、保底、宽限期
- 是否能把 队列优先级 与多租户公平、在线保护和训练效率联系起来
- 是否能通过队列等待、抢占率、SLO 违约、quota 使用率验证判断

**回答框架**
- 先界定题面场景和 队列优先级的判断边界
- 拆关键变量：queue、priority、preemption、quota、borrow/lend、保底、宽限期
- 说明主要风险：无规则抢占会破坏恢复，无保底会饿死关键业务
- 给出验证或落地方式：队列等待、抢占率、SLO 违约、quota 使用率

**追问**
- 保底资源闲置时能否借给别的团队？
- 抢占训练前要检查哪些恢复条件？

**评分要点**
- 及格：能说出 队列优先级的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对多租户公平、在线保护和训练效率的影响

### 26.6.6 Quota 与 Fair Share

**问题**
固定 quota、弹性 quota、DRF/fair share 各适合什么组织场景？如何解释给业务团队？

**考察点**
- 是否能说清 Quota/Fair Share 的关键变量：固定 quota、弹性 quota、DRF、GPU 型号折算、borrowed/preemptible
- 是否能把 Quota/Fair Share 与组织公平、预算和利用率联系起来
- 是否能通过使用率、等待时间、抢占率、成本归因和 showback验证判断

**回答框架**
- 先界定题面场景和 Quota/Fair Share 的判断边界
- 拆关键变量：固定 quota、弹性 quota、DRF、GPU 型号折算、borrowed/preemptible
- 说明主要风险：只按卡数配额会忽略 GPU 型号和多资源占用
- 给出验证或落地方式：使用率、等待时间、抢占率、成本归因和 showback

**追问**
- GPU 型号不同 quota 如何折算？
- 公平调度和高优先级项目冲突谁拍板？

**评分要点**
- 及格：能说出 Quota/Fair Share 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对组织公平、预算和利用率的影响

### 26.6.7 在线服务 Autoscaling

**问题**
大模型推理服务如何 autoscale？为什么只看 CPU 或 QPS 不够？

**考察点**
- 是否能说清 推理 Autoscaling 的关键变量：QPS、token/s、TTFT、ITL、queue time、KV usage、cold start、drain
- 是否能把 推理 Autoscaling 与推理 SLO、成本和容量水位联系起来
- 是否能通过预热扩容、模型加载时间、drain 测试、防抖窗口验证判断

**回答框架**
- 先界定题面场景和 推理 Autoscaling 的判断边界
- 拆关键变量：QPS、token/s、TTFT、ITL、queue time、KV usage、cold start、drain
- 说明主要风险：只看 CPU/QPS 会漏掉 token 长度和 KV 压力
- 给出验证或落地方式：预热扩容、模型加载时间、drain 测试、防抖窗口

**追问**
- QPS 相同但 token 长度不同容量差多少？
- 冷启动慢时如何提前扩容？

**评分要点**
- 及格：能说出 推理 Autoscaling 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对推理 SLO、成本和容量水位的影响

### 26.6.8 训练任务抢占与恢复

**问题**
低优先级训练任务被抢占时，平台要保证哪些 checkpoint、日志和队列状态？

**考察点**
- 是否能说清 训练抢占的关键变量：preemption notice、宽限期、checkpoint 新鲜度、queue position、retry budget
- 是否能把 训练抢占 与低优先级利用率和训练正确性联系起来
- 是否能通过抢占演练、restore smoke test、checkpoint age、失败重试指标验证判断

**回答框架**
- 先界定题面场景和 训练抢占的判断边界
- 拆关键变量：preemption notice、宽限期、checkpoint 新鲜度、queue position、retry budget
- 说明主要风险：没有新鲜 checkpoint 的抢占会浪费大量 GPU 时间
- 给出验证或落地方式：抢占演练、restore smoke test、checkpoint age、失败重试指标

**追问**
- 没有新鲜 checkpoint 能不能抢占？
- 抢占后重新排队是否保留原优先级？

**评分要点**
- 及格：能说出 训练抢占的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对低优先级利用率和训练正确性的影响

### 26.6.9 Notebook 到生产 Job 的边界

**问题**
研究用户习惯在 Notebook 上跑训练。平台如何把它转成可复现、可调度、可审计的 Job？

**考察点**
- 是否能说清 Notebook 生产化的关键变量：代码入口、镜像、依赖、数据版本、参数、Secrets、输出和指标
- 是否能把 Notebook 生产化 与复现性、调度治理和研究效率联系起来
- 是否能通过模板化提交、环境锁定、实验追踪、一键转 Job验证判断

**回答框架**
- 先界定题面场景和 Notebook 生产化的判断边界
- 拆关键变量：代码入口、镜像、依赖、数据版本、参数、Secrets、输出和指标
- 说明主要风险：Notebook 隐式状态会让成功实验不可重跑
- 给出验证或落地方式：模板化提交、环境锁定、实验追踪、一键转 Job

**追问**
- Notebook 隐式状态如何破坏复现？
- 如何兼顾研究灵活性和平台治理？

**评分要点**
- 及格：能说出 Notebook 生产化的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对复现性、调度治理和研究效率的影响

### 26.6.10 平台 API 设计

**问题**
为什么平台 API 应面向训练任务、评测任务和推理服务，而不是直接暴露 Pod、PVC 和节点选择？

**考察点**
- 是否能说清 平台 API 的关键变量：训练任务、评测任务、推理服务、release、quota、artifact、run record
- 是否能把 平台 API 与用户体验、治理和可审计性联系起来
- 是否能通过API 契约评审、escape hatch、审计记录、回滚路径验证判断

**回答框架**
- 先界定题面场景和 平台 API 的判断边界
- 拆关键变量：训练任务、评测任务、推理服务、release、quota、artifact、run record
- 说明主要风险：直接暴露 Pod/PVC 会把平台复杂度推给用户
- 给出验证或落地方式：API 契约评审、escape hatch、审计记录、回滚路径

**追问**
- 什么时候需要暴露底层 Pod 模板？
- API 太薄会把什么复杂度推给用户？

**评分要点**
- 及格：能说出 平台 API 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对用户体验、治理和可审计性的影响

### 26.6.11 多集群与混合云调度

**问题**
GPU 分布在自建集群和云上。如何处理镜像、数据、网络、成本和故障域？

**考察点**
- 是否能说清 多集群调度的关键变量：镜像、数据复制、模型制品、身份、网络、egress cost、故障域
- 是否能把 多集群调度 与成本、SLO 和数据合规联系起来
- 是否能通过按任务类型做 placement、复制校验、跨集群回滚演练验证判断

**回答框架**
- 先界定题面场景和 多集群调度的判断边界
- 拆关键变量：镜像、数据复制、模型制品、身份、网络、egress cost、故障域
- 说明主要风险：跨集群迁移可能被数据重力和网络费用抵消
- 给出验证或落地方式：按任务类型做 placement、复制校验、跨集群回滚演练

**追问**
- 什么时候不该把训练调到云上？
- 跨集群回滚需要哪些一致性？

**评分要点**
- 及格：能说出 多集群调度的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对成本、SLO 和数据合规的影响

### 26.6.12 平台化的反模式

**问题**
一个团队想一次性建设全功能 AI 平台。你会指出哪些过度设计和哪些必须先做？

**考察点**
- 是否能说清 平台演进的关键变量：大而全门户、过早自研调度、无制品闭环、无观测、无成本 owner
- 是否能把 平台演进 与平台交付速度和长期治理联系起来
- 是否能通过路线图用真实痛点、V1 闭环指标和用户迁移成本验证判断

**回答框架**
- 先界定题面场景和 平台演进的判断边界
- 拆关键变量：大而全门户、过早自研调度、无制品闭环、无观测、无成本 owner
- 说明主要风险：组件堆砌不能形成可运营闭环
- 给出验证或落地方式：路线图用真实痛点、V1 闭环指标和用户迁移成本验证

**追问**
- 第一版平台最不该做哪三个功能？
- 如何判断平台抽象已经过度？

**评分要点**
- 及格：能说出 平台演进的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对平台交付速度和长期治理的影响

## 26.7 可观测性、发布、安全、成本与多租户治理

### 26.7.1 AI Infra 可观测性的指标体系

**问题**
请为训练平台和推理平台分别设计核心指标，不要只说 CPU/GPU utilization。

**考察点**
- 是否能说清 指标体系的关键变量：训练 step/MFU/data wait/NCCL/checkpoint，推理 TTFT/ITL/queue/KV/tokens/s，治理 cost/SLO/version
- 是否能把 指标体系 与故障定位、容量规划和成本治理联系起来
- 是否能通过按模型、租户、版本和分位数建立 dashboard 与告警验证判断

**回答框架**
- 先界定题面场景和 指标体系的判断边界
- 拆关键变量：训练 step/MFU/data wait/NCCL/checkpoint，推理 TTFT/ITL/queue/KV/tokens/s，治理 cost/SLO/version
- 说明主要风险：只看 CPU/GPU 平均值无法解释 AI 负载
- 给出验证或落地方式：按模型、租户、版本和分位数建立 dashboard 与告警

**追问**
- 为什么 GPU util 不能作为唯一核心指标？
- 训练和推理的黄金信号分别是什么？

**评分要点**
- 及格：能说出 指标体系的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对故障定位、容量规划和成本治理的影响

### 26.7.2 高基数标签治理

**问题**
推理日志想按 user_id、prompt_id、model_version 全量打标签。你如何控制 cardinality 和成本？

**考察点**
- 是否能说清 高基数治理的关键变量：label cardinality、tenant/model/version、request id、exemplar、sampling、TTL
- 是否能把 高基数治理 与观测成本和查询可用性联系起来
- 是否能通过字段准入、logs/traces 分层、采样和聚合验证判断

**回答框架**
- 先界定题面场景和 高基数治理的判断边界
- 拆关键变量：label cardinality、tenant/model/version、request id、exemplar、sampling、TTL
- 说明主要风险：user_id/prompt_id 全进 metrics 会打爆时序库
- 给出验证或落地方式：字段准入、logs/traces 分层、采样和聚合验证

**追问**
- 哪些字段可以做 metric label？
- 如何排查单用户问题又不打爆 metrics？

**评分要点**
- 及格：能说出 高基数治理的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对观测成本和查询可用性的影响

### 26.7.3 SLO 与错误预算

**问题**
大模型推理服务的 SLO 应该如何定义？TTFT、ITL、错误率和质量指标如何进入错误预算？

**考察点**
- 是否能说清 SLO 的关键变量：availability、错误率、TTFT、ITL、E2E p99、质量采样、burn rate、SLA tier
- 是否能把 SLO 与发布节奏、容量投资和用户体验联系起来
- 是否能通过按模型/tier 计算 burn rate，联动灰度、回滚和限流验证判断

**回答框架**
- 先界定题面场景和 SLO 的判断边界
- 拆关键变量：availability、错误率、TTFT、ITL、E2E p99、质量采样、burn rate、SLA tier
- 说明主要风险：把质量和可用性混成一个数会难以执行
- 给出验证或落地方式：按模型/tier 计算 burn rate，联动灰度、回滚和限流

**追问**
- TTFT 和 ITL 哪个更影响流式体验？
- 质量下降是否消耗同一个错误预算？

**评分要点**
- 及格：能说出 SLO 的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对发布节奏、容量投资和用户体验的影响

### 26.7.4 灰度发布与回滚

**问题**
新模型上线如何设计 canary、shadow、A/B、自动回滚和人工审批？

**考察点**
- 是否能说清 发布治理的关键变量：离线评测、压测、扫描、shadow、canary、A/B、自动回滚、审批
- 是否能把 发布治理 与发布风险、质量和性能稳定联系起来
- 是否能通过版本 diff、门禁指标、canary 阈值、回滚演练验证判断

**回答框架**
- 先界定题面场景和 发布治理的判断边界
- 拆关键变量：离线评测、压测、扫描、shadow、canary、A/B、自动回滚、审批
- 说明主要风险：直接全量切流会放大模型/配置错误
- 给出验证或落地方式：用版本 diff、门禁指标、canary 阈值、回滚演练验证

**追问**
- shadow 和 canary 分别发现什么问题？
- 自动回滚阈值过敏有什么风险？

**评分要点**
- 及格：能说出 发布治理的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对发布风险、质量和性能稳定的影响

### 26.7.5 离线评测与线上质量

**问题**
离线 benchmark 通过但线上用户投诉。你如何设计线上质量采样和反馈闭环？

**考察点**
- 是否能说清 线上质量的关键变量：线上分布、用户反馈、人工审核、LLM judge、任务成功率、安全事件
- 是否能把 线上质量 与模型真实可用性和反馈闭环联系起来
- 是否能通过采样->标注/评测->归因->修复->灰度验证判断

**回答框架**
- 先界定题面场景和 线上质量的判断边界
- 拆关键变量：线上分布、用户反馈、人工审核、LLM judge、任务成功率、安全事件
- 说明主要风险：离线 benchmark 通过不代表线上任务成功
- 给出验证或落地方式：采样->标注/评测->归因->修复->灰度验证

**追问**
- 线上投诉和离线分数冲突时怎么判断？
- LLM judge 的偏差如何监控？

**评分要点**
- 及格：能说出 线上质量的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对模型真实可用性和反馈闭环的影响

### 26.7.6 安全隔离与 Secrets

**问题**
多租户训练任务需要访问数据和模型仓库。如何管理身份、Secrets、网络策略和审计？

**考察点**
- 是否能说清 安全隔离的关键变量：workload identity、短期凭证、最小权限、网络策略、审计、密钥轮换
- 是否能把 安全隔离 与多租户数据安全和事故影响面联系起来
- 是否能通过访问审计、secret 扫描、权限演练、泄露影响面分析验证判断

**回答框架**
- 先界定题面场景和 安全隔离的判断边界
- 拆关键变量：workload identity、短期凭证、最小权限、网络策略、审计、密钥轮换
- 说明主要风险：共享长期密钥和宽网络权限会放大越权
- 给出验证或落地方式：访问审计、secret 扫描、权限演练、泄露影响面分析

**追问**
- 如何防止一个 Job 读其他租户数据？
- Secrets 泄露后如何定位影响面？

**评分要点**
- 及格：能说出 安全隔离的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对多租户数据安全和事故影响面的影响

### 26.7.7 Prompt 与工具调用安全

**问题**
Agent 服务可调用内部工具。如何防止越权、提示注入和不可幂等副作用？

**考察点**
- 是否能说清 工具安全的关键变量：tool whitelist、schema、scoped credential、sandbox、egress、approval、idempotency
- 是否能把 工具安全 与 Agent 越权、副作用和审计联系起来
- 是否能通过工具执行 trace、权限测试、红队 prompt、审批日志验证判断

**回答框架**
- 先界定题面场景和 工具安全的判断边界
- 拆关键变量：tool whitelist、schema、scoped credential、sandbox、egress、approval、idempotency
- 说明主要风险：只靠 prompt 约束无法形成可靠安全边界
- 给出验证或落地方式：工具执行 trace、权限测试、红队 prompt、审批日志

**追问**
- 为什么让模型“不要越权”不可靠？
- 写操作工具如何设计人工审批？

**评分要点**
- 及格：能说出 工具安全的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对 Agent 越权、副作用和审计的影响

### 26.7.8 成本归因与 Chargeback

**问题**
GPU 集群成本很高。你如何把训练、推理、批处理和 idle 成本归因到团队和项目？

**考察点**
- 是否能说清 成本归因的关键变量：tenant/project/model/job/request、GPU-hour、GPU-second、token、storage、network、idle
- 是否能把 成本归因 与预算沟通、配额和利用率联系起来
- 是否能通过成本 dashboard、showback/chargeback、预算告警、单位成本趋势验证判断

**回答框架**
- 先界定题面场景和 成本归因的判断边界
- 拆关键变量：tenant/project/model/job/request、GPU-hour、GPU-second、token、storage、network、idle
- 说明主要风险：idle 和失败重试没人负责会造成成本黑洞
- 给出验证或落地方式：成本 dashboard、showback/chargeback、预算告警、单位成本趋势

**追问**
- 在线保底导致 idle 成本算给谁？
- 如何避免省钱破坏 SLO？

**评分要点**
- 及格：能说出 成本归因的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对预算沟通、配额和利用率的影响

### 26.7.9 多租户噪声隔离

**问题**
一个租户长上下文请求导致其他租户 p99 升高。你如何从队列、cache、配额和限流处理？

**考察点**
- 是否能说清 租户隔离的关键变量：tenant queue、token limit、concurrency、cache partition、priority、SLO tier
- 是否能把 租户隔离与 p99 稳定性和公平性联系起来
- 是否能通过按租户切分 p99、cache usage、queue time 和限流效果验证判断

**回答框架**
- 先界定题面场景和 租户隔离的判断边界
- 拆关键变量：tenant queue、token limit、concurrency、cache partition、priority、SLO tier
- 说明主要风险：长上下文或大租户会污染 cache 并拖慢其他租户
- 给出验证或落地方式：按租户切分 p99、cache usage、queue time 和限流效果

**追问**
- 长上下文租户按请求数还是 token 数限流？
- cache partition 会牺牲哪些利用率？

**评分要点**
- 及格：能说出 租户隔离的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对p99 稳定性和公平性的影响

### 26.7.10 合规审计与数据留存

**问题**
模型训练和推理日志涉及敏感数据。平台应如何做留存、脱敏、访问审计和删除？

**考察点**
- 是否能说清 合规留存的关键变量：敏感等级、脱敏、加密、TTL、访问审计、删除请求、legal hold、血缘
- 是否能把 合规留存 与隐私合规和模型复现联系起来
- 是否能通过日志采样审计、删除演练、血缘影响面查询、访问审批验证判断

**回答框架**
- 先界定题面场景和 合规留存的判断边界
- 拆关键变量：敏感等级、脱敏、加密、TTL、访问审计、删除请求、legal hold、血缘
- 说明主要风险：debug 日志和原始 prompt 最容易绕过策略
- 给出验证或落地方式：日志采样审计、删除演练、血缘影响面查询、访问审批

**追问**
- 删除某用户数据后如何证明已处理？
- 为什么 debug 日志最容易绕过合规？

**评分要点**
- 及格：能说出 合规留存的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对隐私合规和模型复现的影响

### 26.7.11 Incident 复盘模板

**问题**
一次推理服务事故后，你会如何写复盘，确保不是只写“扩容解决”？

**考察点**
- 是否能说清 Incident 复盘的关键变量：时间线、影响面、检测延迟、止血、根因、系统性原因、行动项 owner
- 是否能把 Incident 复盘 与可靠性改进和团队学习联系起来
- 是否能通过复盘行动项用指标、截止时间、演练和告警验证判断

**回答框架**
- 先界定题面场景和 Incident 复盘的判断边界
- 拆关键变量：时间线、影响面、检测延迟、止血、根因、系统性原因、行动项 owner
- 说明主要风险：只写扩容解决无法防复发
- 给出验证或落地方式：复盘行动项用指标、截止时间、演练和告警验证

**追问**
- “扩容解决”为什么不是根因？
- 如何判断行动项降低了复发概率？

**评分要点**
- 及格：能说出 Incident 复盘的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对可靠性改进和团队学习的影响

### 26.7.12 容量规划与预算沟通

**问题**
业务要下季度接入 5 倍流量。你如何把容量、SLO、成本和风险讲给管理层？

**考察点**
- 是否能说清 容量预算的关键变量：token/s、并发、KV 显存、GPU 数、峰值、冗余、SLO 档位、成本
- 是否能把 容量预算 与管理层决策和采购/云策略联系起来
- 是否能通过容量模型、压测、情景方案、风险清单和阶段预算验证判断

**回答框架**
- 先界定题面场景和 容量预算的判断边界
- 拆关键变量：token/s、并发、KV 显存、GPU 数、峰值、冗余、SLO 档位、成本
- 说明主要风险：平均流量倍数不能直接等于容量倍数
- 给出验证或落地方式：容量模型、压测、情景方案、风险清单和阶段预算

**追问**
- 为什么平均流量 5 倍不等于容量 5 倍？
- 预算不足时优先降低哪些服务等级？

**评分要点**
- 及格：能说出 容量预算的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对管理层决策和采购/云策略的影响

## 26.8 综合系统设计与故障排查 Case

### 26.8.1 设计 70B 在线推理平台

**问题**
给你 32 张 H100，设计一个 70B 模型在线推理平台，目标 p99 稳定、成本可控、支持灰度。

**考察点**
- 是否能说清 70B 推理平台的关键变量：模型/context/QPS/token 分布、SLO、GPU、KV、网关、engine、灰度、多租户
- 是否能把 70B 推理平台 与线上延迟、成本和可发布性联系起来
- 是否能通过容量估算、压测、canary、限流、回滚和成本 dashboard验证判断

**回答框架**
- 先界定题面场景和 70B 推理平台的判断边界
- 拆关键变量：模型/context/QPS/token 分布、SLO、GPU、KV、网关、engine、灰度、多租户
- 说明主要风险：只画组件不做容量和故障设计
- 给出验证或落地方式：容量估算、压测、canary、限流、回滚和成本 dashboard

**追问**
- p99 和成本不能同时满足时调什么？
- 如何处理长上下文突刺租户？

**评分要点**
- 及格：能说出 70B 推理平台的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对线上延迟、成本和可发布性的影响

### 26.8.2 设计分布式训练平台

**问题**
设计一个支持 100B 级训练的基础设施，从数据、调度、并行、checkpoint、观测和恢复说明。

**考察点**
- 是否能说清 分布式训练平台的关键变量：数据版本、调度、拓扑、并行策略、checkpoint、观测、恢复、registry
- 是否能把 分布式训练平台 与 100B 训练可完成性和可恢复性联系起来
- 是否能通过小规模 dry run、restore test、nccl-tests、step dashboard验证判断

**回答框架**
- 先界定题面场景和 分布式训练平台的判断边界
- 拆关键变量：数据版本、调度、拓扑、并行策略、checkpoint、观测、恢复、registry
- 说明主要风险：只关注算力不关注数据/恢复/观测会失败
- 给出验证或落地方式：小规模 dry run、restore test、nccl-tests、step dashboard

**追问**
- checkpoint 写入拖慢训练怎么改？
- 拓扑不足时牺牲效率还是排队时间？

**评分要点**
- 及格：能说出 分布式训练平台的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对100B 训练可完成性和可恢复性的影响

### 26.8.3 排查训练吞吐突然下降

**问题**
一个训练任务运行 6 小时后 throughput 下降 40%，没有代码变更。请给出排查剧本。

**考察点**
- 是否能说清 训练吞吐故障的关键变量：影响面、step breakdown、节点健康、网络 counters、数据源、存储、rank timeline
- 是否能把 训练吞吐故障 与训练成本和故障止血联系起来
- 是否能通过迁移节点、固定数据、关闭 checkpoint、nccl-tests、证据保全验证判断

**回答框架**
- 先界定题面场景和 训练吞吐故障的判断边界
- 拆关键变量：影响面、step breakdown、节点健康、网络 counters、数据源、存储、rank timeline
- 说明主要风险：无代码变更也可能有环境/数据/资源变化
- 给出验证或落地方式：迁移节点、固定数据、关闭 checkpoint、nccl-tests、证据保全

**追问**
- 没有代码变更时最易漏掉什么？
- 如何避免排查时污染证据？

**评分要点**
- 及格：能说出 训练吞吐故障的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对训练成本和故障止血的影响

### 26.8.4 排查推理 p99 长尾

**问题**
线上 p99 升高但平均延迟正常。请设计排查和止血流程，覆盖流量、batching、cache、依赖和租户。

**考察点**
- 是否能说清 推理 p99 故障的关键变量：租户、模型、输入长度、queue、TTFT、ITL、KV evict、依赖、版本
- 是否能把 推理 p99 故障 与用户体验和 SLO burn联系起来
- 是否能通过按维度切分、限流长请求、隔离租户、回滚、扩容验证判断

**回答框架**
- 先界定题面场景和 推理 p99 故障的判断边界
- 拆关键变量：租户、模型、输入长度、queue、TTFT、ITL、KV evict、依赖、版本
- 说明主要风险：平均延迟正常会掩盖长尾队列或长上下文问题
- 给出验证或落地方式：按维度切分、限流长请求、隔离租户、回滚、扩容

**追问**
- 平均延迟正常为什么 p99 会坏？
- 如何判断长尾来自队列还是 kernel？

**评分要点**
- 及格：能说出 推理 p99 故障的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对用户体验和 SLO burn的影响

### 26.8.5 从研究脚本到平台化发布

**问题**
团队已有训练脚本和手工部署。请设计 3 个月内的最小平台化路线图。

**考察点**
- 是否能说清平台落地路线的关键变量：代码入口、镜像、数据版本、checkpoint、registry、评测、发布、观测
- 是否能把平台落地路线与三个月平台 V1 和团队迁移联系起来
- 是否能通过月度里程碑、V1 成功指标、用户迁移和回滚演练验证判断

**回答框架**
- 先界定题面场景和平台落地路线的判断边界
- 拆关键变量：代码入口、镜像、数据版本、checkpoint、registry、评测、发布、观测
- 说明主要风险：先做门户或复杂工作流会绕开最小闭环
- 给出验证或落地方式：月度里程碑、V1 成功指标、用户迁移和回滚演练

**追问**
- 如何让研究团队愿意迁移？
- V1 成功指标是什么？

**评分要点**
- 及格：能说出平台落地路线的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对三个月平台 V1 和团队迁移的影响

### 26.8.6 多租户 GPU 平台资源争用

**问题**
两个团队分别跑在线推理和长训练，互相影响。请设计隔离、配额、抢占和沟通机制。

**考察点**
- 是否能说清多租户争用的关键变量：GPU/HBM/网络/存储/队列/cache、保底、borrow/lend、抢占、成本
- 是否能把多租户争用与在线保护、训练效率和组织公平联系起来
- 是否能通过队列策略、SLO tier、抢占演练、成本归因、容量日历验证判断

**回答框架**
- 先界定题面场景和多租户争用的判断边界
- 拆关键变量：GPU/HBM/网络/存储/队列/cache、保底、borrow/lend、抢占、成本
- 说明主要风险：没有明确规则会让冲突变成人工协调
- 给出验证或落地方式：队列策略、SLO tier、抢占演练、成本归因、容量日历

**追问**
- 训练打满 NIC 影响推理怎么隔离？
- 保底闲置时如何提高利用率？

**评分要点**
- 及格：能说出多租户争用的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对在线保护、训练效率和组织公平的影响

### 26.8.7 Agent 平台成本失控

**问题**
Agent 产品上线后工具调用和 reasoning token 成本暴涨。请设计预算、限流、观测和降级方案。

**考察点**
- 是否能说清 Agent 成本的关键变量：reasoning tokens、model calls、tool calls、verifier、session TTL、budget envelope
- 是否能把 Agent 成本与 Agent 单位经济性和可治理性联系起来
- 是否能通过 step 级 trace、reserve/settle、限流、降级和成本归因验证判断

**回答框架**
- 先界定题面场景和 Agent 成本的判断边界
- 拆关键变量：reasoning tokens、model calls、tool calls、verifier、session TTL、budget envelope
- 说明主要风险：按 request 计费会漏掉多步和工具等待
- 给出验证或落地方式：step 级 trace、reserve/settle、限流、降级和成本归因

**追问**
- 预算耗尽时如何避免随机中断？
- 如何判断成本上涨来自工具还是 decode？

**评分要点**
- 及格：能说出 Agent 成本的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对 Agent 单位经济性和可治理性的影响

### 26.8.8 模型供应链事故应急

**问题**
线上加载了未签名或错误 tokenizer 的模型制品。请说明止血、排查、回滚和长期门禁修复。

**考察点**
- 是否能说清供应链事故的关键变量：签名、hash、tokenizer/config、release unit、缓存、admission、审计
- 是否能把供应链事故与线上安全、止血和长期门禁联系起来
- 是否能通过冻结发布、回滚可信版本、影响面查询、运行时校验和门禁演练验证判断

**回答框架**
- 先界定题面场景和供应链事故的判断边界
- 拆关键变量：签名、hash、tokenizer/config、release unit、缓存、admission、审计
- 说明主要风险：未签名或错误 tokenizer 会造成错误输出和信任链断裂
- 给出验证或落地方式：冻结发布、回滚可信版本、影响面查询、运行时校验和门禁演练

**追问**
- 错误 tokenizer 已进入 cache 如何清理？
- 如何证明没有其他未签名制品在线？

**评分要点**
- 及格：能说出供应链事故的基本组成和常见风险
- 良好：能围绕关键变量给出有顺序的判断路径
- 优秀：能用验证证据支撑取舍，并说明对线上安全、止血和长期门禁的影响

## Mock Interview Pack 1：Inference Platform Engineer，60 分钟

- **Warm-up questions**：26.2.2、26.5.1、26.5.2，确认候选人是否能把 GPU 指标、prefill/decode 和 KV Cache 说成容量模型。
- **Deep-dive questions**：26.5.4、26.5.8、26.6.7、26.7.9，围绕 batching、engine 选型、autoscaling 和租户隔离连续追问。
- **Case prompt**：设计 32 张 H100 上的 70B 在线推理平台，支持长上下文、灰度发布和多租户限流。
- **Strong interviewer listens for**：候选人是否区分 TTFT/ITL/吞吐/p99，是否知道 cache 生命周期和队列公平性，是否能把成本、SLO 和发布风险一起讲。

## Mock Interview Pack 2：Training Infrastructure Engineer，60 分钟

- **Warm-up questions**：26.2.5、26.3.1、26.3.3，确认候选人是否理解拓扑、单机瓶颈和 DP 扩展边界。
- **Deep-dive questions**：26.3.5、26.3.8、26.3.10、26.4.4，压测并行策略、straggler 和恢复语义。
- **Case prompt**：设计一个 64 卡训练平台，要求支持 checkpoint 恢复、拓扑感知调度和训练吞吐观测。
- **Strong interviewer listens for**：候选人是否能拆 step time，是否知道 ZeRO/TP/PP 的通信代价，是否把 checkpoint 当成恢复协议而不是文件保存。

## Mock Interview Pack 3：AI Platform Engineer，60 分钟

- **Warm-up questions**：26.1.3、26.1.8、26.4.6，确认候选人是否有平台分层、最小闭环和 registry 思维。
- **Deep-dive questions**：26.6.5、26.6.6、26.7.4、26.7.8，围绕队列、配额、发布和成本治理追问。
- **Case prompt**：为 20 人算法团队设计 3 个月 AI 平台 V1，从训练提交到模型发布形成闭环。
- **Strong interviewer listens for**：候选人是否拒绝组件堆砌，是否能说明哪些能力先做、哪些延后，是否能把平台 API 设计成任务抽象。

## Mock Interview Pack 4：Reliability and Troubleshooting Round，45 分钟

- **Warm-up questions**：26.1.7、26.2.12，确认候选人是否能从告警和错误日志做故障分类。
- **Deep-dive questions**：26.3.12、26.5.14、26.7.11，分别压训练 hang、推理压测和事故复盘。
- **Case prompt**：线上推理 p99 突然升高，训练集群同时出现 NCCL timeout。请在 15 分钟内给出止血和初判。
- **Strong interviewer listens for**：候选人是否先确认影响面，是否能分离止血与根因，是否会保护证据和写出后续修复项。

## Mock Interview Pack 5：AI Infra Tech Lead System Design，90 分钟

- **Warm-up questions**：26.1.1、26.1.4、26.7.12，确认候选人是否能从系统判断、负载画像和预算沟通起步。
- **Deep-dive questions**：26.6.12、26.7.3、26.8.1、26.8.2，覆盖平台演进、SLO、推理系统和训练系统设计。
- **Case prompt**：公司要统一训练、推理、Agent 和制品治理平台。请给出 6 个月路线图、关键取舍、组织边界和风险控制。
- **Strong interviewer listens for**：候选人是否能做阶段化设计，是否能在利用率、SLO、安全和研发效率之间显式取舍，是否能说出不做什么。

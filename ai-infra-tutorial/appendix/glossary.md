# 附录A：AI Infra 术语表

| 术语 | 简要解释 |
|------|----------|
| AI Infra（AI Infrastructure） | 承载 AI 数据、训练、评测、部署、推理、监控与治理的基础设施体系 |
| GPU（Graphics Processing Unit） | 适合高吞吐并行计算的加速设备，常用于训练和推理 |
| 显存（GPU Memory / VRAM） | GPU 上的高速内存，决定模型、batch、缓存能否放下 |
| Checkpoint | 训练过程中的可恢复状态，通常包含模型参数、优化器状态和 step 信息 |
| SafeTensors | 更窄、更安全的张量序列化格式，常用于替代依赖 `pickle` 的权重保存方式 |
| 模型包（Model Artifact） | 面向部署的模型产物，通常包含权重、配置、tokenizer 与推理元数据 |
| 数据并行（Data Parallelism） | 多个设备各自处理不同 batch 分片，并同步梯度 |
| 张量并行（Tensor Parallelism） | 将同一层内部计算切分到多个设备上执行 |
| 流水线并行（Pipeline Parallelism） | 将模型不同层分配到不同设备或阶段执行 |
| Sequence Parallelism | 在张量并行组内继续切分序列维度上的部分计算，以降低激活和显存压力 |
| Context Parallelism | 切分 attention 的序列维度来支持更长上下文训练的并行方式 |
| Ring Attention | 一类用环形通信实现 Context Parallelism 的 attention 方案 |
| FSDP（Fully Sharded Data Parallel） | PyTorch 官方的全分片训练实现，按参数 / 梯度 / 优化器状态切分显存压力 |
| ZeRO（Zero Redundancy Optimizer） | 通过分片优化器状态、梯度和参数来降低单卡显存占用的技术族 |
| MFU（Model FLOPs Utilization） | 实际模型有效计算吞吐占理论峰值的比例，强调“算得值不值” |
| HFU（Hardware FLOPs Utilization） | 从硬件角度衡量总 FLOPs 利用率，通常比 MFU 更宽泛 |
| NCCL（NVIDIA Collective Communications Library） | 常用于 GPU 间通信的集合通信库 |
| All-reduce | 多个进程聚合数据并把结果分发给所有进程的通信操作 |
| Fat-tree Topology | 一类提供较均衡跨节点带宽的数据中心网络拓扑，常见于大规模训练集群 |
| Rail-optimized Topology | 让每个 GPU 或节点优先走固定 rail 的网络设计，用更低成本换可接受带宽 |
| FlashAttention | 通过重排 attention 计算和显存访问，降低显存带宽压力的注意力优化方法 |
| KV Cache（Key-Value Cache） | LLM 推理中缓存历史 key/value，减少 decode 阶段重复计算 |
| Continuous Batching | LLM 服务中动态组织正在生成的请求，提高 decode 吞吐的批处理方式 |
| PagedAttention | 将 KV Cache 分块管理以降低显存碎片和预分配浪费的思路 |
| Prefix Caching | 复用相同输入前缀的 KV Cache，减少重复 prefill 计算 |
| Disaggregated Serving | 将 prefill 与 decode 拆成不同服务层或不同资源池的推理架构 |
| Speculative Decoding | 用小模型先生成草稿、大模型再验证，从而加速解码的推理策略 |
| Inference-Time Compute | 在推理阶段额外投入计算，如思维链、搜索、工具调用或树搜索 |
| MoE（Mixture of Experts） | 由多个专家子网络组成、按 token 路由激活部分专家的模型结构 |
| LoRA（Low-Rank Adaptation） | 通过低秩增量矩阵做参数高效微调的方法 |
| QLoRA（Quantized LoRA） | 把量化和 LoRA 结合起来，以更低显存做微调的方法 |
| Multi-LoRA Serving | 一个 base model 实例挂载多个 LoRA adapter，并按请求切换的服务模式 |
| RLHF（Reinforcement Learning from Human Feedback） | 基于人类反馈训练奖励或策略模型的后训练方法 |
| DPO（Direct Preference Optimization） | 直接利用偏好对做优化、避免在线强化学习环节的对齐方法 |
| PPO（Proximal Policy Optimization） | RLHF 中常见的策略梯度优化算法 |
| GRPO（Group Relative Policy Optimization） | 通过同组多个采样结果的相对奖励做优化、常见于去掉 critic 的后训练路线 |
| SLO（Service Level Objective） | 服务等级目标，用于定义可用性、延迟、错误率等目标 |
| RAG（Retrieval-Augmented Generation） | 检索增强生成，把外部知识检索结果引入模型上下文 |
| Embedding | 将文本、图片等对象映射为向量表示 |
| 向量索引（Vector Index） | 支持近似最近邻检索的数据结构或服务 |
| MIG（Multi-Instance GPU） | NVIDIA 的硬件级 GPU 切分能力，可把一张卡分成多个隔离实例 |
| MPS（Multi-Process Service） | NVIDIA 的多进程共享机制，让多个进程复用同一 GPU 上下文 |
| Time-Slicing | 在平台层按时间片复用 GPU 的方式，隔离弱但门槛低 |
| Straggler | 分布式训练里显著慢于其他 worker、拖慢整体同步节奏的慢节点 |
| Elastic Training | 允许训练过程中动态增减 worker，并保持作业继续推进的能力 |
| Spot Instance | 云上可被抢占的低价实例，适合能从 checkpoint 恢复的离线作业 |
| GGML | 面向轻量推理的张量与推理实现项目，常见于端侧 / CPU 推理生态 |
| GGUF | GGML / llama.cpp 生态常见的模型封装格式，便于本地与端侧分发 |
| Canary Release | 让新版本先接入少量真实流量，再逐步放量的发布方式 |
| Blue-Green Deployment | 准备两套独立环境，通过切流快速完成发布或回滚的部署方式 |
| DRF（Dominant Resource Fairness） | 面向多资源系统的公平分配思路，关注租户占用的“主导资源”比例 |
| SLSA（Supply-chain Levels for Software Artifacts） | 用于提升软件供应链可追溯性和可信度的分级框架 |
| 灰度发布（Progressive Delivery / Canary Rollout） | 让新模型或新服务先接收小比例流量，再逐步放量 |
| 成本归因（Cost Attribution / Chargeback） | 将 GPU、存储、网络等资源成本归属到团队、项目、任务或模型 |
| CPU 流水线（Pipeline） | 把一条指令拆成取指、译码、执行、访存、写回等阶段，让多条指令在不同阶段重叠执行 |
| CPI（Cycles Per Instruction） | 平均每条指令消耗的 CPU cycle 数；真实 CPI 会被 cache miss、分支误预测、依赖和资源冲突抬高 |
| 乱序执行（Out-of-Order / OoO） | CPU 在不破坏程序语义的前提下，绕过暂时阻塞的指令，先执行已就绪指令以提高指令级并行 |
| Register Renaming | 通过物理寄存器重命名消除 WAR/WAW 等假依赖，让 OoO 更容易发掘并行度 |
| ROB（Reorder Buffer） | 记录乱序执行结果并按程序顺序提交的硬件结构，用于保证异常和提交语义正确 |
| 分支预测（Branch Prediction） | CPU 预测条件分支方向和目标地址，减少流水线等待；误预测会导致流水线刷新 |
| BTB（Branch Target Buffer） | 缓存分支目标地址的硬件结构，帮助 CPU 在取指阶段提前跳转到预测路径 |
| SIMD | 单指令多数据并行，一条指令同时处理多个数据元素，常见于 AVX、AVX-512、tokenizer 和 preprocessing 热点 |
| L1 / L2 / L3 Cache | CPU 多级缓存层次，越靠近核心容量越小、延迟越低；L3 通常跨核心或跨核心簇共享 |
| Cache Line | CPU cache 与内存之间传输的最小粒度，常见为 64B；不恰当布局会造成带宽浪费或伪共享 |
| Cache Associativity | 一个内存块可映射到多少个 cache way 的规则；关联度不足会导致冲突 miss |
| MESI | CPU 缓存一致性协议中的 Modified、Exclusive、Shared、Invalid 四状态，用于协调多核对同一 cache line 的读写 |
| 伪共享（False Sharing） | 多个线程修改不同变量，但变量落在同一 cache line 上，导致无意义的一致性失效和性能下降 |
| 虚拟内存（Virtual Memory） | 进程看到的地址空间抽象，由页表映射到物理内存，可支持隔离、mmap、换页和共享 |
| 页表（Page Table） | 记录虚拟页到物理页映射的数据结构，页表遍历开销通常由 TLB 缓解 |
| TLB（Translation Lookaside Buffer） | 缓存虚拟地址到物理地址翻译结果的硬件缓存；TLB miss 会增加内存访问延迟 |
| Page Cache | Linux 用内存缓存文件内容的机制，可显著加速重复读取，也会让存储 benchmark 产生误判 |
| Dirty Page | 已在内存中修改但尚未写回持久介质的页；过多脏页可能放大 checkpoint 尾延迟 |
| Huge Pages / THP | 使用更大页面减少页表和 TLB 压力；THP 自动化更方便，显式 HugeTLB 可控性更强 |
| NUMA | 多 socket 或多内存控制器机器上的非均匀内存访问架构，跨 NUMA 访问会增加延迟并降低带宽 |
| Syscall | 用户态进入内核态请求 OS 服务的调用；高频 syscall 会带来上下文切换和内核路径开销 |
| `epoll` | Linux 事件通知机制，适合大量 socket 的 readiness-based IO 多路复用 |
| `io_uring` | Linux 异步 IO 接口，通过提交队列和完成队列降低 syscall 与上下文切换成本 |
| PCIe Lane | PCIe 链路的基本通道，lane 数和代际共同决定 GPU、NIC、NVMe 与 CPU 之间的带宽上限 |
| DMA（Direct Memory Access） | 设备绕过 CPU 直接读写主存的机制，是高速网卡、NVMe、GPU 数据搬运的基础 |
| Pinned Memory / Page-Locked Memory | 不会被换出的主机内存，常用于 `cudaMemcpyAsync` 和 H2D/D2H 异步拷贝 |
| VFS（Virtual File System） | Linux 把不同文件系统统一成 inode、dentry、file 等抽象的内核层 |
| inode | 文件系统中记录文件元数据和数据块位置的对象，不等同于文件名 |
| dentry | Linux VFS 中目录项缓存，用于把路径名解析到 inode |
| ext4 Journal | ext4 用日志记录元数据或数据提交顺序，提高崩溃恢复能力，但可能引入写放大和提交延迟 |
| XFS B+tree | XFS 用于管理 extent、空闲空间和目录等元数据的 B+tree 结构，适合大文件和并发写场景 |
| ZFS COW（Copy-on-Write） | ZFS 写新块再切换引用的写入语义，便于快照和校验，但会改变写放大与碎片特征 |
| ARC（Adaptive Replacement Cache） | ZFS 的自适应读缓存，用于在最近访问和频繁访问数据之间平衡缓存空间 |
| `fsync` | 要求把文件相关脏数据刷到持久介质的系统调用，是 checkpoint 一致性语义的重要边界 |
| `O_DIRECT` | 尽量绕过 Page Cache 做直接 IO 的打开选项，可减少缓存污染，但带来对齐和吞吐约束 |
| Lustre MDS / OSS | Lustre 中 MDS 负责元数据，OSS/OST 负责对象数据存储；小文件常压 MDS，大文件吞吐看 stripe 和 OSS |
| Stripe | 并行文件系统把一个文件切分到多个存储目标上的布局策略，影响大文件吞吐和恢复行为 |
| TCP CUBIC | Linux 常见 TCP 拥塞控制算法，按丢包和窗口增长调节发送速率 |
| BBR | 基于瓶颈带宽和 RTT 估计的 TCP 拥塞控制算法，常用于改善长距离或特定网络路径吞吐 |
| MTU | 单个链路层帧可承载的最大传输单元，MTU 不一致会造成分片、丢包或性能下降 |
| Jumbo Frame | 通常指 MTU 约 9000 的以太网大帧，可降低大流量传输的包数和 CPU/交换机处理开销 |
| RSS / RPS | 接收端多队列和软件分发机制，用于把网卡收包负载分摊到多个 CPU core |
| RDMA Verbs | RDMA 编程接口抽象，包括 QP、CQ、WR、WC 等，用于提交和完成零拷贝网络操作 |
| QP（Queue Pair） | RDMA 通信端点，包含发送队列和接收队列 |
| CQ（Completion Queue） | RDMA 完成队列，用于报告 Work Request 的完成状态 |
| WR / WC | RDMA Work Request 表示提交的工作项，Work Completion 表示完成结果 |
| RoCE v2 | 在 UDP/IP 之上承载 RDMA 的以太网方案，需要更严格的拥塞和丢包控制 |
| PFC（Priority Flow Control） | 以太网优先级流控，可降低 RoCE 丢包，但配置不当会导致拥塞扩散 |
| ECN（Explicit Congestion Notification） | 网络设备在不丢包的情况下标记拥塞，RoCE 网络常用它触发端侧降速 |
| GPUDirect RDMA | 让 NIC 直接读写 GPU 显存的数据路径，减少 CPU 内存中转和拷贝开销 |
| 梯度压缩（Gradient Compression） | 通过量化、稀疏化或低秩近似减少梯度同步通信量的技术 |
| PowerSGD | 用低秩矩阵近似梯度来降低 all-reduce 传输量的梯度压缩方法 |
| Interleaved Pipeline | 把每个流水线 stage 再切成多个 virtual stage，减少 pipeline bubble 的调度方式 |
| Zero Bubble Pipeline | 通过重排前向、反向和权重梯度计算，尽量填平流水线空泡的并行训练策略 |
| Reward Model（RM） | 对 prompt-response 打分的模型，常用于 PPO/GRPO 等后训练流程的 reward 计算 |
| TTFT（Time To First Token） | 从请求进入服务到返回第一个 token 的时间，主要受排队、prefill、路由和冷启动影响 |
| TPOT（Time Per Output Token） | 输出 token 平均生成间隔，常用于衡量 decode 吞吐和用户等待体验 |
| ITL（Inter-Token Latency） | 流式返回时相邻 token 到达用户侧或 flush 点之间的延迟，更敏感于尾部抖动 |
| Volcano | 面向批任务和 AI 训练的 Kubernetes 调度系统，提供队列、gang scheduling 等能力 |
| Kueue | Kubernetes 原生生态里的任务队列与准入控制组件，常用于批任务配额和 ResourceFlavor 管理 |
| Mermaid | Markdown 中常用的文本化图表语法，可渲染流程图、时序图、状态图和 mindmap |

---

## 使用建议

阅读正文时，如果遇到术语含义不清，优先回到本表查找大致定义；如果需要更深入理解，再回到对应章节阅读上下文。

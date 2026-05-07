# 附录A：AI Infra 术语表

本附录把正文中出现过的术语按"硬件 → CPU/IO → 网络 → 存储 → 训练 → 推理 → 平台 → 后训练 → 可观测"九个分组组织，便于在阅读时快速回查。每个分组只列与该子领域强相关的术语；交叉概念（如 NCCL 既属网络也属训练）按主用场景归类。

---

## A. 硬件与 GPU 体系

| 术语 | 简要解释 |
|------|----------|
| AI Infra（AI Infrastructure） | 承载 AI 数据、训练、评测、部署、推理、监控与治理的基础设施体系 |
| GPU（Graphics Processing Unit） | 适合高吞吐并行计算的加速设备，常用于训练和推理 |
| 显存（GPU Memory / VRAM） | GPU 上的高速内存，决定模型、batch、缓存能否放下 |
| MIG（Multi-Instance GPU） | NVIDIA 的硬件级 GPU 切分能力，可把一张卡分成多个隔离实例 |
| MPS（Multi-Process Service） | NVIDIA 的多进程共享机制，让多个进程复用同一 GPU 上下文 |
| Time-Slicing | 在平台层按时间片复用 GPU 的方式，隔离弱但门槛低 |
| SR-IOV（Single Root I/O Virtualization） | PCIe 设备虚拟化能力，把一个物理设备暴露成多个虚拟功能（VF）给虚拟机或容器使用；GPU / NIC 场景下常用于虚拟化隔离和资源切分，但性能、功能和安全边界取决于厂商实现 |
| GPUDirect RDMA | 让 NIC 直接读写 GPU 显存的数据路径，减少 CPU 内存中转和拷贝开销 |
| GDS（GPUDirect Storage） | 让 NVMe / 存储路径更直接地读写 GPU 显存的数据通路，减少 CPU bounce buffer 和主机内存拷贝，常用于高速数据加载、checkpoint 和权重加载 |
| PCIe Lane | PCIe 链路的基本通道，lane 数和代际共同决定 GPU、NIC、NVMe 与 CPU 之间的带宽上限 |
| ACS（Access Control Services） | PCIe 的访问控制能力，可影响 P2P、IOMMU 隔离和设备间事务路由；AI 服务器里 ACS/BIOS 配置不当可能让 GPUDirect 或 GPU-GPU P2P 走慢路径 |
| IOMMU（Input-Output Memory Management Unit） | 设备 DMA 的地址翻译和隔离单元，用于把设备可见 IOVA 映射到物理内存；能增强隔离，也可能影响 P2P / GPUDirect 路径，需要按平台配置验证 |
| DMA（Direct Memory Access） | 设备绕过 CPU 直接读写主存的机制，是高速网卡、NVMe、GPU 数据搬运的基础 |
| WGMMA（Warp Group Matrix Multiply-Accumulate） | NVIDIA Hopper 及后续架构中的 warp group 级矩阵乘累加指令族，让多个 warp 协作驱动 Tensor Core，是高性能 GEMM / attention kernel 的关键底层能力 |
| TMA（Tensor Memory Accelerator） | NVIDIA Hopper 及后续架构中的张量搬运硬件机制，可把多维 tile 在 global memory 与 shared memory 间异步搬运，降低复杂地址计算和拷贝开销 |
| NVLS（NVLink Switch System） | NVIDIA 节点或机柜级 NVLink 交换系统，把多张 GPU 连接成更大的 NVLink fabric；GB200 NVL72 等系统中用于形成 rack-scale NVLink domain |

---

## B. CPU / 内存 / IO 体系

| 术语 | 简要解释 |
|------|----------|
| CPU 流水线（Pipeline） | 把一条指令拆成取指、译码、执行、访存、写回等阶段，让多条指令在不同阶段重叠执行 |
| CPI（Cycles Per Instruction） | 平均每条指令消耗的 CPU cycle 数；真实 CPI 会被 cache miss、分支误预测、依赖和资源冲突抬高 |
| 乱序执行（Out-of-Order / OoO） | CPU 在不破坏程序语义的前提下，绕过暂时阻塞的指令，先执行已就绪指令以提高指令级并行 |
| Register Renaming | 通过物理寄存器重命名消除 WAR/WAW 等假依赖，让 OoO 更容易发掘并行度 |
| ROB（Reorder Buffer） | 重排序缓冲，记录乱序执行的中间结果并按程序顺序"引退"提交，是保证异常语义和精确中断的关键队列；ROB 容量与发射宽度共同决定 CPU 能挖掘的指令级并行上限，AI Infra 中分析推理 server 的 CPU 端预处理瓶颈时常配合 IPC、bad-speculation 一起看。 |
| 分支预测（Branch Prediction） | CPU 预测条件分支方向和目标地址，减少流水线等待；误预测会导致流水线刷新 |
| BTB（Branch Target Buffer） | 分支目标缓冲，缓存最近见过的分支指令的目标地址，让 CPU 在取指阶段就能跳到预测路径而不必等到执行阶段；BTB 容量不足或工作集过大（如解释器、tokenizer 跳转密集代码）会显著抬高 front-end stall。 |
| RAS（Return Address Stack） | 函数返回地址预测专用栈，CPU 在 `call` 时压栈、`ret` 时弹栈来预测返回地址；递归过深、long jump 或异常展开都会污染 RAS，造成 ret 误预测，AI Infra 中表现为大量 Python/C++ 调用栈下的 retiring 占比下降。 |
| HITM（HIT Modified） | Cache 一致性 snoop 状态之一，表示某个核心读取的 cache line 命中了另一个核心 L1/L2 中的 Modified 副本，需要从对方核心搬运而非走 LLC；`perf c2c` 中 HITM 计数高强烈暗示伪共享或跨核热点共享变量，是诊断多线程数据加载和锁竞争的关键指标。 |
| SIMD | 单指令多数据并行，一条指令同时处理多个数据元素，常见于 AVX、AVX-512、tokenizer 和 preprocessing 热点 |
| L1 / L2 / L3 Cache | CPU 多级缓存层次，越靠近核心容量越小、延迟越低；L3 通常跨核心或跨核心簇共享 |
| Cache Line | CPU cache 与内存之间传输的最小粒度，常见为 64B；不恰当布局会造成带宽浪费或伪共享 |
| Cache Associativity | 一个内存块可映射到多少个 cache way 的规则；关联度不足会导致冲突 miss |
| 冲突缺失（Conflict Miss） | 即使总容量够、也因 cache 关联度不足、多个热地址竞争同一组 set 而被强制驱逐产生的 miss；典型出现在 stride 为 2 的幂或 tile 大小恰好等于 cache way 步距时，AI Infra 中调 GEMM tile、tensor pad、stride 经常需要刻意打散来规避。 |
| MESI | CPU 缓存一致性协议中的 Modified、Exclusive、Shared、Invalid 四状态，用于协调多核对同一 cache line 的读写 |
| 伪共享（False Sharing） | 多个线程修改不同变量，但变量落在同一 cache line 上，导致无意义的一致性失效和性能下降 |
| UPI / Infinity Fabric | Intel UPI（Ultra Path Interconnect）与 AMD Infinity Fabric 分别是两家在多 socket 之间承载 cache 一致性流量、内存访问与 IO 路由的高速互联；其带宽与拓扑决定 NUMA 跨 socket 访问的延迟与吞吐，AI Infra 中 NCCL host-staging、CPU 端数据加载、PCIe peer-to-peer 拓扑都依赖它的容量。 |
| 虚拟内存（Virtual Memory） | 进程看到的地址空间抽象，由页表映射到物理内存，可支持隔离、mmap、换页和共享 |
| 页表（Page Table） | 记录虚拟页到物理页映射的数据结构，页表遍历开销通常由 TLB 缓解 |
| TLB（Translation Lookaside Buffer） | 缓存虚拟地址到物理地址翻译结果的硬件缓存；TLB miss 会增加内存访问延迟 |
| Page Cache | Linux 用内存缓存文件内容的机制，可显著加速重复读取，也会让存储 benchmark 产生误判 |
| Dirty Page | 已在内存中修改但尚未写回持久介质的页；过多脏页可能放大 checkpoint 尾延迟 |
| Huge Pages / THP | 使用更大页面减少页表和 TLB 压力；THP 自动化更方便，显式 HugeTLB 可控性更强 |
| NUMA | 多 socket 或多内存控制器机器上的非均匀内存访问架构，跨 NUMA 访问会增加延迟并降低带宽 |
| Pinned Memory（锁页内存） | 通过 `mlock`、`cudaHostAlloc` 等接口固定在物理内存中、不会被换出的页面；DMA 引擎可直接以物理地址访问而不必走 bounce buffer，是 `cudaMemcpyAsync`、RDMA 注册内存、NVMe DMA 的前提，AI Infra 训练 dataloader 与推理 KV cache 拷贝路径上常需要预先 pin 一块大缓冲区。 |
| Syscall | 用户态进入内核态请求 OS 服务的调用；高频 syscall 会带来上下文切换和内核路径开销 |
| `epoll` | Linux 事件通知机制，适合大量 socket 的 readiness-based IO 多路复用 |
| `io_uring` | Linux 异步 IO 接口，通过提交队列和完成队列降低 syscall 与上下文切换成本 |
| io_uring SQE / CQE | SQE（Submission Queue Entry）是用户态写入 io_uring 提交队列的请求项，描述本次 IO 操作；CQE（Completion Queue Entry）是内核完成后写入完成队列的结果项，含返回值与用户携带数据；通过批量提交一组 SQE 然后批量收割 CQE，可以把数千次 IO 压缩成一次 syscall 甚至零 syscall，常用于高并发存储与网络框架。 |
| libfabric | OpenFabrics Interface（OFI）的用户态高性能通信抽象层，把 verbs、UCX、tcp、shared memory 等底层 provider 统一成 endpoint/AV/CQ/MR 模型；MPI、NCCL、Mercury 等上层都可以用 libfabric 写一次代码跑在 InfiniBand、RoCE、EFA、Slingshot 等多种 fabric 上，AI Infra 中 AWS EFA 与多云互联场景常见。 |
| rendezvous protocol | 大消息 RDMA 三阶段握手协议：发送端先发一个小 RTS（Ready-To-Send）告知大小与地址，接收端准备好缓冲区后回 CTS（Clear-To-Send），最后发送端发起 RDMA WRITE 把数据直送过去；与 eager 协议相比避免了大消息的预拷贝和缓冲区耗尽，NCCL、MPI、UCX 在超过 eager 阈值的张量传输上都走 rendezvous。 |
| VFS（Virtual File System） | Linux 把不同文件系统统一成 inode、dentry、file 等抽象的内核层 |

---

## C. 网络与 RDMA

| 术语 | 简要解释 |
|------|----------|
| NCCL（NVIDIA Collective Communications Library） | 常用于 GPU 间通信的集合通信库 |
| All-reduce | 多个进程聚合数据并把结果分发给所有进程的通信操作 |
| Fat-tree Topology | 一类提供较均衡跨节点带宽的数据中心网络拓扑，常见于大规模训练集群 |
| Rail-optimized Topology | 让每个 GPU 或节点优先走固定 rail 的网络设计，用更低成本换可接受带宽 |
| RDMA Verbs | RDMA 编程接口抽象，包括 QP、CQ、WR、WC 等，用于提交和完成零拷贝网络操作 |
| HCA（Host Channel Adapter） | InfiniBand / RoCE 网卡在 RDMA 语境下的常用称呼，负责队列、DMA、RDMA verbs 和链路通信 |
| GID（Global Identifier） | RDMA / InfiniBand 地址标识，RoCE 中常用于选择源地址、VLAN / 子网和路由语义；GID index 配错会导致通信失败或走错网络 |
| QP（Queue Pair） | RDMA 通信端点，包含发送队列和接收队列 |
| CQ（Completion Queue） | RDMA 完成队列，用于报告 Work Request 的完成状态 |
| WR / WC | RDMA Work Request 表示提交的工作项，Work Completion 表示完成结果 |
| RoCE v2 | 在 UDP/IP 之上承载 RDMA 的以太网方案，需要更严格的拥塞和丢包控制 |
| PFC（Priority Flow Control） | 链路级反压机制，按 802.1Qbb 的 8 个优先级分别 PAUSE，让 RoCE 的 RDMA 流量可以"无损"通过以太网；配置不当会引发拥塞扩散与 PFC storm，AI Infra 部署 RoCE 集群时通常要把 PFC 与 ECN 联调，并隔离 RDMA 队列与普通 TCP 队列。 |
| ECN（Explicit Congestion Notification） | IP 层显式拥塞通知，交换机在队列即将拥塞时把包头的 ECN 位标记为 CE，由接收端通过 ACK/CNP 回送给发送端触发降速；RoCE 中常用 ECN + DCQCN 替代纯 PFC 做端到端拥塞控制，减少反压扩散。 |
| DSCP（Differentiated Services Code Point） | IP 头 ToS 字段中 6 bit 的流量分类码，用于在交换机/路由器上把流量映射到不同的队列与 PFC/ECN 策略；AI Infra 训练集群常给 RDMA、控制面、对象存储分配不同 DSCP，再在交换机做 priority/queue 映射，确保 RoCE 流量不被普通流量挤占。 |
| sm_priority（IB Subnet Manager Priority） | InfiniBand 子网管理器（SM）的选举优先级配置项，多台 SM 在同一 IB 子网中运行时数值越大越优先成为主 SM；AI Infra 中通常给硬件交换机 SM 给低值、给 OpenSM 节点设较高值或反之，明确主备避免脑裂或路由表抖动。 |
| NCCL_TOPO_FILE | NCCL 的拓扑提示文件环境变量，指向一份描述 GPU、NIC、PCIe Switch、NVLink 连接关系的 XML；当 NCCL 自动探测的拓扑与实际机型不符（如自研服务器、异构 NIC 排布）时，通过 NCCL_TOPO_FILE 注入正确拓扑可以让 ring/tree 算法选对路径，避免误走 QPI 或慢速 NIC。 |
| TCP CUBIC | Linux 常见 TCP 拥塞控制算法，按丢包和窗口增长调节发送速率 |
| BBR | 基于瓶颈带宽和 RTT 估计的 TCP 拥塞控制算法，常用于改善长距离或特定网络路径吞吐 |
| MTU | 单个链路层帧可承载的最大传输单元，MTU 不一致会造成分片、丢包或性能下降 |
| Jumbo Frame | 通常指 MTU 约 9000 的以太网大帧，可降低大流量传输的包数和 CPU/交换机处理开销 |
| RSS / RPS | 接收端多队列和软件分发机制，用于把网卡收包负载分摊到多个 CPU core |

---

## D. 存储与文件系统

| 术语 | 简要解释 |
|------|----------|
| inode | 文件系统中记录文件元数据和数据块位置的对象，不等同于文件名 |
| dentry | Linux VFS 中目录项缓存，用于把路径名解析到 inode |
| ext4 Journal | ext4 用日志记录元数据或数据提交顺序，提高崩溃恢复能力，但可能引入写放大和提交延迟 |
| XFS B+tree | XFS 用于管理 extent、空闲空间和目录等元数据的 B+tree 结构，适合大文件和并发写场景 |
| ZFS COW（Copy-on-Write） | ZFS 写新块再切换引用的写入语义，便于快照和校验，但会改变写放大与碎片特征 |
| ARC（Adaptive Replacement Cache） | ZFS 的自适应替换缓存，通过同时跟踪"最近一次访问"（MRU）与"频繁访问"（MFU）两条链以及它们各自的 ghost 链，自动在偏好新数据与偏好热数据之间调节缓存空间；相比 LRU 在混合工作负载下命中率更稳，AI Infra 中训练数据反复扫读 + 偶发大文件访问的场景非常受益。 |
| log-structured FS（日志结构文件系统） | 把所有写入按顺序追加到一个大日志段中、再用 GC 回收旧段的文件系统设计，如 LFS、F2FS、部分 SSD FTL；优点是把随机写转化为顺序写、吞吐高且对 SSD 友好，缺点是需要持续 GC、读路径与崩溃恢复更复杂；AI Infra 中 checkpoint shard、对象存储底层和 NVMe 设备内部都借鉴了这类结构。 |
| 写放大（Write Amplification） | 实际写入存储介质的字节数与用户原始写入字节数之比，公式上 WA = 实际写 / 用户写；SSD 的 GC、文件系统日志、RAID 5/6、LSM tree compaction 都会贡献写放大；AI Infra 中 checkpoint 频繁覆写、训练日志大量小写都会推高 WA，进而缩短 SSD 寿命并占用 IO 带宽。 |
| 纠删码（Erasure Coding） | 把 k 个数据块编码出 m 个校验块，任意丢失 ≤ m 块都可恢复的冗余方案，存储开销 (k+m)/k 远低于 3 副本；常见 Reed-Solomon、LRC，广泛用于对象存储（S3、Ceph、HDFS EC）；AI Infra 中冷模型权重、归档数据集、长 checkpoint 历史经常用 EC 而非副本来降低存储成本。 |
| `fsync` | 要求把文件相关脏数据刷到持久介质的系统调用，是 checkpoint 一致性语义的重要边界 |
| `O_DIRECT` | 尽量绕过 Page Cache 做直接 IO 的打开选项，可减少缓存污染，但带来对齐和吞吐约束 |
| Lustre MDS / OSS | Lustre 中 MDS 负责元数据，OSS/OST 负责对象数据存储；小文件常压 MDS，大文件吞吐看 stripe 和 OSS |
| Stripe | 并行文件系统把一个文件切分到多个存储目标上的布局策略，影响大文件吞吐和恢复行为 |

---

## E. 训练并行与显存

| 术语 | 简要解释 |
|------|----------|
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
| MFU（Model FLOPs Utilization） | 实际模型有效计算吞吐占理论峰值的比例，强调"算得值不值" |
| HFU（Hardware FLOPs Utilization） | 从硬件角度衡量总 FLOPs 利用率，通常比 MFU 更宽泛 |
| 梯度压缩（Gradient Compression） | 通过量化、稀疏化或低秩近似减少梯度同步通信量的技术 |
| PowerSGD | 用低秩矩阵近似梯度来降低 all-reduce 传输量的梯度压缩方法 |
| Interleaved Pipeline | 把每个流水线 stage 再切成多个 virtual stage，减少 pipeline bubble 的调度方式 |
| Zero Bubble Pipeline | 通过重排前向、反向和权重梯度计算，尽量填平流水线空泡的并行训练策略 |
| Straggler | 分布式训练里显著慢于其他 worker、拖慢整体同步节奏的慢节点 |
| Elastic Training | 允许训练过程中动态增减 worker，并保持作业继续推进的能力 |
| Spot Instance | 云上可被抢占的低价实例，适合能从 checkpoint 恢复的离线作业 |

---

## F. 推理与 serving

| 术语 | 简要解释 |
|------|----------|
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
| GGML | 面向轻量推理的张量与推理实现项目，常见于端侧 / CPU 推理生态 |
| GGUF | GGML / llama.cpp 生态常见的模型封装格式，便于本地与端侧分发 |
| TTFT（Time To First Token） | 从请求进入服务到返回第一个 token 的时间，主要受排队、prefill、路由和冷启动影响 |
| TPOT（Time Per Output Token） | 输出 token 平均生成间隔，常用于衡量 decode 吞吐和用户等待体验 |
| ITL（Inter-Token Latency） | 流式返回时相邻 token 到达用户侧或 flush 点之间的延迟，更敏感于尾部抖动 |

---

## G. 平台 / 调度 / 治理

| 术语 | 简要解释 |
|------|----------|
| RAG（Retrieval-Augmented Generation） | 检索增强生成，把外部知识检索结果引入模型上下文 |
| Embedding | 将文本、图片等对象映射为向量表示 |
| 向量索引（Vector Index） | 支持近似最近邻检索的数据结构或服务 |
| Volcano | 面向批任务和 AI 训练的 Kubernetes 调度系统，提供队列、gang scheduling 等能力 |
| Kueue | Kubernetes 原生生态里的任务队列与准入控制组件，常用于批任务配额和 ResourceFlavor 管理 |
| Gang Scheduling | 要求分布式训练的一组 Pod 同时获得资源后再启动，避免只启动部分 rank 占住 GPU 空转 |
| PodGroup | Volcano / Kueue 等调度系统用于表达一组必须一起调度的 Pod、最小副本数和队列属性的对象 |
| ResourceFlavor | Kueue 中描述资源来源或硬件类型的抽象，例如 H100、A100、MIG 分片、spot 或特定拓扑节点池 |
| ClusterQueue | Kueue 中跨 namespace 汇聚资源配额、准入、借用和抢占策略的集群级队列 |
| Borrow / Lend | 队列之间临时借用或出借未使用配额的策略，要求同时定义上限、归还和抢占语义 |
| Preemption | 为更高优先级或更符合配额策略的 workload 腾出资源而中止或驱逐已有 workload 的机制 |
| Topology-aware Scheduling | 调度时把 GPU/NIC/CPU socket/NVLink/PCIe switch 等拓扑关系纳入约束，降低跨 NUMA 或跨交换路径开销 |
| DRA（Dynamic Resource Allocation） | Kubernetes 动态资源分配机制，用结构化方式申请、分配和绑定 GPU、加速器或其他扩展资源 |
| CDI（Container Device Interface） | 容器运行时暴露设备的标准化描述接口，使 GPU/MIG/NIC 等设备注入更可审计、可移植 |
| DRF（Dominant Resource Fairness） | 面向多资源系统的公平分配思路，关注租户占用的"主导资源"比例 |
| Canary Release | 让新版本先接入少量真实流量，再逐步放量的发布方式 |
| Canary | 同 Canary Release，强调发布单元在小流量真实请求下接受 SLO、质量和成本门禁验证 |
| Shadow Traffic | 将真实请求复制给候选版本但不返回其结果，用于比较延迟、错误、成本和模型质量差异 |
| Rollback Target | 发布前确认可切回的模型、engine、镜像、路由、索引、缓存和配置组合 |
| Eval Gate | 发布状态机中的评测门禁，只有质量、安全、回归、成本或延迟阈值达标后才允许进入下一阶段 |
| Blue-Green Deployment | 准备两套独立环境，通过切流快速完成发布或回滚的部署方式 |
| 灰度发布（Progressive Delivery / Canary Rollout） | 让新模型或新服务先接收小比例流量，再逐步放量 |
| SLSA（Supply-chain Levels for Software Artifacts） | 用于提升软件供应链可追溯性和可信度的分级框架 |
| SBOM（Software Bill of Materials） | 软件物料清单，列出镜像、依赖、模型构建链路中的组件、版本和来源 |
| Provenance | 构建来源证明，记录产物由哪个源码、数据、参数、构建器和流水线步骤生成 |
| Attestation | 对构建、扫描、测试或发布事实的可验证声明，常与签名一起进入供应链证据 |
| Cosign | Sigstore 生态中的镜像或工件签名与验证工具，常用于校验镜像、SBOM 和 attestation |
| OPA / Gatekeeper | 基于 Open Policy Agent 的 Kubernetes 准入控制方案，用策略阻止不合规资源进入集群 |
| Kyverno | Kubernetes 原生策略引擎，可做准入校验、默认值注入、资源变更和策略报告 |
| Secret Encryption | 对 secret 在存储、传输和运行时挂载路径上的加密与权限控制，防止明文泄漏扩大影响 |
| Prompt Injection | 外部输入通过提示词内容诱导模型越权、泄露数据或错误调用工具的攻击方式 |
| Guardrails | 围绕模型输入、输出、工具调用和策略执行的约束层，用于降低越权、泄露和不安全响应风险 |
| Token-level Quota | 以 token、请求、GPU-second 或模型调用成本为粒度的租户限额和熔断策略 |
| 成本归因（Cost Attribution / Chargeback） | 将 GPU、存储、网络等资源成本归属到团队、项目、任务或模型 |
| Mermaid | Markdown 中常用的文本化图表语法，可渲染流程图、时序图、状态图和 mindmap |

---

## H. 后训练与 RLHF

| 术语 | 简要解释 |
|------|----------|
| RLHF（Reinforcement Learning from Human Feedback） | 基于人类反馈训练奖励或策略模型的后训练方法 |
| DPO（Direct Preference Optimization） | 直接利用偏好对做优化、避免在线强化学习环节的对齐方法 |
| PPO（Proximal Policy Optimization） | RLHF 中常见的策略梯度优化算法 |
| GRPO（Group Relative Policy Optimization） | 通过同组多个采样结果的相对奖励做优化、常见于去掉 critic 的后训练路线 |
| Reward Model（RM） | 对 prompt-response 打分的模型，常用于 PPO/GRPO 等后训练流程的 reward 计算 |

---

## I. 可观测性与可靠性

| 术语 | 简要解释 |
|------|----------|
| SLO（Service Level Objective） | 服务等级目标，用于定义可用性、延迟、错误率等目标 |
| SLI（Service Level Indicator） | 衡量服务状态的具体指标，例如 TTFT、TPOT、错误率、成功率、排队时长或恢复时间 |
| Error Budget | SLO 允许消耗的错误额度，用来约束发布速度、变更风险和故障后的冻结策略 |
| Burn Rate | 错误预算消耗速度，常用于判断是否需要告警、止血、暂停发布或降级 |
| Trace Sampling | 在请求 trace 中按比例、规则或异常条件采样，平衡排障可见性与存储成本 |
| High Cardinality | 指标标签取值过多导致存储、查询和告警成本急剧上升的现象，例如把 request id 当 label |
| Runbook | 面向值班和故障处理的操作手册，包含触发条件、判断路径、止血动作、retest 和升级联系人 |
| Postmortem | 事故后复盘文档，记录时间线、影响面、根因、修复、预防动作和平台规则沉淀 |

> 注：可观测性相关的更细分指标（TTFT / TPOT / ITL 等）按主用场景归在 §F 推理；CPU 端可观测性指标（CPI、HITM、bad-speculation）按主用场景归在 §B。

---

## J. 跨章节工程契约

| 术语 | 简要解释 |
|------|----------|
| EvidenceBundle | 一次诊断或发布判断所需的最小证据包，包含 symptom、scope、workload、version、evidence、hypothesis、action、retest 和 rollback。 |
| CapacityLedger | 跨训练、推理、RAG、平台和成本治理复用的容量账本，记录 workload shape、硬件、利用率、goodput、存储、网络、缓存、队列、成本和 headroom。 |
| ReleaseUnit | 一次可审计发布的最小单元，绑定模型、tokenizer、prompt、adapter、engine、image、router、index、cache、eval gate 和 rollback target。 |
| StateManifest | 描述数据集、checkpoint、模型版本、索引、缓存或 agent session 的状态清单，至少包含 immutable id、alias、lineage、schema version、owner、status、timestamp 和 validation result。 |
| RestoreLevel | 描述恢复语义的等级，包括 true resume、same-shape restore、reshard restore、model-only warm start、serving conversion 和 rollback。 |
| CacheKeyContract | 缓存复用必须满足的键空间约束，至少绑定 tenant、ACL/auth scope、model/version、prompt/template、index、tool schema、adapter/base 和 runtime 口径。 |
| TenantBudget | 租户级预算与降级策略对象，记录 token、GPU-second、cache、warm pool、storage、egress、queue priority 和 soft landing 动作。 |
| BenchmarkProtocol | 性能数字的复现协议，记录 hardware、software version、model、input distribution、warmup、cache state、command、metric definition、confidence window 和 counterfactual。 |

---

## 使用建议

阅读正文时，如果遇到术语含义不清，优先回到本表查找大致定义；如果需要更深入理解，再回到对应章节阅读上下文。建议按章节涉及的子领域定位到对应分组（A–I），而不是全表线性查找。

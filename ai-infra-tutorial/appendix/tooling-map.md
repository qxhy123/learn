# 附录B：工具生态地图

> 本附录只做类别速查，不把任何工具列为唯一推荐。实际选型应先看团队约束、规模、预算和维护能力。

## 训练与框架

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 深度学习框架 | PyTorch、JAX、TensorFlow | 定义模型、训练循环、自动求导 |
| 分布式训练 | PyTorch DDP、FSDP、DeepSpeed、Megatron-LM | 多卡 / 多机训练、状态切分、并行策略 |
| 加速库 | CUDA、cuDNN、NCCL、Triton | GPU 计算、通信与算子优化 |

## CPU 与主机侧性能分析

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| CPU 采样分析 | `perf record`、`perf report`、`perf top` | 找到 DataLoader、tokenizer、preprocessing 或服务网关里的 CPU 热点 |
| CPU 计数器统计 | `perf stat`、Intel VTune、AMD uProf | 观察 IPC/CPI、branch-misses、cache-misses、cycles、instructions 等硬件事件 |
| Cache / 伪共享分析 | `perf c2c`、`perf mem`、VTune Memory Access | 定位 cache line 竞争、NUMA 远端访问和 false sharing |
| 编译器向量化报告 | `-Rpass=loop-vectorize`、`-fopt-info-vec`、LLVM-MCA | 判断 SIMD 是否生效，以及循环为何未被自动向量化 |
| NUMA 观察与绑定 | `numactl`、`numastat`、`lscpu`、`hwloc-ls` | 检查 CPU core、内存、GPU、NIC 是否跨 NUMA 访问 |

## 内存、IO 与 PCIe 观察

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 内存状态 | `free`、`vmstat`、`sar -B`、`/proc/meminfo` | 观察 Page Cache、脏页、换页、page fault 和内存压力 |
| Huge Page / TLB | `/proc/meminfo`、`perf stat -e dTLB-load-misses`、`turbostat` | 判断 THP/HugeTLB 是否生效，以及 TLB miss 是否成为瓶颈 |
| PCIe 拓扑 | `lspci -tv`、`nvidia-smi topo -m`、`hwloc-ls` | 识别 GPU、NIC、NVMe 与 CPU socket 的连接关系 |
| GPU 拷贝链路 | `nvidia-smi dmon`、Nsight Systems、CUDA profiler | 验证 H2D/D2H、pinned memory、copy/compute overlap 是否符合预期 |

## 对齐训练与偏好优化

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| RLHF / DPO 训练框架 | TRL、OpenRLHF、DeepSpeed-Chat | 组织 SFT、奖励建模、PPO / DPO 等后训练流程 |
| 偏好数据与评审 | 自建 preference pipeline、W&B Tables | 管理偏好样本、judge 结果和实验对照 |

## 参数高效微调

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| Adapter 训练接口 | PEFT | 统一 LoRA、QLoRA、Adapter 等微调接口 |
| 高效 LoRA 训练 | Unsloth | 降低 LoRA 训练的显存和吞吐开销 |
| Multi-LoRA 服务 | LoRAX、vLLM Multi-LoRA | 让多个 adapter 共享同一 base model 服务 |

## 数据、实验与工件

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 数据版本 | DVC、LakeFS、对象存储版本策略 | 数据集追踪与版本管理 |
| 实验追踪 | MLflow、Weights & Biases、ClearML | 记录配置、指标、工件和运行上下文 |
| 模型仓库 | MLflow Model Registry、Hugging Face Hub、自建 registry | 管理模型版本、状态和元数据 |
| 特征管理 | Feast、自建 feature store | 训练和线上特征一致性 |

## 文件系统与存储压测

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 块设备与系统 IO | `iostat`、`pidstat -d`、`sar -d`、`blktrace` | 观察磁盘利用率、队列深度、await、吞吐和进程级 IO |
| 通用 IO 压测 | `fio`、`dd`、`diskspd` | 构造顺序/随机、读/写、direct/buffered、不同 block size 的基准 |
| 文件系统基准 | `fsbench`、`mdtest`、`ior` | 测试小文件元数据、大文件吞吐、并行文件系统 stripe 等场景 |
| 文件系统观察 | `df`、`du`、`stat`、`filefrag`、`xfs_info`、`zpool iostat` | 查看容量、extent、XFS/ZFS 状态、碎片和池级吞吐 |
| 对象存储压测 | `s5cmd`、`aws s3`、`rclone`、自建 multipart benchmark | 验证对象存储 list、get、put、multipart、并发和尾延迟 |
| 并行文件系统工具 | Lustre `lfs`、GPFS `mmlsfs` / `mmdiag`、BeeGFS `beegfs-ctl` | 查看 stripe、MDS/OSS 状态、客户端挂载和服务端健康 |

## 调度与平台

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 容器 | Docker、containerd | 封装运行环境 |
| 编排 | Kubernetes | 统一运行和调度容器工作负载 |
| 工作流 | Argo Workflows、Airflow、Kubeflow Pipelines | 编排训练、评测、数据处理流程 |
| 队列调度 | Volcano、Kueue、Slurm | 管理批任务、GPU 队列和资源公平性 |

## 推理与服务

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 通用模型服务 | KServe、BentoML、Triton Inference Server | 模型部署、服务化、扩缩容 |
| LLM Serving | vLLM、TensorRT-LLM、TGI | LLM 推理优化、batching、缓存管理 |
| API 网关 | Envoy、NGINX、Kong | 路由、限流、鉴权、灰度 |

## AI Gateway

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| Provider 聚合网关 | LiteLLM、Portkey | 统一多 LLM provider API、路由、fallback 和日志 |
| 策略编排层 | 自建 gateway policy、feature flag 平台 | 管理模型路由、配额、降级和不可变 prompt / bundle 版本选择，不应把 prompt 文本当成可热改 flag 值 |

## 模型压缩与量化

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 通用量化运行库 | bitsandbytes | 提供 8-bit / 4-bit 量化与低精度优化能力 |
| GPTQ 生态 | AutoGPTQ | 面向 GPTQ 路线的模型量化与加载 |
| AWQ 生态 | AutoAWQ | 面向 AWQ 路线的权重量化与推理集成 |

## 边缘 / 端侧推理

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 本地 LLM 推理 | llama.cpp、GGUF / GGML | 适合 CPU、本地设备和轻量边缘部署 |
| 移动端推理 | ONNX Runtime Mobile | 面向移动和嵌入式环境的推理运行时 |
| 端侧模型框架 | TensorFlow Lite | 面向手机、IoT 和端侧加速器的轻量推理框架 |

## 可观测性与治理

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 指标监控 | Prometheus、Grafana | 指标采集、dashboard、告警 |
| 日志 | Loki、Elasticsearch / OpenSearch | 日志检索与故障追溯 |
| Trace | OpenTelemetry、Jaeger | 跨服务链路追踪 |
| 成本与审计 | 云账单系统、自建 cost attribution、审计日志 | 成本归因、权限审计 |

## 网络、RDMA 与链路验收

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| TCP / socket 观察 | `ss`、`ip route`、`nstat`、`tcpdump`、`iperf3` | 排查连接状态、重传、路由、MTU、吞吐和包级异常 |
| 网卡配置 | `ethtool`、`ip link`、`tc` | 查看速率、offload、队列、RSS、MTU、ECN/PFC 相关配置 |
| RDMA 设备状态 | `ibstat`、`ibv_devinfo`、`rdma link`、`rdma resource` | 查看 HCA、端口、GID、QP/CQ 和 RDMA 资源 |
| RDMA 性能测试 | `ib_write_bw`、`ib_read_bw`、`ib_send_bw`、`perftest` | 验证 RDMA 带宽、延迟、消息大小和双向性能 |
| InfiniBand / RoCE 管理 | `iblinkinfo`、`ibnetdiscover`、`perfquery`、`mlxlink` | 检查链路错误、速率、拓扑、PFC/ECN、交换机端口状态 |
| NCCL 网络验证 | `nccl-tests`、`NCCL_DEBUG=INFO`、`NCCL_TOPO_DUMP_FILE` | 验证 AllReduce 性能、识别 socket fallback、查看 NCCL 拓扑选择 |

## 安全工具与合规

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 漏洞扫描 | Trivy | 扫描镜像、文件系统和依赖漏洞 |
| 镜像签名与校验 | cosign | 对镜像和制品做签名、验证和 provenance 关联 |
| Secret 管理 | Vault | 集中管理动态凭据、密钥和敏感配置 |

> 延伸阅读：如果你想把 OpenTelemetry 从“工具名称”继续学到“统一埋点模型、Collector 管道、跨信号关联与生产治理”，可阅读 [OpenTelemetry 教程](../../opentelemetry-tutorial/README.md)，尤其是 [OpenTelemetry Collector 基础](../../opentelemetry-tutorial/part5-collector-and-pipelines/13-opentelemetry-collector-basics.md) 与 [搭建一个观测栈](../../opentelemetry-tutorial/part8-advanced-and-capstone/24-build-an-observability-stack.md)。

## RAG 与向量基础设施

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 向量数据库 | Milvus、Weaviate、Qdrant、pgvector | 向量索引、检索与过滤 |
| 文档处理 | 自建 ETL、Unstructured、解析器集合 | 文档清洗、切分、元数据抽取 |
| 评测与反馈 | 自建评测集、人工审核、LLM-as-judge 工作流 | 检索与生成质量评估 |

## 文档、图表与 Mermaid 渲染

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| Mermaid 渲染 | Mermaid CLI、`@mermaid-js/mermaid-cli`、markdown preview | 把正文中的 flowchart、sequenceDiagram、stateDiagram、mindmap 渲染为图 |
| Markdown 静态站点 | MkDocs、Docusaurus、VitePress、自建转换脚本 | 把教程 markdown 组织为可浏览站点 |
| 图表回归检查 | Playwright screenshot、HTML build smoke test | 检查 mermaid 图是否渲染、是否溢出、是否在浅色主题下可读 |
| 代码块与链接检查 | markdownlint、lychee、自建 `rg` 检查 | 发现标题、链接、代码块 fence、相对路径和附录引用错误 |

## 版本线 / 关键里程碑速记

> 版本变化很快，这里不把某一天的精确版本号当成长期事实，而只给出“常见主线或关键里程碑”做直觉定位；真正选型前仍应回到官方 release 页面确认。

| 工具 | 当前更稳妥的理解方式 | 适合怎么理解 |
|------|----------------------|--------------|
| vLLM | `0.x` 快速迭代主线 | 适合重点关注 batching、KV cache、disaggregated serving 等运行时能力演进 |
| TensorRT-LLM | `1.x` 主线 | NVIDIA 生态里的高性能推理主线 |
| bitsandbytes | `0.x` 持续维护线 | 低精度训练 / 推理常见基础库 |
| AutoGPTQ | 历史生态里程碑 | 更适合做 GPTQ 路线理解，而不是默认唯一工具 |
| AutoAWQ | 仓库已归档（archived） | 更适合作为 AWQ 路线的里程碑参考，而不是长期唯一依赖 |
| llama.cpp | 常以构建编号持续演进 | 适合观察本地 / 端侧推理生态 |
| ONNX Runtime | `1.x` 主线 | 通用跨平台推理运行时主线 |
| TensorFlow / TensorFlow Lite | `v2.21.0` | TFLite 通常跟随 TensorFlow 主版本线演进 |
| Trivy | `v0.70.0` | 镜像与依赖扫描常见基线工具 |
| cosign | `v3.0.6` | 镜像签名与验证工具主线 |
| Vault | `v2.0.0` | Secret 管理主版本线，升级前应重点看兼容与运维变更 |

---

## 选型建议

1. 先明确约束：规模、延迟、成本、合规、团队维护能力
2. 先跑通最小闭环，再引入复杂平台
3. 优先选择能进入现有工作流的工具
4. 对核心路径保留可替换边界，避免过早绑定单一实现

const TUTORIAL = [
  {
    "part": "开始之前",
    "chapters": [
      {
        "id": "preface",
        "title": "前言：如何使用本教程",
        "path": "preface.html"
      }
    ]
  },
  {
    "part": "Part 0 · 体系结构基础",
    "chapters": [
      {
        "id": "0a",
        "title": "第 0a 章 · CPU 微架构总览",
        "path": "part0/0a-cpu-microarchitecture.html"
      },
      {
        "id": "0a1",
        "title": "第 0a-1 章 · 流水线（Pipeline）",
        "path": "part0/0a1-pipeline.html"
      },
      {
        "id": "0a2",
        "title": "第 0a-2 章 · 乱序执行、Register Renaming 与 ROB",
        "path": "part0/0a2-out-of-order-execution.html"
      },
      {
        "id": "0a3",
        "title": "第 0a-3 章 · 分支预测",
        "path": "part0/0a3-branch-prediction.html"
      },
      {
        "id": "0a4",
        "title": "第 0a-4 章 · SIMD：SSE、AVX、AVX-512",
        "path": "part0/0a4-simd.html"
      },
      {
        "id": "0a5",
        "title": "第 0a-5 章 · Cache 层级",
        "path": "part0/0a5-cache-hierarchy.html"
      },
      {
        "id": "0a6",
        "title": "第 0a-6 章 · MESI 一致性协议",
        "path": "part0/0a6-mesi-coherence.html"
      },
      {
        "id": "0a7",
        "title": "第 0a-7 章 · 伪共享（False Sharing）",
        "path": "part0/0a7-false-sharing.html"
      },
      {
        "id": "0a8",
        "title": "第 0a-8 章 · CPU 综合排障 Worked Example",
        "path": "part0/0a8-cpu-worked-example.html"
      },
      {
        "id": "0b",
        "title": "第 0b 章 · 内存、虚拟内存与 IO 导览",
        "path": "part0/0b-memory-virtual-memory-and-io.html"
      },
      {
        "id": "0b1",
        "title": "第 0b1 章 · 虚拟内存、页表、TLB 与 Page Fault",
        "path": "part0/0b1-virtual-memory-page-tables-and-tlb.html"
      },
      {
        "id": "0b2",
        "title": "第 0b2 章 · Page Cache、脏页回写与 Huge Pages",
        "path": "part0/0b2-page-cache-writeback-and-huge-pages.html"
      },
      {
        "id": "0b3",
        "title": "第 0b3 章 · NUMA、PCIe、DMA 与 Pinned Memory",
        "path": "part0/0b3-numa-pcie-dma-and-pinned-memory.html"
      },
      {
        "id": "0b4",
        "title": "第 0b4 章 · Syscall、Epoll、io_uring 与 IO 服务模型",
        "path": "part0/0b4-syscall-epoll-io-uring-and-service-io.html"
      },
      {
        "id": "0c",
        "title": "第 0c 章 · 文件系统与存储内核导览",
        "path": "part0/0c-filesystems-and-storage-internals.html"
      },
      {
        "id": "0c1",
        "title": "第 0c1 章 · VFS、inode/dentry 与 Block Layer",
        "path": "part0/0c1-vfs-inode-dentry-and-block-layer.html"
      },
      {
        "id": "0c2",
        "title": "第 0c2 章 · ext4、XFS、ZFS 与本地文件系统选择",
        "path": "part0/0c2-local-filesystems-ext4-xfs-zfs.html"
      },
      {
        "id": "0c3",
        "title": "第 0c3 章 · fsync、Direct IO 与 Checkpoint 语义",
        "path": "part0/0c3-storage-semantics-fsync-direct-io-and-checkpoints.html"
      },
      {
        "id": "0c4",
        "title": "第 0c4 章 · 对象存储、并行文件系统与 Dataset IO",
        "path": "part0/0c4-object-storage-parallel-filesystems-and-dataset-io.html"
      },
      {
        "id": "0d",
        "title": "第 0d 章 · 网络协议栈基础导览",
        "path": "part0/0d-network-stack-fundamentals.html"
      },
      {
        "id": "0d1",
        "title": "第 0d1 章 · Linux 网络栈、TCP 与 MTU",
        "path": "part0/0d1-linux-network-stack-tcp-and-mtu.html"
      },
      {
        "id": "0d2",
        "title": "第 0d2 章 · NIC Offload、队列与服务网络 IO",
        "path": "part0/0d2-nic-offload-queues-and-service-network-io.html"
      },
      {
        "id": "0d3",
        "title": "第 0d3 章 · RDMA、RoCE/IB 与 GPUDirect 导览",
        "path": "part0/0d3-rdma-roce-infiniband-and-gpudirect.html"
      },
      {
        "id": "0d3a",
        "title": "第 0d3a 章 · RDMA Verbs、内存注册与队列模型",
        "path": "part0/0d3a-rdma-verbs-memory-registration-and-queues.html"
      },
      {
        "id": "0d3b",
        "title": "第 0d3b 章 · RoCE/InfiniBand、无损网络与拥塞控制",
        "path": "part0/0d3b-roce-infiniband-lossless-fabric-and-congestion.html"
      },
      {
        "id": "0d3c",
        "title": "第 0d3c 章 · GPUDirect RDMA、GPU/NIC 拓扑与诊断",
        "path": "part0/0d3c-gpudirect-rdma-gpu-nic-topology-and-diagnostics.html"
      },
      {
        "id": "0d4",
        "title": "第 0d4 章 · NCCL Collective 与网络诊断",
        "path": "part0/0d4-nccl-collectives-and-network-diagnostics.html"
      }
    ]
  },
  {
    "part": "Part 1 · 基础概念",
    "chapters": [
      {
        "id": "01",
        "title": "第 1 章 · 什么是 AI Infra",
        "path": "part1/01-what-is-ai-infra.html"
      },
      {
        "id": "02",
        "title": "第 2 章 · 算力、存储与网络",
        "path": "part1/02-compute-storage-network.html"
      },
      {
        "id": "03",
        "title": "第 3 章 · 从模型实验到生产系统",
        "path": "part1/03-from-model-to-production.html"
      }
    ]
  },
  {
    "part": "Part 2 · 系统栈",
    "chapters": [
      {
        "id": "04",
        "title": "第 4 章 · GPU 与加速器导览",
        "path": "part2/04-gpu-and-accelerators.html"
      },
      {
        "id": "04a",
        "title": "第 4a 章 · GPU 执行模型与 Tensor Core",
        "path": "part2/04a-gpu-execution-model-and-tensor-cores.html"
      },
      {
        "id": "04b",
        "title": "第 4b 章 · HBM、显存预算与 Roofline",
        "path": "part2/04b-hbm-memory-and-roofline.html"
      },
      {
        "id": "04c",
        "title": "第 4c 章 · GPU 互联与系统形态",
        "path": "part2/04c-gpu-interconnect-and-systems.html"
      },
      {
        "id": "04d",
        "title": "第 4d 章 · GPU 选型、虚拟化与异构加速器",
        "path": "part2/04d-gpu-selection-virtualization-and-heterogeneous-accelerators.html"
      },
      {
        "id": "05",
        "title": "第 5 章 · 内存、互联与 IO 导览",
        "path": "part2/05-memory-interconnect-io.html"
      },
      {
        "id": "05a",
        "title": "第 5a 章 · 内存与存储层级、数据驻留",
        "path": "part2/05a-memory-storage-hierarchy-and-data-residency.html"
      },
      {
        "id": "05b",
        "title": "第 5b 章 · Host-Device IO、PCIe、NUMA 与重叠",
        "path": "part2/05b-host-device-io-pcie-numa-and-overlap.html"
      },
      {
        "id": "05c",
        "title": "第 5c 章 · RDMA、Collective 与集群拓扑",
        "path": "part2/05c-rdma-collectives-and-cluster-topology.html"
      },
      {
        "id": "05d",
        "title": "第 5d 章 · 训练存储、Checkpoint 与 IO 诊断",
        "path": "part2/05d-training-storage-checkpoint-and-io-diagnostics.html"
      },
      {
        "id": "06",
        "title": "第 6 章 · CUDA、运行时与算子执行导览",
        "path": "part2/06-cuda-runtime-and-kernels.html"
      },
      {
        "id": "06a",
        "title": "第 6a 章 · Framework Dispatch、Runtime 与 Kernel Launch",
        "path": "part2/06a-framework-dispatch-runtime-and-kernel-launch.html"
      },
      {
        "id": "06b",
        "title": "第 6b 章 · Stream、同步与 CUDA Graph",
        "path": "part2/06b-streams-synchronization-and-cuda-graphs.html"
      },
      {
        "id": "06c",
        "title": "第 6c 章 · 算子库、融合与 SM 资源边界",
        "path": "part2/06c-kernel-libraries-fusion-and-sm-resource-limits.html"
      },
      {
        "id": "06d",
        "title": "第 6d 章 · Profiling、Debugging 与性能排障 SOP",
        "path": "part2/06d-profiling-debugging-and-performance-sop.html"
      }
    ]
  },
  {
    "part": "Part 3 · 训练基础设施",
    "chapters": [
      {
        "id": "07",
        "title": "第 7 章 · 单机训练系统",
        "path": "part3/07-single-node-training.html"
      },
      {
        "id": "08",
        "title": "第 8 章 · 数据并行",
        "path": "part3/08-data-parallel.html"
      },
      {
        "id": "09",
        "title": "第 9 章 · 模型并行与流水并行",
        "path": "part3/09-model-pipeline-parallel.html"
      },
      {
        "id": "09e",
        "title": "第 09e 章 · MoE 训练基础设施",
        "path": "part3/09e-moe-training-infrastructure.html"
      },
      {
        "id": "10",
        "title": "第 10 章 · 内存优化、检查点与恢复",
        "path": "part3/10-memory-checkpointing-and-recovery.html"
      },
      {
        "id": "10b",
        "title": "第 10b 章 · 对齐训练与后训练基础设施",
        "path": "part3/10b-alignment-and-post-training.html"
      },
      {
        "id": "10c",
        "title": "第 10c 章 · Fine-Tuning 基础设施与多 Adapter 服务",
        "path": "part3/10c-finetuning-and-multi-adapter.html"
      }
    ]
  },
  {
    "part": "Part 4 · 数据与存储",
    "chapters": [
      {
        "id": "11",
        "title": "第 11 章 · 数据管道总览",
        "path": "part4/11-data-pipeline.html"
      },
      {
        "id": "11a",
        "title": "第 11a 章 · 数据采集与摄入",
        "path": "part4/11a-data-ingestion.html"
      },
      {
        "id": "11b",
        "title": "第 11b 章 · 数据清洗、去重与质量治理",
        "path": "part4/11b-data-cleaning-dedup-quality.html"
      },
      {
        "id": "11c",
        "title": "第 11c 章 · Tokenization、切分与训练 Dataset 格式",
        "path": "part4/11c-tokenization-and-dataset-formats.html"
      },
      {
        "id": "11d",
        "title": "第 11d 章 · 流式读取与 DataLoader 工程化",
        "path": "part4/11d-streaming-and-dataloader-engineering.html"
      },
      {
        "id": "11e",
        "title": "第 11e 章 · 数据版本、血缘与谱系",
        "path": "part4/11e-data-versioning-and-lineage.html"
      },
      {
        "id": "11f",
        "title": "第 11f 章 · 数据飞轮与持续学习闭环",
        "path": "part4/11f-data-flywheel-online-learning.html"
      },
      {
        "id": "12",
        "title": "第 12 章 · 制品、模型与检查点管理总览",
        "path": "part4/12-artifacts-and-checkpoints.html"
      },
      {
        "id": "12a",
        "title": "第 12a 章 · Model Registry 体系",
        "path": "part4/12a-model-registry.html"
      },
      {
        "id": "12b",
        "title": "第 12b 章 · Checkpoint 工程化",
        "path": "part4/12b-checkpoint-engineering.html"
      },
      {
        "id": "12c",
        "title": "第 12c 章 · 制品版本治理与发布门禁",
        "path": "part4/12c-release-governance.html"
      },
      {
        "id": "12d",
        "title": "第 12d 章 · 制品供应链与签名",
        "path": "part4/12d-supply-chain-and-signing.html"
      },
      {
        "id": "13",
        "title": "第 13 章 · 特征、向量与缓存总览",
        "path": "part4/13-feature-vector-and-cache.html"
      },
      {
        "id": "13a",
        "title": "第 13a 章 · Feature Store 体系",
        "path": "part4/13a-feature-store.html"
      },
      {
        "id": "13b",
        "title": "第 13b 章 · 向量索引算法",
        "path": "part4/13b-vector-index-algorithms.html"
      },
      {
        "id": "13c",
        "title": "第 13c 章 · 向量数据库选型与运维",
        "path": "part4/13c-vector-db-selection-and-operations.html"
      },
      {
        "id": "13d",
        "title": "第 13d 章 · RAG 工程化",
        "path": "part4/13d-rag-engineering.html"
      },
      {
        "id": "13e",
        "title": "第 13e 章 · Embedding 工程与缓存层",
        "path": "part4/13e-embedding-and-cache-layer.html"
      }
    ]
  },
  {
    "part": "Part 5 · 推理基础设施",
    "chapters": [
      {
        "id": "14",
        "title": "第 14 章 · 在线推理架构",
        "path": "part5/14-online-inference-architecture.html"
      },
      {
        "id": "15",
        "title": "第 15 章 · 批处理、调度与 KV Cache",
        "path": "part5/15-batching-scheduling-and-kv-cache.html"
      },
      {
        "id": "16",
        "title": "第 16 章 · 量化、编译与推理引擎",
        "path": "part5/16-quantization-compilation-and-engines.html"
      },
      {
        "id": "16a",
        "title": "第 16a 章 · vLLM 内部机制深入",
        "path": "part5/16a-vllm-internals.html"
      },
      {
        "id": "16b",
        "title": "第 16b 章 · SGLang 内部机制深入",
        "path": "part5/16b-sglang-internals.html"
      },
      {
        "id": "16c",
        "title": "第 16c 章 · TensorRT-LLM 内部机制深入",
        "path": "part5/16c-trt-llm-internals.html"
      },
      {
        "id": "17",
        "title": "第 17 章 · 多租户与成本治理",
        "path": "part5/17-multitenancy-and-cost.html"
      }
    ]
  },
  {
    "part": "Part 6 · 平台与编排",
    "chapters": [
      {
        "id": "18",
        "title": "第 18 章 · 容器与运行时",
        "path": "part6/18-containers-and-runtime.html"
      },
      {
        "id": "18a",
        "title": "第 18a 章 · AI 镜像与 CUDA 兼容矩阵",
        "path": "part6/18a-ai-images-and-cuda-compatibility.html"
      },
      {
        "id": "18b",
        "title": "第 18b 章 · 容器运行时与设备注入",
        "path": "part6/18b-container-runtime-and-device-injection.html"
      },
      {
        "id": "18c",
        "title": "第 18c 章 · 制品供应链与镜像治理",
        "path": "part6/18c-artifact-supply-chain-and-image-governance.html"
      },
      {
        "id": "18d",
        "title": "第 18d 章 · 运行时故障排除",
        "path": "part6/18d-runtime-troubleshooting.html"
      },
      {
        "id": "19",
        "title": "第 19 章 · Kubernetes for AI",
        "path": "part6/19-kubernetes-for-ai.html"
      },
      {
        "id": "19a",
        "title": "第 19a 章 · AI 工作负载对象建模",
        "path": "part6/19a-kubernetes-ai-workloads.html"
      },
      {
        "id": "19b",
        "title": "第 19b 章 · GPU 调度与拓扑感知",
        "path": "part6/19b-gpu-scheduling-and-topology.html"
      },
      {
        "id": "19c",
        "title": "第 19c 章 · AI CRD 与 Operator",
        "path": "part6/19c-ai-crd-and-operators.html"
      },
      {
        "id": "19d",
        "title": "第 19d 章 · Kubernetes AI 排障 SOP",
        "path": "part6/19d-kubernetes-ai-troubleshooting.html"
      },
      {
        "id": "20",
        "title": "第 20 章 · 队列、配额与自动扩缩容",
        "path": "part6/20-queues-quotas-and-autoscaling.html"
      },
      {
        "id": "20a",
        "title": "第 20a 章 · 队列、配额、优先级与公平调度",
        "path": "part6/20a-queues-quotas-priority-and-fairness.html"
      },
      {
        "id": "20b",
        "title": "第 20b 章 · GPU 资源切分与共享",
        "path": "part6/20b-gpu-partitioning-and-sharing.html"
      },
      {
        "id": "20c",
        "title": "第 20c 章 · 推理 Autoscaling",
        "path": "part6/20c-inference-autoscaling.html"
      },
      {
        "id": "20d",
        "title": "第 20d 章 · 容量与排障 SOP",
        "path": "part6/20d-capacity-and-troubleshooting-sop.html"
      }
    ]
  },
  {
    "part": "Part 7 · 可靠性与安全",
    "chapters": [
      {
        "id": "21",
        "title": "第 21 章 · 可观测性与容量规划",
        "path": "part7/21-observability-and-capacity.html"
      },
      {
        "id": "22",
        "title": "第 22 章 · 评测、发布与故障处理",
        "path": "part7/22-evaluation-release-and-incident.html"
      },
      {
        "id": "23",
        "title": "第 23 章 · 安全、隔离与治理",
        "path": "part7/23-security-isolation-and-governance.html"
      }
    ]
  },
  {
    "part": "Part 8 · 高级主题与 Capstone",
    "chapters": [
      {
        "id": "24",
        "title": "第 24 章 · 构建一个 AI 平台",
        "path": "part8/24-build-an-ai-platform.html"
      },
      {
        "id": "25",
        "title": "第 25 章 · AI Agent 与推理时计算基础设施",
        "path": "part8/25-agent-and-inference-time-compute.html"
      }
    ]
  },
  {
    "part": "附录",
    "chapters": [
      {
        "id": "glossary",
        "title": "附录 A · AI Infra 术语表",
        "path": "appendix/glossary.html"
      },
      {
        "id": "tooling-map",
        "title": "附录 B · 工具生态地图",
        "path": "appendix/tooling-map.html"
      },
      {
        "id": "checklists",
        "title": "附录 C · 上线与排障检查清单",
        "path": "appendix/checklists.html"
      },
      {
        "id": "answers",
        "title": "附录 D · 练习题详细参考解答",
        "path": "appendix/answers.html"
      }
    ]
  }
];

if (typeof window !== 'undefined') window.TUTORIAL = TUTORIAL;
if (typeof module !== 'undefined') module.exports = TUTORIAL;

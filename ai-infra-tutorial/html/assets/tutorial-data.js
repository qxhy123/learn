const TUTORIAL = [
  {
    "part": "Part 0 · 体系结构基础",
    "chapters": [
      {
        "id": "0a",
        "title": "第 0a 章 · CPU 微架构",
        "path": "part0/0a-cpu-microarchitecture.html"
      },
      {
        "id": "0b",
        "title": "第 0b 章 · 内存、虚拟内存与 IO",
        "path": "part0/0b-memory-virtual-memory-and-io.html"
      },
      {
        "id": "0c",
        "title": "第 0c 章 · 文件系统与存储内核",
        "path": "part0/0c-filesystems-and-storage-internals.html"
      },
      {
        "id": "0d",
        "title": "第 0d 章 · 网络协议栈基础",
        "path": "part0/0d-network-stack-fundamentals.html"
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
        "title": "第 4 章 · GPU 与加速器",
        "path": "part2/04-gpu-and-accelerators.html"
      },
      {
        "id": "05",
        "title": "第 5 章 · 内存、互联与 IO",
        "path": "part2/05-memory-interconnect-io.html"
      },
      {
        "id": "06",
        "title": "第 6 章 · CUDA、运行时与算子执行",
        "path": "part2/06-cuda-runtime-and-kernels.html"
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
        "title": "第 11 章 · 数据管道",
        "path": "part4/11-data-pipeline.html"
      },
      {
        "id": "12",
        "title": "第 12 章 · 制品、模型与检查点管理",
        "path": "part4/12-artifacts-and-checkpoints.html"
      },
      {
        "id": "13",
        "title": "第 13 章 · 特征、向量与缓存",
        "path": "part4/13-feature-vector-and-cache.html"
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
        "id": "19",
        "title": "第 19 章 · Kubernetes for AI",
        "path": "part6/19-kubernetes-for-ai.html"
      },
      {
        "id": "20",
        "title": "第 20 章 · 队列、配额与自动扩缩容",
        "path": "part6/20-queues-quotas-and-autoscaling.html"
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

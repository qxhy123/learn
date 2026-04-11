# 附录B：工具生态地图

> 本附录只做类别速查，不把任何工具列为唯一推荐。实际选型应先看团队约束、规模、预算和维护能力。

## 训练与框架

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 深度学习框架 | PyTorch、JAX、TensorFlow | 定义模型、训练循环、自动求导 |
| 分布式训练 | PyTorch DDP、FSDP、DeepSpeed、Megatron-LM | 多卡 / 多机训练、状态切分、并行策略 |
| 加速库 | CUDA、cuDNN、NCCL、Triton | GPU 计算、通信与算子优化 |

## 数据、实验与工件

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 数据版本 | DVC、LakeFS、对象存储版本策略 | 数据集追踪与版本管理 |
| 实验追踪 | MLflow、Weights & Biases、ClearML | 记录配置、指标、工件和运行上下文 |
| 模型仓库 | MLflow Model Registry、Hugging Face Hub、自建 registry | 管理模型版本、状态和元数据 |
| 特征管理 | Feast、自建 feature store | 训练和线上特征一致性 |

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

## 可观测性与治理

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 指标监控 | Prometheus、Grafana | 指标采集、dashboard、告警 |
| 日志 | Loki、Elasticsearch / OpenSearch | 日志检索与故障追溯 |
| Trace | OpenTelemetry、Jaeger | 跨服务链路追踪 |
| 成本与审计 | 云账单系统、自建 cost attribution、审计日志 | 成本归因、权限审计 |

> 延伸阅读：如果你想把 OpenTelemetry 从“工具名称”继续学到“统一埋点模型、Collector 管道、跨信号关联与生产治理”，可阅读 [OpenTelemetry 教程](../../opentelemetry-tutorial/README.md)，尤其是 [OpenTelemetry Collector 基础](../../opentelemetry-tutorial/part5-collector-and-pipelines/13-opentelemetry-collector-basics.md) 与 [搭建一个观测栈](../../opentelemetry-tutorial/part8-advanced-and-capstone/24-build-an-observability-stack.md)。

## RAG 与向量基础设施

| 类别 | 常见示例 | 主要作用 |
|------|----------|----------|
| 向量数据库 | Milvus、Weaviate、Qdrant、pgvector | 向量索引、检索与过滤 |
| 文档处理 | 自建 ETL、Unstructured、解析器集合 | 文档清洗、切分、元数据抽取 |
| 评测与反馈 | 自建评测集、人工审核、LLM-as-judge 工作流 | 检索与生成质量评估 |

---

## 选型建议

1. 先明确约束：规模、延迟、成本、合规、团队维护能力
2. 先跑通最小闭环，再引入复杂平台
3. 优先选择能进入现有工作流的工具
4. 对核心路径保留可替换边界，避免过早绑定单一实现


# 从零到高阶的 vLLM 教程

## 项目简介

本教程旨在为学习者提供一套系统、完整的 vLLM 学习路径，从大语言模型推理的基本挑战出发，逐步覆盖 vLLM 的核心机制（PagedAttention、连续批处理、调度器）、模型管理与量化、高级推理特性（投机解码、前缀缓存、结构化输出）、分布式推理、生产部署，以及源码架构与扩展开发。

**本教程的独特之处**：每章都包含可运行的 Python 代码示例和「动手实验」部分，不仅讲"怎么用"，更讲"为什么这样设计"——从内存管理的物理直觉到调度策略的工程权衡，帮助读者建立对高性能 LLM 推理系统的系统性理解。

---

## 目标受众

- 希望部署大语言模型推理服务的工程师
- 已有 PyTorch / Transformers 经验，想理解高性能推理引擎底层原理的开发者
- 需要在生产环境中优化 LLM 服务吞吐、延迟和成本的平台工程师
- 对 LLM 推理系统设计（内存管理、调度、并行策略）感兴趣的研究者
- 希望从"会调 API"进阶到"理解引擎内部"的 AI 应用开发者

---

## 章节导航目录

### 开始之前

- [前言：如何使用本教程](./00-preface.md)

### 第一部分：LLM 推理基础

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第1章 | [LLM 推理的挑战](./part1-inference-fundamentals/01-llm-inference-challenges.md) | 自回归生成、计算与内存瓶颈、延迟vs吞吐 | 分析一次生成请求的时间分布 |
| 第2章 | [KV Cache 原理](./part1-inference-fundamentals/02-kv-cache.md) | 注意力计算复用、KV Cache 结构、显存占用分析 | 手算 KV Cache 显存并与实际对照 |
| 第3章 | [vLLM 概览与定位](./part1-inference-fundamentals/03-vllm-overview.md) | vLLM 设计目标、与其他引擎对比、架构总览 | 运行第一个 vLLM 推理并对比 HF 基线 |

### 第二部分：快速上手

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第4章 | [安装与环境配置](./part2-quick-start/04-installation.md) | pip/conda/Docker 安装、GPU 驱动、常见问题排查 | 验证安装并运行健康检查 |
| 第5章 | [离线批量推理](./part2-quick-start/05-offline-inference.md) | LLM 类、SamplingParams、批量生成、输出解析 | 用不同参数生成并比较结果 |
| 第6章 | [OpenAI 兼容服务器](./part2-quick-start/06-openai-compatible-server.md) | API 服务器启动、Chat/Completions 端点、流式输出 | 用 curl 和 OpenAI SDK 调用 vLLM 服务 |

### 第三部分：核心机制

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第7章 | [PagedAttention](./part3-core-mechanisms/07-paged-attention.md) | 虚拟内存类比、物理块与逻辑块、显存碎片消除 | 可视化块分配与显存利用率 |
| 第8章 | [连续批处理](./part3-core-mechanisms/08-continuous-batching.md) | 静态vs动态批处理、iteration-level调度、吞吐提升 | 对比静态批处理与连续批处理的吞吐 |
| 第9章 | [调度器与请求管理](./part3-core-mechanisms/09-scheduler.md) | 调度策略、抢占机制、等待队列、公平性 | 观察高并发下的调度行为 |
| 第10章 | [内存管理与块引擎](./part3-core-mechanisms/10-memory-management.md) | BlockSpaceManager、块分配/释放、Copy-on-Write | 跟踪请求生命周期中的块变化 |

### 第四部分：模型与量化

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第11章 | [模型加载与支持列表](./part4-models-and-quantization/11-model-loading.md) | 支持的模型架构、权重格式、信任远程代码 | 加载不同架构的模型并验证 |
| 第12章 | [量化推理](./part4-models-and-quantization/12-quantization.md) | GPTQ、AWQ、SqueezeLLM、FP8、量化原理与精度权衡 | 比较量化前后的显存、速度与质量 |
| 第13章 | [LoRA 多适配器服务](./part4-models-and-quantization/13-lora-serving.md) | 动态 LoRA 加载、多适配器并发、显存开销 | 同时服务多个 LoRA 适配器 |
| 第14章 | [多模态模型推理](./part4-models-and-quantization/14-multimodal.md) | 视觉语言模型、图像/视频输入、多模态 API | 用 vLLM 运行 VLM 并处理图文输入 |

### 第五部分：高级推理特性

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第15章 | [投机解码](./part5-advanced-inference/15-speculative-decoding.md) | 草稿模型、验证机制、加速原理、配置方法 | 测量投机解码对延迟的改善 |
| 第16章 | [前缀缓存与 Prompt 复用](./part5-advanced-inference/16-prefix-caching.md) | 自动前缀缓存、hash 匹配、适用场景 | 对比有无前缀缓存时的 TTFT |
| 第17章 | [结构化输出](./part5-advanced-inference/17-structured-output.md) | JSON Schema 约束、正则引导、guided decoding 原理 | 生成严格符合 Schema 的 JSON |
| 第18章 | [采样参数与解码策略](./part5-advanced-inference/18-sampling-and-decoding.md) | temperature、top-p/top-k、beam search、最佳实践 | 不同策略对生成质量与多样性的影响 |

### 第六部分：分布式推理

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第19章 | [张量并行](./part6-distributed-inference/19-tensor-parallelism.md) | 张量切分策略、通信开销、GPU 间同步 | 用多 GPU 启动张量并行推理 |
| 第20章 | [流水线并行](./part6-distributed-inference/20-pipeline-parallelism.md) | 层间切分、micro-batch、气泡开销 | 对比张量并行与流水线并行 |
| 第21章 | [多节点分布式部署](./part6-distributed-inference/21-multi-node-deployment.md) | Ray 集群、跨节点通信、网络配置 | 搭建双节点 vLLM 推理集群 |

### 第七部分：生产部署

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第22章 | [性能调优](./part7-production-deployment/22-performance-tuning.md) | GPU 利用率、批大小、并发、引擎参数调优 | 系统化压测与瓶颈定位 |
| 第23章 | [监控与可观测性](./part7-production-deployment/23-monitoring.md) | Prometheus 指标、日志、请求追踪、告警 | 搭建 Grafana 监控面板 |
| 第24章 | [容器化与 Kubernetes 部署](./part7-production-deployment/24-containerization-k8s.md) | Docker 镜像、K8s Deployment、HPA、GPU 调度 | 编写 K8s 部署清单并上线 |
| 第25章 | [多模型服务与路由](./part7-production-deployment/25-multi-model-routing.md) | 模型路由、A/B 测试、版本管理、网关集成 | 构建多模型服务网关 |

### 第八部分：架构深入与扩展

| 章节 | 标题 | 主要内容 | 实验重点 |
|------|------|----------|----------|
| 第26章 | [vLLM 源码架构](./part8-architecture-and-extensions/26-source-architecture.md) | 模块划分、请求生命周期、执行引擎、Worker | 阅读关键路径源码并绘制调用链 |
| 第27章 | [自定义模型接入](./part8-architecture-and-extensions/27-custom-model-integration.md) | 模型注册、自定义层、注意力后端适配 | 将一个自定义模型接入 vLLM |
| 第28章 | [前沿进展与生态](./part8-architecture-and-extensions/28-frontier-and-ecosystem.md) | Disaggregated Prefill、chunked prefill、FlashInfer、SGLang 对比 | 跟踪社区 Roadmap 与关键 PR |

### 附录

| 附录 | 标题 | 内容说明 |
|------|------|----------|
| 附录A | [vLLM CLI 与 API 速查](./appendix/api-reference.md) | 启动参数、SamplingParams、API 端点一览 |
| 附录B | [性能调优速查表](./appendix/performance-cheatsheet.md) | 常见场景的推荐配置与调优 checklist |
| 附录C | [练习答案汇总](./appendix/answers.md) | 各章练习题的要点与常见误区 |

---

## 学习路径建议

### 路径一：零基础入门

适合第一次接触 LLM 推理引擎的学习者：

1. 按顺序学习第 1-6 章，建立推理基础并跑通第一个服务
2. 学习第 7-8 章，理解 vLLM 的核心创新
3. 学习第 12 章和第 18 章，掌握量化部署与采样调参
4. 按需学习第 22 章，了解性能调优基本方法

### 路径二：AI 应用开发者

适合已有 LLM 应用开发经验，需要部署自己的推理服务的开发者：

1. 快速阅读第 1-3 章，理解推理引擎的设计动机
2. 重点学习第 4-6 章，掌握 vLLM 的使用方式
3. 学习第 11-14 章，掌握模型管理、量化和多模态
4. 学习第 15-17 章，掌握投机解码、前缀缓存和结构化输出
5. 按需阅读第 22-25 章，进入生产部署

### 路径三：推理系统工程师

适合需要深入理解和优化 LLM 推理系统的工程师：

1. 系统学习第 1-3 章，建立推理性能的理论框架
2. 重点学习第 7-10 章，深入理解 PagedAttention、调度器和内存管理
3. 学习第 15-16 章，理解投机解码和前缀缓存的系统设计
4. 深入学习第 19-21 章，掌握分布式推理
5. 学习第 22-23 章，掌握性能调优与监控
6. 研读第 26-28 章，进入源码级理解

### 路径四：平台与运维工程师

适合负责 LLM 推理平台建设和运维的工程师：

1. 快速阅读第 1-3 章和第 5-6 章
2. 学习第 12 章，理解量化对资源的影响
3. 重点学习第 19-25 章，系统掌握分布式部署、性能调优、监控和 K8s 部署
4. 参考附录 B 建立运维手册

---

## 前置要求

学习本教程建议具备以下基础：

- **必需**：Python 编程基础，能读懂类、装饰器和异步代码
- **必需**：对大语言模型有基本了解（知道 Transformer、token、生成过程）
- **推荐**：PyTorch 基础（张量操作、模型加载）
- **推荐**：Linux 命令行基本操作
- **可选**：Docker / Kubernetes 经验（学习第 24 章时会更轻松）
- **可选**：分布式系统基础知识（学习第 19-21 章时会更轻松）

本教程默认**不要求**你有 vLLM 使用经验；前三章会补齐必要的推理背景知识。

---

## 环境配置

本教程的示例以 vLLM 官方发布版为主，推荐环境如下：

```bash
# 推荐环境
NVIDIA GPU（Compute Capability >= 7.0，即 V100 及以上）
CUDA Toolkit >= 12.1
Python >= 3.9
vLLM >= 0.6.0
PyTorch >= 2.4
```

快速安装：

```bash
# 创建虚拟环境
python -m venv vllm-env
source vllm-env/bin/activate

# 安装 vLLM（会自动安装 PyTorch 和相关依赖）
pip install vllm

# 验证安装
python -c "import vllm; print(vllm.__version__)"

# 可选：安装开发和测试工具
pip install openai httpx rich tqdm
```

Docker 快速启动：

```bash
# 使用官方镜像
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct
```

重要说明：

- vLLM **仅支持 NVIDIA GPU**（CUDA），部分实验性支持 AMD ROCm
- 建议至少 **16GB 显存**（运行 7B 模型），**24GB+ 显存**可获得更好体验
- 如果没有本地 GPU，可使用云 GPU（如 AWS、GCP、Lambda Labs）
- 部分章节涉及多 GPU，需要至少 2 块 GPU

---

## 如何使用本教程

1. **先跑通再深入**：每章的动手实验至少亲手运行一遍
2. **理解设计动机**：不仅要知道"怎么用"，更要理解"为什么这样设计"
3. **用数据说话**：性能优化要用 benchmark 和 profiler，不要凭直觉猜测
4. **对比实验**：比如量化前后、有无前缀缓存、不同并行策略
5. **读源码**：对关键机制，教程会指出源码位置，鼓励你直接阅读

---

## 教程特色

- **28 章完整内容**：从推理基础到源码级架构分析
- **实验驱动学习**：每章都有可运行的代码和动手实验
- **深度与广度兼顾**：既讲使用方法，也讲底层原理和设计权衡
- **生产导向**：覆盖量化、分布式、监控、K8s 部署等工程主题
- **前沿跟踪**：涵盖投机解码、Disaggregated Prefill 等最新技术
- **中文编写**：术语统一，强调直觉解释和工程经验

---

## 与仓库其他教程的关系

本教程与本仓库其他系列教程形成互补关系：

- 如果你不熟悉 Transformer 架构和注意力机制，建议先阅读 [Transformer 教程](../transformer-tutorial/)
- 如果你想了解 CUDA 底层编程和 GPU 执行模型，可参考 [CUDA 教程](../cuda-tutorial/)
- 如果你对 AI 基础设施全景（训练、数据管道、平台）感兴趣，可参考 [AI Infra 教程](../ai-infra-tutorial/)
- 如果你需要 Python 异步编程基础（理解 vLLM 的异步 API），可参考 [Python 教程的 asyncio 部分](../python-tutorial/part9-asyncio/)

---

## 许可证

本项目采用 MIT 许可证开源。

---

*如有建议或发现错误，欢迎提交 Issue 或 Pull Request。*

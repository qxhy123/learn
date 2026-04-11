# 从零到高阶的 OpenTelemetry 教程

## 项目简介

本教程旨在为学习者提供一套系统、完整、厂商中立的 OpenTelemetry 学习路径，从可观测性的基本直觉出发，逐步覆盖 traces、metrics、logs、上下文传播、语义规范、API 与 SDK 区别、Collector 管道、生产部署、成本治理，以及与后端工程和平台可观测性密切相关的实践问题。

**本教程的独特之处**：它不把 OpenTelemetry 写成某个具体后端或某家厂商的接入手册，而是尽量围绕“为什么这样设计”“什么时候该这样接”“哪些边界最容易踩坑”来组织内容；同时在合适的位置使用 TypeScript / Node.js 与 OpenTelemetry Collector YAML 代码片段，帮助你把概念落实到真实工程场景中。

---

## 本教程的定位

本教程在仓库中的定位介于几类内容之间：

- 它不像 `typescript-tutorial/` 那样重点讲语言和后端开发基础
- 它不像 `ai-infra-tutorial/` 那样从更高层讨论整个平台的稳定性、容量与治理
- 它也不像某些产品文档那样只告诉你把哪个 exporter 指向哪个地址

它更像一张 **“把应用、Collector、后端与生产可观测性连起来的总装图”**：

- 往下，能连接请求链路、HTTP 客户端、数据库、队列、Kubernetes 和运行时环境
- 往上，能连接 SLI/SLO、告警、排障、成本、采样和数据治理
- 横向，能连接 traces、metrics、logs 三类信号，以及它们在一个分布式系统中如何协同工作

因此，本教程特别强调三种能力：

1. **信号直觉**：知道 traces、metrics、logs 各自解决什么问题，以及为什么要互相关联
2. **管道直觉**：知道应用、SDK、Collector、OTLP、backend 分别负责哪一层
3. **生产直觉**：知道什么时候该自动注入、什么时候该手动埋点、什么时候该控制 cardinality 和成本

---

## 目标受众

- 希望系统理解分布式追踪、指标和日志统一接入方式的后端工程师
- 想把“会看面板”进阶到“会设计可观测性体系”的平台工程师
- 对 OpenTelemetry、Collector、语义规范、上下文传播感兴趣的学生与研究者
- 需要在微服务、消息队列、Kubernetes 环境中建设可观测体系的技术负责人
- 已经听说过 OpenTelemetry，但还没有建立完整心智模型的学习者

如果你已经接触过 tracing、metrics、structured logging，但总觉得它们像几块彼此独立的组件，而不是一套协调工作的系统，那么这套教程就是为你写的。

---

## 章节导航目录

### 开始之前

- [前言：如何使用本教程](./00-preface.md)

### 第一部分：OpenTelemetry 基础认知

| 章节 | 标题 | 主要内容 | 实践重点 |
|------|------|----------|----------|
| 第1章 | [为什么需要 OpenTelemetry](./part1-foundations/01-why-opentelemetry.md) | 可观测性痛点、厂商中立、三种信号协同 | 建立整体问题意识 |
| 第2章 | [三种信号与 OTel 架构](./part1-foundations/02-signals-and-architecture.md) | traces、metrics、logs、API/SDK/Collector/backend、OTLP | 建立管道心智模型 |
| 第3章 | [第一个端到端链路](./part1-foundations/03-first-pipeline-app-collector-backend.md) | 应用到 Collector 再到后端的最小闭环 | 看懂最小数据流 |

### 第二部分：Trace 与上下文传播

| 章节 | 标题 | 主要内容 | 实践重点 |
|------|------|----------|----------|
| 第4章 | [Trace、Span 与 Context](./part2-traces-and-propagation/04-traces-spans-and-context.md) | trace/span、事件、状态、父子关系、span kind | 学会读懂一条 trace |
| 第5章 | [上下文传播与 Baggage](./part2-traces-and-propagation/05-context-propagation-and-baggage.md) | W3C Trace Context、Baggage、跨 HTTP/RPC/异步传播 | 看懂 trace 为什么会断 |
| 第6章 | [Span 设计与链路建模](./part2-traces-and-propagation/06-span-design-and-trace-modeling.md) | span 边界、错误记录、链路层级与采样直觉 | 设计可解释的 trace |

### 第三部分：Metrics、Logs 与语义规范

| 章节 | 标题 | 主要内容 | 实践重点 |
|------|------|----------|----------|
| 第7章 | [Metrics 模型与常见 Instrument](./part3-metrics-logs-and-semantics/07-metrics-model-and-instruments.md) | Counter、Gauge、Histogram、同步/异步 instrument | 选对指标类型 |
| 第8章 | [Logs 与跨信号关联](./part3-metrics-logs-and-semantics/08-logs-and-cross-signal-correlation.md) | 结构化日志、trace/log correlation、日志边界 | 让日志真正可关联 |
| 第9章 | [Resources、Attributes 与 Semantic Conventions](./part3-metrics-logs-and-semantics/09-resources-attributes-and-semantic-conventions.md) | 资源身份、属性、语义规范、cardinality | 设计稳定的 telemetry schema |

### 第四部分：Instrumentation 与 SDK 使用

| 章节 | 标题 | 主要内容 | 实践重点 |
|------|------|----------|----------|
| 第10章 | [API、SDK 与 Library Guidelines](./part4-instrumentation-and-sdks/10-api-vs-sdk-and-library-guidelines.md) | 应用 vs 库、API/SDK 边界、依赖方式 | 避免把 telemetry 绑死 |
| 第11章 | [自动注入、手动埋点与混合策略](./part4-instrumentation-and-sdks/11-auto-instrumentation-and-manual-instrumentation.md) | zero-code、框架集成、手动 instrumentation | 决定该怎么接入 |
| 第12章 | [用 Node.js / TypeScript 接入 OTel](./part4-instrumentation-and-sdks/12-nodejs-otel-instrumentation-in-practice.md) | HTTP 服务、客户端、数据库、中间件接入 | 做出最小可运行方案 |

### 第五部分：Collector 与 Telemetry Pipeline

| 章节 | 标题 | 主要内容 | 实践重点 |
|------|------|----------|----------|
| 第13章 | [OpenTelemetry Collector 基础](./part5-collector-and-pipelines/13-opentelemetry-collector-basics.md) | receiver、processor、exporter、agent/gateway | 看懂 Collector 的职责 |
| 第14章 | [Processor、Sampling 与 Routing](./part5-collector-and-pipelines/14-processors-sampling-and-routing.md) | batch、filter、transform、tail sampling、多路导出 | 设计合理 pipeline |
| 第15章 | [在生产环境部署 Collector](./part5-collector-and-pipelines/15-deploying-collector-in-production.md) | 本地、容器、Kubernetes、sidecar/daemonset/gateway | 选择部署拓扑 |

### 第六部分：生产环境运维与治理

| 章节 | 标题 | 主要内容 | 实践重点 |
|------|------|----------|----------|
| 第16章 | [Cardinality、成本与性能](./part6-production-operations/16-cardinality-cost-and-performance.md) | 属性高基数、采样、开销、成本控制 | 控制 telemetry 成本 |
| 第17章 | [安全、隐私与治理](./part6-production-operations/17-security-privacy-and-governance.md) | PII、敏感字段、保留策略、租户治理 | 降低观测数据风险 |
| 第18章 | [调试断裂链路与缺失信号](./part6-production-operations/18-debugging-broken-telemetry.md) | trace 断裂、指标缺失、日志无法关联、时钟偏移 | 建立系统化排障能力 |

### 第七部分：生态、异步系统与平台整合

| 章节 | 标题 | 主要内容 | 实践重点 |
|------|------|----------|----------|
| 第19章 | [Exporter、Backend 与 OTLP 策略](./part7-ecosystem-and-platforms/19-exporters-backends-and-otlp.md) | OTLP、backend 选择、直连与 Collector 中转 | 做出厂商中立方案 |
| 第20章 | [消息队列、批处理与异步系统](./part7-ecosystem-and-platforms/20-messaging-batch-and-async-systems.md) | queue、job、cron、link 与 parent-child | 处理非同步请求链路 |
| 第21章 | [Kubernetes 与平台级可观测性](./part7-ecosystem-and-platforms/21-kubernetes-and-platform-observability.md) | Operator、资源检测、sidecar、平台统一接入 | 理解平台侧接入路径 |

### 第八部分：高级设计与完整项目

| 章节 | 标题 | 主要内容 | 实践重点 |
|------|------|----------|----------|
| 第22章 | [为微服务系统设计可观测性](./part8-advanced-and-capstone/22-observability-design-for-microservices.md) | span 设计、服务边界、指标体系、schema 约束 | 做出可维护的可观测设计 |
| 第23章 | [从旧监控/日志体系迁移到 OpenTelemetry](./part8-advanced-and-capstone/23-migrating-and-rolling-out-opentelemetry.md) | 增量接入、双写、灰度、回滚与验收 | 规划迁移路径 |
| 第24章 | [搭建一个观测栈](./part8-advanced-and-capstone/24-build-an-observability-stack.md) | 应用、Collector、backend 的端到端蓝图 | 形成完整系统视角 |

### 附录

| 附录 | 标题 | 内容说明 |
|------|------|----------|
| 附录A | [术语表](./appendix/glossary.md) | Trace、Span、Resource、OTLP、Baggage、Cardinality 等核心术语 |
| 附录B | [Collector Pipeline 速查表](./appendix/collector-pipeline-cheatsheet.md) | 常见 receiver / processor / exporter 组合模板 |
| 附录C | [Semantic Conventions 速查](./appendix/semantic-conventions-quick-reference.md) | HTTP、DB、messaging、异常、服务身份常见字段 |
| 附录D | [练习答案汇总](./appendix/answers.md) | 各章练习题的提示、要点与常见误区 |

---

## 学习路径建议

### 路径一：后端工程师快速入门

适合已经会写服务，但对 OpenTelemetry 还没有系统认知的学习者：

1. 学习第 1-6 章，建立 traces、propagation 和 span 设计基础
2. 学习第 7-9 章，补齐 metrics、logs 与语义规范
3. 学习第 11-15 章，理解 instrumentation 与 Collector 接入
4. 最后按需阅读第 16-24 章进入生产与平台话题

### 路径二：平台与可观测性工程导向

适合需要建设统一 telemetry 管道和生产治理体系的工程师：

1. 学习第 1-3 章，建立 OpenTelemetry 全景图
2. 重点学习第 9-18 章，掌握 schema、Collector、部署、成本和排障
3. 补充第 19-24 章，进入 backend 策略、平台整合与迁移设计

### 路径三：Node.js / TypeScript 实战导向

适合已经在 Node.js 服务里工作，希望快速建立 OTel 实操能力的学习者：

1. 学习第 1-5 章，理解最核心的 trace 与 propagation 模型
2. 重点学习第 10-15 章，掌握 API/SDK、自动/手动埋点与 Collector
3. 用第 12 章作为主实践入口，再回头补第 7-9 章和第 16-18 章

### 路径四：完整体系学习

适合希望从概念、设计到生产实践全部走一遍的学习者：

1. 按章节顺序完整学习
2. 每章完成 OTel实验 与练习题
3. 最后以第 24 章为蓝图，画出自己系统的 telemetry 架构图

---

## 前置要求

学习本教程建议具备以下基础：

- **必需**：基本后端编程经验，能读懂 HTTP 服务、日志和配置文件
- **推荐**：理解微服务、数据库、消息队列等分布式系统常见组件
- **推荐**：会阅读 YAML、JSON 和基本 TypeScript / JavaScript 代码
- **可选**：有 Prometheus、Jaeger、Grafana 或云厂商观测产品使用经验

如果你希望补齐相关背景，可以配合本仓库的以下教程一起学习：

- [TypeScript 教程](../typescript-tutorial/README.md)
- [AI Infra 教程](../ai-infra-tutorial/README.md)
- [计算机网络教程](../computer-network-tutorial/README.md)

---

## 如何使用本教程

1. **先建立模型，再抄配置**：先搞清 API、SDK、Collector、backend 的职责边界，再看 YAML 和代码
2. **先保证信号正确，再讨论数据量**：先确认 trace 连得起来、metrics 含义正确、logs 可关联
3. **把 schema 当成接口设计**：属性名、resource 字段、语义规范不是细节，而是长期兼容性问题
4. **把 Collector 当成管道系统**：不要把它只理解成“转发器”
5. **多做对照实验**：例如自动注入 vs 手动埋点、直连 backend vs 经 Collector、低 cardinality vs 高 cardinality

---

## 教程特色

- **厂商中立**：重点讲 OpenTelemetry 的概念、规范与工程边界，而非绑定某家后端
- **三种信号统一视角**：不是只讲 traces，也覆盖 metrics、logs 及其关联
- **生产导向**：强调成本、采样、cardinality、安全、治理和排障
- **平台连接能力**：覆盖 Collector、Kubernetes、异步系统和迁移策略
- **与仓库其他教程互补**：不重复写后端基础，而是把 observability 系统性讲透

---

## 与仓库其他教程的关系

本教程与本仓库其他系列教程形成互补关系：

- 如果你还不熟悉后端服务和 Node.js 工程结构，可先阅读 [TypeScript 教程中的完整项目章节](../typescript-tutorial/part8-fullstack/24-complete-project.md)
- 如果你希望从更高层理解可观测性在平台稳定性中的位置，可结合 [AI Infra 教程中的可观测性与容量规划](../ai-infra-tutorial/part7-reliability-security/21-observability-and-capacity.md)
- 如果你想理解 trace 穿过代理、TLS 与服务边界时发生了什么，可配合 [计算机网络教程](../computer-network-tutorial/README.md) 一起学习

---

## 许可证

本项目采用 MIT 许可证开源。

---

*如有建议或发现错误，欢迎提交 Issue 或 Pull Request。*

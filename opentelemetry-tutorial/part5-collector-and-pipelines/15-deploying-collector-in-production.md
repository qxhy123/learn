# 第15章：在生产环境部署 Collector

## 学习目标

- 理解本地开发、单机容器、Kubernetes 等不同环境下部署 Collector 的目标差异
- 掌握 sidecar、daemonset、gateway 三种常见部署拓扑分别解决什么问题
- 学会判断什么时候适合单层部署，什么时候适合 agent + gateway 分层部署
- 理解生产部署中高可用、资源限制、配置管理与安全出口的基本要求
- 建立“Collector 既是数据通道，也是平台组件”的部署直觉

## 1. 部署 Collector，先回答“它在这个环境里负责什么”

很多团队一谈 Collector 部署，马上开始选 YAML、镜像和 Helm 参数，但真正先要搞清楚的是：**你希望这层 Collector 在当前环境里承担什么职责。**

在不同环境里，Collector 的职责并不一样：

- 在本地开发环境里，它更像一个可见的数据调试点
- 在单机容器环境里，它更像应用与远端 backend 之间的近端出口
- 在 Kubernetes 里，它往往既承担节点或 Pod 邻近采集，也承担平台统一治理
- 在大型组织中，它常常被拆成接入层 agent 和集中层 gateway

因此，部署方式不是“Collector 的安装细节”，而是对可观测性架构的具体表达。

一个很实用的判断问题是：

1. 应用应该发给谁，才能最稳定地导出 telemetry？
2. 哪些处理要靠近应用做，哪些处理要集中做？
3. 节点、容器、Kubernetes 元数据由谁补？
4. 谁负责出口认证、多后端导出与组织级治理？

这些问题的答案，最终会决定你是用单实例、sidecar、daemonset、gateway，还是它们的组合。

## 2. 本地开发与单机容器：目标是最小闭环，而不是过早复制生产复杂度

### 2.1 本地开发：先让数据看得见、流得通

在开发机上部署 Collector，主要不是为了“模拟完整生产系统”，而是为了建立最小闭环。

一个典型目标是：

- 本地 Node.js 服务通过 OTLP 发数据
- 本地 Collector 接收并打印到 `debug` exporter
- 必要时再转发到远端测试后端

示意配置如下：

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  batch:
    timeout: 2s

exporters:
  debug:
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug]
```

这类本地 Collector 的价值在于：

- 帮你判断问题是出在应用埋点，还是出在远端后端
- 让你能直接观察 span、resource、attributes 是否符合预期
- 让本地开发先依赖 OTLP，而不是依赖具体厂商 endpoint

### 2.2 单机容器：让应用不要直接依赖远端 backend

在 Docker Compose 或单机容器编排中，Collector 很适合作为应用的同环境邻近出口。这样做有两个明显好处：

- 应用只需要知道 `collector:4317` 之类的内部地址
- 远端 backend 地址、TLS、认证、重试等细节由 Collector 负责

这种模式特别适合：

- 本地联调
- 测试环境
- 小型服务部署
- 迁移期双写或出口切换

### 2.3 一个 Node.js 应用指向近端 Collector 的示意

```ts
import { NodeSDK } from '@opentelemetry/sdk-node'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-grpc'

const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: 'http://collector:4317',
  }),
})

sdk.start()
```

这里最重要的不是某个包名，而是部署边界：应用只依赖 OTLP 和近端 Collector，而不是把后端产品细节直接写入业务服务。

## 3. 进入生产后，核心选择通常是 sidecar、daemonset、gateway

在生产环境里，Collector 部署方式本质上是在选择 telemetry 管道拓扑。最常见的三种模式分别是 sidecar、daemonset、gateway。

### 3.1 Sidecar：最贴近单个 Pod 或服务实例

sidecar 指的是把 Collector 作为同一个 Pod 内的伴生容器，与应用实例一起部署。

它的特点是：

- 离应用最近，网络路径最短
- 配置可以针对单个工作负载定制
- 很适合每个 Pod 都需要独立处理能力的场景

适合场景：

- 某些强隔离需求的服务
- 需要为单个工作负载提供特殊 exporter、认证或本地处理
- 对网络路径和租户隔离非常敏感的场景

代价也很明显：

- 每个 Pod 都多一个 Collector 容器，资源成本上升
- 升级面扩大
- 平台统一治理更复杂
- 对大规模集群而言，sidecar 数量可能非常多

所以 sidecar 不是“更专业的默认方案”，而是适合隔离性强、定制性强的特殊场景。

### 3.2 DaemonSet：每个节点一个近端 agent

DaemonSet 模式是在 Kubernetes 每个节点上部署一个 Collector 实例，它通常扮演 agent 角色。

优势包括：

- 每个节点只需一个 Collector，资源成本通常低于 sidecar 全覆盖
- 节点上的多个 Pod 可以共享本地 Collector
- 很适合采集主机指标、容器日志、节点级元数据
- 应用统一发往节点本地地址，接入体验较一致

这也是为什么很多平台团队把 DaemonSet 看作默认的接入层形态。它往往能同时支持：

- OTLP 接收来自应用的 traces、metrics、logs
- 主机/节点指标采集
- 容器日志接入
- 近端缓冲与轻量批处理

但它也有边界：

- 对需要完整 trace 视图的 tail sampling 并不理想
- 对复杂多租户路由、组织级出口控制并不总是方便
- 节点间分散部署意味着重治理能力会被重复执行

### 3.3 Gateway：集中治理与统一出口

gateway 模式通常把 Collector 部署成集群内共享服务，多个应用或多个 agent 把数据汇聚到这里，再由它统一导出。

它最适合承担：

- 统一出口认证
- 多 backend 导出
- 集中脱敏与 transform
- tail sampling
- 租户级路由与流量分级

gateway 的价值在于“集中治理”，但它并不适合承担一切：

- 如果应用直接跨网络连远端 gateway，路径更长，局部抖动更明显
- 如果 gateway 同时承担所有近端吸收与集中治理，扩容压力会更大

因此，很多成熟部署会采用两层拓扑：

- 近端 DaemonSet 或 sidecar 作为 agent
- 中央 Deployment 形式的 Collector 作为 gateway

## 4. 怎么在 sidecar、daemonset、gateway 之间做选择

没有一种拓扑适合所有系统。更重要的是理解它们各自优化的目标。

### 4.1 一个简化对照表

| 模式 | 更贴近谁 | 更适合做什么 | 主要代价 |
|------|----------|--------------|----------|
| sidecar | 单个 Pod / 单个服务 | 强隔离、特定服务定制、本地特殊处理 | 实例数量多、资源成本高 |
| daemonset | 节点 | 统一接入、节点指标、容器日志、近端 agent | 全局治理能力有限 |
| gateway | 集群共享入口 | tail sampling、统一出口、路由、脱敏、多后端导出 | 需要高可用与扩容设计 |

### 4.2 一些实用判断题

如果你更关心以下问题，通常更偏向某种模式：

- “每个 Pod 都要有自己独立的认证或出口策略” -> 更偏 sidecar
- “我想让所有节点上的应用都先发到本地 Collector” -> 更偏 daemonset
- “我想集中做 tail sampling 和多租户 routing” -> 更偏 gateway

### 4.3 为什么很多生产环境是组合式，而不是单选题

因为现实中的需求往往同时存在：

- 应用需要近端低抖动接入
- 平台需要集中治理
- 节点层又有自己的主机和容器信号

这就使得“agent + gateway”成为非常常见的组合：

1. 应用先把数据发到 sidecar 或 daemonset
2. 近端 Collector 做轻量处理，如 `memory_limiter`、少量 `batch`、节点资源补充
3. 再转发到 gateway
4. gateway 做 tail sampling、routing、fan-out 和统一出口

这种分层让职责更清晰：

- 接入层负责近端稳定性
- 治理层负责组织级策略

## 5. Kubernetes 中的生产直觉：Collector 不是“加一个 Pod”这么简单

Kubernetes 里的 Collector 部署，真正难的不是 YAML 能不能跑起来，而是如何与平台元数据、调度边界和资源治理结合。

### 5.1 节点、Pod、命名空间元数据很重要

在 Kubernetes 中，很多排障和路由都依赖平台上下文，例如：

- `k8s.namespace.name`
- `k8s.pod.name`
- `k8s.node.name`
- `k8s.container.name`
- `deployment.environment.name`

这些字段不是为了“好看”，而是决定了：

- 你能否快速定位异常 Pod
- 你能否按命名空间或工作负载治理数据
- 你能否把平台层与应用层问题区分开

### 5.2 资源限制、队列和 backpressure 要提前考虑

Collector 本身就是生产路径中的组件，所以必须给它明确的资源边界：

- CPU / memory requests 与 limits
- exporter 队列与重试配置
- HPA 或其他扩缩容机制
- PodDisruptionBudget 与滚动升级策略

否则一个很常见的失败模式是：

- 平台希望 Collector 做所有治理
- 但没有给足资源与扩容机制
- 最终 Collector 自己成了瓶颈

### 5.3 生产里要观测 Collector 自己

必须持续关注 Collector 自身的：

- CPU、内存
- 接收吞吐
- exporter 失败率
- 队列积压
- dropped spans / logs / datapoints

否则你只是把系统中的一个关键中间层当成黑箱来运行。

## 6. 生产部署的几个常见原则

### 6.1 应用尽量只指向 OTLP 与近端地址

一个健康的做法是让应用只知道：

- 使用 OTLP
- 发给近端或组织规定的 Collector 地址

不要让每个服务都分别维护：

- 不同 backend 的地址
- 各自的认证配置
- 多重 exporter 双写逻辑

### 6.2 重处理尽量集中，轻处理尽量就近

通常更推荐：

- 就近做 `memory_limiter`、少量 `batch`、必要的资源补充
- 集中做 tail sampling、复杂 transform、routing、fan-out

这样既减少节点侧重复成本，也使策略更容易统一管理。

### 6.3 优先选择可滚动升级、可灰度的方式

不管是容器环境还是 Kubernetes，Collector 都不应该成为“只能整站切换”的组件。更好的做法是：

- 新老配置并行验证
- 先对部分工作负载灰度
- 在 `debug` 或测试后端验证后再切主出口

### 6.4 配置要版本化，变更要可回滚

Collector 配置本质上也是生产代码。它应该具备：

- 版本管理
- 审核流程
- 变更记录
- 快速回滚能力

否则一次看似无害的 filter、sampling 或 routing 修改，就可能让整类 telemetry 消失。

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| 本地部署 | 重点是建立最小闭环，验证应用到 Collector 的数据流 |
| 单机容器 | 适合作为应用邻近出口，屏蔽远端 backend 细节 |
| sidecar | 最贴近单个服务，隔离性强，但资源和运维成本高 |
| daemonset | 适合作为 Kubernetes 节点级 agent，统一接入和平台采集 |
| gateway | 适合作为集中治理层，承担 tail sampling、routing 与统一出口 |
| 常见生产形态 | 不是单选，而是 agent + gateway 分层 |
| 关键意识 | Collector 本身也是生产组件，必须有资源、扩容、监控与回滚设计 |

## OTel实验

### 实验目标

通过同一套 Node.js 服务，分别体验本地单实例、容器邻近 Collector 和 Kubernetes 式两层拓扑的差异，建立部署直觉。

### 实验步骤

1. 本地先运行一个最小 Collector，让 Node.js 服务把 traces 发到 `localhost:4317`，用 `debug` exporter 确认数据路径。
2. 把同一服务和 Collector 放进容器编排环境，让应用改为发到容器网络内的 `collector:4317`，体验“应用只依赖近端 OTLP 地址”的好处。
3. 假设进入 Kubernetes：
   - 第一层使用 daemonset 作为节点 agent，负责接收应用流量与补充节点相关资源信息
   - 第二层使用 gateway，负责 tail sampling 和统一出口
4. 比较三种阶段中应用配置、Collector 配置复杂度和平台职责的变化。
5. 思考如果把 gateway 的能力全部塞回每个节点 agent，会带来哪些资源与运维代价。

### 预期现象

- 本地单实例最适合理解基础数据流
- 容器邻近 Collector 有助于隔离应用与远端 backend 的耦合
- Kubernetes 两层拓扑最能体现“接入层”和“治理层”的分工

## 练习题

1. 为什么说部署 Collector 前，先要回答“它在当前环境里承担什么职责”？
2. sidecar、daemonset、gateway 三种模式分别更适合解决什么问题？它们最大的代价又是什么？
3. 为什么 tail sampling 通常更适合放在 gateway，而不是每个节点本地的 agent？
4. 如果一个团队把所有应用都直接连到远端 backend，而不经过近端 Collector，可能会失去哪些工程上的好处？
5. 在 Kubernetes 中，如果你要设计一个可灰度、可回滚的 Collector 升级流程，你最希望控制哪些环节？

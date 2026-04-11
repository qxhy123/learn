# 第21章：Kubernetes 与平台级可观测性

## 学习目标

- 理解 Kubernetes 场景下应用可观测性与平台可观测性为什么必须同时设计
- 掌握 Operator、资源检测、k8s 元数据补充以及统一接入路径的基本思路
- 区分 sidecar、daemonset、gateway 在 Kubernetes 平台中的不同角色
- 学会判断哪些元数据适合进入资源属性，哪些会带来高基数或治理负担
- 建立平台侧统一接入、统一治理、统一排障的可观测性直觉

## 1. Kubernetes 里的可观测性，不只是把应用跑起来再接 OTLP

在单机环境里，应用可观测性很多时候可以围绕服务本身展开：

- 服务名是什么
- 请求延迟如何
- 下游依赖是否报错

但到了 Kubernetes，平台本身就成为系统行为的重要组成部分。很多故障不再只是“代码慢了”，还可能是：

- Pod 被频繁重建
- 节点资源紧张
- 调度不均衡
- DaemonSet 异常
- Service、Ingress 或网络策略引入额外问题

因此在 Kubernetes 中，真正成熟的可观测性至少要同时看两层：

1. **应用层可观测性**：请求、依赖、业务任务、错误、日志、指标
2. **平台层可观测性**：Pod、Node、Namespace、容器运行时、调度、网络和平台组件状态

OpenTelemetry 在这里的重要价值，不是只给应用加 trace，而是帮助你把应用层和平台层放到同一套资源与管道视角里理解。

## 2. 平台统一接入的核心问题：谁来帮应用接入、谁来补平台上下文

Kubernetes 里的应用数量通常很多，生命周期也更短，因此平台侧最关心的往往不是“某一个服务怎么手工接入”，而是：

- 如何让大量工作负载以一致方式接入
- 如何减少每个团队各写一套接入逻辑
- 如何补充稳定、可解释的 Kubernetes 资源属性
- 如何把平台治理能力集中起来

### 2.1 为什么平台要追求统一接入

如果每个业务团队都：

- 自己决定 exporter 指向哪里
- 自己决定资源属性怎么命名
- 自己决定日志和 trace 怎么关联
- 自己决定是直连后端还是走中间层

最终很容易出现：

- `service.name` 命名不一致
- `deployment.environment.name` 缺失或拼写不统一
- 有的服务能看到 Pod 上下文，有的完全没有
- 平台无法统一做 sampling、routing、脱敏和出口治理

因此平台统一接入的真正目标，不只是省事，而是让整个集群里的 telemetry 更可治理。

### 2.2 平台层真正想收束的，是“接入边界”

一个典型的平台侧目标是：

- 应用统一通过 OTLP 发往规定入口
- 资源身份使用统一约定
- 平台负责补充 Kubernetes 环境上下文
- 更复杂的处理在 Collector 侧集中完成

这使得应用团队更聚焦于：

- 业务 span 设计
- 合理的 metrics 和 logs
- 少量必要属性

而平台团队则更聚焦于：

- 接入标准
- Collector 部署拓扑
- 平台级 metadata enrich
- 多租户与安全治理

## 3. Operator 的价值：把接入和 Collector 生命周期管理平台化

在 Kubernetes 中，Operator 的价值并不只是“少写一点 YAML”，而是让 OpenTelemetry 相关资源能以更平台化的方式被管理。

### 3.1 Operator 更像接入与运维的控制面

从工程视角看，Operator 通常帮助团队把这些事情标准化：

- Collector 实例的声明式部署与升级
- 自动注入或统一接入策略
- 平台约定的配置模板分发
- 工作负载与观测配置之间的绑定关系

这对于大规模集群尤其重要，因为你不可能靠手工方式长期维护成百上千个工作负载的观测接入细节。

### 3.2 Operator 解决的是规模化一致性问题

举例来说，平台团队往往希望做到：

- 新服务接入时有统一默认值
- Collector 升级时具备可灰度、可回滚的路径
- 某些命名空间自动获得平台规定的资源检测与元数据补充能力
- 平台能够审计谁启用了什么接入策略

这些能力本质上都属于“控制面问题”，而不是单个应用的 SDK 使用问题。

### 3.3 不要把 Operator 误解成“自动产生正确可观测性”

Operator 可以帮助你统一接入，但它不会自动替你解决：

- 业务 span 是否设计合理
- 指标维度是否高基数
- 日志是否包含无意义噪声
- schema 是否长期稳定

换句话说，Operator 帮你把接入规模化，但语义质量仍然需要应用和平台共同治理。

## 4. 资源检测与 k8s 元数据：平台上下文为什么重要

Kubernetes 里的资源属性是平台排障和平台治理的关键基础。没有这些字段，很多问题几乎没法快速定位。

### 4.1 哪些平台属性最常见也最有用

常见而有价值的 Kubernetes 平台属性包括：

- `k8s.namespace.name`
- `k8s.pod.name`
- `k8s.node.name`
- `k8s.container.name`
- `service.name`
- `deployment.environment.name`

这些字段能帮助你回答很多关键问题：

- 问题只发生在某个命名空间吗
- 是某个 Pod 异常，还是整类服务异常
- 是否集中在某个节点
- 某个滚动发布后是否开始出现错误

### 4.2 一个资源检测与平台属性补充的示意

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  resourcedetection:
    detectors: [env, system]
  resource/platform:
    attributes:
      - key: deployment.environment.name
        value: production
        action: upsert
  batch:
    timeout: 5s

exporters:
  debug:
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [resourcedetection, resource/platform, batch]
      exporters: [debug]
```

这里的重点不是某个检测器清单，而是一个设计原则：**应用自己知道业务身份，平台帮助补充环境身份。**

### 4.3 元数据不是越多越好

和前面几章一样，Kubernetes 元数据也要有边界。某些字段对排障有帮助，但如果被拿去做高频聚合、routing 或长期 dashboard 主维度，就可能带来高基数问题。

例如：

- `k8s.pod.name` 对定位单 Pod 故障很有价值
- 但如果把它当成几乎所有指标的主要聚合维度，系列数可能迅速膨胀

因此要区分：

- **稳定聚合维度**：如 `service.name`、`deployment.environment.name`、`k8s.namespace.name`
- **瞬时定位维度**：如 `k8s.pod.name`、`container.id`

## 5. Sidecar、DaemonSet、Gateway：在 Kubernetes 里分别扮演什么角色

这一章必须把平台部署拓扑和平台接入目标放在一起理解。

### 5.1 Sidecar：单工作负载强隔离路径

在 Kubernetes 中，sidecar 适合：

- 某些安全边界很强的工作负载
- 需要每个 Pod 自带独立 Collector 处理能力的场景
- 某些特殊协议转换或局部治理需求

它的优点是近、隔离强、定制性高；代价则是实例数量多、运维开销高。

### 5.2 DaemonSet：平台默认的节点级接入层

DaemonSet 往往是 Kubernetes 平台里最自然的 agent 形态。它特别适合：

- 接收节点上多个 Pod 的 OTLP 流量
- 采集容器日志和主机指标
- 补充节点或容器邻近上下文
- 给应用提供统一、近端、低抖动的接入地址

也就是说，DaemonSet 更偏“平台接入层”。

### 5.3 Gateway：平台治理与统一出口层

Gateway 更适合承担：

- 统一出口认证
- tail sampling
- 多租户 routing
- 统一脱敏和 transform
- 多 backend fan-out

它更像“平台治理层”，而不是“每个工作负载都直接绑定的邻近组件”。

### 5.4 为什么平台统一接入常常是 DaemonSet + Gateway

这是非常常见的一种分层：

- 应用统一发给节点上的 DaemonSet Collector
- DaemonSet 负责近端接收、轻量处理和部分平台 metadata enrich
- 再由它转发到 Gateway
- Gateway 负责组织级策略和统一出口

这样分层的好处是：

- 应用接入一致
- 平台上下文更容易补充
- 重治理集中执行
- 更利于多团队、多命名空间和多环境统一管理

## 6. 平台级可观测性的真正目标：统一接入、统一治理、统一排障

当我们说“平台级可观测性”时，重点不是把所有服务的数据都收上来，而是让整个平台具备一致的观测语言。

### 6.1 统一接入

统一接入意味着：

- 新服务知道应该把数据发给哪里
- 平台知道如何为它补足环境上下文
- 接入方式不依赖某个团队的私有脚本或口口相传经验

### 6.2 统一治理

统一治理意味着：

- 平台能统一执行 sampling、脱敏、routing、出口控制
- 服务之间的资源属性和语义命名更一致
- Collector 的部署、升级和回滚具备组织级流程

### 6.3 统一排障

统一排障意味着：

- 可以用 `service.name + namespace + deployment.environment.name` 等稳定字段快速聚合问题
- 也可以下钻到 `pod.name`、节点、容器层面定位个体故障
- trace、metrics、logs 和平台元数据能够互相支撑，而不是各自为政

### 6.4 不要把平台可观测性做成“全靠平台团队兜底”

平台可以统一接入与治理，但应用团队仍然需要负责：

- 业务 span 和指标设计
- 日志结构化与关联字段
- 避免高基数字段滥用
- 保持服务级 schema 稳定

平台与应用不是替代关系，而是分层协作关系。

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| Kubernetes 可观测性 | 必须同时关注应用层与平台层 |
| Operator | 更像规模化接入与生命周期管理的控制面 |
| 资源检测 | 帮助补充环境身份，让平台问题可定位 |
| 元数据使用原则 | 稳定字段用于聚合，瞬时字段用于定位 |
| sidecar | 适合强隔离、强定制的单工作负载场景 |
| daemonset | 适合作为 Kubernetes 节点级接入层 |
| gateway | 适合作为集中治理与统一出口层 |
| 平台统一路径 | 常见做法是 DaemonSet/agent + Gateway 分层 |

## OTel实验

### 实验目标

从“单个服务可观测”升级到“平台统一接入可观测”的视角，设计一套 Kubernetes 环境中的 OpenTelemetry 接入路径。

### 实验步骤

1. 先假设只有一个 Node.js 服务，列出它最少需要哪些资源属性才能在 Kubernetes 中被正确识别。
2. 设计一条平台统一接入路径：
   - 应用通过 OTLP 发给节点上的 DaemonSet Collector
   - DaemonSet 做轻量处理与资源补充
   - Gateway 做 tail sampling、routing 和统一出口
3. 思考如果某个高安全工作负载不能共享节点级 agent，是否需要 sidecar，以及它会带来什么代价。
4. 设计一份最小平台字段清单，区分：
   - 稳定聚合字段
   - 瞬时定位字段
5. 再思考如果没有统一接入，这些能力会如何碎片化。

### 预期现象

- 只看应用服务名，不足以解决 Kubernetes 平台中的排障问题
- 加入 namespace、node、pod 等上下文后，问题定位会更快
- DaemonSet + Gateway 的分层最能体现“接入层”和“治理层”的平台化思路

## 练习题

1. 为什么 Kubernetes 中的可观测性必须同时覆盖应用层和平台层？
2. Operator 在平台侧最重要的价值是什么？为什么它更像控制面，而不是“自动产生观测质量”的工具？
3. 为什么 `k8s.pod.name` 很适合故障定位，却不一定适合作为长期 dashboard 的核心聚合维度？
4. 在 Kubernetes 中，sidecar、daemonset、gateway 三种 Collector 模式分别更适合承担什么角色？
5. 如果一个平台没有统一接入路径，每个团队都自己接入 OTel，长期最容易出现哪些治理问题？

# 第9章：Resources、Attributes 与 Semantic Conventions

## 学习目标

- 理解 resource 与 attribute 在 OTel 数据模型中分别代表什么
- 说清资源身份、信号字段与语义规范之间的关系
- 理解 semantic conventions 为什么是长期可维护性的基础，而不只是命名建议
- 学会区分哪些信息适合放在 resource，哪些更适合放在 span、metric、log attributes
- 建立稳定 schema、控制 cardinality 与保持 vendor-neutral 的设计意识

## 1. 为什么 telemetry schema 需要被认真设计

很多团队在接入 OpenTelemetry 的最初阶段，会把字段命名当成“以后再说的细节”。但真正进入多服务、多团队、长期运行的系统后，你会发现：

- 同一个服务名有三种写法
- 同一个环境字段有人叫 `env`，有人叫 `environment`
- 同一个 HTTP 路由有人记录模板，有人记录原始路径
- 同一种支付方式在不同服务里命名完全不一致

这类问题短期像小事，长期会直接影响：

- 查询能否统一
- 仪表盘能否复用
- 告警规则能否稳定
- 跨服务排障是否顺畅
- 成本是否可控

因此，telemetry schema 不是附属问题，而是接口设计问题。只不过它的“接口”不是给业务调用的，而是给你的观测系统和组织协作使用的。

## 2. Resource：谁产生了这批 telemetry

Resource 用来描述 telemetry 的来源实体，也就是“这些数据是谁产生的”。

它通常更适合承载的是相对稳定的身份信息，例如：

- `service.name`
- `service.namespace`
- `service.version`
- `deployment.environment.name`
- `host.name`
- `k8s.namespace.name`
- `k8s.pod.name`

### 2.1 为什么 service.name 放在 resource 而不是每个 span attribute

因为 `service.name` 描述的是整批 telemetry 的来源身份，而不是某一个 span 的临时属性。把它放在 resource，有几个好处：

- 避免每条数据重复塞同样的信息
- 更符合“谁产生了数据”这一语义
- 更方便在不同信号间统一服务身份

一个简化的 Node.js 资源配置示例如下：

```ts
import { resourceFromAttributes } from '@opentelemetry/resources'
import {
  ATTR_SERVICE_NAME,
  ATTR_SERVICE_VERSION,
} from '@opentelemetry/semantic-conventions'

const resource = resourceFromAttributes({
  [ATTR_SERVICE_NAME]: 'checkout-service',
  [ATTR_SERVICE_VERSION]: '1.2.0',
  'deployment.environment.name': 'production',
})
```

这段代码的重点不在 API 本身，而在于它表达了：

- 服务身份应该稳定声明
- 环境信息应该统一命名
- 这些字段属于数据来源身份，而不是某个业务步骤的局部信息

### 2.2 Resource 的工程边界

Resource 更适合放：

- 服务名
- 环境名
- 版本号
- 部署位置
- 节点或容器身份

通常不适合放：

- 当前用户是谁
- 当前订单号是多少
- 当前请求属于哪个业务分支

因为这些都不是“数据来源实体”的身份，而是某次具体执行的上下文。

## 3. Attributes：这条 telemetry 在什么条件下发生

如果说 resource 回答的是“谁产生了数据”，那么 attribute 更像是在回答：

- 这条 span / metric / log 是在什么条件下产生的
- 当前操作属于什么类别
- 当前执行阶段有哪些关键上下文

### 3.1 span attributes

span attributes 常用于表达：

- HTTP 方法和路由
- 数据库系统类型
- 支付方式
- 业务渠道
- 错误类别

例如：

- `http.request.method = POST`
- `http.route = /checkout`
- `payment.method_type = card`
- `checkout.channel = mobile`

### 3.2 metric attributes

metric attributes 的语义也类似，但它更强调长期聚合，因此应尽量选择：

- 稳定
- 类别有限
- 业务解释清晰

例如：

- `deployment.environment.name`
- `service.name`
- `http.route`
- `user.tier`

而不是：

- `user.id`
- `order.id`
- `session.id`
- 原始 URL 参数

### 3.3 log attributes

日志属性的空间通常比 metric 更灵活一些，但也仍然需要稳定命名和治理。它适合承载：

- trace/log 关联字段
- 错误类别
- 业务阶段
- 请求上下文的抽象信息

但不应因此把日志变成原始对象转储站。

## 4. Semantic Conventions：为什么要尽量用标准语义

Semantic Conventions 可以简单理解为：

> OpenTelemetry 推荐的一组标准字段名和语义约定，用来减少不同团队和工具之间的命名碎片化。

### 4.1 它解决的不是“写代码省一点事”，而是长期兼容问题

如果你的 HTTP span 字段有人写：

- `method`
- `http_method`
- `request.method`
- `http.request.method`

那查询、告警、仪表盘、跨服务复用都会变得很痛苦。Semantic Conventions 的价值就在于尽量把这些公共领域用统一语言表达出来。

### 4.2 哪些领域特别值得遵循规范

最常见的包括：

- HTTP
- 数据库
- messaging
- RPC
- 异常与错误
- 服务与部署身份

这些都是跨团队、跨语言最容易重复建设的地方，也是最需要统一命名的地方。

### 4.3 规范不是限制你表达业务语义

这点很重要。使用语义规范不代表所有字段都只能来自标准库。更准确地说：

- 通用领域优先遵循标准语义
- 业务领域可以补充自定义字段
- 自定义字段也应保持稳定、克制、可解释

例如：

- HTTP 方法、路由、状态码优先用通用语义
- 支付方式、订单渠道、库存来源等业务字段由你自己设计

这种方式既能享受标准化带来的兼容性，又能保留业务表达能力。

## 5. 怎样区分 Resource、Attributes 与业务字段边界

这是设计里最常见的困惑之一。可以用三个问题来帮助判断。

### 5.1 这个字段描述的是谁在产出数据吗

如果是，优先考虑 Resource。

例如：

- `service.name`
- `deployment.environment.name`
- `service.version`

### 5.2 这个字段描述的是某次执行的上下文吗

如果是，优先考虑 span / metric / log attributes。

例如：

- `payment.method_type`
- `checkout.channel`
- `http.route`

### 5.3 这个字段真的值得长期保留和统一查询吗

如果不值得，可能更适合：

- 不采集
- 仅在日志中短期保留
- 只在局部调试场景使用

这个问题特别重要，因为不是所有“看起来有用”的字段都应该进入长期 telemetry schema。

### 5.4 一个简化判断表

| 字段 | 更适合放在哪里 | 原因 |
|------|----------------|------|
| `service.name` | Resource | 描述来源服务身份 |
| `deployment.environment.name` | Resource | 描述部署环境 |
| `http.route` | Span / Metric Attribute | 描述本次请求属于哪类路径 |
| `payment.method_type` | Span / Metric Attribute | 描述当前业务分类条件 |
| `trace_id` | Log Attribute / Context 关联字段 | 用于跨信号关联 |
| `user.id` | 通常不适合高频聚合字段 | 高基数且有隐私风险 |
| `order.id` | 谨慎用于局部诊断 | 适合单次定位，不适合长期聚合 |

## 6. schema 设计中的常见反模式

### 6.1 反模式一：同义字段到处重复

例如：

- `service`
- `service_name`
- `svc`
- `app_name`

都在表达同一件事。这样做会让查询和治理迅速失控。

### 6.2 反模式二：把实例主键当成通用属性

像 `user.id`、`order.id`、`session.id` 这类字段，如果在 metrics、logs、traces 中被无差别扩散，通常会带来高基数和合规风险。

### 6.3 反模式三：完全忽略语义规范

如果 HTTP、DB、messaging 这类通用领域每个团队都自己命名，那么后续统一面板、跨团队排障和迁移会非常困难。

### 6.4 反模式四：只想着“现在够用”，不考虑长期稳定

telemetry schema 一旦进入生产并被仪表盘、告警、团队习惯依赖，后续改名和迁移的成本会非常高。因此字段命名不应随意变动。

### 6.5 反模式五：把 Resource、Attribute、Baggage、Log Field 混成一类

这会导致所有东西都往一个地方塞，结果既不清晰，也不利于治理。不同层次字段的价值，本来就不同。

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| Resource | 描述是谁产生了这批 telemetry，适合稳定来源身份 |
| Attributes | 描述某条 telemetry 发生时的上下文条件 |
| Semantic Conventions | 通过标准语义减少命名碎片化，支撑长期兼容与复用 |
| 设计重点 | 先分清身份字段与上下文字段，再考虑标准语义与成本治理 |
| 自定义字段原则 | 可以扩展业务语义，但要稳定、克制、可解释 |
| 常见风险 | 字段漂移、同义重复、高基数、长期 schema 失控 |

## OTel实验

### 实验目标

为一个简单服务设计一套最小 telemetry schema，练习区分 Resource、Attributes 与标准语义字段。

### 实验步骤

1. 选择一个场景，例如 `checkout-service` 或 `search-service`。
2. 为该服务先定义最小 Resource：
   - `service.name`
   - `service.version`
   - `deployment.environment.name`
3. 再为一个关键业务 span 设计 3-5 个 attributes，例如：
   - `http.route`
   - `checkout.channel`
   - `payment.method_type`
4. 判断这些字段中哪些应该优先采用标准语义，哪些属于业务自定义字段。
5. 再列出几项你故意不放进 schema 的字段，例如：
   - `user.id`
   - `order.id`
   - 完整请求体
6. 说明你不采集或不长期保留它们的原因。

### 预期现象

- 你会更清楚哪些字段属于来源身份，哪些属于执行上下文。
- 你会发现 semantic conventions 解决的是长期一致性问题，而不只是命名美观问题。
- 你会更容易识别高基数和字段漂移风险。

## 练习题

1. Resource 与 attribute 在语义上最大的区别是什么？
2. 为什么 `service.name` 更适合放在 Resource，而不是每个 span 的普通 attribute 中？
3. Semantic Conventions 解决的核心工程问题是什么？
4. 为什么通用领域优先使用标准语义，而业务领域再补充自定义字段，是一种更稳妥的做法？
5. 请判断以下字段更适合放在哪里，并说明理由：`deployment.environment.name`、`http.route`、`payment.method_type`、`user.id`、`service.version`。

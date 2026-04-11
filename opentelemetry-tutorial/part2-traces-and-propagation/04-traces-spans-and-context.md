# 第4章：Trace、Span 与 Context

## 学习目标

- 理解 trace、span、context 三个核心概念分别代表什么
- 能区分 span name、attributes、events、status、span kind 等关键元素的职责
- 理解父子关系、层级结构与调用边界如何共同构成一条 trace
- 建立“context 是当前执行语义载体”而不只是“某个变量”的直觉
- 学会从一条 trace 中读出系统路径、瓶颈与错误位置

## 1. Trace 与 Span：一条请求是如何被拆开的

OpenTelemetry 里最重要的基础概念，就是 trace 和 span。

可以先用一句话建立直觉：

- **trace** 表示一次请求、任务或操作在系统中的整体执行路径
- **span** 表示这条路径中的一个具体步骤或时间片段

如果把一次请求看成一部电影，那么：

- trace 是整部电影
- span 是电影中的一个场景

例如一个下单请求可能经过：

```text
gateway -> auth -> checkout -> inventory -> payment
```

对应到 trace 里，你可能看到：

- `POST /checkout`
- `auth.validate_token`
- `checkout.submit_order`
- `inventory.reserve`
- `payment.charge`

这些 span 组合在一起，才构成了一条可以解释问题的 trace。

### 1.1 为什么 span 是 OTel tracing 的基本单位

因为排障与性能分析很少发生在“整条系统”这种模糊层次，而往往发生在：

- 哪个步骤最慢
- 哪个依赖失败了
- 哪一段重试了
- 哪一层父子关系不合理

span 正好提供了这个层次的可观测粒度。

### 1.2 trace 不是“很多日志拼起来”

日志也能描述事件，但 trace 的关键不同在于它天然带有：

- 开始与结束时间
- 父子关系
- 上下文关联
- 跨服务传播能力

因此 trace 不是更花哨的日志，而是一种更适合表达执行结构的模型。

## 2. 一个 span 里通常有什么

span 不是只有一个名字和耗时。一个真正有解释力的 span，通常至少包含以下几个要素。

### 2.1 span name：这个步骤叫什么

span name 应该回答的是“当前正在做什么”，而不是“这段代码在哪个文件里”。

例如：

- `checkout.submit_order`
- `inventory.reserve`
- `payment.charge`
- `GET /orders/:id`

好的 span name 通常具备两个特点：

1. 能反映动作或阶段
2. 长期稳定，便于查询和比较

### 2.2 attributes：这个步骤有哪些关键上下文

attributes 用于补充 span 的解释信息，例如：

- `service.name`
- `http.request.method`
- `http.route`
- `db.system`
- `payment.method_type`
- `checkout.channel`

attributes 的价值在于让你不只知道“有个 span”，还知道“它在什么条件下发生”。

但要注意，attributes 不是越多越好。高基数字段和敏感字段如果放得不当，会带来成本和治理问题。

### 2.3 events：时间线上的离散事件

events 适合记录某个 span 生命周期中的关键时刻，例如：

- 开始重试
- 命中回退逻辑
- 某个异步阶段完成
- 收到特定错误响应

它的定位是“span 内部的重要节点”，而不是另起一个新 span。

### 2.4 status：这个步骤最终成功还是失败

status 用于表达 span 的执行结果，尤其是在错误场景中非常重要。

例如在 Node.js 中：

```ts
import { trace, SpanStatusCode } from '@opentelemetry/api'

const tracer = trace.getTracer('checkout-service', '1.0.0')

export async function chargePayment(orderId: string) {
  return tracer.startActiveSpan('payment.charge', async (span) => {
    try {
      span.setAttribute('order.present', Boolean(orderId))
      throw new Error('payment gateway timeout')
    } catch (error) {
      span.recordException(error as Error)
      span.setStatus({ code: SpanStatusCode.ERROR, message: 'charge failed' })
      throw error
    } finally {
      span.end()
    }
  })
}
```

这里最重要的不是语法，而是语义边界：

- `recordException()` 记录异常细节
- `setStatus(ERROR)` 声明这个 span 的结果语义

两者常常一起使用，但并不等价。

### 2.5 start/end time：耗时分析的基础

span 一定有开始和结束时间，否则就无法构成性能分析。很多 trace 的阅读价值，本质上就来自：

- 哪个 span 最长
- 谁阻塞了谁
- 是否存在异常长尾

## 3. 父子关系、层级结构与 Span Kind

trace 不只是 span 的列表，而是有结构的。

### 3.1 父子关系表达“谁触发了谁”

在最常见的同步调用链中，父子关系通常表示：

- 父 span 代表更高层业务阶段
- 子 span 代表该阶段内部的下游调用或步骤

例如：

```text
POST /checkout
  ├─ checkout.submit_order
  │   ├─ inventory.reserve
  │   └─ payment.charge
```

这个结构会直接影响你如何理解链路：

- 如果 `payment.charge` 很慢，你可以判断慢发生在订单提交流程内部
- 如果 `inventory.reserve` 报错，你也能知道错误属于哪个上层业务动作

### 3.2 Span Kind：当前 span 站在什么角色上

Span Kind 用来描述一个 span 在通信关系中的位置。最常见的几类包括：

- `SERVER`：服务端收到一个请求
- `CLIENT`：当前服务向下游发起一个请求
- `PRODUCER`：向消息系统发送消息
- `CONSUMER`：从消息系统消费消息
- `INTERNAL`：进程内部的业务步骤

这个区分很重要，因为它能帮助你更准确地读一条链路。

例如：

- `SERVER` span 更像服务入口
- `CLIENT` span 更像对外部依赖的调用
- `INTERNAL` span 更像业务阶段建模

### 3.3 不是所有函数都值得变成 span

span 的目标是表达执行结构，而不是覆盖每一行代码。

更适合建 span 的位置通常是：

- 服务入口
- 下游远程调用
- 关键业务阶段
- 容易失败或性能敏感的边界

如果把每个函数都变成 span，trace 会变得很吵，阅读成本上升，真正重要的结构反而被淹没。

## 4. Context：为什么 span 能连成一条 trace

很多初学者能理解 span，却会把 context 理解得很模糊。实际上，context 是 tracing 能成立的关键基础。

### 4.1 context 是“当前执行语义”的载体

可以先把 context 理解成：

> 当前这段代码正在处于哪个 trace / span 的执行环境中。

它通常承载的不是业务数据本身，而是：

- 当前活动 span
- 当前 trace 上下文
- propagation 需要带下去的信息
- baggage 等跨边界元数据

如果没有 context，系统就不知道：

- 新建的 span 应该挂到谁下面
- 出站请求该注入哪个 trace id
- 当前日志该关联到哪条链路

### 4.2 active span 为什么重要

在很多语言里，OTel 都有“当前活动上下文”这个概念。它的核心意义是：

- 当前代码块里创建的新 span，默认能接到正确的父 span 上
- 当前日志可以读到正确的 trace_id / span_id
- 当前出站请求可以继承正确的传播上下文

如果 active context 丢了，你通常会看到：

- trace 突然断裂
- 同一个请求里的 span 变成多条 trace
- 日志无法关联到当前链路

### 4.3 Context 不是普通业务上下文字段

很多工程师第一次听到 context，容易把它和：

- HTTP request 对象
- 函数参数对象
- 某个框架的 request context

混为一谈。

这些对象当然也能携带信息，但 OTel 的 context 更强调的是：

- 执行作用域
- 传播关系
- telemetry 关联语义

所以它的重点不只是“数据能传下去”，而是“trace 关系能传下去”。

## 5. 学会读一条 trace，而不是只会看耗时数字

很多人第一次打开 tracing 页面，最先看的是哪个 span 最慢。这当然有用，但真正有价值的阅读方式通常更完整。

### 5.1 先看结构，再看时长

一条 trace 先要回答：

- 入口在哪里
- 经过哪些服务或步骤
- 父子关系是否合理
- 是否有异常分叉或缺失

如果结构本身就不合理，例如业务 span 缺失、异步任务错误挂父、服务边界混乱，那么只看耗时数字很容易误判。

### 5.2 再看关键属性和错误语义

接着要看：

- 哪些 span 标记了错误状态
- 有没有异常记录
- 是否带有足够解释力的属性
- 慢请求是否集中在某个 route、某种支付方式、某类租户

这一步决定你能否从“慢了”走向“为什么慢”。

### 5.3 最后再决定需要用 metrics 还是 logs 补充

如果 trace 只能说明“这次付款失败发生在 `payment.charge`”，你可能还需要：

- 用 metrics 看最近 30 分钟失败率是否整体升高
- 用 logs 看下游返回的具体错误文本或业务分支信息

这也说明了 trace 的边界：它非常强，但并不打算单独解决所有问题。

## 6. 常见误区与工程边界

### 6.1 把 trace 当成完整业务审计流水

trace 更适合表达执行结构，不适合承载所有业务原文、完整输入输出或长期审计信息。

### 6.2 只会看自动生成的基础 span

自动 instrumentation 很有帮助，但如果没有业务级 `INTERNAL` span，很多链路虽然“能看”，却依旧“不够解释”。

### 6.3 把 status、event、attribute 混着用

- 错误结果语义主要用 status 表达
- 生命周期中的离散时刻主要用 event 表达
- 稳定上下文信息主要用 attribute 表达

这三者职责不同，混用后可读性会快速下降。

### 6.4 认为 context 只是框架细节

事实上，context 几乎决定了 tracing 是否能成立。传播、父子关系、日志关联，最后都离不开正确的 context 管理。

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| trace | 表示一次请求或任务的整体执行路径 |
| span | 表示路径中的一个具体步骤或时间片段 |
| span 关键元素 | name、attributes、events、status、time 构成基本解释能力 |
| 父子关系 | 决定 trace 的结构与因果阅读方式 |
| span kind | 帮助区分入口、出站依赖、内部步骤、消息边界 |
| context | 是当前执行语义与关联关系的载体，决定 span 如何连成一条 trace |
| 阅读 trace 方法 | 先看结构，再看时长、属性和错误语义 |

## OTel实验

### 实验目标

通过一个简单的订单请求，观察 trace、span 与 context 在同一条链路中分别扮演什么角色。

### 实验步骤

1. 在一个 Node.js 服务里创建一个 `POST /checkout` 接口。
2. 让该接口内部再手动创建两个业务 span，例如 `inventory.reserve` 与 `payment.charge`。
3. 给其中一个子 span 故意注入一次失败，并记录 exception 与 error status。
4. 发送几次请求，观察 trace 中：
   - 根 span 是谁
   - 子 span 如何挂接
   - 错误在哪个 span 上出现
   - attributes 与 events 分别提供了什么信息
5. 再把一个业务 span 从 `startActiveSpan` 改成不带活动上下文的错误写法，观察链路是否出现断裂或层级异常。

### 预期现象

- 你会看到 trace 是由多个有层级关系的 span 组成的。
- error status 和 exception 会让错误位置比单看日志更直观。
- 一旦活动 context 管理错误，span 关系就可能异常，甚至直接断裂。

## 练习题

1. trace 与 span 的关系是什么？为什么 span 是 tracing 的基本分析单位？
2. span name、attribute、event、status 分别更适合表达什么信息？
3. 为什么说 context 是 tracing 能成立的关键基础，而不是一个可有可无的附加概念？
4. `SERVER`、`CLIENT`、`INTERNAL` 三种 span kind 分别更适合出现在哪些场景？
5. 请说明为什么“给每个函数都加 span”通常不是一个好的链路设计策略。

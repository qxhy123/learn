# 第6章：Span 设计与链路建模

## 学习目标

- 理解为什么“能出 trace”不等于“trace 有解释力”
- 学会判断 span 应该放在什么边界上，而不是给所有代码都加埋点
- 理解同步调用、异步任务、批处理等不同场景下的链路建模差异
- 掌握错误记录、命名、属性设计在 trace 可读性中的作用
- 建立兼顾排障价值、成本和长期稳定性的 span 设计直觉

## 1. 为什么 span 设计决定 trace 是否有价值

很多团队接入 OpenTelemetry 后，第一阶段的问题通常不是“完全没 trace”，而是“有 trace 但看不懂”。

常见表现包括：

- span 特别多，但没有清晰的业务阶段
- 所有 span 都是自动生成的 HTTP、SQL、Redis 调用
- span 名称不稳定，今天一种写法，明天一种写法
- 错误信息散在日志里，trace 本身解释力很弱

这说明问题不在于 tracing 功能有没有打开，而在于 span 设计是否合理。

一个有价值的 trace，通常应该让你比较快地回答：

- 这次请求的主要业务阶段是什么
- 哪一段最慢
- 哪一段失败了
- 失败是在调用外部依赖，还是在内部逻辑判断
- 当前 trace 的结构是否符合系统真实执行模型

而这些问题，最后都落到 span 设计与链路建模上。

## 2. span 应该放在哪些边界上

span 的目标不是覆盖所有代码，而是表达系统的关键执行边界。一个常见且实用的判断标准是：**这个步骤是否值得在诊断和性能分析中被单独看见。**

### 2.1 适合放 span 的典型边界

#### 服务入口

例如：

- HTTP server 请求
- gRPC server 调用
- 消费一条消息
- 启动一个后台 job

这些通常是 trace 的重要起点。

#### 出站依赖调用

例如：

- 调用支付网关
- 查询数据库
- 请求库存服务
- 访问缓存或消息系统

这些步骤通常决定了系统大部分外部等待时间，也是错误高发位置。

#### 关键业务阶段

例如：

- `checkout.submit_order`
- `pricing.calculate_final_price`
- `risk.evaluate`
- `inventory.reserve`

这些 span 让链路从“基础 I/O 列表”变成“业务流程结构图”。

#### 需要单独统计或排障的内部步骤

如果某个内部阶段经常慢、经常失败，或者在业务上非常关键，也值得独立建 span。

### 2.2 通常不适合放 span 的位置

- 每个普通工具函数
- 很短、很频繁、诊断价值很低的内部调用
- 只为了“多一点可见性”而硬加的局部细节
- 本应是 attribute 或 event 的信息，却被误做成独立 span

如果所有微小步骤都变成 span，trace 会迅速膨胀，既难读，又增加开销。

## 3. 一个好 span 要如何命名与补充语义

### 3.1 span 名称要稳定、动作导向、层次清楚

span 名称最怕两种问题：

- 太技术实现化，例如 `handleOrderWithStrategyV2`
- 太随意，命名风格完全不统一

更好的做法通常是：

- 用稳定动作表达阶段
- 尽量体现领域含义
- 保持同一服务内命名风格一致

例如：

- `checkout.submit_order`
- `inventory.reserve`
- `payment.charge`
- `risk.evaluate`

这种命名方式的好处是，你以后做：

- trace 查询
- dashboard 聚合
- 慢调用分析
- 错误率对比

都会更稳定。

### 3.2 attribute 要服务于解释，而不是堆砌信息

好的 span attribute 应该帮助你回答：

- 当前 span 发生在哪种业务条件下
- 当前调用属于哪个分类维度
- 当前错误或慢请求是否和某类场景相关

例如：

- `payment.method_type = card`
- `checkout.channel = web`
- `inventory.source = warehouse`
- `user.tier = premium`

而不应该优先塞入：

- `order.id`
- `user.id`
- 全量 SQL
- 完整请求体

前者更利于长期聚合与解释，后者更容易引发高基数、隐私和成本问题。

### 3.3 error 语义要明确落在哪个 span 上

一个常见反模式是：

- 业务失败了
- 但只有根 span 被标红
- 真正出错的子阶段没有任何错误语义

更好的做法是：

- 在真正失败的那个 span 上记录异常与错误状态
- 必要时在更高层业务 span 上补一个总结性错误状态

这样你看 trace 时能更快区分：

- 是支付网关超时
- 是库存预占失败
- 还是订单校验本身没通过

## 4. 同步链路、异步链路与批处理该怎么建模

链路建模最容易踩坑的地方，不是单个 HTTP 请求，而是异步系统。

### 4.1 同步请求：通常更接近树状 parent-child

在同步 HTTP / RPC 场景中，最常见的建模方式是父子结构：

```text
POST /checkout
  ├─ checkout.submit_order
  │   ├─ inventory.reserve
  │   └─ payment.charge
```

这种结构很适合表达明确的调用层级。

### 4.2 异步任务：不要强行假装还是同步调用

如果用户请求提交后，只是把任务放入队列，由后台 worker 稍后处理，那么：

- 用户请求的生命周期
- 后台任务的生命周期

就不一定适合完全当作同步 parent-child 关系。

这时更重要的是表达：

- 它们有因果关联
- 但执行时序和资源边界已经分离

在这种场景里，设计重点就不只是“挂到谁下面”，而是“如何让后续排障能看清从请求到任务的关系”。

### 4.3 批处理：一个输入对多个输出，或多个输入汇成一个处理

批处理系统里经常出现：

- 一个批次处理 1000 条消息
- 一个任务聚合多个来源事件
- 一个下游操作对应多个上游请求

这类场景如果仍然僵硬地套成普通 parent-child 树，trace 结构往往会很怪。你要先明确系统的真实执行模型，再决定如何表达关联关系，而不是为了“看起来像树”去牺牲可解释性。

### 4.4 设计原则：优先还原真实边界

一个实用原则是：

- 同步等待关系强时，优先 parent-child
- 执行时序明显分离时，优先把边界表达清楚，而不是强行塞成一个长调用栈

这也是为什么 trace 建模不是纯粹技术问题，而是系统建模问题。

## 5. 用一个 Node.js 例子看“有解释力”的 span 设计

下面用一个简化的订单提交流程说明什么叫“自动 span 之外，还需要业务 span”。

```ts
import { trace, SpanStatusCode } from '@opentelemetry/api'

const tracer = trace.getTracer('checkout-service', '1.0.0')

export async function submitOrder(input: {
  orderId: string
  channel: 'web' | 'mobile'
  paymentMethod: 'card' | 'wallet'
}) {
  return tracer.startActiveSpan('checkout.submit_order', async (span) => {
    span.setAttribute('checkout.channel', input.channel)
    span.setAttribute('payment.method_type', input.paymentMethod)

    try {
      await tracer.startActiveSpan('inventory.reserve', async (childSpan) => {
        try {
          await reserveInventory(input.orderId)
        } finally {
          childSpan.end()
        }
      })

      await tracer.startActiveSpan('payment.charge', async (childSpan) => {
        try {
          await chargePayment(input.orderId)
        } catch (error) {
          childSpan.recordException(error as Error)
          childSpan.setStatus({ code: SpanStatusCode.ERROR, message: 'charge failed' })
          throw error
        } finally {
          childSpan.end()
        }
      })

      return { ok: true }
    } catch (error) {
      span.recordException(error as Error)
      span.setStatus({ code: SpanStatusCode.ERROR, message: 'submit order failed' })
      throw error
    } finally {
      span.end()
    }
  })
}

async function reserveInventory(orderId: string) {
  return orderId
}

async function chargePayment(orderId: string) {
  return orderId
}
```

这个例子有几个设计点值得注意：

- 根业务 span 是 `checkout.submit_order`
- 核心阶段被拆成 `inventory.reserve` 和 `payment.charge`
- 业务分类信息放在 attributes 中
- 真正失败的子阶段被明确标记错误

如果只有自动 instrumentation，你可能只能看到：

- 一个 HTTP server span
- 一个数据库调用
- 一个 HTTP client 调用

而很难直接看出“这次订单提交慢，是因为支付阶段超时，且发生在钱包支付路径”。业务 span 的价值，正是在这里。

## 6. span 设计中的常见反模式

### 6.1 反模式一：所有东西都交给自动 instrumentation

自动 instrumentation 很适合基础覆盖，但它无法替你表达领域阶段。只靠自动 span，链路往往更像 I/O 列表，而不是业务流程图。

### 6.2 反模式二：所有函数都加 span

这会制造大量噪音，阅读体验很差，也会提高开销。span 应该服务于系统解释，而不是代码行级可视化。

### 6.3 反模式三：span 命名完全不稳定

如果 span 名称随着版本、实现、开发者习惯频繁变化，长期趋势分析和查询会非常痛苦。

### 6.4 反模式四：高基数主键滥用

把 `user.id`、`order.id`、`session.id` 普遍加到 span 和相关聚合字段里，会带来成本与治理问题。要优先选择分类维度而不是实例主键。

### 6.5 反模式五：只标记根 span 错误，不标记具体失败步骤

这样看起来“整条 trace 出错了”，但并不能快速解释错在哪里。好的 trace 应该让错误尽量靠近真实失败边界。

## 本章小结

| 主题 | 核心结论 |
|------|----------|
| span 设计目标 | 让 trace 可解释，而不只是有数据 |
| 适合建 span 的边界 | 服务入口、出站依赖、关键业务阶段、性能敏感环节 |
| 命名原则 | 稳定、动作导向、领域清晰 |
| attribute 原则 | 用分类信息增强解释，避免滥用高基数主键 |
| 异步与批处理 | 先理解真实执行模型，再选择建模方式 |
| 错误设计 | 让错误尽量落在真实失败步骤上，而不是只停留在根 span |

## OTel实验

### 实验目标

对比“只有自动 instrumentation”与“补充业务级 span”两种链路结构，感受 span 设计对 trace 可读性的影响。

### 实验步骤

1. 准备一个简单订单流程，至少包含库存预占和支付两个步骤。
2. 第一轮只开启自动 instrumentation，观察 trace 中主要出现哪些 span。
3. 第二轮手动加入 `checkout.submit_order`、`inventory.reserve`、`payment.charge` 三个业务 span。
4. 在支付步骤故意制造一次失败，并在对应 span 上记录 exception 与 error status。
5. 比较两轮 trace 在以下问题上的差异：
   - 哪个阶段最慢是否更容易看出来
   - 错误位置是否更容易定位
   - 业务语义是否更清楚
6. 再尝试给每个辅助函数都加 span，观察 trace 是否开始变得噪音过多。

### 预期现象

- 只有自动 instrumentation 时，链路能看见基础 I/O，但业务语义较弱。
- 加入业务 span 后，trace 会更像真正的处理流程图。
- 过度埋点会迅速降低 trace 的可读性。

## 练习题

1. 为什么说“有 trace”不等于“trace 有解释力”？
2. 哪些边界通常最适合建 span，哪些边界通常不适合？
3. 为什么 span 名称应尽量动作导向、长期稳定？
4. 在异步任务场景里，为什么不能机械照搬同步 HTTP 的 parent-child 心智模型？
5. 请举一个例子说明：把错误明确记录在真实失败步骤上，比只标记根 span 错误更有价值。

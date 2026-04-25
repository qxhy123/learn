# 第10c章：Fine-Tuning 基础设施与多 Adapter 服务

> 生产环境里最常见的训练，不是从头预训练，而是围绕一个已有 base model 做大量小而快的微调任务。

> **关联章节**：本章与 [第10章](./10-memory-checkpointing-and-recovery.md) 的 checkpoint、[第10b章](./10b-alignment-and-post-training.md) 的后训练流程、[第12章](../part4-data-and-storage/12-artifacts-and-checkpoints.md) 的制品管理、[第14章](../part5-serving-infra/14-online-inference-architecture.md) 的推理架构、[第17章](../part5-serving-infra/17-multitenancy-and-cost.md) 的多租户成本控制密切相关。adapter 既是训练产物，也是服务期的一等制品。

## 学习目标

完成本章学习后，你将能够：

1. 理解为什么 fine-tuning 需要独立于 pretraining 讨论
2. 区分全量微调、LoRA、QLoRA 的资源形态与平台含义
3. 画出从数据快照到 adapter 上线的完整 fine-tuning 流程
4. 理解 Fine-Tuning as a Service 的控制面、调度面和制品面
5. 看懂 Multi-LoRA Serving 的显存、缓存和路由设计
6. 设计 adapter 的版本管理、兼容性校验与 A/B 测试流程
7. 识别 fine-tuning / multi-adapter 系统里的典型失败模式
8. 为多租户场景设计更稳妥的观测指标与容量规则

---

## 本章导读

很多人第一次接触这部分时，会把 fine-tuning 理解成：

- “就是把训练规模缩小一点”
- “反正最后也是导出一个模型”
- “服务时只要把 LoRA 文件挂上去就行”

这些理解都不算完全错，但会遗漏最关键的工程事实：

- fine-tuning 的**作业形态**和 pretraining 完全不同
- fine-tuning 的**交付物**经常不是完整模型，而是 adapter
- fine-tuning 的**平台目标**不是极限吞吐，而是高并发、低成本、短反馈回路
- fine-tuning 的**服务形态**会直接反过来约束训练配置、制品格式和版本管理

所以，本章不是单独讲“LoRA 是什么”，也不是只讲“怎么把多个 adapter 挂到一个服务上”，而是把整条链路连起来：

```text
数据快照
  -> 训练作业
  -> adapter 制品
  -> 评测与注册
  -> 多 adapter 路由
  -> 线上灰度 / 回滚
```

从平台视角看，真正要解决的问题是：

> **如何让大量短周期、低成本、强版本绑定的微调任务，可以被稳定训练、可靠登记、快速上线，并在共享推理池里安全服务。**

换句话说，fine-tuning 和 multi-adapter serving 不应被拆成“两套彼此无关的系统”。
平台真正交付的是一条完整链路：先把 adapter 训练出来，再把 adapter 当作一等制品去登记、评测、治理，最后让服务层能够按租户、版本和流量策略稳定选择并加载它。
如果训练阶段不记录清楚 base model、模板和元数据，服务阶段就无法安全热加载；如果服务阶段没有 registry、灰度和缓存策略，训练阶段产出的 adapter 也很难变成可运营的线上能力。

---

## 正文内容

### 10c.1 为什么 fine-tuning 需要独立讲

把 fine-tuning 当成“缩小版 pretraining”通常会误导平台设计。
两者最关键的差异不是“参数少了一点”，而是**目标函数、作业节奏、交付物和服务方式都变了**。

| 维度 | 全量预训练 / 全量微调 | 参数高效微调 |
|------|----------------------|--------------|
| 核心目标 | 学通用能力或重写整个模型 | 在固定 base model 上做定向增强 |
| 训练参数 | 大部分甚至全部参数 | 少量 adapter 参数 |
| 任务时长 | 天到周 | 分钟到小时 |
| 作业形态 | 少量长作业 | 大量短作业 |
| 交付物 | checkpoint / 完整模型包 | adapter + 元数据 |
| 平台重点 | 吞吐、扩展效率、恢复 | 多租户、低成本、快速交付 |
| 服务形态 | 独立部署一套模型 | 共享 base model + 动态挂载 adapter |

一个非常重要的组织视角是：

- pretraining 更像“研发主线工程”
- fine-tuning 更像“平台化产品能力”

原因在于 fine-tuning 往往天然走向：

- 多团队共享同一组 base model
- 多个租户高频提交微调任务
- 每次训练只改极小一部分参数
- 训练完成后很快进入评测、灰度和上线

因此平台通常会把 fine-tuning 放进独立队列，而不是和 pretraining 混跑：

- 使用更小 GPU 或更便宜实例
- 接受更高任务并发
- 更重视等待时延和任务周转时间
- 强调 artifact 管理、兼容性约束和上线路径

如果把两者强行塞进同一资源池，通常会同时损害两边：

- 长作业会压住短作业，导致用户等待时间恶化
- 短作业会打碎大作业的容量规划
- 调度器很难同时优化“平均吞吐”和“首个结果返回速度”

### 10c.2 全量微调、LoRA、QLoRA 到底在交换什么

LoRA 的核心思想是：冻结 base model，只训练插入到特定层中的低秩矩阵。
QLoRA 则进一步把 base model 量化到更低比特表示，以进一步降低训练显存。

一个很有帮助的直觉是：LoRA 不是“把大模型变小”，而是**只训练一小块可控增量**。

如果某个线性层权重为：

$$
W \in \mathbb{R}^{d_{out} \times d_{in}}
$$

LoRA 往往把更新写成：

$$
\Delta W = BA
$$

其中：

- \(A \in \mathbb{R}^{r \times d_{in}}\)
- \(B \in \mathbb{R}^{d_{out} \times r}\)
- \(r\) 是远小于 \(d_{in}, d_{out}\) 的 rank

于是该层新增的可训练参数量大约是：

$$
r(d_{in} + d_{out})
$$

而不是原始的：

$$
d_{out} \times d_{in}
$$

这就是为什么 adapter 往往远小于完整模型。

#### 10c.2.1 从平台视角看三种路径

| 方案 | Base model 形态 | 训练成本 | 服务期形态 | 是否适合 Multi-Adapter | 常见风险 |
|------|-----------------|----------|------------|------------------------|----------|
| 全量微调 | 全精度可训练 | 最高 | 常常变成一个新完整模型 | 较差 | 制品大、发布慢、难共享 |
| LoRA | Base 冻结，adapter 可训练 | 低很多 | 可合并，也可动态挂载 | 很适合 | adapter 数量膨胀、版本管理复杂 |
| QLoRA | Base 量化 + adapter 可训练 | 更低 | 训练更省，但服务仍需考虑兼容引擎 | 适合，但约束更多 | 训练与服务环境不一致、精度波动 |

平台在选择时，看的不只是“便不便宜”，而是下面几件事是否同时成立：

- 训练是否能落在更便宜的 GPU 档位
- 制品是否足够小，便于存储和传输
- 服务期能否挂载到共享 base model 上
- 回滚是否简单
- A/B 测试是否方便

#### 10c.2.2 一个粗略的资源直觉

| 7B 级模型场景 | 粗略显存量级 | 工程含义 |
|---------------|--------------|----------|
| 全量微调 | 常见 35-50 GB 起 | 往往逼近高端训练卡 |
| LoRA | 常见十几到二十多 GB | 单卡中高端 GPU 就可覆盖很多任务 |
| QLoRA | 常见可进一步下降到十几 GB 内外 | 更适合成本敏感平台 |

这张表的重点不是精确数字，而是平台形态变化：

- **全量微调** 常常意味着“单个任务占用昂贵资源更久”
- **LoRA** 常常意味着“同样资源池里可以容纳更多短任务”
- **QLoRA** 常常意味着“原本上不了 GPU 的任务，也能进入平台化自助微调”

#### 10c.2.3 rank 和 target modules 为什么是基础设施参数

很多人把 `rank`、`target_modules` 看成纯算法超参，但在平台上它们也会影响：

- 训练期显存
- 训练期吞吐
- adapter 文件大小
- 服务期加载时延
- 多 adapter 共存时的缓存压力

一个实用判断是：

- `rank` 更像“单个 adapter 的容量旋钮”
- `target_modules` 更像“这个 adapter 侵入模型多少位置”

它们调大并不只是“效果可能更好”，也意味着：

- 训练成本更高
- 服务挂载更慢
- registry 中的制品更难统一治理

### 10c.3 FTaaS 控制面与端到端 pipeline

平台里的 fine-tuning 不应被抽象成“跑完一个 train.py 就结束”。
在生产环境里，FTaaS 更像一个围绕 adapter 生命周期构建的小型控制面，它负责把一次训练请求安全地推进到“可被线上服务加载”的状态。

一个更完整的端到端视角是：

```text
[1] 数据快照 / 数据清洗
      ↓
[2] 选择 base model、模板、训练参数
      ↓
[3] 提交 FTaaS 任务并做准入控制
      ↓
[4] 执行训练，产出 checkpoint / 最终 adapter
      ↓
[5] 自动评测、安全检查、兼容性校验
      ↓
[6] 注册到 adapter registry
      ↓
[7] 触发 Multi-LoRA 服务热加载
      ↓
[8] staging / canary / production 灰度发布
      ↓
[9] 线上观测、A/B、回滚、下线
```

每一步都对应不同的平台责任：

| 阶段 | 主要输入 | 平台要保证什么 | 典型输出 |
|------|----------|----------------|----------|
| 数据快照 | 训练样本、清洗规则 | 数据可追踪、可复现 | dataset version |
| 训练提交 | base model、超参、租户信息 | 鉴权、配额、资源估算 | training job |
| 训练执行 | GPU、checkpoint 存储 | 稳定运行、失败恢复 | adapter / checkpoint |
| 自动评测 | 基准集、判分器、门禁规则 | 不让“训练成功但效果退化”的版本进入上线 | eval report |
| 注册 | 制品元数据 | 版本绑定、状态管理、审计 | adapter record |
| 热加载 | serving pool、路由信息 | 可观测、可回退、不重启 | loaded adapter |
| 发布 | 灰度策略、SLO、告警规则 | 可回滚、权限正确、流量可控 | online version |

平台里经常要明确区分三种“成功”：

1. **训练成功**：job 没挂，产出了 adapter
2. **评测成功**：adapter 达到了离线门槛并通过兼容性检查
3. **发布成功**：adapter 被正确加载，并在真实流量里表现稳定

这三者不是同义词。
很多系统事故，都出在“把训练成功误当成可上线”。

#### 10c.3.1 控制面和数据面分别管什么

平台化 fine-tuning 的典型形态是：

- 多个租户共享同一个 base model
- 每个任务只训练自己的 adapter
- 训练完成后把 adapter 作为独立制品保存
- 用户预期是“提交任务后较快拿到一个可测试版本”

因此 FTaaS 真正要建设的，不只是训练脚本，而是一套能驱动制品流转的控制面。

| 平面 | 主要职责 |
|------|----------|
| 控制面 | API、认证、配额、任务排队、状态机、registry、发布门禁、热加载触发 |
| 数据面 | 训练 worker、对象存储、评测 worker、serving loader、日志与监控 |

一个成熟的 FTaaS 往往至少要能回答：

- 这个任务是谁提交的
- 它可以用哪几个 base model
- 它最多能占多大资源
- 失败后能否自动重试
- 训练产物会被登记到哪里
- 哪些版本允许被自动热加载

#### 10c.3.2 一个很常见的任务状态机

```text
submitted
  -> admitted
  -> staging_data
  -> running
  -> evaluating
  -> registered
  -> loading
  -> deployable
```

同时还要允许下面这些转移：

- `running -> failed`
- `running -> cancelled`
- `running -> retrying`
- `evaluating -> rejected`
- `loading -> load_failed`
- `deployable -> deprecated`

为什么状态机很重要？
因为平台不是只给“训练框架开发者”看，而是要给：

- 提交任务的业务团队
- 做排障的 SRE / 平台工程师
- 做发布决策的 reviewer
- 做审计和回滚的人

每个人看到的都不是“程序退出码”，而是这个任务当前处于什么业务状态。

#### 10c.3.3 从训练完成到热加载的闭环

FTaaS 控制面最容易被低估的部分，是“训练完之后还要继续做的事情”。
一个稳妥的默认流程通常是：

1. 训练 worker 上报 `running -> evaluating`，并冻结最终 adapter、训练配置、数据版本与 base model 标识。
2. 评测 worker 自动拉起离线评测，包括任务指标、安全检查、兼容性检查。
3. 评测通过后，把 adapter 与 metadata 写入 adapter registry，状态推进到 `registered`。
4. registry 发出事件，通知 Multi-LoRA serving pool 预加载或热加载该 adapter。
5. loader 完成加载后，把状态推进到 `deployable`，再由流量系统做 staging / canary / production 放量。

这条链路里的失败处理必须是显式设计，而不是靠人工补救：

- **训练失败重试**：区分瞬时错误和确定性错误。前者如节点重启、对象存储抖动，可以按次数和退避策略自动重试；后者如配置非法、数据缺失，应直接失败并返回明确原因。
- **评测不通过回退**：训练成功但评测未达标时，adapter 可以保留在 registry 中作为审计对象，但状态必须停在 `rejected` 或 `non_deployable`，不能继续触发线上加载。
- **加载失败告警**：如果注册成功但服务热加载失败，控制面应把版本标记为 `load_failed`，触发告警，并维持旧版本路由不变，避免半上线状态。
- **灰度期快速回滚**：一旦 canary 流量的错误率、延迟或业务指标异常，路由层应直接切回上一稳定 adapter，而不是重新等待训练作业。

从这个角度看，FTaaS 控制面的职责并不是“把作业塞到 GPU 上”，而是**把一个 adapter 从训练产物推进成可治理的线上版本**。

#### 10c.3.4 调度器真正关注什么

FTaaS 调度器和 pretraining 调度器看的维度并不一样：

| 调度维度 | 为什么重要 |
|----------|------------|
| 预计显存 | 决定能否落到某个 GPU 档位 |
| base model 本地性 | 减少重复下载和冷启动时间 |
| checkpoint / 数据位置 | 影响作业启动耗时 |
| 租户配额与优先级 | 防止一个团队把短作业池打满 |
| 是否可抢占 | 决定能否放到更便宜资源上 |
| 任务时长预估 | 帮助调度器减少碎片和排队抖动 |

这说明 FTaaS 的关键优化目标不是“单作业跑到极致”，而是：

- 平均等待时间
- 单位 GPU 上的作业周转数
- 成功率
- 用户拿到第一个可用版本的时间

### 10c.4 Multi-LoRA Serving

训练完成后的 adapter，往往不会单独起一套 base model 服务，而是挂载到共享实例上。

一个常见架构可以粗略画成：

```text
request
  -> gateway / auth
  -> router(tenant, task, adapter_version)
  -> model pool(shared base model)
  -> adapter loader/cache
  -> inference
```

它的核心思想是：

- base model 权重只保留一份
- 多个 adapter 共用同一组推理副本
- adapter 按请求或按会话被动态选择

#### 10c.4.1 它为什么能显著降成本

| 组成部分 | 在服务期做什么 | 资源关注点 |
|----------|----------------|------------|
| Base model | 共享主权重 | 大头显存、预热成本 |
| Adapter | 叠加到请求路径 | 增量显存、切换延迟 |
| Router | 按 tenant / task / version 选 adapter | 路由正确性和隔离 |
| Loader | 热加载 / 热卸载 adapter | 不重启、低抖动 |

Multi-LoRA Serving 的关键收益是：

- base model 不需要为每个小任务复制一份
- 新 adapter 上线不必重新部署整套模型
- 多个业务实验可以共享 warm pool
- 灰度和回滚粒度可以细到 adapter 版本

这也是为什么它常常和 [第17章](../part5-serving-infra/17-multitenancy-and-cost.md) 的多租户治理绑在一起讲。

#### 10c.4.2 但它不是“免费午餐”

Multi-LoRA Serving 同时引入了很多新的复杂度：

- adapter 缓存淘汰策略
- 热加载失败回滚
- 请求路由和权限控制
- 不同 adapter 请求是否还能高效 batch
- 某个热租户是否把 cache 全部挤爆

尤其是 batching，会出现一个很工程化的问题：

> 如果两个请求对应不同 adapter，它们还能不能共用同一批次、同一份 KV cache、同一条高效执行路径？

不同 serving engine 的答案不完全一样，但平台要默认认为：

- **跨 adapter 的批处理通常更难做**
- **adapter 越多，调度与缓存越容易成为瓶颈**

#### 10c.4.3 一个实用的缓存分层

很多生产系统会把 adapter 做成三层缓存：

| 层级 | 典型位置 | 作用 |
|------|----------|------|
| Hot | GPU 显存 | 当前高频 adapter，追求最低切换延迟 |
| Warm | CPU 内存或本地 NVMe | 避免每次都回对象存储拉取 |
| Cold | 对象存储 / registry | 作为真实来源与长期归档 |

这个分层非常重要，因为 adapter 文件虽然比完整模型小，但一旦数量上百上千：

- 热点分布会变得高度不均匀
- 全量常驻显存不现实
- 冷加载时延会直接拉高 TTFT

所以 Multi-LoRA Serving 的瓶颈经常不是“显存绝对不够”，而是：

- 哪些 adapter 值得常驻
- 冷热切换时延能否被业务接受
- cache miss 会不会形成抖动风暴

#### 10c.4.4 Multi-LoRA 显存管理

从显存账本的角度看，Multi-LoRA Serving 的核心不是“adapter 很小，所以问题不大”，而是要把共享成本、增量成本和并发成本分开核算。
最粗略、但非常实用的一条估算公式是：

$$
\text{总显存} \approx \text{base\_model\_size} + N \times \text{adapter\_size} + \text{concurrent\_requests} \times \text{kv\_cache\_per\_request}
$$

这里有三个完全不同的部分。
第一部分是 `base_model_size`，它通常是固定大头，也是 Multi-LoRA 能省钱的根本原因，因为无论挂多少 adapter，base model 权重理论上都只保留一份常驻副本。
第二部分是 `N × adapter_size`，表示当前被放进 GPU 热层的 adapter 数量乘以单个 adapter 的增量参数体积。单个 LoRA adapter 往往远小于完整模型，但当热点 adapter 从个位数涨到几十个、上百个时，这部分就会从“可以忽略”变成显存预算里的硬约束。
第三部分是 `concurrent_requests × kv_cache_per_request`。很多团队一开始只盯着 adapter 大小，却忽略了请求级 KV cache 往往会随着并发数、上下文长度、batching 策略快速增长，在长上下文和高并发场景下，它甚至可能比 adapter 总和更先把显存吃满。

因此，Multi-LoRA 服务真正要做的不是“把尽可能多的 adapter 塞进显存”，而是预留一个动态预算：

$$
\text{GPU 总容量} \ge \text{base 常驻} + \text{热点 adapter 常驻} + \text{峰值 KV cache} + \text{安全余量}
$$

举个直觉化的例子：如果某个 base model 常驻后占掉 16 GB，单个 adapter 平均 150 MB，GPU 上常驻 48 个热点 adapter 就要再吃掉约 7.2 GB；如果同时有 32 个并发请求，每个请求平均需要 200 MB KV cache，又是 6.4 GB。这样即使不算框架额外开销、碎片化、临时 buffer，总量也已经接近 30 GB。放在 40 GB 卡上看似还有空间，但只要上下文长度上升、某些请求更大、或者 loader 需要短时双拷贝，OOM 风险就会迅速上来。

这也是为什么 adapter 数量上限从来不只是一个“文件数”问题，而通常同时受三类约束决定：

- **显存上限**：热层能容纳多少 adapter，最终受 base 常驻、KV cache 波峰和安全余量共同限制。
- **切换延迟**：即使显存还能塞下更多 adapter，热加载和热卸载带来的时延抖动也会拉高 TTFT，影响用户体验。
- **路由复杂度**：adapter 越多，按租户、版本、实验桶、权限去选择正确 adapter 的规则就越复杂，batching 和排队也更容易碎片化。

生产系统里常见的做法是把 base model 设为绝对常驻，把少量高频 adapter 放在 GPU 热层，其余 adapter 放在 CPU 内存、本地 NVMe 或对象存储，再通过热加载和热卸载机制做动态迁移。
其中热加载并不是简单 `load(adapter)`，而是要配合 admission control 判断当前显存是否还允许接纳新 adapter；热卸载也不能只看“最近有没有请求”，因为有些 adapter 虽然暂时冷却，但可能正准备参与 canary 或被某个高价值租户周期性访问。

因此缓存层通常会引入 **LRU 驱逐** 或其变体：当 GPU 热层接近上限时，优先驱逐最近最少使用、且当前没有活跃请求绑定的 adapter。
但单纯 LRU 往往还不够，因为平台还要考虑业务优先级、租户权重、灰度任务和短时热点回潮。
很多系统会再叠加 **预加载（preload）** 机制：在新版本灰度开始前、已知大客户流量高峰前，或者发现某个 adapter 命中率持续抬升时，提前把它从 warm 层搬到 GPU 热层，降低第一批请求的冷启动抖动。

所以 Multi-LoRA 显存管理的本质不是一个静态容量公式，而是一套持续平衡机制：共享 base 节省固定成本，adapter 热层控制增量成本，KV cache 吞吐决定并发成本，而热加载、热卸载、LRU 和预加载共同决定这套系统在真实流量下是否稳定。

### 10c.5 合并部署和动态挂载是两种不同策略

很多团队在 adapter 上线时，会在两条路之间选择：

| 路径 | 做法 | 优点 | 代价 | 更适合 |
|------|------|------|------|--------|
| 合并部署（merge） | 把 adapter 合并进权重，再导出一个独立模型版本 | 推理路径简单、单模型性能更稳定 | 失去热切换能力，制品变大 | 单租户、固定版本、吞吐优先 |
| 动态挂载（side-load） | base model 常驻，adapter 按需加载 | 多租户友好、热上线快、成本低 | 缓存、路由、权限更复杂 | 多版本并存、频繁实验、平台化服务 |

这两条路没有绝对对错，关键看平台目标：

- 如果你只有少量稳定版本，**merge** 往往更省心
- 如果你有大量实验和多租户需求，**dynamic adapter loading** 几乎是必选项

这也解释了一个常见误区：

> “我们训练时用的是 LoRA，所以服务时一定用 Multi-LoRA。”

其实不一定。
LoRA 是训练形态，Multi-LoRA 是服务形态；两者常一起出现，但不是强绑定。

### 10c.5a Adapter 与 Base Model 版本兼容性

很多人会下意识认为：LoRA adapter 只是“贴在 base model 上的一层增量”，所以只要新旧 base model 规模相同、层名看起来差不多，adapter 就应该还能继续用。
这个判断在工程上通常过于乐观。
LoRA 的本质是对特定 base 权重空间做低秩增量拟合，训练出来的矩阵不是抽象地绑定在“某个 7B 模型品类”上，而是绑定在**某一版具体 base 权重**上。
只要 base model 升级了 checkpoint、替换了 tokenizer、调整了 chat template，甚至只是做了看似温和的后训练，adapter 对应的最优增量方向都可能已经变化。

这也是为什么 base model 升级后，LoRA adapter 通常需要重新训练，至少也要重新做完整兼容性验收，而不能默认直接复用。
最危险的情况不是“完全加载失败”，因为那反而容易发现；真正麻烦的是 shape 还能对上、服务也能启动，但语义空间已经漂移，导致离线指标和线上行为同时变差。

一个稳妥的平台策略，是把版本绑定从“字符串匹配”提升到“元数据契约”。
adapter metadata 至少要记录：

- `base_model_id`
- `base_model_hash` 或 checkpoint digest
- tokenizer 版本
- chat template 版本
- target modules / adapter schema
- 训练时间、训练代码版本、评测报告版本

其中 `base_model_hash` 很关键，因为仅靠 `model_name=v1` 这样的标签往往不足以区分“同名但内容已更新”的情况。
registry 在注册或加载前，应该优先校验 hash 是否精确匹配；只要 base hash 变了，就默认进入重新训练或重新验收流程，而不是直接放行。

在实际升级中，更好的做法通常不是原地替换旧 base，而是让新旧 base 并存一段时间。
例如保留 `base-v1 + adapter-v1-*` 这一整套服务池，同时新建 `base-v2 + adapter-v2-*` 的训练与服务链路。
这样平台就可以做灰度升级和流量分割：一部分租户继续走旧 base，另一部分租户或实验桶切到新 base；同一个业务也可以先在 shadow / canary 流量上验证 `adapter-v2`，观察效果后再扩大比例。
这种分流策略的价值在于，升级 base model 不是只切一份大权重，而是切换一整套与之绑定的 adapter 生态。
只有把新旧 base 并存、adapter 重训、路由分流和回滚预案一起设计，版本升级才不会变成一次高风险的整池替换。

### 10c.6 Adapter Registry、兼容性约束与 A/B 测试

adapter 不能只当一个小文件看待，它必须和运行它所需的上下文一起管理。

#### 10c.6.1 一个 adapter 至少要绑定什么

| 要管理的对象 | 为什么重要 |
|--------------|------------|
| Base model 版本 | base 一变，adapter 可能立刻失效 |
| Tokenizer / chat template | 输入格式变了，行为可能明显漂移 |
| Adapter 格式与目标模块 | 层名、shape、target modules 不匹配会直接加载失败 |
| 训练配置与数据版本 | 没有元数据就难以复现 |
| 评测结果 | 方便做上线门禁和回归判断 |
| 租户与权限信息 | 防止路由到不该访问的 adapter |

换句话说，registry 里登记的不是：

- “一个 `.safetensors` 文件”

而应该是：

- “这个 adapter 能在哪个 base model、哪种模板、哪种服务引擎上被谁以什么状态使用”

#### 10c.6.2 一个稳妥的兼容性检查

在允许 adapter 进入 deployable 状态前，平台至少应校验：

- `base_model_id` 是否完全匹配
- target modules 和层名是否存在
- tokenizer / chat template 是否在允许范围内
- serving engine 是否支持该 adapter 格式
- 关键离线评测是否通过
- 当前租户是否有权把它挂到指定路由上

这里最容易被忽略的，不是“有没有文件”，而是：

- **文件和服务环境是不是同一套契约**

#### 10c.6.3 一个安全的 A/B 路径

adapter 的发布最好不要一步切全量，而应经过：

| 阶段 | 目的 |
|------|------|
| 离线评测 | 先挡住明显退化版本 |
| Shadow / Replay | 看真实流量下的输出差异，不影响用户 |
| Canary | 小流量真实放量，观察错误率和延迟 |
| 扩量 | 逐步从 5% / 10% 提升到更高比例 |
| 全量 / 回滚 | 达标则放量，不达标立刻回滚 |

当 base model 升级时，平台应默认假设：

> 现有 adapter 很可能需要重新训练或重新验收，而不是直接“兼容沿用”。

### 10c.7 观测指标、容量规划与典型失败模式

fine-tuning 和 multi-adapter 系统要观测的，不只是 GPU 利用率。

#### 10c.7.1 训练侧更关心什么

| 指标 | 为什么重要 |
|------|------------|
| 作业等待时间 | 决定 FTaaS 的用户体验 |
| 作业成功率 / 重试率 | 反映平台稳定性 |
| tokens/s 或 samples/s | 判断训练吞吐是否异常 |
| 单作业显存峰值 | 帮助做 GPU 档位与 admission control |
| checkpoint / artifact 上传耗时 | 经常是短作业的隐藏瓶颈 |
| eval pass rate | 防止“大量训练成功但上线不可用” |

#### 10c.7.2 服务侧更关心什么

| 指标 | 为什么重要 |
|------|------------|
| adapter load latency P50 / P95 | 决定切换体验和冷启动抖动 |
| adapter cache hit rate | 决定显存与 TTFT 的稳定性 |
| route miss / permission reject | 反映路由规则和权限是否健康 |
| TTFT / tokens per second | 观察挂载 adapter 后的真实服务代价 |
| OOM / eviction rate | 直接反映缓存策略是否过激 |
| 按 adapter 维度的错误率 | 方便发现坏版本或坏租户 |

#### 10c.7.3 常见失败模式

| 失败模式 | 典型表现 | 更根本的问题 |
|----------|----------|--------------|
| Base model 不匹配 | adapter 加载失败或效果异常 | 版本绑定不严 |
| Tokenizer / template 变化 | 线上输出风格突变 | 输入契约未被登记 |
| target modules 不匹配 | 启动时报 shape / key 错误 | 训练与服务环境不一致 |
| 热点 adapter 频繁被淘汰 | 延迟剧烈波动 | cache 策略没有考虑热度分布 |
| 训练成功但 registry 元数据缺失 | 无法上线或无法回滚 | 制品流转链路不完整 |
| 路由越权 | 请求拿到错误 adapter | 多租户隔离没进入控制面 |

这些故障说明一个共同事实：

> adapter 系统的问题，很少只是“模型训练得好不好”，更多是“训练、制品、路由、缓存、权限有没有被当成同一条链路设计”。

### 10c.8 工程建议

- 资源队列上明确区分“全量训练”和“fine-tuning”任务
- adapter 作为一等制品进入 registry，而不是只存训练输出目录
- 在训练完成和允许发布之间插入兼容性校验与离线评测
- 先把 base model、tokenizer、template 的版本绑定做严，再谈 Multi-LoRA Serving
- 在生产前压测 adapter 热加载 / 热卸载延迟，而不只是看单次训练是否成功
- 对高并发场景，优先设计 adapter 缓存与淘汰策略，而不是只关注单次训练成本
- 让多租户配额、权限和路由进入控制面，否则 adapter 服务很容易变成“能跑但不可治理”

#### 本章涉及的常见工具

| 概念 | 常见工具 / 命令 | 备注 |
|------|-----------------|------|
| 参数高效微调 | PEFT | Hugging Face 生态的统一入口 |
| 量化微调 | bitsandbytes、QLoRA | 常用于降低单卡显存要求 |
| 高效 LoRA 训练 | Unsloth | 面向更快的单机 / 单卡 LoRA 训练 |
| Multi-LoRA Serving | vLLM Multi-LoRA、LoRAX | 关注热加载与共享 base model |
| 制品管理 | MLflow、Weights & Biases、自建 registry | 关键是版本绑定和可追踪性 |

---

## 本章小结

| 主题 | 关键点 |
|------|--------|
| Fine-tuning 本质 | 高频、短任务、多租户，与 pretraining 完全不同 |
| LoRA / QLoRA | 通过冻结或量化 base model 显著降低训练成本 |
| FTaaS | 重点是队列、配额、状态机、制品管理，而不是极限扩展效率 |
| Multi-LoRA Serving | 通过共享 base model 降成本，但会引入缓存、路由和权限复杂度 |
| 兼容性约束 | adapter 必须和 base model、tokenizer、模板、评测结果一起管理 |
| 观测重点 | 训练看周转与成功率，服务看加载延迟、缓存命中和路由正确性 |

---

## 练习题

1. 为什么 fine-tuning 任务不应该和 pretraining 任务共用同一资源队列？
2. LoRA 和 QLoRA 的平台意义分别是什么？它们分别改变了训练侧和服务侧的哪些成本结构？
3. 从平台角度看，为什么说 adapter 是“一等制品”，而不是训练脚本顺手导出的一个小文件？
4. 画出一个 FTaaS 的任务状态机，并解释为什么 `training succeeded`、`evaluation passed`、`deployable` 必须是三个不同状态。
5. 训练完成后，为什么还要插入自动评测、兼容性检查、registry 注册和服务热加载，而不是直接把 adapter 交给业务方使用？
6. 如果某个 FTaaS 任务因为节点重启而失败，控制面应如何区分“自动重试”与“直接失败”的条件？
7. 请基于公式 `总显存 ≈ base_model_size + N × adapter_size + concurrent_requests × kv_cache_per_request`，说明在 Multi-LoRA 场景下为什么只盯 adapter 文件大小会低估显存风险。
8. 假设一个 base model 常驻占 18 GB，单个 adapter 为 120 MB，GPU 热层放 40 个 adapter，同时有 24 个并发请求、每个请求平均消耗 180 MB KV cache。粗略估算总显存占用，并讨论是否还需要额外安全余量。
9. Multi-LoRA 中 adapter 数量上限为什么不只由显存决定，还会被切换延迟和路由复杂度约束？
10. 设计一个 adapter 热加载 / 热卸载策略，说明 LRU 驱逐与预加载分别解决什么问题，各自可能带来什么副作用。
11. 为什么 base model 升级后，已有 LoRA adapter 往往需要重新训练或重新验收？请从“权重空间绑定”角度解释。
12. 如果平台要做 base model 灰度升级，为什么更稳妥的方式是让新旧 base 并存，而不是直接原地替换？请说明流量分割和回滚策略。
13. 设计一个 adapter A/B 测试流程，说明离线评测、shadow、canary、扩量和回滚各自的作用。
14. 当 registry 里积累了数千个 adapter 后，你会如何设计 adapter 清理策略？请同时考虑审计留存、热度分布、租户权限和对象存储成本。

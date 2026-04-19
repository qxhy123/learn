# TaskCore + TaskCLI Part 4 Runtime Upgrade

## 从 Part 3 基线出发：项目当时已经稳了什么

Part 3 的 `TaskCore + TaskCLI v1` 已经把项目从单 executable 推进成一个真实 Swift package：

- `TaskCore` 负责领域模型与核心状态变化
- `TaskCLI` 负责命令入口与文本输出
- XCTest 直接锁定 core behavior
- CLI 仍然建立在同步、内存态、seeded store 的教学起点上

这条基线非常重要，因为 Part 4 不是推翻它，而是在这套边界上做 runtime upgrade。换句话说，Part 3 解决的是 package engineering；Part 4 解决的是 runtime engineering。

## Part 4 真正改变了什么

这一部分没有把项目改名，也没有把读者带去框架旅游，而是把同一条 `TaskCore + TaskCLI` 主线推进到更现代的 Swift 工程判断上。

### 1. 从同步命令路径，升级到异步运行时路径

Part 3 的 CLI 心智是：

- 拿到参数
- 对一个内存中的 `TaskStore` 做同步修改
- 立即输出结果

Part 4 则要求读者开始按 `async` / `await` 的方式思考：

- 任务状态可能需要异步加载
- 状态修改后可能需要异步保存
- “命令完成”不再只是内存改好了，而是运行时路径真正完成了加载、提交或失败处理

### 2. 从“能异步调用”，升级到“共享状态有隔离边界”

只要项目开始出现 load / mutate / save 这样的异步链路，共享可变状态就成为现实风险。Part 4 因此把 Actor、隔离和 `Sendable` 放到主线上，要求读者能判断：

- 哪些状态应该被 actor 保护
- 哪些值适合以 snapshot 形式跨边界传递
- 为什么 `TaskStore` 这类 value-oriented core model 会让并发设计更稳

### 3. 从“类和 struct 的语法区别”，升级到 ownership 与 lifetime 判断

Part 2 讲过值与引用的基础区别；Part 4 则把它推进到 runtime 后果：

- repository、coordinator、后台任务和闭包会怎样延长生命周期
- 哪些对象应该拥有资源，哪些只该传值
- cancel、`deinit`、`defer` 等清理策略为什么会影响可靠性

这让读者开始把 ARC 看成工程边界问题，而不只是记忆 `weak` / `strong` 规则。

### 4. 从“项目还小所以先别管性能”，升级到复制与测量心智

Part 4 把 Swift 性能放回语义背景里理解：

- `TaskStore`、`[Task]`、字符串渲染和 actor hop 都有成本
- 复制不总是坏事，很多 snapshot 正是并发安全与边界清晰的代价
- 真正稳的做法是先识别 cost surface，再用 representative workload 测量，而不是靠感觉乱优化

### 5. 从“有错误类型”，升级到可解释的 failure surface

Part 3 的 `TaskStoreError` 主要覆盖领域失败；Part 4 则要求读者继续扩展 runtime contract：

- 输入层失败
- 领域层失败
- 持久化 / 运行时失败
- cancellation 与 partial completion

这意味着项目的“成功”定义被收紧了：不仅要修改对，还要说明何时真正提交、失败后留下什么状态、取消是否被区分对待。

## 并发、性能与可靠性现在怎样影响项目

经过 Part 4，`TaskCore + TaskCLI` 不再只是“一个有测试的命令行包”，而是开始具备现代 Swift runtime 项目的核心判断力。

### 并发影响

- `TaskCore` 依然负责领域规则，但运行时协调不再适合散在 CLI 入口里
- runtime boundary 需要承认挂起点、隔离边界和可发送值
- CLI 不应再假设自己面对的是无等待、无竞争的同步世界

### 性能影响

- 值语义、快照、字符串构建、全量编码与 actor hop 都会留下成本
- 性能讨论必须与边界设计、复制语义和 workload 一起看
- 优化优先级应由测量决定，而不是由语言偏见决定

### 可靠性影响

- CLI 输出需要更准确地区分输入错误、领域错误、运行时错误和取消
- “成功”意味着更严格的提交语义，而不只是内存状态改过
- 失败和取消路径同样需要被定义、测试和文档化

## 这一里程碑为什么重要

Part 4 的价值，不在于让项目突然拥有很多新功能，而在于让读者开始把 `TaskCore + TaskCLI` 当成一个真实运行中的 Swift 系统来判断。

经过这一部分之后，读者应能更清楚地回答：

- 为什么 `async` / `await` 首先是在暴露挂起点，而不是在炫并行
- 为什么 Actor 和 `Sendable` 会直接影响项目边界
- 为什么 ARC、ownership、复制和性能是连在一起的
- 为什么可靠性不是“加几个 catch”，而是定义 failure surface 和 runtime contract

这就是 Part 4 相对 Part 3 的真正升级：项目不只是工程结构更清楚了，运行时判断也开始成熟了。

## 对下一阶段的交接

接下来无论进入 `TaskFlow`，还是继续扩展 `TaskCore` 的其他客户端，新的表面都不应该站在一个“只有语法正确”的 core 上，而应站在一个：

- 懂异步与隔离
- 懂 ownership 与资源清理
- 懂复制成本与测量
- 懂 cancellation 与 failure surface

的共享核心之上。

这正是 Part 4 作为 runtime upgrade 的意义。它没有换项目线，却显著抬高了读者理解和扩展这条项目线的工程上限。

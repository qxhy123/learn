# Part 3 综合实验：把 `TaskCore + TaskCLI v1` 做成真的工程表面

## 对应部分与项目阶段

- 对应部分：Part 3 `part3-packages-testing-and-cli-engineering`
- 对应项目阶段：`TaskCore + TaskCLI v1`
- 关联章节：第 11 章到第 15 章

Part 3 的任务不是“继续给 CLI 加命令”，而是把你前面做出的建模判断，沉淀成一个可构建、可测试、可维护的 Swift package。你在这里要练的，是 package boundary、target responsibility、XCTest surface、CLI 组织方式，而不是“文件变多所以看起来更专业”。

## 使用方式

建议你直接围绕 `swift-tutorial/projects/taskcore-taskcli/starter` 思考，但把这份 lab 当成一次工程审查：

- 哪些逻辑必须进 `TaskCore`
- 哪些逻辑必须留在 `TaskCLI`
- 哪些行为必须先被测试锁住
- 哪些“工程感操作”其实只是噪音

## Integrated Exercises

### 综合练习 1：重新画出 package 责任图

请先不写代码，只画出 `TaskCore + TaskCLI v1` 的最小责任分布：

- `TaskCore`：领域模型、核心状态变换、可测试业务规则
- `TaskCLI`：参数读取、命令解释、文本输出、错误翻译
- `Tests`：优先锁定 core behavior，而不是先做 CLI 快照测试狂欢

然后完成一个最小 package 设计说明，至少回答：

- 为什么 `TaskCore` 不该依赖 CLI 文本。
- 为什么 `TaskCLI` 不该偷偷拥有领域规则。
- 为什么这一步叫 `TaskCore + TaskCLI v1`，而不是“最终架构”。

### 综合练习 2：为核心行为补齐 XCTest 闭环

至少写出这几类测试：

- 新增任务成功
- 重复任务失败
- 标记完成成功
- 找不到任务时失败

每个测试都要写清楚：

- 输入状态
- 调用动作
- 预期结果

附加要求：

- 测试命名要直接表达行为，不要命名成 `test1`、`testHappyPath` 这种弱标签。
- 如果一个测试失败，你能从名字看出 contract 被破坏在哪。

### 综合练习 3：把 CLI 从“入口堆逻辑”改成“翻译层”

实现或重构一个最小流程：

1. parse command
2. call core
3. render output
4. map error to user-facing message

重点是分清三类语言：

- core language：领域语义
- CLI language：命令和 usage
- test language：断言行为

## Debugging Tasks

### 调试任务 1：测试明明失败，CLI 却看起来正常

常见弱设计：

- `TaskCLI` 捕获所有错误后只打印“something went wrong”
- 测试只断言进程没崩，而不检查核心行为

你的任务：

- 找出哪里把真实失败面吞掉了。
- 说明为什么 Part 3 应优先测 core，而不是先被 CLI 文本牵着走。
- 给出一个更强的错误传播路径。

### 调试任务 2：`main.swift` 越拆越乱

观察这种症状：

- `main.swift` 变成调用多个 helper 的调度中心
- helper 又彼此隐式共享状态
- 命令解析、执行、渲染互相耦合

你需要：

- 指出哪些职责分离是假的。
- 重新划出 parse / execute / render 的边界。
- 判断某些 helper 是否应该删除而不是继续保留。

## Refactoring / Design Tasks

### 设计任务 1：为目标边界做一次逆向审查

请逐个检查当前类型，并回答：

- 如果把它放进 `TaskCore`，是否会把 CLI 细节带进核心？
- 如果把它留在 `TaskCLI`，是否会把领域规则挤回入口层？
- 如果它同时被两个 target 需要，是否说明它应成为共享 core contract？

把结果写成一个清单。这个练习的价值在于：你要学会用依赖方向判断边界，而不是用目录名判断边界。

### 设计任务 2：为 CLI usage 建立可维护格式

设计一个最小 usage / help 方案，要求：

- `list`、`add`、`done` 的说明风格一致
- 出错时能指出用户错在哪，不只重复 usage
- 不引入完整命令框架

写出一段说明：为什么 Part 3 先自己设计这个表面，比立刻依赖第三方 CLI 库更有教学价值。

## Challenge Tasks

### 挑战 1：增加一个存储接缝，但不要急着做完整持久化系统

设计一个最小 `TaskStorePersistence` 或等价概念，让 core 可以被不同存储策略接上。

约束：

- 当前阶段只需要表达“边界”，不需要实现完整数据库。
- 优先让测试可替换，而不是先追求生产能力。
- 如果 protocol 已经足够，就不要再套 protocol + type erasure + factory。

### 挑战 2：加入 `import` 命令的 CLI 骨架

新命令：`import <path>`

要求：

- `TaskCLI` 负责参数与路径解释。
- `TaskCore` 负责批量导入后的语义校验。
- 错误要区分“路径问题”“格式问题”“领域规则问题”。

这个挑战真正练的是边界，不是文件 I/O 本身。

## 退出标准

完成这份 lab 后，你应该能明确说出：

- 为什么 Part 3 的项目阶段名必须是 `TaskCore + TaskCLI v1`。
- 为什么 Swift Package Manager 在这里是工程边界工具，不只是构建命令。
- 为什么一个 package 真正变强，靠的是责任清楚和测试闭环，而不是 target 数量。

## 复盘问题

1. 你最想继续塞回 `TaskCLI` 的那段逻辑，为什么其实更像 core responsibility？
2. 你的测试有没有验证 contract，还是只在验证“今天输出恰好长这样”？
3. 如果 Part 4 要开始谈运行时与可靠性，你现在的 package 表面已经准备好了哪些条件？

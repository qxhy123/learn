# Part 8 综合实验：完成 `Capstone`，把整套 Swift 教程收束成一个可信系统

## 对应部分与项目阶段

- 对应部分：Part 8 `part8-capstone-and-next-steps`
- 对应项目阶段：`Capstone`
- 关联章节：第 33 章到第 36 章

Part 8 的综合实验不再是单点强化，而是毕业层的系统收束。你现在面对的不只是某一章的语言主题，而是整套教程主线：`TaskCLI Lite -> TaskCore + TaskCLI -> TaskFlow -> Capstone`。这份 lab 的目标，是让你证明自己已经能把 CLI、共享核心、SwiftUI 客户端和下一步成长路线统一成一个连贯判断。

## 使用方式

把这份 lab 当成一次真正的 graduation review。不要急着一上来就“重构到满意”，而要按顺序完成：

1. 盘点现状
2. 定义成功标准
3. 加固 core 与 CLI
4. 加固 `TaskFlow`
5. 统一验证链
6. 写出下一步路线

## Integrated Exercises

### 综合练习 1：写一份 capstone inventory

请对三条表面分别盘点：

- `TaskCLI`
- `TaskCore`
- `TaskFlow`

每一项至少回答：

- 当前最强的 contract 是什么
- 当前最脆的边界是什么
- 哪些历史设计在当时合理，但现在应该升级

输出格式建议：

`Surface` / `Strong Contract` / `Weak Edge` / `Capstone Action`

### 综合练习 2：定义 Capstone minimum success bar

为本次 `Capstone` 写出最小成功标准，至少覆盖：

- core contract 清楚
- CLI 只是翻译层
- `TaskFlow` 不再补核心空白
- 关键失败面可验证
- 至少一条端到端场景能被证明

要求：

- 每一条都必须可验证，不接受“更优雅”“更高级”这种主观措辞。
- 明确哪些内容故意不纳入本次 capstone。

### 综合练习 3：做一次跨客户端统一演练

选一个真实任务流，例如“新增任务并完成，再在 UI 中看到一致结果”，要求你同时说明：

- `TaskCore` 暴露了什么 contract
- `TaskCLI` 如何翻译
- `TaskFlow` 如何消费
- 测试 / preview / 手动验证如何证明这条链成立

这个练习不是在追求“每层都做一次同样的事”，而是在验证共享核心真的成为多客户端系统的中心。

## Debugging Tasks

### 调试任务 1：CLI 和 UI 对同一失败给出不同语义

症状：

- CLI 说“task not found”
- UI 显示成“save failed”
- 实际两者都在处理同一个 domain failure

你的任务：

- 找出 failure naming 在哪一层开始分叉。
- 判断哪些失败应由 core 命名，哪些应由客户端翻译。
- 修到同一语义在不同客户端有不同表达，但不再有不同含义。

### 调试任务 2：Capstone 之后测试更多了，但系统反而更不敢改

这通常说明：

- 测试过度绑定实现细节
- 多层都在重复测同一个表象
- 缺少关键 contract 级测试，只有大量脆弱快照

请你：

- 标出三类应保留的测试
- 标出三类应删除、合并或降级的测试
- 说明为什么“更多测试”不自动等于“更高可信度”

## Refactoring / Design Tasks

### 设计任务 1：做一次 Capstone 级 contract 收紧

选择一个共享 contract，例如：

- task snapshot
- task mutation result
- domain error
- filter / sort semantics

要求：

- 收紧命名和返回值
- 删除至少一个模糊中间层
- 写出影响面：CLI、`TaskFlow`、tests 各自怎么调整

### 设计任务 2：写一页“这套系统现在长什么样”

请用教程语言写一页简短设计说明，覆盖：

- 三条表面各自职责
- 共享核心位置
- 运行时与失败面位置
- 为什么这套形状足以支持下一步深入，而不是又回到大杂烩

这是对外说明，也是对自己做架构压缩。

## Challenge Tasks

### 挑战 1：设计一条毕业后的延伸功能，但必须先证明不该现在做

可以任选：

- cloud sync
- reminders / notifications
- widgets
- server-backed tasks
- collaboration

要求：

- 先写为什么当前教程故意不做它。
- 再写如果毕业后要做，第一步应该从哪个 contract 开始，而不是从哪个页面或 API 开始。

### 挑战 2：做一次“端到端但不失控”的综合演练

设计一个最小 E2E 场景，包含：

- core 行为
- CLI 调用
- `TaskFlow` 状态消费
- 错误或恢复路径

重点：

- 不求覆盖所有功能
- 只求证明共享 contract、客户端翻译和验证链已经闭环

## 退出标准

完成这份 lab 后，你至少应能清楚说明：

- 为什么 Part 8 的项目阶段名就是 `Capstone`。
- 为什么毕业层的价值不在“再学几个 API”，而在把整套教程能力压成可信系统判断。
- 为什么下一步路线应该按方向深入，而不是继续随机追热点。

## 复盘问题

1. 整套教程里，哪一个共享 contract 是你现在最有把握维护的？
2. 如果要把本教程成果带进真实工作项目，你最先复用的是哪种判断，而不是哪段代码？
3. 你现在最需要继续深挖的方向是并发、SwiftUI、系统 API 还是包设计？为什么？

# Part 6 综合实验：让 `TaskFlow` 的数据流和架构真正能增长

## 对应部分与项目阶段

- 对应部分：Part 6 `part6-swiftui-dataflow-and-app-architecture`
- 对应项目阶段：`TaskFlow` 架构强化阶段
- 关联章节：第 25 章到第 28 章

Part 5 让 `TaskFlow v1` 站住，Part 6 则要求它开始像一个真正应用那样处理 app state、持久化、异步更新、preview 和测试。这里的核心问题不再是“会不会写 SwiftUI 组件”，而是“当应用开始增长时，数据从哪里来、往哪里去、谁负责解释、谁负责持久化”。

## 使用方式

完成这份 lab 时，请把每个任务都写成“状态流图”而不是“页面任务单”。你需要盯住的是：

- app state 与 feature state 的边界
- SwiftUI 客户端与 `TaskCore` 的接口
- preview / test double / persistence 之间的关系

## Integrated Exercises

### 综合练习 1：为应用状态画出单向数据流

至少覆盖这几条路径：

- 初次加载任务
- 新增任务
- 完成任务
- 失败提示
- 刷新或重载

要求：

- 画出状态变化顺序，不只是页面跳转顺序。
- 明确哪些状态属于 app-level，哪些只属于单个 feature。
- 说明为什么 `TaskFlow` 仍是 `TaskCore` 的 client，而不是平行核心。

### 综合练习 2：给持久化接缝一个可测试表面

设计一个最小 persistence 接口或 adapter，让 `TaskFlow` 能在不污染 View 的前提下加载与保存任务。

要求：

- 持久化细节不要直接穿进 View。
- 测试和 preview 能替换成假实现。
- 错误能沿着 app model 回到 UI，而不是散落在 button action 里。

### 综合练习 3：把异步更新放回 app model

请实现一个真实的异步场景，例如：

- 下拉刷新
- 启动时加载
- 批量同步

要求：

- 异步工作先进入 model / store / coordinator，而不是 View 直接裸开任务。
- UI 能清楚表现 loading / success / failure。
- 写出说明：为什么 `.task` 只是触发点，不是业务逻辑容器。

## Debugging Tasks

### 调试任务 1：Preview 能显示，真机一跑就乱

常见原因：

- preview 数据路径和真机路径完全不同
- preview 直接构造 View，而真实应用还要经过 app model / dependency
- 某些状态只在 preview 人工伪造，真实流里根本不会出现

你的任务：

- 指出 preview 为什么应是结构检查器，而不是平行实现。
- 给出一个更强的 preview 依赖组织方式。

### 调试任务 2：持久化成功，但列表没有同步更新

这里通常暴露的不是“磁盘 API 不稳定”，而是：

- app state 没更新
- 更新后没有发布
- View 读的不是唯一真源（single source of truth）

请明确：

- 真正的数据源在哪里。
- 为什么“保存成功”不自动等于“UI 一致”。

## Refactoring / Design Tasks

### 设计任务 1：做一次 feature boundary 审查

把当前 `TaskFlow` 按 feature 重新看一遍，例如：

- task list
- task editor
- settings / sync / filter

要求：

- 分析哪些边界应按 feature 切，哪些应按共享 app state 切。
- 删除“为了整洁而层层转发”的空壳类型。
- 说明这种切分如何让未来功能增长更稳。

### 设计任务 2：统一 preview、test double 与真实依赖注入

做一张对照表：

`Context` / `Data Source` / `Why`

至少覆盖：

- App runtime
- Preview
- Unit test
- UI-like integration test

目标不是完全统一实现，而是统一契约和进入方式。

## Challenge Tasks

### 挑战 1：设计离线恢复路径

假设应用在保存到一半时被杀掉或磁盘写入失败。请设计：

- 启动后的恢复策略
- 用户可见反馈
- 如何避免 UI 看起来成功、底层其实失败

这个挑战重点是 recovery 语义，不是具体文件格式。

### 挑战 2：新增跨界面共享筛选与排序

需求：

- 列表页、统计页、搜索页共享同一筛选条件
- 不允许每页各自维护一份互不一致的筛选状态

你需要说明：

- 这属于 app state 还是 feature state
- 状态改变后如何传播
- 为什么这题一旦做错，`TaskFlow` 会迅速退化成“多页面共享全局混乱”

## 退出标准

完成这份 lab 后，你应该能清楚回答：

- 为什么 Part 6 的 `TaskFlow` 已经不是“几个 SwiftUI 页面”，而是一条应用架构线。
- 为什么 app data flow 比 View 数量更能决定系统是否可增长。
- 为什么 preview、testing、persistence 会反过来塑造 architecture。

## 复盘问题

1. 你系统里是否已经有且只有一个真正的任务真源？
2. 你哪里最容易把异步副作用写回 View？
3. 如果 Part 7 要开始统一三条项目线，你现在的 `TaskFlow` 哪些 contract 已经足够清楚？

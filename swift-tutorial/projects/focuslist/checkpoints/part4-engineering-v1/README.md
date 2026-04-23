# Part 4 Checkpoint：工程化 v1

## 这一阶段的目标状态

Part 4 结束时，`FocusList` 应该已经不只是“功能和结构都不错的 App”，而是一条拥有清楚验证面和依赖边界的工程线。

## 你应该已经具备的工程能力

- `SwiftPM` 工作区和 target 角色清楚
- 关键共享规则有行为测试
- `FocusCore`、`FocusListApp`、`focusctl` 依赖方向稳定
- CLI 可以复用共享核心，而不是复制逻辑

## 重点检查点

至少确认：

- 测试命名描述的是行为
- `FocusCore` 不反向依赖 UI
- CLI 新能力建立在核心 API 之上

## 手动检查路径

1. 跑测试
2. 运行 `focusctl`
3. 对比 App 和 CLI 对同一规则的表现是否一致
4. 检查新增功能是否落在正确 target

## 如果这阶段没做稳，会有什么症状

- 测试只围着实现细节转
- CLI 为了工作复制了一套业务逻辑
- 共享核心开始依赖页面状态或 SwiftUI

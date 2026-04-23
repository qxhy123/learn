# Part 1 Checkpoint：FocusList v1

## 这一阶段应该交付什么

Part 1 结束时，`FocusList` 还不需要复杂功能，但必须像一个真正能继续长大的产品起点。最低目标是：

- 应用能启动
- 有稳定的根视图和导航骨架
- `Inbox` 能展示和新增任务
- `Settings` 已经有明确入口

## 你应该在代码里看到什么

- `FocusListApp` 持有应用级初始状态
- `FocusListRootView` 建立侧栏和详情结构
- `InboxView` 里有局部草稿状态和最小新增流
- `FocusStore` 提供最基础的新增与切换完成行为

## 手动检查路径

1. 启动 App
2. 进入 `Inbox`
3. 新增一条任务
4. 切换它的完成状态
5. 进入 `Projects` 和 `Settings`

如果这条路径还不顺，说明 Part 1 的产品骨架还没真正成立。

## 进入下一阶段前要明确什么

你应该已经能解释：

- 为什么草稿文本留在局部 `@State`
- 为什么 `FocusStore` 现在还很轻
- 为什么侧栏/详情结构现在就值得建立

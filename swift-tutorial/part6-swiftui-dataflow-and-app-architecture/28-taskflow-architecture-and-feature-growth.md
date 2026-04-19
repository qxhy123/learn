# 第28章：`TaskFlow` 架构与功能增长

> Part 6 到这里要完成一件真正重要的收束工作：证明 `TaskFlow` 不是“会动的 SwiftUI demo”，而是一条能够继续增长、仍然站在共享核心之上的 app architecture 线。前面几章分别处理了 app state、持久化集成、异步更新、预览和测试；本章要把它们统一成一个能继续扩展的架构判断。

## 为什么这一章现在出现

如果没有这一章，Part 6 很容易留下一个隐患：你学了很多关于 SwiftUI app 的局部正确做法，但还不知道当功能继续增加时，系统要靠什么原则避免再次塌回“哪里能写就写哪里”的混乱状态。

而任务管理领域天然会继续长大：

- 今天只有列表、详情、创建
- 明天可能有标签、优先级、筛选保存、批量操作
- 后天可能有同步、冲突解决、统计面板、跨设备状态

若没有架构判断，feature 增长就只会不断往现有 screen 和 model 上叠逻辑。

## 从一个较弱起点开始：随着 feature 变多，持续堆大 screen 和大 model

这是几乎所有 SwiftUI 新项目都会遇到的路径：

- 首页最开始只有列表，于是所有逻辑都写在 `TaskHomeScreen`
- 加入创建功能后，screen 再多几个状态字段和几个 async 方法
- 再加入详情、筛选、错误提示、导航路径后，model 变成巨大协调器

代码大致会朝这个方向滑：

```swift
@Observable
final class TaskHomeModel {
    var tasks: [Task] = []
    var selectedTask: Task?
    var draftTitle = ""
    var filter: TaskFilter = .all
    var isLoading = false
    var errorMessage: String?

    func load() async { ... }
    func createTask() async { ... }
    func markDone(_ task: Task) async { ... }
    func delete(_ task: Task) async { ... }
    func navigateToStats() { ... }
    func presentSettings() { ... }
}
```

这类大 model 最危险的地方不是“文件长”，而是 feature 边界开始消失。列表、编辑、导航、过滤、同步、错误处理全部缠在一起后，任何一个改动都会穿透整块 app 表面。

## 更强的方向：按 feature 和数据职责共同切分

Part 6 结束时，你应该开始形成这样的架构判断：

- 共享核心按领域职责稳定存在
- 数据边界按 load/save/sync 等运行时职责组织
- app 层按 feature 划分 screen state 与交互
- View 保持声明式消费状态

对于 `TaskFlow`，一种更稳的增长路线会是：

- `TaskCore`：任务领域模型与核心规则
- `TaskFlowData` 或 runtime layer：repository、persistence adapter、sync coordination
- feature models：`TaskListModel`、`TaskDetailModel`、`TaskComposerModel`
- app coordinator state：导航路径、全局筛选、跨 feature 同步触发
- SwiftUI views：screen 与 reusable subviews

这不是为了“像架构图”，而是为了保证 feature 增长时仍能找到明确落点。

## 功能增长时，优先新增 feature 边界，而不是膨胀共享核心

另一个常见误区是：既然强调共享核心，就把所有 app 相关逻辑也不断塞进 `TaskCore`。这同样不对。

共享核心最适合沉淀的是：

- 跨客户端共享的领域模型
- 跨客户端共享的核心规则
- 明确不依赖某个 UI 客户端的业务语义

而下面这些更像 `TaskFlow` app line 的职责：

- 某个列表 screen 的筛选展示状态
- SwiftUI 导航路径组织
- preview 与 app-specific feature composition

真正成熟的复用不是“什么都放共享层”，而是知道哪些东西值得共享，哪些东西应留在客户端边界。

## 让 CLI 线和 SwiftUI 线继续并存，而不是相互吞掉

本任务有一个需要刻意守住的教学目标：`TaskFlow` 是共享任务核心的 client，不是来取代 `TaskCLI` 线。

这意味着你在做架构判断时，应能明确说出：

- CLI 线擅长命令组织、文本输出、脚本化与自动化入口
- SwiftUI 线擅长列表、导航、交互反馈与 app 生命周期
- 两者复用同一任务领域时，系统整体反而更强

也就是说，课程现在不是从“CLI 阶段升级到 UI 阶段然后抛弃旧线”，而是在同一共享核心上长出多个 client surface。这个认识非常关键，因为它决定你未来会不会写出真正可复用的 Swift 工程。

## Feature growth 时要警惕的四种膨胀

随着 `TaskFlow` 继续扩展，最常见的架构退化通常来自这四种膨胀：

### 1. 状态膨胀

所有状态都被抬到最顶层，feature 失去局部自治能力。

### 2. 责任膨胀

screen model 同时承担导航、存储、领域规则、错误映射和样式判断。

### 3. 依赖膨胀

每个 feature 都直接碰 repository、存储和全局 coordinator，导致依赖图越来越乱。

### 4. 文档膨胀

项目里程碑不再说明阶段差异，最终让读者不知道 starter、part5 milestone、part6 architecture milestone 与 final 的关系。

本章的价值，就是让你在功能还没有爆炸前，先学会识别这些退化方向。

## `TaskFlow` 在 Part 6 结束时应达到什么架构成熟度

一个合格的 Part 6 终点，不需要把所有未来 feature 都做完，但应该能清楚说明：

- 共享核心如何继续服务多个客户端
- app data flow 和持久化边界如何配合
- feature state 如何拆分
- 预览与测试如何围绕状态流工作
- 后续新增功能准备落在哪些边界

这就是“architecture milestone”的含义。它强调的不是功能清单，而是增长路径已经从脆弱堆叠变成可解释演进。

## 双语关键词

- architecture：架构
- feature growth：功能增长
- boundary：边界
- coordinator：协调者
- shared core client：共享核心客户端
- dependency graph：依赖图
- evolution path：演进路径
- architecture milestone：架构里程碑

## 常见错误

### 1. 一旦强调共享核心，就把所有 app 逻辑都塞进去

共享核心要共享的是领域价值，不是所有客户端细节。

### 2. 功能一多，就继续给现有 screen/model 加字段和方法

这会让 feature 边界越来越模糊，最终改一处牵一片。

### 3. 认为有了 SwiftUI app 就可以自然淘汰 CLI 线

这会破坏教程主线，也会削弱“多客户端共享核心”的工程判断。

### 4. 只记录功能完成情况，不记录项目阶段结构

没有清楚的 milestone 文档，读者很难理解 starter、阶段成果和 final state 之间的连续性。

## English Recap

The goal of Part 6 is not just to add more SwiftUI features, but to give `TaskFlow` an architecture that can keep growing without replacing the shared task core. A strong design separates domain rules, data/runtime boundaries, feature state, and UI views, while keeping CLI and SwiftUI as parallel clients of the same core.

## Drills

1. 列出你认为 `TaskFlow` 下一步最可能增长的三个 feature，并分别说明它们应落在哪一层边界。
2. 用一段话解释为什么“共享核心”不等于“所有客户端逻辑都必须放进 core”。
3. 对比 `TaskCLI` 与 `TaskFlow`，说明“多客户端共享核心”为什么比“迁移后只保留一个客户端”更有工程价值。

## Project Handoff

Part 6 到这里完成时，`TaskFlow` 已经从 SwiftUI 基础 demo 进化成有明确数据流、持久化边界、异步更新策略、预览/测试入口和 feature growth 判断的 app client。接下来的高级 Swift 与综合设计部分，不会抛开这条线，而会把 CLI、共享核心和 SwiftUI client 一起放回更高层的系统设计视角中复盘。

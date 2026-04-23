# FocusList：从零到高阶的现代 Swift 教程

## 这套教程是什么

这不是一份 Swift 语法速记，也不是把 SwiftUI 组件按名字排成目录的说明书。它是一套完整的项目驱动教程：你会沿着同一个跨平台应用 `FocusList`，从第一个能跑的 SwiftUI 界面，一路走到共享核心、测试、并发、打磨和发布准备。

默认读者已经写过别的语言，但没系统学过 Swift。也就是说，这里不会用很长篇幅解释“什么是变量”，而会把重点放在 Swift 和 Apple 应用工程里那些真正会绊住你的地方：状态到底该放哪、为什么 `View` 会重算、什么时候该抽共享核心、失败面怎么处理、为什么跨平台不是把一个页面简单拉伸。

## 从这里开始

如果你今天刚打开这个目录，按这个顺序推进：

1. 读 [前言](./00-preface.md)，确认环境和学习方式
2. 进入 `Part 1`，跟着 starter 工程把第一个 `FocusList` 跑起来
3. 每完成一个 Part，就做一次 `labs/` 里的综合实验
4. 不跳过 `projects/focuslist/`，因为正文和项目线是一体的

## 六部分学习地图

### Part 1：App-first Foundations

你会先做出真正的应用壳，而不是先绕去写一堆和产品无关的练习。这里会讲 `App`、`Scene`、`WindowGroup`、`View`、`@State`、`@Bindable`、`List`、`Form` 和基础导航。目标不是“会几个控件”，而是做出第一个能浏览、能录入、能继续扩展的 `FocusList v1`。

### Part 2：Feature Growth and UI Organization

第二部分让应用从 demo 变成产品。你会加入任务分组、标签、编辑流、筛选、搜索和更清楚的界面组织。重点不只是加功能，而是训练你在 SwiftUI 里做“产品层面的结构判断”：哪些页面应该分开，哪些状态应该共享，哪些组件真的值得抽。

### Part 3：Data Modeling, Persistence, and Shared Core

当应用开始有更多数据和更长生命周期之后，我们再进入模型、持久化和共享核心。你会学到如何用 `SwiftData` 管理长期状态，怎样把查询、失败路径和产品规则放到正确边界里，以及为什么 `FocusCore` 应该在这时出现，而不是更早也不是更晚。

### Part 4：Engineering, Testing, and Modularization

这里开始把 `FocusList` 当成代码库而不是示例项目来对待。你会用 `SwiftPM` 组织共享代码、用 `Swift Testing` 锁定关键行为、给产品画清依赖方向，并且构建一个轻量命令行工具 `focusctl`，用来证明 `FocusCore` 不是只给 SwiftUI 服务的内部细节。

### Part 5：Concurrency, Reliability, and Cross-Platform Polish

真实产品不会只有同步按钮和即时成功。第五部分会处理刷新、搜索、后台任务、取消、错误反馈、批量操作，以及 `Observation` 的刷新成本。你还会把相同产品在 `iOS` 和 `macOS` 上都打磨到更合理的体验，而不是只求“两个平台都能显示”。

### Part 6：Capstone and Shipping Readiness

最后一部分是毕业层。你会回头做结构收束、测试和预览补强、无障碍检查、发布前整理和能力复盘。它的目标不是把目录读完，而是让你真的知道自己现在已经具备哪些 Swift 工程判断，下一步该往哪条线继续深入。

## FocusList 项目线

整套教程围绕一个连续项目推进：

- `FocusList`：用户真正使用的应用表面
- `FocusCore`：中后段抽出来的共享规则中心
- `focusctl`：建立在共享核心之上的轻量 CLI

这三者不是三个独立项目，而是同一个产品线的三个层面。前半段重点是把 App 做出来；中段开始讨论如何把产品规则和界面规则拆开；后半段再用测试、CLI 和验证脚本把整个工程面锁住。

## 目录怎么配合阅读

- `part*/`：主线正文
- `projects/focuslist/`：starter、checkpoints 和 final 说明
- `labs/`：每个 Part 的综合实验
- `appendix/`：环境、速查、FAQ 和答案索引
- `scripts/`：结构、内容和 starter 工程的验证脚本

你可以把这套教程理解成一门“边做产品边学 Swift”的课。正文负责解释，项目负责落地，labs 负责检验，脚本负责自证它不是一堆看起来很像教程的 Markdown。

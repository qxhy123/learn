# FocusList：从零到高阶的现代 Swift 教程

## 教程定位

这是一套全新的 Swift 教程产品。它不继承旧 `swift-tutorial` 的章节结构，也不再用 CLI 作为开场主角，而是直接把读者带进一个真实的 `iOS + macOS` 效率应用里学习 Swift。教程的核心目标不是让你背更多语法点，而是让你能把 Swift、SwiftUI、数据流、测试、并发和工程边界一起用在一个能持续演进的产品上。

## 适合谁

默认读者是已经学过其他语言、但没系统学过 Swift 的开发者。你可能写过 Python、TypeScript、Go、Java 或 Kotlin，知道函数、类型、模块、测试这些概念，但一进入 Swift 就会被 `View`、状态、Observation、SwiftData、并发和 Apple 平台约束打断原有直觉。这套教程的任务，就是把这些内容重新连成一条完整的工程路径。

## 六部分能力地图

### Part 1：App-first Foundations

先做出第一个可用的 `FocusList`。这一部分只讲当前必须知道的 Swift 和 SwiftUI：应用壳、`View`、状态、列表、表单、导航，以及它们为什么足以支撑你进入真正的 App 开发。

### Part 2：Feature Growth and UI Organization

应用开始变得像产品，而不是 demo。你会加入任务分组、标签、编辑流、筛选和搜索，并学习如何让界面结构和组件边界一起变得清晰。

### Part 3：Data Modeling, Persistence, and Shared Core

当应用复杂度开始逼迫你重想模型和存储时，我们再进入 `SwiftData`、查询边界、失败路径和共享核心 `FocusCore`。模块化在这里不是美学，而是被业务压力逼出来的决定。

### Part 4：Engineering, Testing, and Modularization

这一部分把产品推进成真正的代码库：`SwiftPM`、`Swift Testing`、功能边界、依赖关系，以及一个轻量命令行工具 `focusctl`。CLI 在这里是工程辅助面，不是主产品。

### Part 5：Concurrency, Reliability, and Cross-Platform Polish

你会处理刷新、搜索、后台任务、取消、错误反馈、批量操作和 `Observation` 成本，同时把同一个产品在 `iOS` 和 `macOS` 上都打磨到可用、稳定、可解释。

### Part 6：Capstone and Shipping Readiness

最后一部分不是简单总结，而是一次成品化收束。我们会做架构复盘、测试补强、预览和无障碍质量检查、发布前整理，以及一条清楚的后续进阶路线。

## FocusList 项目主线

整套教程围绕一个持续演进的效率应用 `FocusList` 展开。它一开始只是最小可运行的任务应用，随后逐步拥有项目、标签、筛选、搜索、持久化和跨平台体验。到了中段，教程会从 App 里抽出共享核心 `FocusCore`，再在其上补一个 `focusctl` 工具，用来讲清代码复用、测试和边界设计。项目线始终连续，读者看到的是同一个产品如何被逐步做对，而不是一串互不相干的小 demo。

## 如何学习

最推荐的走法是严格按 Part 1 到 Part 6 的顺序推进。每读完一个 Part，就进入对应的 `labs/` 做一次综合实验，把这一部分分散的能力重新拼起来。如果你已经会一些 Swift，可以跳读，但仍然建议至少完整走完 Part 1、Part 3 和 Part 4，因为这三部分决定了你能不能把 Swift 写成一个可维护的应用，而不是只会堆界面。

## 教程特色

- App-first：先进入真实产品，再补全语言与工程判断
- 单一连续项目线：`FocusList -> FocusCore -> focusctl`
- 纯中文工程讲解：API、类型名和代码保持英文，正文保持中文判断力
- 现代技术栈优先：`Swift 6`、`SwiftUI`、`Observation`、`SwiftData`、`Swift Testing`
- 产品与工程并重：界面、数据流、失败面、测试、并发、发布准备都在主线里

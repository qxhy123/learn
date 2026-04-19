# TaskCLI Lite

## 这个项目是干什么的

`TaskCLI Lite` 是 Swift 教程项目主线的第一块着陆区。它不追求“像真实产品那样完整”，而追求另一件更重要的事：把 Part 1 的 Swift 基础语义真正压到一个可运行的命令行程序里。读者在这里第一次把 `let` / `var`、`String`、`Array`、控制流、函数、`Optional`、`enum`、`struct` 这些语言点接成一个最小闭环，而不是停留在零散语法片段。

这也解释了它为什么叫 `Lite`。Part 1 的目标不是一口气造出持久化、模块边界、测试分层、复杂参数解析都齐备的 CLI，而是先让程序员建立对 Swift 语言本体的稳定直觉：数据是什么、命令如何分支、状态如何变化、输出如何组织。只要这一步稳住，Part 2 才有资格讨论更严肃的建模与类型设计。

## 它在 Parts 1-2 里如何演进

在 Part 1，`TaskCLI Lite` 保持为一个小而完整的 SwiftPM executable。你会从命令行拿到参数，根据命令做最基本的任务列表、添加、完成操作，并用 XCTest 锁住最核心的行为。这里的重点是“基础能力第一次落地”，所以代码故意保持扁平、直接、可讲解。

进入 Part 2 以后，教程不会假装 Part 1 已经拥有成熟架构，而是明确承认：这个小 CLI 已经完成了它的第一阶段使命，下一步该把被挤在一起的职责拆开。也就是说，Part 2 不是凭空发明新项目，而是把 `TaskCLI Lite` 继续推向 `TaskCore + TaskCLI` 这条更工程化的主线。

## 目录说明

- `starter/`：Part 1 读者可以直接运行和阅读的 SwiftPM 起点。
- `milestones/part1-v1.md`：Part 1 完成时，`TaskCLI Lite v1` 到底达到了什么状态。
- `final/README.md`：站在 Part 1 终点回看，这个项目当前稳定在哪里，以及它如何衔接 Part 2。

## 如何运行 starter package

先进入 starter 目录：

```bash
cd swift-tutorial/projects/task-cli-lite/starter
```

构建与测试：

```bash
swift build
swift test
```

运行最小命令：

```bash
swift run TaskCLILite list
swift run TaskCLILite add "write chapter notes"
swift run TaskCLILite done "read chapter 01"
```

当前 Part 1 版本使用的是内存中的 seed tasks，而不是磁盘持久化。这是刻意设计，不是缺漏：教程希望你先看清语言语义和程序结构，再在后续部分处理更强的工程边界。

## 你在这里应该学到什么

如果你读完 Part 1 再回来看这个目录，应该能明确说出下面几件事：

- 为什么 `TaskCLI Lite v1` 故意保持简单，而不是一开始就拆模块。
- 为什么一个最小 CLI 足够承接 Swift 基础语法，而不是“玩具例子”。
- 为什么 Part 2 的建模升级是从这里长出来的，而不是另起炉灶。

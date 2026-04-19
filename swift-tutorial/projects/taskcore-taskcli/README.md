# TaskCore + TaskCLI

## 这个项目为什么现在出现

`TaskCLI Lite` 完成了 Part 1 的使命：它把 Swift 基础语义压进一个最小可运行程序里。到了 Part 3，项目线必须进入真正的工程地带。读者已经看过类型、值语义、协议、泛型和错误建模，如果代码还继续全部挤在一个 executable target 里，这些能力就很难变成可维护的判断。

因此这一阶段把项目正式升级为 `TaskCore + TaskCLI`。这不是为了显得更“架构化”，而是为了让 Swift Package Manager、模块边界、XCTest、命令组织和后续运行时强化都能落在一个真实 package 上。

## 什么应该放进 `TaskCore`

`TaskCore` 是 library target。它承接任务领域本身，以及那些不该依赖命令行入口的稳定行为：

- 领域模型，例如 `Task`、`TaskStatus`
- 核心状态变换，例如 `TaskStore.add(title:)`、`TaskStore.markDone(title:)`
- 与 CLI 文本无关、但值得被测试锁定的业务规则
- 后面 Part 4 还会继续加强的失败面、运行时行为、存储接缝

更直接地说，`TaskCore` 关注“任务系统本身怎样工作”，而不是“命令行今天如何把它展示给用户”。

## 什么应该放进 `TaskCLI`

`TaskCLI` 是 executable target。它只负责命令行入口层的工作：

- 读取 `CommandLine.arguments`
- 解释 `list` / `add` / `done`
- 把 `TaskCore` 的行为结果组织成用户可读的 CLI 输出
- 维持当前阶段最小但清楚的 usage 文本

它不应该反过来拥有核心领域状态，也不应该把 `Task` 的真实规则藏在 `main.swift` 里。否则模块拆分就会退化成“目录变多了，但职责没变清楚”。

## 这个拆分如何连接 Part 3 和 Part 4

Part 3 的工作，是先把 package shape、module boundary、test surface 和 command organization 立住。也就是说，系统要先变成“像一个真正的 Swift package”。

Part 4 不会推翻这个拆分，而是沿着它继续强化 runtime behavior。届时我们会开始认真处理更强的失败建模、潜在的异步与 I/O 压力、可靠性判断，以及哪些状态变化必须在更严格的边界上运行。没有 Part 3 的 `TaskCore + TaskCLI` 基础，Part 4 的运行时讨论只能重新掉回单文件脚本。

## 目录说明

- `starter/`：Part 3 读者实际运行、构建和测试的 SwiftPM 起点。
- `milestones/part3-v1.md`：记录当前 package boundary 已经稳定到什么程度。
- `final/README.md`：站在 Part 3 终点看，当前版本为什么足够进入 Part 4，但又为什么还没有把运行时问题做满。

## 如何运行 starter package

```bash
cd swift-tutorial/projects/taskcore-taskcli/starter
swift build
swift test
swift run TaskCLI list
swift run TaskCLI add "write chapter 15"
swift run TaskCLI done "read chapter 11"
```

当前 starter 仍然使用内存内的 seeded tasks，而不是磁盘持久化。这不是偷懒，而是教学边界：Part 3 先把包、模块、测试和 CLI layering 立稳；Part 4 再把更强的运行时行为推上来。

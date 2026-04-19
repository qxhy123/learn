# 第30章：结果构建器、宏与 API 表面判断

> 学到这里，读者已经见过 `@Observable`、`#Preview`、SwiftUI DSL，也已经能看懂一些泛型和协议边界。现在终于可以正面讨论一个更容易被误学的话题：**高级 Swift 的“漂亮表面”到底什么时候值得造，什么时候只是把系统包装得更难理解。**

## 为什么这一章现在出现

如果太早讲 `result builder` 和 macro，读者通常只会得到两种错觉：

- “Swift 高级 API 的秘诀就是把调用写得像 DSL”
- “遇到重复就应该造 macro，把样板自动展开”

这两种直觉都危险，因为在 Part 1 到 Part 6 的大部分阶段，我们真正努力建立的，是边界、状态、并发、失败面和共享核心。没有这些基础，任何表面糖衣都只会掩盖问题。

现在讲它们，时机才对。因为我们已经可以拿真实系统来判断：

- `TaskCLI` 是否真的需要 DSL 式命令声明
- `TaskFlow` 的某些视图组合是否值得用 `result builder`
- `@Observable`、`#Preview` 这类宏为什么在这里有价值
- 什么样的公共 API 表面会降低理解成本，什么样的只是在制造“高级感”

## 从一个较弱起点开始：先被“调用很漂亮”诱惑

看一个典型弱起点。有人想让 CLI 帮助文本更“声明式”，于是写出：

```swift
let help = TaskHelpScreen {
    Section("Commands") {
        Command("list", summary: "Show tasks")
        Command("add", summary: "Create task")
        Command("done", summary: "Complete task")
    }
}
```

又或者有人想让任务过滤条件看起来像小语言：

```swift
let query = TaskQuery {
    Status(.todo)
    Sort(.createdAt)
    Limit(20)
}
```

这些表面看上去都很顺滑，但问题是：**你还没证明这些调用表面比普通初始化器或普通函数更清楚。**

如果底层语义只是一组固定配置，下面这种写法可能反而更强：

```swift
let query = TaskQuery(
    status: .todo,
    sort: .createdAt,
    limit: 20
)
```

高级 API 表面判断的第一原则，就是不要把“写起来像 DSL”误认为“设计更好”。

## `result builder` 真正适合解决什么问题

`result builder` 的强项，不是“让任何 API 都变得像 SwiftUI”，而是处理这类调用面：

- 调用方天然要按块组织多个子项
- 块结构本身有语义意义
- 顺序、嵌套、分支会影响最终组合结果

SwiftUI View 组合就是典型例子，所以它用 `ViewBuilder` 非常合理。

把这个判断带回我们的项目线，比较自然的候选场景是：

- `TaskFlow` 里某些由 section 组成的界面构造
- preview 配置集合
- CLI 的帮助文档树，如果它真的存在清晰层级关系

例如：

```swift
@resultBuilder
enum TaskHelpBuilder {
    static func buildBlock(_ sections: HelpSection...) -> [HelpSection] {
        sections
    }
}

func makeHelp(@TaskHelpBuilder _ content: () -> [HelpSection]) -> HelpDocument {
    HelpDocument(sections: content())
}
```

这类 builder 的价值并不在于“少写括号”，而在于它把“帮助文档由若干 section 组成”这个结构关系直接写进了 API。

## 先问“普通函数够不够”，再问“builder 值不值得”

这是本章最重要的判断之一。

很多 API 如果换成普通初始化器，其实已经足够清楚：

```swift
let section = HelpSection(
    title: "Commands",
    commands: [
        .init(name: "list", summary: "Show tasks"),
        .init(name: "add", summary: "Create task")
    ]
)
```

如果这个版本已经：

- 容易读
- 容易搜索
- 容易调试
- 不会显著增加样板

那就没有必要为了“风格统一”强行上 builder。

对教程读者尤其要守住这点。因为初学高级 Swift 时，一个非常常见的误判是：**越像框架，越像高级设计。** 真实工程恰恰不是这样。高级 API 的目标是降低误用和维护成本，不是提升视觉戏剧性。

## macro 的价值：消除机械样板，但不替你做架构判断

Swift macro 最容易被误解成“以后重复代码都交给编译器展开”。这个理解过于轻浮。

macro 真正擅长的是：

- 机械、可预测、规则清楚的代码生成
- 能从声明位推导出重复样板的地方
- 能在编译期提供额外检查或派生能力的地方

这就是为什么 `@Observable` 在 `TaskFlow` 里有价值。它处理的是一类高度机械、重复、又和声明位置强相关的样板。

同理，`#Preview` 的价值也不是“语法新潮”，而是它把 preview 入口变成了更短、更清楚的声明式结构。

但 macro 不会自动回答下面这些更难的问题：

- `TaskFlow` 的状态边界应该放在哪里
- `TaskCLI` 的错误映射该如何分层
- `TaskCore` 的共享抽象是否值得提炼

这些仍然是系统设计判断。macro 最多只能帮你减少样板，不能代替你决定边界。

## 从项目线看，哪些地方值得用 macro，哪些地方不值得

对当前主线，一个相对稳妥的判断可以这样分：

### 值得考虑 macro 的地方

- `TaskFlow` 中声明式 preview 和状态样本
- 某些明显机械、规则稳定的模型辅助声明
- 测试里少量重复且模式极稳定的构造工具

### 不值得优先用 macro 的地方

- 领域规则本身
- 运行时失败与恢复逻辑
- CLI 命令执行主流程
- 仍在快速变化中的架构边界

原因很直接。越接近系统语义核心，越不应该过早把逻辑藏进生成层。因为那会让读者和维护者同时失去“它到底如何工作”的透明度。

## API surface judgment：公开表面应尽量窄、尽量可解释

无论是 builder 还是 macro，最终都要回到更大的问题：**你的 API surface 应该长什么样。**

对共享核心来说，公开表面最重要的不是“够酷”，而是：

- 调用者能快速知道怎么正确使用
- 错误用法尽量难以发生
- 隐藏实现细节不会损伤必要的语义透明度

例如，下面是较弱的 API：

```swift
func perform(_ options: [String: Any]) async throws -> Any
```

而更强的 API 会更像：

```swift
func perform(_ mutation: TaskMutation) async throws -> TaskMutationResult
```

如果再上一层的 `TaskFlow` 只需要某个窄表面，那它就不应被暴露到“整个 runtime 万能入口”。

这和 builder / macro 其实是同一个主题：**公共表面越强，内部越有资格复杂；公共表面越花哨，内部越应该谨慎。**

## `result builder` 最怕的不是复杂，而是偷偷扩大语义

一个经常被忽略的问题是：builder 非常容易让 API 语义偷偷膨胀。

例如最开始你只是想拼 section：

```swift
makeHelp {
    HelpSection(
        title: "Commands",
        commands: [
            .init(name: "list", summary: "Show tasks")
        ]
    )
}
```

后来慢慢加出：

- if / else
- loops
- fallback content
- environment mutation
- hidden defaults

最后 builder 虽然还“好看”，但调用者已经很难直观看出它到底在做什么。

所以一个成熟判断是：**builder 的语义要比普通函数更收敛，而不是更放肆。** 如果用了 builder，就更要限制它到底允许调用方表达什么。

## macro 最怕的不是魔法，而是把“重要逻辑”埋进生成物

macro 的风险也类似。真正危险的不是它“自动生成代码”，而是：

- 读 API 表面的人不知道它生成了哪些行为
- 运行时错误实际上来自编译期展开逻辑
- 团队开始把架构问题伪装成“可以生成”的问题

比如，如果有人想用 macro 给 `TaskStore` 自动生成错误映射、自动生成 CLI 输出、自动生成并发隔离包装，这通常已经是危险信号。因为这些地方包含的是业务与系统判断，不是纯机械样板。

与其那样，不如老老实实保留显式代码。教程在这里必须刻意反直觉：**显式，常常比聪明更高级。**

## 把 `@Observable`、`#Preview` 放回本教程语境

这一章如果只泛泛谈 macro，会很空。更有价值的，是回头解释我们已经见过的两个例子为什么成立：

### `@Observable`

它在 `TaskFlow` 中有价值，因为它减少的是与“状态发布”直接相关的机械样板，而不会替你定义状态边界本身。

### `#Preview`

它在 `TaskFlow` 中有价值，因为它让 preview 成为显式、短小、可组合的设计检查入口，而不是一堆散落的临时演示代码。

这两个例子都符合一个重要标准：**它们优化了声明表面，却没有掩盖核心系统判断。**

这正是本章想让读者建立的判断力。

## API 表面判断也会反过来约束包边界

一旦你开始认真对待 builder、macro 和公共表面，包边界（package boundary）也会被反向塑形。

例如：

- 如果某个 builder 完全服务 `TaskFlow`，它不该被塞进 `TaskCore`
- 如果某个 macro 只为了 app model 样板，它不该污染 CLI 包边界
- 如果共享核心的公共 API 因为照顾 DSL 变得难以解释，那就是表面设计已经开始反噬核心

所以高级 Swift 表面从来不是“语言趣味”。它最终仍然要回到系统工程的老问题：**什么东西该共享，什么东西该留在客户端。**

## 双语关键词

- result builder：结果构建器
- macro：宏
- API surface：API 表面
- DSL：领域特定语言
- boilerplate：样板代码
- explicitness：显式性
- generated code：生成代码
- declaration site：声明位置
- misuse-resistant API：抗误用 API

## 常见错误

### 1. 因为 SwiftUI 很漂亮，就觉得所有 API 都该做成 builder

SwiftUI 的成功不意味着 DSL 是所有问题的默认答案。

### 2. 把 macro 当成架构捷径

macro 只能消除机械重复，不能替你决定边界和责任分配。

### 3. 为了减少几行代码，牺牲调试性和可搜索性

如果调用变漂亮了，但行为更难定位，那往往是退步。

### 4. 让 builder 或 macro 跨越本不该共享的包边界

客户端表面工具不该反向污染共享核心。

### 5. 只看写代码时是否顺手，不看读代码时是否清楚

API 设计首先服务读者和维护者，其次才是调用时的爽感。

## English Recap

Result builders and macros are useful only when they improve a real API surface without hiding essential system meaning. Builders fit block-structured composition; macros fit predictable, mechanical code generation. Neither should be used to compensate for unclear boundaries, and both must be judged by whether they make the shared core and client surfaces easier to understand.

## Drills

1. 选一个你想做成 builder 的调用面，先写出普通初始化器版本，再判断 builder 是否真的更强。
2. 对 `TaskFlow` 中的一个状态模型，说明 `@Observable` 帮你消除了什么机械样板，又没有替你决定什么架构问题。
3. 找一个你直觉上想“用 macro 自动生成”的地方，解释为什么它可能其实属于显式代码更好的区域。

## Project Handoff

高级 Swift 的漂亮表面现在已经被放回了真实判断里。下一章会继续向系统边界推进：当 `TaskCore + TaskCLI + TaskFlow` 需要和 Foundation、文件系统、日志、用户默认值等系统 API 打交道时，互操作（interop）与包边界的取舍该怎么做，才能不让共享核心被平台细节吞掉。

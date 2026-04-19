# 第9章：泛型、关联类型与类型驱动 API 设计

> 现在 `TaskCLI Lite` 已经不只是有模型、有边界，它开始面临一个更高级的问题：同样的结构关系出现了不止一次，我们应该继续复制，还是让类型系统直接表达这些关系？

## 为什么这一章现在出现

如果你沿着前几章继续写，很快就会遇到两类重复：

- 同一种操作在不同模型上反复出现
- API 的正确使用方式只能靠注释和记忆维持

例如，今天你可能只处理 `Task`，明天就会出现 `TaskGroup`、`TaskID`、解析结果、渲染结果。此时如果继续靠“写一份差不多的函数”“用 `String` / `Int` / `Any` 先糊住”，代码表面上能跑，类型信息却在不断流失。

这就是泛型（generics）和关联类型（associated types）现在必须出现的原因。它们不是为了让代码“更学术”，而是为了让 API 直接表达：

- 这个操作适用于哪一类值
- 输入类型和输出类型之间有什么关系
- 哪些组合在编译期就应该被禁止

对于来自动态语言的程序员，这一章通常会挑战一种旧习惯：很多关系以前靠运行时约定也能凑合；在 Swift 里，更强的做法是尽量把关系写进类型系统。

对于来自 Java、C#、Go、Kotlin 的程序员，这一章的挑战则不同：Swift 的泛型和协议会更紧密地协作，尤其是关联类型，会让“一个协议内部自己携带类型关系”这件事变得非常自然。

## 从一个典型弱起点开始：复制函数，或者退回 `Any`

先看两种常见但不够强的写法。

第一种是复制函数：

```swift
func findTask(id: Int, in tasks: [Task]) -> Task? {
    tasks.first { $0.id == id }
}

func findTaskGroup(id: Int, in groups: [TaskGroup]) -> TaskGroup? {
    groups.first { $0.id == id }
}
```

第二种是为了“通用”，退回到模糊类型：

```swift
func find(id: Int, in values: [Any]) -> Any? {
    // ...
    nil
}
```

前者的问题是扩展性差，后者的问题是类型信息直接塌掉。

Swift 想训练你的，不是二选一，而是第三种路径：**把共同结构提炼出来，同时保留类型关系**。

## 泛型先解决“同一个操作适用于一类类型”

如果 `Task` 和 `TaskGroup` 都有 `id`，你就可以先抽出一个共同协议：

```swift
protocol IdentifiedModel {
    associatedtype ID: Hashable
    var id: ID { get }
}
```

然后让具体模型遵循它：

```swift
struct Task: IdentifiedModel {
    let id: Int
    // ...
}

struct TaskGroup: IdentifiedModel {
    let id: UUID
    let name: String
}
```

接着写一个泛型函数：

```swift
func find<Model: IdentifiedModel>(
    _ id: Model.ID,
    in values: [Model]
) -> Model? {
    values.first { $0.id == id }
}
```

这段代码值得慢一点看，因为它把三件事同时表达出来了：

- 函数适用于任何遵循 `IdentifiedModel` 的类型
- 传入的 `id` 类型必须和这个模型自己的 `ID` 一致
- 返回值仍然是原来的具体模型，而不是被抹平成某种公共父类

这就是泛型真正的价值。它不是为了省几行代码，而是为了把“这几个类型共享某个结构关系”直接写进签名。

## 关联类型（associated type）解决的是“协议内部带着自己的类型关系”

上面的关键点其实不只是泛型，而是 `associatedtype ID`。

为什么这里不用简单的 `var id: Int { get }`？因为不同模型未必共享同一种 ID 表示。`Task` 也许用 `Int`，`TaskGroup` 也许用 `UUID`，将来别的类型也许用自定义 `TaskID`。

关联类型让协议可以说：

- 我要求你有一个 `id`
- 但 `id` 的具体类型由遵循者决定
- 同时这个类型关系会继续流入所有使用该协议的泛型 API

这是 Swift 类型系统里非常核心的一点。很多别的语言需要靠额外模板、泛型接口、或者比较笨重的参数化语法才能表达；Swift 用 protocol + associated type 的组合，让它非常自然。

如果你把它翻成工程语言，就是：

“这个协议描述的不只是字段列表，还描述了一组随具体类型变化的类型关系。”

## 用类型系统约束 CLI 参数解析，而不是靠注释提醒

类型驱动 API 设计（type-driven API design）最有用的，不是写更泛的库，而是让日常 API 更不容易被误用。

看一个贴近 `TaskCLI Lite` 的例子。命令行参数天然是字符串数组，但我们并不希望系统内部长期维持字符串态。一个很实用的泛型工具，是把“某个字符串能否解析为某种目标类型”写成通用能力：

```swift
func parseValue<T: LosslessStringConvertible>(
    _ raw: String,
    as type: T.Type = T.self
) -> T? {
    T(raw)
}
```

现在：

```swift
let id: Int? = parseValue("42")
let ratio: Double? = parseValue("3.14")
```

这看起来很小，但它体现的是一个重要方向：不要让调用方记住“这里应该自己转 Int、那里应该自己转 Double”，而是提供一个签名就能告诉调用方如何使用的 API。

把它放回任务命令解析：

```swift
func parseDoneCommand(arguments: [String]) -> TaskCommand? {
    guard
        let rawID = arguments.dropFirst().first,
        let id: Int = parseValue(rawID)
    else { return nil }

    return .done(id: id)
}
```

此时你的 API 已经在向读者表达：

- 命令行进来时是文本
- 进入领域层前应该转换成更具体的类型
- 解析失败属于显式分支，而不是悄悄吞掉

## 泛型不是为了“抽象一切”，而是为了保留正确的具体性

这句话很重要，因为很多人第一次学泛型时，会误以为目标是让所有代码都变得更一般化。实际上，好的泛型往往是在保留正确具体性的同时，抽出真正共享的结构。

上面的 `find`：

- 没有把返回值改成 `IdentifiedModel`
- 没有把集合改成 `[Any]`
- 没有为了通用而丢掉模型自己的 `ID` 类型

也就是说，它更通用了，但没有更模糊。

这正是 Swift 风格 API 的一个重要特征：能用类型把关系说清楚时，不要退回动态容器、字符串键、注释约定。

## 再看一个和协议边界直接相连的例子

上一章我们写了命令解析协议。如果把它继续推进一点，就会自然遇到关联类型：

```swift
protocol CommandParsing {
    associatedtype Command
    func parse(arguments: [String]) -> Command?
}

struct TaskCommandParser: CommandParsing {
    func parse(arguments: [String]) -> TaskCommand? {
        // ...
        nil
    }
}
```

为什么这里值得用关联类型？

因为不同 parser 可以解析出不同命令类型。`TaskCommandParser` 解析的是 `TaskCommand`；将来如果你有别的子系统 parser，它们不必被迫返回同一种统一命令枚举。

这就是关联类型的第二个典型用途：让“某种能力”和“该能力产出的具体类型”绑定在一起。

当然，这也带来一个后面会经常遇到的现实：带关联类型的协议，通常更适合通过泛型约束来使用，而不是立刻当作简单存储属性。这不是缺点，而是 Swift 在提醒你，类型关系是这个抽象的一部分，不能被轻易抹平。

## 一个更贴近工程判断的 API 对比

看下面两种接口：

弱版本：

```swift
func complete(_ payload: [String: Any]) -> [String: Any]
```

强版本：

```swift
func complete(taskID: Int, in list: TaskList) -> Result<Task, TaskMutationError>
```

第二个版本当然还引用了下一章要讲的 `Result` 和错误类型，但光从签名就已经能看出类型驱动设计的差别：

- 它不要求调用方猜字典里该放什么键
- 它明确说明需要 `taskID` 和 `TaskList`
- 它明确说明成功时得到 `Task`
- 它明确说明失败时得到 `TaskMutationError`

这就是“API 用类型表达正确使用方式”的含义。调用者越少依赖注释、文档和猜测，API 就越稳。

## 泛型在 `TaskCLI Lite` 当前阶段该用到什么程度

这一章也很容易被学歪成“马上写一个超抽象框架”。我们不要这么做。

当前阶段更合理的目标是：

- 能看出哪些重复是结构性重复，而不只是业务细节相似
- 能写出几个小而清楚的泛型工具
- 能理解关联类型在协议里表达关系的价值
- 能开始用类型约束改进日常 API 签名

还不需要做的事情包括：

- 为了练语法，硬发明大型泛型容器
- 把简单直接的具体代码过早改成高度参数化框架
- 在还没有模块压力前，就做复杂 type erasure

Part 2 的职责是建立判断，不是提前炫技术。

## 为什么这一步会直接影响 Part 3

进入 Part 3 后，你会开始认真做 package、测试和 CLI 工程化。那时一个持续出现的问题是：哪些东西值得抽成共享核心，哪些东西只是当前 executable 的偶然细节。

泛型和类型驱动 API 会直接帮助你回答这个问题，因为它们迫使你把“共同结构”和“具体实现”区分清楚。

例如：

- 一个按 `ID` 查找元素的能力，很可能属于共享核心
- 一个把参数数组转成 `TaskCommand` 的 parser，更接近 CLI 边界

这种区分能力，会让你在 Part 3 做模块划分时不只是“按文件搬家”，而是真正按语义拆分。

## 双语关键词

- 泛型：generics
- 泛型约束：generic constraint
- 关联类型：associated type
- 类型驱动 API 设计：type-driven API design
- 协议遵循：protocol conformance
- 具体类型：concrete type
- 写时约束：compile-time constraint
- 抹平类型：type erasure / erase type information

## 常见错误

### 1. 一看到重复就立刻抽象成大泛型框架

泛型应该提炼真实结构关系，不应该把简单问题变成参数化秀场。

### 2. 为了通用而退回 `Any`、字典或字符串键

这通常不是“灵活”，而是在丢掉编译器本来可以帮你检查的关系。

### 3. 不理解关联类型，只把协议看成字段和方法清单

很多 Swift 协议的重要价值，正来自它能携带“这个能力关联哪种具体类型”的信息。

### 4. 让 API 看起来很抽象，却没有保留具体返回类型

好的泛型常常更通用，但不会更模糊。返回值如果被抹平成过于宽泛的类型，调用体验往往会变差。

### 5. 试图在还没需要时就解决所有 `any` / type erasure 问题

这些是后续更工程化阶段才值得认真展开的话题。当前章节先把关系看清楚，比把每种高级技巧都预支更重要。

## English Recap

This chapter shows how Swift uses generics and associated types to preserve relationships instead of erasing them. Generic APIs should generalize real structure while keeping concrete types meaningful, and associated types let protocols describe model-specific type relationships such as “this parser produces this command” or “this model uses this ID type”.

## Drills

1. 写一个 `IdentifiedModel` 协议和一个泛型 `find(_:in:)`，让它至少能用于 `Task`。
2. 写一个 `parseValue<T: LosslessStringConvertible>`，分别解析 `Int` 和 `Double`。
3. 检查你现在的某个 API：它有没有用 `String`、字典或 `Any` 偷偷承载本可以由类型表达的关系？

## Project Handoff

到这里，`TaskCLI Lite` 已经开始从“会写模型和边界”走向“会用类型表达关系和正确用法”。下一章要补上的，是另一块同样关键的工程语义：失败（failure）如何建模。因为一旦 API 变得更强，`nil`、`Bool` 和临时打印就不再足以表达所有失败面了。

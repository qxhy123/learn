# 第2章：值、类型与可变性

> 现在我们已经知道 Swift 程序怎么被运行，下一步必须回答：程序里到底装的是什么数据，这些数据为什么有类型限制，哪些地方应该允许变化，哪些地方最好一开始就不要改。

## 为什么这一章现在出现

如果上一章只解决了“代码怎么跑”，这一章要解决的就是“代码里流动的东西到底是什么”。对会别的语言的程序员来说，这一步尤其关键，因为你很容易把 Swift 的值和变量理解成旧语言里的模糊容器，然后在真正开始写项目时，把所有东西都塞进可变状态里。

`TaskCLI Lite` 很快就要处理任务标题、任务列表、命令参数、完成状态。这里面每一件事都要求你对值（value）、类型（type）和可变性（mutability）有清楚判断。不然你会得到一种表面上能跑、实际上非常脆弱的代码：所有东西都能改、哪里都能塞字符串、数字之间混着算、状态变化没有边界。

## 一个典型的弱起点

很多人从别的动态语言过来，会自然写出类似这样的心智：

- 变量就是一个名字，先放进去再说
- 反正后面要变，所以默认都可变
- 类型出错时再修，不需要一开始就明确

这种心智在 Swift 里会很快撞墙。比如：

```swift
var taskCount = 3
var progress = 0.5
```

这两行看起来普通，但 Swift 其实已经在替你做判断：

- `taskCount` 被推断为 `Int`
- `progress` 被推断为 `Double`

当你继续写：

```swift
let total = taskCount + progress
```

Swift 会拒绝，因为它不会偷偷把 `Int` 和 `Double` 混在一起。对初学者来说这像“麻烦”，但对工程代码来说这是好事：类型系统在帮你提前暴露含糊不清的意图。

## `let` 和 `var`：先问“应不应该变”

Swift 最值得你尽早习惯的一个判断是：先默认不可变（immutable），只有确实需要变化时才使用可变（mutable）。

```swift
let tutorialName = "TaskCLI Lite"
var completedCount = 0
```

这里的区别不只是语法：

- `let` 表示绑定后不再重新赋值
- `var` 表示后续允许变化

这是一种设计判断，而不是打字偏好。`tutorialName` 作为一个固定标识，不应该在程序中途被改掉；而 `completedCount` 明显会随着操作变化，所以用 `var` 才合理。

很多程序员会说：“我先都写 `var`，后面再改。”这个习惯在 Swift 里代价很高，因为它会把你的程序默认推向到处都能被修改的状态。对于后面的 `TaskCLI Lite` 来说，这意味着你会把本来应该清晰的数据流，写成四处散开的可变脚本。

## 类型推断（type inference）和显式标注（explicit annotation）

Swift 很擅长类型推断：

```swift
let title = "read chapter 02"
let targetMinutes = 25
let isImportant = true
```

这里分别推断出 `String`、`Int`、`Bool`。在简单情境里，推断足够清晰，就没必要硬写类型名。但你也应该知道何时显式写出来更稳：

```swift
let completionRatio: Double = 0.75
let taskTitle: String = "build TaskCLI Lite"
```

当值的语义可能被误读、或者你想把教程示例讲得更清楚时，显式标注会更有教学价值。Part 1 不是要你逢变量必写类型，而是要你形成判断：哪里靠推断更自然，哪里靠显式类型更清楚。

## 值是如何变强的

让我们把一个弱程序往强一点的状态推进。先看一个很弱的例子：

```swift
var currentTask = "read chapter 02"
currentTask = "write notes"
currentTask = "test again"
```

这段代码当然能跑，但它的问题是：你失去了“程序当前在表达什么”的清晰度。`currentTask` 既像当前任务标题，又像一个临时编辑框，所有信息都被覆盖掉了。

更强一点的状态会开始区分“固定配置”和“运行时变化”：

```swift
let projectName = "TaskCLI Lite"
let initialTaskCount = 3
var completedTaskCount = 0

completedTaskCount += 1
```

再进一步，开始利用不同类型表达不同含义：

```swift
let canReleaseV1 = completedTaskCount == initialTaskCount
print("Can release \(projectName): \(canReleaseV1)")
```

现在程序的可读性已经强很多了。你能看出哪些值是稳定背景，哪些值是状态变化，哪些表达式产出的是 `Bool` 判断结果。这正是 CLI 项目真正需要的基础。

## 数值类型与布尔值别混着用

Swift 对数值非常认真。`Int`、`Double`、`Float` 不是“反正都能算”的一团东西。你要学会显式转换：

```swift
let completed = 2
let total = 3
let ratio = Double(completed) / Double(total)
```

这看起来比某些语言啰嗦，但它逼你承认一个事实：你到底想要整数语义，还是浮点语义。对于工程代码，这种明确性远比“省几个字符”更值钱。

## 这和项目有什么关系

`TaskCLI Lite` 的第一版虽然小，但它已经需要这些判断：

- 任务标题是 `String`
- 完成状态是 `Bool`
- 任务列表数量天然是整数语义
- 哪些值是命令执行过程中的临时变化，哪些值是固定配置

如果你现在没有把值、类型、可变性理顺，后面写 `list`、`add`、`done` 时就会不断把问题写回字符串拼接和随手可变状态里，程序表面上像在工作，实际却没有稳定形状。

## 双语关键词

- 值：value
- 类型：type
- 可变性：mutability
- 不可变：immutable
- 类型推断：type inference
- 类型标注：type annotation
- 布尔值：Boolean / `Bool`

## 常见错误

### 1. 默认把所有绑定都写成 `var`

这会让程序的状态边界变得模糊。更稳的默认是：先用 `let`，只有在确实需要变化时才用 `var`。

### 2. 以为 Swift 会自动帮你混合 `Int` 和 `Double`

Swift 不愿意替你猜。类型转换虽然显式，但它迫使你把数值语义想清楚。

### 3. 把类型标注理解成“写得越多越专业”

并不是。简单场景下，推断已经足够清楚；只有在教学、建模或可读性需要时，显式类型才更有价值。

## English Recap

This chapter moves from execution to data modeling basics. You learned how `let` and `var` express intent, how Swift infers types, when explicit annotations help, and why Swift refuses sloppy numeric mixing. The main lesson is that clean programs start by being honest about what can change and what each value means.

## Drills

1. 写一个小程序，定义 `let totalTasks = 5`、`var doneTasks = 2`，输出当前完成比例。
2. 故意尝试把 `Int` 和 `Double` 直接相加，观察编译器报错，再用显式转换修正。
3. 找出你练习代码里三个其实不需要变化的绑定，把它们从 `var` 改成 `let`。

## Project Handoff

现在我们已经有了 `TaskCLI Lite` 最基本的数据直觉：标题、数量、状态不再是模糊容器，而是带有明确类型和可变性边界的值。下一章会把这些值组织起来，进入字符串（`String`）、集合（collections）和控制流（control flow），因为一个 CLI 不会只处理单个值，它要处理一组任务和一组命令分支。

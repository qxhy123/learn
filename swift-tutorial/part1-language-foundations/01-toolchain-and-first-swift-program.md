# 第1章：工具链与第一个 Swift 程序

> Part 1 的第一步不是背关键字，而是先建立一个稳定的实验闭环：你知道 Swift 代码怎么写、怎么跑、怎么编译、怎么读最基本的错误输出。

## 为什么这一章现在出现

很多已经会别的语言的程序员，一上来学 Swift 时最容易犯的错，不是语法记不住，而是把 Swift 错看成“只有在 Xcode 里点 Run 才能工作的语言”。这种起手式会直接削弱后面的学习，因为你会把工具链、编译、命令行执行、包结构这些最基础的工程事实全部外包给 IDE。

本教程刻意从 toolchain 开始，是因为后面整条项目主线都要建立在一个简单但可靠的前提上：你能在终端里运行 `swift`、理解 `swiftc`、区分脚本执行和编译可执行文件，并且知道自己的程序到底是“解释式地跑了一遍”，还是“被编译成了一个 executable”。如果这一步不稳，后面的 `TaskCLI Lite` 很容易被你误学成“某个按钮背后的魔法”。

## 从一个过弱的起点开始

初学者常见的弱起点大概长这样：

- 只知道 Swift 是 Apple 生态语言，但不知道本地有哪些命令可用
- 看到 `swift` 和 `swiftc` 时，分不清谁负责编译、谁负责直接执行
- 把“写出 Hello, World!”当成全部目标，却不知道程序入口、标准输出、编译产物各自意味着什么

如果你带着这种状态继续往前走，接下来每学一个语法点都会漂在空中。你会写出一些能运行的行，但不会真的形成“程序是怎样被组织和执行的”这个心智模型。

## 先把工具链看清楚

对 Part 1 而言，你暂时只需要认识三样东西：

- `swift`：Swift driver，可以直接执行脚本，也能代理一些编译相关动作
- `swiftc`：Swift compiler，负责编译 Swift 源码并产出可执行文件
- `swift package` / `swift build` / `swift test`：Swift Package Manager（SPM）入口，后面会成为项目主线的工程基础

先做最小检查：

```bash
swift --version
swiftc --version
```

这一步不是形式主义。它的意义是确认：你的环境里真的有 Swift toolchain，而且终端能直接调用它。很多“代码写错了”的误判，实际上是环境根本没接通。

## 第一个 Swift 程序：先弱后强

最弱但可运行的起点是一个脚本文件：

```swift
print("Hello, Swift")
```

把它保存成 `hello.swift` 之后，你可以这样跑：

```bash
swift hello.swift
```

这一步的价值不是打印一句话，而是让你第一次把“源文件 -> 命令 -> 输出”这条链路打通。你应该立刻问自己三个问题：

1. `print` 在做什么？
2. 为什么不需要显式写 `main`？
3. 这次运行和真正编译成一个 executable 有什么不同？

现在把程序往强一点的状态推进。比如，不只是打印固定文本，而是把值插入输出里：

```swift
let learnerName = "Ming"
let chapterNumber = 1

print("Hello, \(learnerName). You are starting chapter \(chapterNumber).")
```

这里第一次出现了两个重要事实。第一，Swift 程序不是只能写“直接输出文本”的脚本，它立刻就会牵涉值（value）和字符串插值（string interpolation）。第二，哪怕是 Hello World 级别的例子，也已经在暗示后面项目主线需要的能力：拿到一些数据，组织成输出。

## `swift` 和 `swiftc` 到底差在哪

到这里，很多读者会说：“能跑就行，为什么还要区分 `swift` 和 `swiftc`？”因为这两者对应的是两种不同的心智：

- `swift hello.swift` 更像快速执行一个脚本
- `swiftc hello.swift -o hello` 则明确告诉你：我要把源码编成一个二进制可执行文件

你可以试一遍：

```bash
swiftc hello.swift -o hello
./hello
```

现在程序不再只是“解释式地被跑一下”，而是被显式编译成了一个产物。后面当我们进入 `TaskCLI Lite` 时，这个差别会很重要，因为一个真正的 CLI 项目不可能长期停留在“随手跑脚本”的阶段。

## 为什么这一步还不够

虽然你现在已经能运行和编译 Swift 代码，但程序仍然非常弱。它的弱，不是因为功能少，而是因为它还没有开始处理真实程序一定会遇到的问题：

- 数据是否有类型（type）约束？
- 哪些值应该不可变（immutable），哪些应该允许变化（mutable）？
- 命令输入如何转成程序状态？
- 多个任务数据如何被组织？

这正是下一章要马上进入 `values`, `types`, `mutability` 的原因。工具链只是把门打开；真正的 Swift 判断，从值的语义开始。

## 双语关键词

- 工具链：toolchain
- 编译器：compiler
- 可执行文件：executable
- 标准输出：standard output
- 源文件：source file
- 字符串插值：string interpolation
- 程序入口：entry point

## 常见错误

### 1. 把 `swift` 和 `swiftc` 当成完全一样的东西

它们都和执行 Swift 代码有关，但语义并不相同。前者更适合快速运行脚本或代理命令，后者明确面向编译产物。后面进入项目阶段时，这种差别会直接影响你对 build 过程的理解。

### 2. 一开始就把 Swift 学成“只能在 Xcode 里点按钮”

Xcode 很重要，但 Part 1 的主线是命令行与语言基础。你如果不先建立终端里的执行闭环，后面看到错误时很容易不知道问题出在代码、工具链还是工程组织。

### 3. 看到能输出结果就停止思考

“程序跑了”不等于“你已经理解它”。写完第一个程序后，应该立刻追问：这里有哪些值、有哪些类型、为什么现在不需要显式 `main`、编译产物在哪里。

## English Recap

This chapter establishes the execution loop before real language study begins. You used `swift` to run a small script, used `swiftc` to build an executable, and learned why Swift in this tutorial starts from the command line rather than from IDE magic. The main takeaway is simple: before designing programs, you need a stable toolchain mental model.

## Drills

1. 新建一个 `greeting.swift`，定义 `let name = "Swift"` 与 `let day = 1`，用字符串插值输出一句完整的话。
2. 分别用 `swift greeting.swift` 和 `swiftc greeting.swift -o greeting && ./greeting` 运行它，观察两种方式的差别。
3. 故意把文件名写错一次，读一遍终端错误输出，确认你能分辨“找不到文件”和“代码写错了”这两类失败。

## Project Handoff

`TaskCLI Lite` 还没有正式开始，但这章已经搭好了它的最低地基：你可以在终端中运行 Swift 代码，并且知道一个命令行程序最终要落到 executable 上。下一章我们会处理更关键的语言问题：值（value）、类型（type）和可变性（mutability），因为没有这些心智，CLI 的数据和状态根本立不住。

# 第3章：字符串、集合与控制流

> 到这里，Swift 代码已经不该只会处理单个值了。真正的命令行程序一定要面对三件事：文本输入怎么表示、一组数据怎么组织、不同命令怎么分支。

## 为什么这一章现在出现

上一章把“值是什么、类型怎么约束、哪些地方允许变化”这件事打稳了，但 `TaskCLI Lite` 还差更接近真实程序的三块骨架：

- 任务标题是文本，所以必须认真处理 `String`
- 任务不只一个，所以必须引入集合（collections）
- 用户会输入不同命令，所以必须使用控制流（control flow）

如果没有这一章，教程就会停在“能写几个变量”的层面。你当然可以继续写更多单值示例，但那不会把你带到 CLI 项目上。`TaskCLI Lite v1` 要求你第一次面对数据成组和分支执行，这正是 `String`、`Array`、`if`、`switch`、循环（loop）现在必须出现的原因。

## 从一个过于脆弱的脚本开始

先看一个典型弱起点：

```swift
print("1. read chapter 01")
print("2. write notes")
print("3. build task cli")
```

这段代码的问题不是“太简单”，而是它根本没有数据结构。你打印的是三行文本，而不是一组可以被程序操作的任务。程序无法：

- 新增一个任务
- 遍历任务列表
- 根据条件决定输出不同内容

换句话说，这根本还不是 CLI，只是三句硬编码的输出。

## 先把文本当成真正的数据：`String`

在命令行项目里，字符串不是装饰，而是主角之一。任务标题、命令参数、错误提示、帮助文本，几乎全都和 `String` 有关。

最小起点：

```swift
let title = "read chapter 03"
print(title)
print("Task title: \(title)")
```

这里你应该立刻建立一个更稳的直觉：字符串不是“随手拼上去的字面量集合”，而是一种被频繁创建、拼接、比较、插值的核心数据类型。

比如，CLI 很快就会需要把用户输入的多个参数拼成标题：

```swift
let words = ["write", "chapter", "summary"]
let title = words.joined(separator: " ")
print(title)
```

这已经非常接近 `add <title>` 这种命令的真实需求了。

## 从单值走向一组数据：`Array`

一旦程序要处理多个任务，你就不能继续靠一个又一个独立变量硬撑：

```swift
let task1 = "read chapter 01"
let task2 = "write notes"
let task3 = "review let and var"
```

这仍然太弱，因为它无法自然扩展。更强的状态是把同类数据组织成一个 `Array`：

```swift
var tasks = [
    "read chapter 01",
    "write notes",
    "review let and var"
]
```

现在程序第一次具备“列表”的形状了。你可以追加（append）、遍历（iterate）、按索引读取（index access）。例如：

```swift
tasks.append("build TaskCLI Lite")

for task in tasks {
    print(task)
}
```

这就是从“几行文本”进化到“可操作的数据集合”的关键一步。

## 控制流：程序从这里开始真正像程序

CLI 的本质之一，是根据不同输入走不同路径。只会顺序执行几行代码还不够，你必须开始用条件和分支表达意图。

先看一个简单判断：

```swift
let command = "list"

if command == "list" {
    print("Show all tasks")
} else {
    print("Unknown command")
}
```

这已经比纯打印更像一个程序了。但当命令开始变多时，`switch` 往往更清楚：

```swift
switch command {
case "list":
    print("Show all tasks")
case "add":
    print("Add a new task")
case "done":
    print("Mark a task as complete")
default:
    print("Unknown command")
}
```

对于 `TaskCLI Lite` 这种只有几个离散命令的 Part 1 项目，`switch` 是很好的教学入口，因为它会迫使你显式列出“程序此刻认得哪些命令”。

## 让弱脚本进化成更强的形状

把字符串、数组和控制流拼起来，我们就能得到一个真正过渡性的 CLI 雏形：

```swift
var tasks = [
    "read chapter 01",
    "write notes"
]

let command = "list"

switch command {
case "list":
    for (index, task) in tasks.enumerated() {
        print("\(index + 1). \(task)")
    }
case "add":
    tasks.append("practice Swift arrays")
    print(tasks)
default:
    print("Unknown command")
}
```

这段代码仍然不够强，因为命令还是硬编码的、任务还只是字符串数组、状态和显示逻辑都还挤在一起，但它已经明显比一开始的三个 `print` 靠近真实项目了。Part 1 的教程设计，就是不断做这种“弱起点 -> 更强状态”的推进，而不是一上来就把最终答案贴给你。

## 为什么这里还没用 `struct`

你可能已经隐约感到不舒服：任务只是字符串，似乎无法表达“已完成 / 未完成”。这个不舒服是正确的。也正因为如此，下一章才必须进入函数（functions）、Optional、枚举（enum）和结构体（struct）。当前这一章的任务，是先让你会组织文本、集合和分支；下一章再把这些原始材料升级为更像样的数据模型。

## 双语关键词

- 字符串：string / `String`
- 集合：collections
- 数组：array / `Array`
- 遍历：iteration
- 控制流：control flow
- 条件分支：conditional branching
- 模式分支：`switch`

## 常见错误

### 1. 用一堆独立变量代替 `Array`

当数据数量本来就是“可以变多的一组”时，独立变量只会让程序无法扩展。

### 2. 把字符串处理看成拼接小技巧

对 CLI 而言，字符串是输入输出的主要媒介。标题、错误、帮助文本、命令参数都和 `String` 紧密相关。

### 3. 分支越来越多，却还坚持堆 `if` / `else if`

对于少量离散命令，`switch` 往往更清晰，也更符合“把可接受命令列明”的教学目标。

### 4. 忽略越界和空集合问题

Part 1 里我们不会把所有边界问题一次讲完，但你至少要意识到：集合访问不是“永远安全”的。后面 Optional 会专门处理这类不确定性。

## English Recap

This chapter turns isolated values into program-shaped data. You used `String` for task text, `Array` for lists of tasks, loops for iteration, and `if`/`switch` for command branching. The key lesson is that a CLI becomes real only when text, collections, and control flow start working together.

## Drills

1. 写一个字符串数组，包含三个任务标题，用 `for` 循环逐项打印。
2. 定义一个 `command` 常量，分别尝试 `"list"`、`"add"`、`"done"` 和未知值，用 `switch` 输出不同结果。
3. 把一个单词数组通过 `joined(separator: " ")` 拼成完整任务标题，模拟 `add <title>` 的输入效果。

## Project Handoff

现在 `TaskCLI Lite` 已经从“几句硬编码输出”推进到了“有文本、有列表、有命令分支”的雏形。但它仍然缺少真正的数据模型和更稳的程序形状。下一章我们会引入函数、Optional、枚举和结构体，把这些原材料收束成可复用、可解释、可测试的基础程序结构。

# 第13章：解析、渲染与存储接缝

> 包和测试都已经立住之后，系统会暴露出另一个更像真实工程的问题：哪些地方正在承受变化压力，却还没有被明确命名？这正是接缝（seam）现在必须出现的原因。

## 为什么这一章现在出现

一旦项目进入 `TaskCore + TaskCLI` 阶段，代码不再只是“放在哪个目录里”的问题。你会开始注意到三类经常变化、但性质完全不同的工作：

- 解析（parsing）：把命令行参数转成系统内部能理解的命令
- 渲染（rendering）：把领域状态转成用户可读的 CLI 文本
- 存储（storage）：让任务状态能够在一次运行之外继续存在

如果不在现在识别这些接缝，系统会发生两种退化。

第一种退化是所有工作重新挤回 `main.swift`。这样改动一个命令文案、引入一种新输入格式、或者准备接文件存储时，都会把 CLI 入口变成大团块。

第二种退化则是过度抽象：刚刚闻到一点变化压力，就立刻造出一整套 parser hierarchy、renderer registry、storage plugin system。那会让 Part 3 从工程训练滑向架构表演。

本章的任务不是“把所有接缝做完”，而是让你会识别它们、命名它们，并且知道当前阶段该推进到什么强度。

## 从一个还算能跑、但接缝全挤在一起的版本开始

看当前 starter package 的 CLI 入口，会发现这种形状：

```swift
switch command {
case "list":
    return render(tasks: store.tasks)
case "add":
    let title = normalizedTitle(from: arguments)
    let task = try store.add(title: title)
    return "Added: \(task.title)\n" + render(tasks: store.tasks)
case "done":
    let title = normalizedTitle(from: arguments)
    let task = try store.markDone(title: title)
    return "Completed: \(task.title)\n" + render(tasks: store.tasks)
default:
    return "Unknown command: \(command)\n\(usage)"
}
```

这个版本在 Part 3 初段是合理的，因为它足够短，读者还能完整 hold 住上下文。但它已经开始显露三个压力点：

- 参数解释与命令逻辑仍然绑在一起
- 文本渲染规则目前只有一个版本，但未来变化概率很高
- store 还只是内存态，迟早要面对更真实的持久化路径

也就是说，接缝已经出现，只是还没有被显式命名。

## 先理解“接缝”到底是什么意思

在工程语境里，seam 不是“任何函数边界”，而是“一个局部可以替换、调整、单独测试，而不必推倒整个系统的地方”。

对于当前项目，下面这些就属于真正的 seam：

- 命令行字符串如何变成内部命令值
- `[Task]` 如何被组织成终端文本
- `TaskStore` 将来如何从内存实现过渡到文件实现

而下面这些通常还不算 seam：

- `Task.id` 是不是 `Int`
- `TaskStatus` 是不是 `enum`
- `Task.cliLine` 的存在本身

前者承受的是变化压力，后者目前更像稳定领域建模。学会区分这一点，你才不会把“抽象边界”变成“到处套 protocol”。

## 解析接缝：让命令行文本先有内部形状

CLI 系统最容易被低估的一件事，是“解析不是业务规则本身”。命令行参数进来时只是 `[String]`，而 core 系统真正想处理的，通常是更稳定的命令值。

一个很自然的下一步，是引入最小命令枚举：

```swift
enum TaskCommand {
    case list
    case add(title: String)
    case done(title: String)
}
```

然后把解析集中起来：

```swift
func parse(arguments: [String]) -> TaskCommand? {
    guard let command = arguments.first else { return nil }

    switch command {
    case "list":
        return .list
    case "add":
        let title = arguments.dropFirst().joined(separator: " ")
        return .add(title: title)
    case "done":
        let title = arguments.dropFirst().joined(separator: " ")
        return .done(title: title)
    default:
        return nil
    }
}
```

注意，这里我们还没有引入一个完整的 parsing protocol。因为当前阶段命令种类很少，直接用一个清楚函数或一个小 parser type 就足够了。更重要的是这个判断本身：**先把命令从原始字符串态提升为内部命令态。**

这会给后续章节带来两个收益：

- CLI 入口能逐渐从“读字符串 + 做事情”进化为“解析命令 + 执行命令”
- 测试可以更自然地围绕解析失败和命令形状做断言

## 渲染接缝：不要让输出格式散成无名字符串

当前 starter package 的渲染是一个私有 helper：

```swift
private static func render(tasks: [Task]) -> String
```

这比把所有字符串拼接直接写在 `switch` 里已经强很多，但还没有完全把渲染规则命名出来。

渲染为什么值得被视为 seam？因为它通常比领域行为更容易变：

- 标题要不要显示编号
- 状态符号是不是 `[x]` / `[ ]`
- 未来是否会加 summary line
- 是否要区分 plain text 与其他输出格式

当前阶段更强但仍然克制的推进方式，可以是一个具体渲染器：

```swift
struct PlainTextTaskRenderer {
    func render(tasks: [Task]) -> String {
        let lines = tasks.enumerated().map { index, task in
            "\(index + 1). \(task.cliLine)"
        }

        return (["Today's tasks"] + lines).joined(separator: "\n")
    }
}
```

为什么这里先用 concrete type，而不是一上来就写 `TaskRendering` protocol？因为我们眼下只知道“渲染是一条变化轴”，但还没有多个渲染实现真的同时存在。先把它作为命名清楚的 seam 提出来，比提前接口化更稳。

## 存储接缝：现在先看见，下一部分再做强

最容易被误做过头的，就是 storage seam。因为一旦你承认任务不该永远停在 seeded memory state，直觉就会催你立刻把文件 I/O、JSON 编码、目录管理、异常恢复一次全做完。

教程在这里要刻意踩刹车。Part 3 需要的是：

- 承认 `TaskStore.seeded()` 只是当前阶段的起点
- 明确“状态从哪里来”与“状态怎样变”终究要分开
- 给 Part 4 留下真实的 runtime pressure

所以，当前更稳的认知不是“马上写持久化”，而是先能说清楚未来 seam 在哪里：

- `TaskStore` 现在管理的是内存态任务数组
- 将来会需要一个把任务 load/save 到外部介质的边界
- 这个边界不应被埋在 `main.swift`

也就是说，storage seam 现在要先被**看见**，而不是被**做满**。

## 强接缝不是强架构，命名压力才是重点

很多程序员会把“看到接缝”错误理解成“马上引入一套框架”。真实工程里更稳的顺序通常是：

1. 先看到变化压力
2. 给压力点一个准确名字
3. 把最容易纠缠的逻辑拉开
4. 只有当变化实现真的开始增多时，再考虑 protocol 或更强抽象

对 `TaskCore + TaskCLI v1` 而言，这意味着：

- 解析接缝：值得开始收口
- 渲染接缝：值得开始命名
- 存储接缝：值得开始预留位置

但它还不意味着：

- 现在就需要一整套插件化 storage system
- 现在就需要 renderer registry
- 现在就需要 command bus 或 dependency injection 容器

工程判断最难的地方，常常不是看不见变化压力，而是看见了以后还能忍住不过度设计。

## 双语关键词

- seam：接缝
- parsing：解析
- rendering：渲染
- storage：存储
- plain text：纯文本
- command value：命令值
- runtime pressure：运行时压力
- concrete type：具体类型
- abstraction pressure：抽象压力

## 常见错误

### 1. 把所有变化压力重新塞回 `main.swift`

如果解析、渲染、存储的压力都继续混在入口层，CLI 很快就会重新长成一个大开关脚本。

### 2. 一闻到变化就立刻上 protocol

当前阶段真正需要的是识别 seam，不是把所有 seam 都做成框架。没有多个实现时，concrete type 往往更清楚。

### 3. 提前做完存储系统

Part 3 现在的任务是 package engineering，不是 runtime completion。把 I/O 与可靠性一次做满，会模糊与 Part 4 的边界。

### 4. 把渲染规则误当成领域规则

`Task.status` 属于领域；CLI 标题行、编号格式、更友好的 usage 文本则属于输出组织。二者相关，但不是一回事。

## English Recap

This chapter identifies three real seams in the project: parsing, rendering, and storage. The key lesson is to name change pressure before over-engineering it. Part 3 should expose these seams clearly, but only strengthen them as far as the current package architecture needs.

## Drills

1. 用一句话区分“领域规则”和“渲染规则”。
2. 写出一个最小的 `TaskCommand` 枚举，说明它比直接在 `switch` 里读字符串强在哪里。
3. 解释为什么 storage seam 在 Part 3 应该先被看见，而不应该一次被做满。

## Project Handoff

接缝一旦被识别出来，下一步自然会落到 CLI 自身的组织问题上：命令入口应该怎样生长，才不会重新变回“巨大的 `main.swift`”？下一章我们就专门处理命令组织与 CLI architecture，但仍然坚持一个原则：先做清楚，再做复杂。

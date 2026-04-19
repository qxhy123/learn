# 第19章：性能、复制与测量心智

> 到了这里，`TaskCore + TaskCLI` 已经具备了模块边界、异步路径、Actor 隔离和 ownership 直觉。很多读者会在这个阶段突然分成两个极端：一类人开始过早优化，另一类人则继续把所有性能问题当成“以后再说”。这一章的目标，是建立一套更稳的 Swift 性能判断心智：看复制、看热点、看测量，而不是看感觉。

## 为什么这一章现在出现

Part 4 之前，项目规模足够小，很多成本都还像“理论问题”：

- `TaskStore` 里只有三条 seeded task
- `render(tasks:)` 拼几行字符串几乎感觉不到
- `markDone(title:)` 线性扫描数组也完全够用

但只要你把项目从教程最小态推进到更真实的运行态，性能问题就会不再抽象：

- 任务数可能从 3 变成 3,000 或 30,000
- 每次命令可能都要 load / decode / mutate / encode / save
- Actor hop、值快照复制、字符串构建和 JSON 序列化都会带来成本

更关键的是，Swift 的性能问题往往与语义设计绑得很紧。值语义、Copy-on-Write、ARC、actor isolation 这些前面章节讲过的东西，并不是和性能分离的世界；它们正是性能行为的来源。

所以这章不是“最后补一点优化技巧”，而是把前面几章的语义判断收束成运行时成本判断。

## 从一个“能跑所以应该没问题”的弱状态开始

看当前项目，很容易产生一种合理但危险的错觉：既然 CLI 很小，性能应该不是问题。

例如，现有 `TaskStore` 里有两个典型操作：

```swift
guard let index = tasks.firstIndex(where: { $0.title == normalized }) else {
    throw TaskStoreError.taskNotFound(title: normalized)
}
```

以及：

```swift
private var nextID: Int {
    (tasks.map(\.id).max() ?? 0) + 1
}
```

在 starter state 里，这完全没问题。但如果你把它们直接外推到更真实的运行态，就要开始问：

- 任务数量上涨后，线性扫描是否成为热点？
- `map(\.id)` 每次添加时都新建数组，这个复制是否值得？
- `render(tasks:)` 每条命令都重新生成整段文本，是否会成为主要开销？
- 任务保存如果每次都全量编码，I/O 与序列化谁更贵？

性能心智不是“看到 O(n) 就紧张”，也不是“先不管，反正现在才三条”。它要求你先识别**可能的成本面（cost surface）**，再用测量把真假区分开。

## Swift 性能判断首先要看数据形状与复制语义

`TaskCore + TaskCLI` 这个项目特别适合练习 Swift 特有的性能直觉，因为它同时包含：

- 值类型任务模型
- 数组存储
- 字符串拼接
- actor 边界上的快照传递
- 可能的持久化编码与解码

对这些操作，第一步不是马上改数据结构，而是问：**这里有没有复制，复制发生在什么层？**

例如 `TaskStore` 作为 struct，内部持有 `[Task]`。这通常意味着：

- 读取快照时，不一定立刻深拷贝整个数组
- 多份值共享底层存储直到发生写入，这就是 Copy-on-Write（CoW）
- 一旦某个副本发生 mutation，真正复制成本才可能出现

这个语义非常重要。因为很多别的语言背景的人会对 Swift 的值类型走向两个误区：

- 误区 A：struct 一定很便宜，所以复制可以不管
- 误区 B：struct 一复制就一定非常贵，所以应该尽快全改成 class

这两个结论都太粗糙。真正该做的是：知道 CoW 会延迟成本，然后在热点路径上测它是否真的显现出来。

## 先识别 `TaskCore + TaskCLI` 的潜在热点

如果把 Part 4 的项目运行态想象成“异步加载 + actor 协调 + CLI 输出 + 文件保存”，那么最可能值得观察的热点通常在这几类地方：

### 1. 查找和更新路径

`markDone(title:)` 当前按标题线性扫描：

- 小数据集：简单、清楚、完全合理
- 大数据集或高频操作：可能成为累计热点

### 2. ID 生成路径

`nextID` 当前每次 `add` 都 `map + max`：

- 数据量小时，这点成本微不足道
- 若命令执行频繁，且每次都扫全表，可能开始变得可见

### 3. 渲染路径

CLI 每次都重新生成完整列表文本：

- 这让输出逻辑很清楚
- 但也意味着字符串分配、拼接和中间数组生成会反复发生

### 4. 持久化路径

真正 runtime 升级后，load/save 可能涉及：

- `Data` 分配
- 编码 / 解码
- 磁盘 I/O
- 失败恢复或临时文件策略

### 5. 并发边界路径

一旦引入 actor：

- 每次跨 actor 调用都有 hop 成本
- 返回大型快照时会出现额外传递与可能复制
- 过细的 actor API 可能导致“逻辑没多做什么，但来回切边界很多次”

这正是为什么前几章一直强调边界设计。边界如果切得过碎，性能上也会付出代价。

## 更强的状态：先测量，再决定哪里值得动手

Swift 工程里一个非常重要的成熟信号，是你开始把“我猜这里慢”替换成“我证明这里慢”。对当前项目，一个最低成本、但很有训练价值的做法，是先建立测量习惯。

例如，针对渲染或批量添加，你可以用 `ContinuousClock` 做非常轻量的测量：

```swift
import Foundation

let clock = ContinuousClock()
let duration = clock.measure {
    var store = TaskStore(tasks: largeFixture)

    for i in 0..<10_000 {
        _ = try? store.add(title: "task-\(i)")
    }
}

print(duration)
```

这类测量当然还不是完整 benchmark，但它已经比“凭感觉决定优化方向”强得多。它会逼你回答几个关键问题：

- 测的是不是代表性工作负载（representative workload）？
- 测量里是否包含 I/O，还是只测 CPU 路径？
- 数据集规模是不是只停留在 starter 的 3 条任务？

也就是说，**测量不是为了得到一个好看的数字，而是为了让你确认自己到底在测什么。**

## 别把“复制”理解成单纯坏事

很多性能讨论会把复制当成绝对负面，好像任何 snapshot、任何数组传递都是原罪。对 Swift 来说，这种判断太粗。

在 `TaskCore + TaskCLI` 里，复制有时反而是可靠性的朋友：

- 返回值快照能避免上层拿到可变内部状态
- actor 边界上的值传递能减少共享引用风险
- 渲染使用稳定 snapshot，能避免一边输出一边状态变化

这意味着性能判断必须同时考虑语义收益。你不能只看“有没有复制”，还要看“这次复制换来了什么正确性和隔离好处”。

成熟工程的做法通常是：

1. 先用复制换来清楚边界与正确语义
2. 若测量证明它成为热点，再做定点优化
3. 优化时尽量维持原有边界，而不是把整个系统倒回共享可变引用

这套顺序非常重要。否则你很容易为了减少一份快照，重新把 Actor 与 Sendability 的安全优势全丢掉。

## 一个具体例子：渲染优化应该如何被触发

假设你测量后发现 CLI 输出在大列表场景下确实占据明显时间，那么更强的版本可能不是“先改架构”，而是对局部热路径做定点处理：

```swift
func render(tasks: [Task]) -> String {
    var lines: [String] = []
    lines.reserveCapacity(tasks.count + 1)
    lines.append("Today's tasks")

    for (index, task) in tasks.enumerated() {
        lines.append("\(index + 1). \(task.cliLine)")
    }

    return lines.joined(separator: "\n")
}
```

这里的优化思路非常典型：

- 没改项目边界
- 没改变 `TaskCore` / `TaskCLI` 职责
- 只是减少中间数组增长与重复分配

这就是性能优化最理想的形状：**小、局部、可证明。**

相比之下，如果还没测量就因为担心字符串拼接，把整个渲染系统改成复杂 streaming architecture，那就已经脱离当前项目规模了。

## Actor hop、批量 API 与“过碎边界”的成本

前一章我们说过，actor 边界是安全优势；但它也不是零成本。

比如下面这种 CLI 路径：

```swift
let task = try await runtime.add(title: title)
let tasks = try await runtime.list()
```

它当然很清楚，但如果运行态更复杂、每个命令都需要多个跨 actor 往返，你就要开始问：这些边界是否切得太碎？

更强的设计有时会变成：

```swift
let result = try await runtime.addAndRender(title: title)
```

或者：

```swift
let snapshot = try await runtime.perform(.add(title))
```

重点不在于你一定要把 API 改成哪种，而在于建立一个性能判断：**安全边界本身也有成本，过细粒度的交互会积累 hop 开销。**

这说明性能优化并不只发生在 for-loop 或数组操作里，也发生在边界设计层。一个太碎的 runtime API，即使每个函数都“很干净”，整体上也可能变慢。

## 测量心智比微优化技巧更重要

真正进入工程后，性能改进的大头通常不是靠背几条低层小技巧，而是靠这套顺序：

1. 用真实或接近真实的数据规模识别热点
2. 分清 CPU、分配、I/O、并发边界哪一类成本更显著
3. 只改已证明值得改的局部路径
4. 优化后再次测量，确认收益真实存在

这套顺序会让你避开两类典型失误：

- 为了不存在的热点把代码写复杂
- 明明热点已经出现，却继续靠“应该还行”自我安慰

对 `TaskCore + TaskCLI` 当前阶段，最重要的不是立刻得出“应该换字典”或“应该缓存 rendered output”，而是会问：**我们有证据吗？当前瓶颈到底在哪里？**

## Swift 性能判断与前几章怎样连起来

这章如果读完以后仍然像独立技巧章，那就说明还没真正吃透。它和前几章其实是连续的：

- `async`/`await` 让等待点变得可见，也让 I/O 成本变得可测
- Actor / Sendability 让安全边界变清楚，也带来 hop 与 snapshot 成本
- ARC / ownership 让生命周期更稳，也影响对象数量、分配与释放成本

所以性能不是另一个世界，它只是前面所有语义选择在运行时的账单。

一个成熟的 Swift 工程师不是看到“账单”就后悔用了值语义、Actor 和清晰边界，而是会判断：哪些账单是在合理范围内，哪些账单已经大到值得优化。

## 双语关键词

- performance hotspot：性能热点
- cost surface：成本面
- Copy-on-Write / CoW：写时复制
- representative workload：代表性工作负载
- allocation：分配
- actor hop：跨 Actor 边界切换成本
- micro-optimization：微优化
- benchmark：基准测试
- measurement mindset：测量心智
- snapshot copy：快照复制

## 常见错误

### 1. 看见小项目就默认“性能还早”

Starter state 很小，不代表未来 runtime state 也永远小。你不需要现在就优化，但需要现在就学会识别成本面。

### 2. 一看值语义就认定复制一定贵

Swift 的 CoW 语义会延迟很多复制成本。真正重要的是测量实际热点，而不是从类型名字直接推断结论。

### 3. 把性能优化做成架构重写

如果局部热点只需要局部修正，就不要动整个系统边界。大多数可靠优化都应该是小而可证明的。

### 4. 只看 CPU，不看 I/O 和边界往返

在 runtime 升级后的 CLI 里，慢点可能来自磁盘、编码、Actor hop 或字符串构建。别把性能理解成只有算法复杂度。

## English Recap

This chapter teaches performance as a measurement discipline tied to Swift semantics. In `TaskCore + TaskCLI`, the relevant costs come from lookup paths, string rendering, persistence, actor hops, and copy-on-write behavior. The right order is to identify likely cost surfaces, measure representative workloads, and then apply small, evidence-backed optimizations.

## Drills

1. 结合当前项目，列出一个你认为“现在没问题但未来可能成为热点”的路径，并说明原因。
2. 解释为什么“有复制”不自动等于“设计错了”。
3. 设计一个最小测量实验，用来比较“只改内存中的 `TaskStore`”与“改完后再编码保存”的成本差异。

## Project Handoff

并发、隔离、ownership 和性能都已经进入视野，但真正让系统像工程而不是 demo 的，仍然是可靠性：取消时会怎样，部分失败时怎样，用户应该看到哪一层错误，哪些失败可以恢复，哪些必须中止。下一章我们就把 Part 4 的这些 runtime 主题收束到可靠性、取消与 failure surface 上。

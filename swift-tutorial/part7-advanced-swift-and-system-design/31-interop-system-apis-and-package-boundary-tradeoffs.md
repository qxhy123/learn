# 第31章：互操作、系统 API 与包边界取舍

> 到了高级阶段，Swift 工程真正难受的地方常常不在语言本体，而在边界交接处。你的共享核心要不要直接 `import Foundation`？CLI 和 SwiftUI 客户端什么时候该碰 `FileManager`、`UserDefaults`、日志系统、系统时间、通知中心？这一章要处理的，就是这些不够炫、却决定系统能否长期演进的互操作判断。

## 为什么这一章现在出现

前面几部分已经让我们建立了几条重要事实：

- `TaskCore` 应该是共享领域核心
- `TaskCLI` 和 `TaskFlow` 是两个不同客户端
- runtime、repository、preview、testing 都在持续塑造边界

只要系统开始真正碰外部世界，互操作（interop）压力就一定出现：

- CLI 要落盘，需要文件系统 API
- `TaskFlow` 要保存用户偏好，需要某种本地存储
- 系统要记录运行错误，需要日志 API
- 某些时间、路径、编码、解码能力来自 Foundation

如果没有包边界判断，这些东西会很快渗透到整个项目，最后让“共享核心”只剩名义上的共享。

## 从一个较弱起点开始：哪里用到系统能力，哪里就直接 `import`

弱状态通常非常自然，也非常危险：

```swift
import Foundation

struct TaskStore {
    mutating func save() throws {
        let data = try JSONEncoder().encode(tasks)
        try data.write(to: URL(filePath: "/tmp/tasks.json"))
    }
}
```

这段代码的问题不是“不能工作”，而是它把三种本来应该分开的东西揉在了一起：

- 领域模型 `TaskStore`
- 持久化格式与编码策略
- 文件系统路径与系统 API

一旦这样做，后面立刻会出现连锁反应：

- `TaskCore` 被迫依赖某个平台的路径约定
- `TaskFlow` 若想复用 `TaskStore`，也被顺手拖入文件系统细节
- 测试变得更难，因为系统 API 直接嵌在核心对象里

这就是为什么互操作问题不能被当成“只是实现细节”。它会直接重塑包边界。

## 更强的第一步：先把“领域事实”与“系统接触面”分开

对当前教程主线，更强的结构通常是：

- `TaskCore` 表达任务事实、规则、快照、变更
- adapter / repository 层承担与系统 API 的接触
- CLI 和 SwiftUI 各自消费经过边界整理后的能力

例如：

```swift
protocol TaskPersistence {
    func loadSnapshot() async throws -> TaskSnapshot
    func saveSnapshot(_ snapshot: TaskSnapshot) async throws
}
```

而具体系统实现可以放在边界外：

```swift
import Foundation

struct JSONFileTaskPersistence: TaskPersistence {
    let fileURL: URL

    func loadSnapshot() async throws -> TaskSnapshot {
        let data = try Data(contentsOf: fileURL)
        return try JSONDecoder().decode(TaskSnapshot.self, from: data)
    }

    func saveSnapshot(_ snapshot: TaskSnapshot) async throws {
        let data = try JSONEncoder().encode(snapshot)
        try data.write(to: fileURL)
    }
}
```

这样一来：

- `TaskCore` 不需要知道 `URL` 如何构造
- `TaskCLI` 可以选择磁盘文件实现
- `TaskFlow` 可以选择别的本地存储实现，甚至先用内存实现

共享核心因此保持共享，而不是被某个系统 API 拖偏。

## `Foundation` 不是洪水猛兽，但也不是默认应渗透 everywhere

很多 Swift 工程师在这个问题上会走两个极端：

- 极端一：`Foundation` 到处都 import，反正很方便
- 极端二：为了“纯净”，连 `Date`、`URL`、`Data` 都拒绝使用

对本教程，比较成熟的态度是：

- `Foundation` 是重要基础设施，不必妖魔化
- 但越靠近共享核心，越要谨慎决定是否直接依赖它

一个实用判断是：

- 若某类型表达的是通用数据现实，例如 `Date`、`UUID`、`Data`，进入共享边界可能是合理的
- 若某 API 表达的是平台或环境细节，例如 `FileManager`、`UserDefaults`、通知中心、UI 生命周期，则更适合停留在 adapter 或 client 边界

也就是说，问题从来不是“能不能 import Foundation”，而是**这份依赖会不会把核心语义和环境细节绑死在一起。**

## CLI 与 SwiftUI 客户端应怎样各自处理系统 API

这条主线里，一个非常重要的系统设计意识是：**不同客户端可以接触不同系统 API，但不应因此分叉领域现实。**

例如：

### `TaskCLI`

- 更自然地接触文件路径、标准输出、退出码、进程环境
- 更可能使用 `FileHandle`、命令行参数、日志输出

### `TaskFlow`

- 更自然地接触 app 生命周期、用户默认值、界面状态恢复、预览环境
- 更可能使用 `UserDefaults`、scene state、平台 UI API

这两条线各自有自己的系统表面，但它们不应该把这些平台差异重新写进 `TaskCore`。共享核心只该看到被整理后的领域快照、查询和变更接口。

## 包边界（package boundary）首先是依赖方向，不是目录分组

当互操作压力出现时，很多项目会本能地“再拆一个 package”。但真正重要的问题不是有没有更多目录，而是依赖方向是否清楚。

一个更稳的方向会是：

- `TaskCore` 不依赖客户端包
- `TaskCLI` 和 `TaskFlow` 依赖 `TaskCore`
- 系统适配层依赖 `TaskCore` 所定义的协议或快照形状
- 客户端只拿到自己需要的窄接口

这会让你在设计 package 时更容易做判断：

- 某个 Foundation adapter 是不是应该放在共享基础设施包
- 某个 SwiftUI 专用工具是不是只能留在 `TaskFlow`
- 某个 CLI logger 是否根本不该被 UI 端看到

真正成熟的包边界，是让系统 API 的使用被限制在正确层级，而不是让每一层都能直接摸到外部世界。

## 互操作时要特别警惕“类型便利性”反向污染语义边界

系统 API 经常会给你一种很强的便利感，比如：

- `URL` 很好用，于是所有资源定位都变成 `URL`
- `UserDefaults` 很简单，于是所有偏好与缓存都直接读写
- `NotificationCenter` 很顺手，于是跨层通信靠广播补丁

这些都很危险，因为它们容易让“技术上方便”的类型，反向占领“语义上应该清楚”的边界。

例如，对任务系统来说，“当前筛选条件”是领域或应用状态问题，不应因为 `UserDefaults` 容易用，就直接把 app model 写成：

```swift
filter = UserDefaults.standard.string(forKey: "filter")
```

更强的做法是：

- app model 持有清楚的 `TaskFilter`
- 偏好适配层负责把 `TaskFilter` 和 `UserDefaults` 互转

这类边界整理看似啰嗦，实际上是在保护系统长期可改。

## 系统 API 最好通过“语义适配器（semantic adapter）”进入系统

所谓语义适配器，不是额外造层，而是把外部技术表面翻译成系统自己的语言。

例如：

- 文件系统读写被翻译成 `loadSnapshot` / `saveSnapshot`
- 用户默认值被翻译成 `loadPreferences` / `savePreferences`
- 日志框架被翻译成 `record(event:)`

对教程读者来说，这类适配器尤其重要，因为它能持续提醒你：

- 共享核心说的是“任务系统如何工作”
- 外部 API 说的是“操作系统如何提供能力”

这两者的语言不应该混成一套。

## 互操作设计也要考虑测试与 preview

只要边界整理得足够清楚，测试和 preview 的好处会立刻体现出来。

例如：

- `TaskCLI` 的持久化测试可以用临时目录适配器或内存替身
- `TaskFlow` 的 preview 可以完全跳过 `UserDefaults` 和真实文件系统
- 共享核心测试根本不需要任何系统 API

这正是包边界判断最实用的收益之一：不是架构图更漂亮，而是验证成本显著下降。

## 从 Part 7 回看前六部分，系统 API 到底应该停在哪里

把整条主线串起来看，可以得到一个很有用的停靠图：

### `TaskCLI Lite`

早期允许更直接，因为目标是建立语言直觉，不是系统纯度。

### `TaskCore + TaskCLI`

开始严格区分 core behavior 与持久化、日志、环境依赖。

### `TaskFlow`

客户端可以拥抱平台 API，但要通过 app adapter 与状态层整理后再接入共享核心。

### Capstone 前夜

此刻已经可以认真重构“哪些系统依赖该留在边界、哪些共享语义值得往核心沉淀”。

这也解释了为什么本章必须出现在 Part 7 而不是更早。因为只有当三条项目线都出现后，你才能真正看见互操作和包边界的拉扯。

## 双语关键词

- interop：互操作
- system API：系统 API
- package boundary：包边界
- adapter：适配器
- semantic adapter：语义适配器
- dependency direction：依赖方向
- infrastructure：基础设施
- environment detail：环境细节
- persistence backend：持久化后端

## 常见错误

### 1. 哪里需要系统能力，哪里就直接把系统 API 写进核心类型

这会让共享核心迅速失去可移植性和可测试性。

### 2. 为了“纯净”而完全拒绝 `Foundation`

成熟工程要做的是控制依赖位置，而不是做象征性禁欲。

### 3. 把目录拆分误当成包边界已经合理

真正关键的是依赖方向和语义停靠点，而不是文件夹数量。

### 4. 让方便的系统类型反向决定领域模型形状

技术便利性不应凌驾于语义清晰度之上。

### 5. 让 CLI 与 SwiftUI 客户端各自偷偷实现一套系统交互逻辑

客户端可以接触不同系统 API，但不应各自改写共享核心含义。

## English Recap

Interop design is really boundary design. System APIs such as Foundation, file I/O, defaults, and logging should enter through adapters that translate environment details into shared domain-facing operations. Good package boundaries preserve dependency direction: the shared core stays focused on domain semantics, while CLI and SwiftUI clients interact with platform-specific APIs at their own edges.

## Drills

1. 在你的心里画出 `TaskCore`、CLI 持久化、SwiftUI 偏好存储三者的依赖方向，确认谁不该依赖谁。
2. 选一个系统 API，例如 `UserDefaults` 或 `FileManager`，把它翻译成一组更贴近任务系统语义的接口名。
3. 想一想如果 `TaskCore` 直接负责文件路径和 JSON 编解码，会给 `TaskFlow` 带来哪两类额外耦合。

## Project Handoff

现在我们已经把高级 Swift 从类型系统一路推进到了系统 API 与包边界。Part 7 的最后一章会进一步收束：当三条项目线都积累了自己的局部合理性后，哪些共享抽象真正值得提炼，系统又该如何做一次克制但清楚的 redesign，为 Part 8 的 capstone 铺路。

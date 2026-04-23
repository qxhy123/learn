# 第 13 章：组织 Swift Package 工作区

## 现在讨论目录结构，终于有现实意义了

如果在 Part 1 就开始大谈模块和 target，读者很容易把工程结构误解成一种格式化仪式。走到 Part 4，情况已经不一样了。你手里的 `FocusList` 不再只是一个 SwiftUI 页面集合，它开始有三条不同表面：

- 用户真正操作的 App
- 承接规则和模型的共享核心
- 用来验证复用边界的 CLI

这时工作区结构才真正决定未来修改是否痛苦。

## 先读懂当前 `Package.swift`

starter 已经给出了一个最小但正确的方向：

```swift
products: [
    .library(name: "FocusCore", targets: ["FocusCore"]),
    .executable(name: "FocusListApp", targets: ["FocusListApp"]),
    .executable(name: "focusctl", targets: ["focusctl"])
],
targets: [
    .target(name: "FocusCore"),
    .executableTarget(name: "FocusListApp", dependencies: ["FocusCore"]),
    .executableTarget(name: "focusctl", dependencies: ["FocusCore"]),
    .testTarget(name: "FocusCoreTests", dependencies: ["FocusCore"])
]
```

这段配置表达了一个非常重要的工程判断：

- `FocusCore` 是被复用的中心。
- `FocusListApp` 和 `focusctl` 都站在核心之上。
- 测试先围绕共享规则落脚，而不是围绕页面截图。

只要这条依赖方向保持稳定，后面的并发、测试和发布准备都会轻松很多。

## 用目录映射理解职责

你可以把工作区先看成这样一张图：

```text
Sources/
  FocusCore/
    models + rules + query objects
  FocusListApp/
    SwiftUI scenes + feature views + platform presentation
  focusctl/
    CLI commands + formatting
Tests/
  FocusCoreTests/
    stable behavior checks
```

现在最值得建立的习惯是：每想新增一个文件，先问它属于哪一层，而不是先问“放哪最顺手”。

## 一个实用判断法：看变化节奏

目录边界最容易做坏的时候，不是你完全不知道怎么分，而是你把变化节奏不同的代码放到了一起。判断一个东西该属于哪层时，可以问：

1. 它是不是产品规则？如果是，更接近 `FocusCore`。
2. 它是不是平台表达？如果是，更接近 `FocusListApp`。
3. 它是不是命令行输出或命令解析？如果是，更接近 `focusctl`。
4. 它会不会被 App 和 CLI 复用？如果会，优先考虑共享核心。

例如，“任务标题不能为空”是共享规则；“macOS 上 toolbar 怎么排按钮”是平台表达。把两者塞进同一层，后面每次改动都会互相牵扯。

## 工作区整理时别顺手制造新耦合

常见错误不是文件放错一次，而是为了图省事加出反向依赖。例如：

```text
FocusListApp  --->  FocusCore
focusctl      --->  FocusCore
FocusCore     -/->  FocusListApp
```

最后一行必须始终禁止。如果 `FocusCore` 开始依赖 `SwiftUI` 或某个页面视图，你的核心就已经不是核心了，只是换个目录继续耦合。

## 做完这一章的检查项

至少确认下面四件事：

1. `FocusCore` 不导入 `SwiftUI`。
2. App 和 CLI 都只从 `FocusCore` 读取共享规则。
3. 测试能在不启动 UI 的情况下验证关键行为。
4. 你能清楚说出每个 target 的职责边界。

## 本章小结

工作区组织的价值从来不是目录更整齐，而是让“规则、表面、验证”三件事开始各归其位。只要这层边界站稳，`FocusList` 才真正开始像一个可维护的代码库。

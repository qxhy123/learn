# 环境准备

本教程默认读者是已经会别的语言、现在要系统进入 Swift 的程序员。因此这里不从“怎么安装一个编辑器”开始，而是从“为了把整套教程顺利跑通，你最低需要哪些环境表面”开始讲。你不需要一次把所有 Apple 生态工具都装满，但你必须分清：

- 哪些内容只需要 Swift toolchain + SPM
- 哪些内容需要 Xcode
- 哪些内容只有在 macOS 上才能完整完成

## 先说结论

- Part 1 到 Part 4：可以以 `swift`、`swiftc`、Swift Package Manager 为主。
- Part 5 到 Part 8：如果要完整做 SwiftUI、Preview、Simulator 与 Xcode 调试，建议使用 macOS + Xcode。
- 整套教程的最稳妥环境：一台装有 Xcode 和 Swift toolchain 的 macOS 机器。

## 路线 A：以 macOS 为主的完整环境

这是最推荐路线，因为它能覆盖整套教程。

### 1. 安装 Xcode

用途：

- 提供 Swift compiler、SDK、Simulator、Instruments、SwiftUI Preview、XCTest 图形化工作流
- 为 Part 5 之后的 `TaskFlow` 提供最顺手的开发环境

安装后建议做三件事：

1. 首次启动 Xcode，接受许可协议并等待附加组件安装。
2. 打开 `Settings` / `Locations`，确认 Command Line Tools 指向当前 Xcode。
3. 在终端运行：

```bash
xcode-select -p
swift --version
xcodebuild -version
```

你至少应看到：

- `xcode-select -p` 能返回 Developer 目录
- `swift --version` 能正常输出版本
- `xcodebuild -version` 能返回 Xcode 与 Build version

### 2. 确认命令行工具可用

本教程中最常用的命令是：

```bash
swift --version
swift package --version
swift repl
swift build
swift test
swift run
```

如果这些命令在终端不可用，先不要进入教程正文。先修好工具链路径，因为 Part 1 的第一件事就是把 `swift` 和 `swiftc` 当真工具来理解。

### 3. 为 SwiftUI 准备 Simulator

如果你打算完整做 Part 5 到 Part 8，建议至少打开一次：

- Xcode
- Simulator
- 一个空白 SwiftUI App 模板

这样做的目的不是让你先写 app，而是验证：

- SDK 已安装
- 模拟器能启动
- Preview 和设备运行的基础链路没有明显问题

## 路线 B：先做 CLI / Core 的轻量环境

如果你当前只想完成 Part 1 到 Part 4，可以先用更轻量的方式进入：

- 安装 Swift toolchain
- 使用终端 + 你熟悉的编辑器
- 用 SPM 构建和测试 package

你仍然应该先验证：

```bash
swift --version
swift package init --type executable
cd <your-package>
swift build
swift run
swift test
```

如果这些环节跑不通，先不要进入 `TaskCLI Lite` 或 `TaskCore + TaskCLI` 的章节。

## Xcode 与 SPM 在本教程里的分工

很多新读者会混淆这两者的职责。可以这样记：

- Xcode：IDE、SDK、Simulator、调试和 Apple 平台工作流入口
- SPM：package、target、dependency、build/test/run 的工程组织工具

本教程刻意让你先通过 SPM 理解工程表面，再进入 SwiftUI 和 Xcode。原因很简单：如果你先把 Xcode 当成一切，后面很容易学成“会点按钮，不会解释边界”。

## 推荐的最小终端检查表

在进入正文之前，至少跑一遍：

```bash
swift --version
swift package --version
swift package init --type executable
swift build
swift run
```

再进入一个带测试的 package，确认：

```bash
swift test
```

对 Part 1 到 Part 4 来说，这比“装了多少 Apple 工具”更重要。

## 推荐的最小 Xcode 检查表

在进入 Part 5 前，至少确认：

- Xcode 可以打开 package 或 project
- SwiftUI Preview 能编译
- Simulator 能启动一个最小 app
- XCTest 可以在 Xcode 内运行

你不必先把所有 Xcode 功能学完，但你至少要确认工具链没有在基本路径上阻塞你。

## 常见环境问题

### 1. `swift` 和 Xcode 版本看起来对不上

通常原因：

- 系统里装过多个 Xcode
- 当前 `xcode-select` 没指向你想用的那个版本
- 终端读到的不是你预期的 toolchain

先检查：

```bash
xcode-select -p
which swift
swift --version
```

### 2. `swift build` 能跑，SwiftUI Preview 却不工作

这很常见，因为两者依赖的环境不一样：

- `swift build` 主要验证 CLI / package 构建链
- Preview 还依赖 Xcode、SDK、目标平台和 UI 构建链

不要把前者通过误认为后者也一定没问题。

### 3. Package 在终端能测试，Xcode 里却打不开或索引异常

先做基础排查：

- 重启 Xcode
- 重新打开 `Package.swift`
- 确认 Command Line Tools 指向当前 Xcode
- 清理 Derived Data

只有在这些基础动作都无效时，才值得继续深挖 Xcode 特定问题。

## macOS 之外的说明

Swift 的语言和 SPM 不只存在于 macOS，但本教程后半段包含 SwiftUI、Preview、Simulator 与 Apple 平台工作流，因此：

- 如果你只做 CLI / Core，可以在非 macOS 环境先完成前半段
- 如果你要完整完成 `TaskFlow` 与 `Capstone`，仍建议使用 macOS + Xcode

这不是平台偏见，而是因为 SwiftUI 和相关工具链本来就属于 Apple 平台表面。

## 进入教程前的最后检查

在开始 Part 1 前，你至少要能做到：

- 运行 `swift`
- 用 SPM 新建、构建、运行、测试一个 package

在开始 Part 5 前，你还要能做到：

- 打开 Xcode
- 运行最小 SwiftUI app 或 preview

如果这两组检查都通过，说明你的环境已经足以支撑这套教程。

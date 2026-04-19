# Swift Package Manager 速查

这份速查不试图覆盖 SPM 的每个角落，只聚焦本教程里真正高频、真正值得反复记住的日常命令和模式。你应该把它当成工程动作速查，而不是 API 字典。

## 最常用命令

### 初始化 package

```bash
swift package init --type executable
swift package init --type library
```

什么时候用：

- `executable`：Part 1 的 `TaskCLI Lite`
- `library`：只想先搭共享模块骨架时

### 构建、运行、测试

```bash
swift build
swift run
swift run TaskCLI list
swift test
```

记忆方式：

- `build` 看能不能编
- `run` 看能不能执行
- `test` 看 contract 有没有被守住

### 查看 package 信息

```bash
swift package describe
swift package dump-package
```

用途：

- `describe`：人类可读地看 package shape
- `dump-package`：更结构化地看 manifest 结果

### 解析与更新依赖

```bash
swift package resolve
swift package update
```

使用原则：

- `resolve`：让依赖解析到当前约束允许的结果
- `update`：主动尝试更新到更新版本

教程前期一般不鼓励无必要加依赖，所以这两个命令不会像 `build` / `test` 那样高频。

### 清理构建产物

```bash
swift package clean
```

当你怀疑构建缓存或中间产物影响结果时再用，不要把它当护身符。

## `Package.swift` 最常见骨架

### 一个 library + 一个 executable + 一个 test target

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TaskSystem",
    products: [
        .library(name: "TaskCore", targets: ["TaskCore"]),
        .executable(name: "TaskCLI", targets: ["TaskCLI"]),
    ],
    targets: [
        .target(name: "TaskCore"),
        .executableTarget(
            name: "TaskCLI",
            dependencies: ["TaskCore"]
        ),
        .testTarget(
            name: "TaskCoreTests",
            dependencies: ["TaskCore"]
        ),
    ]
)
```

这是本教程 Part 3 以后最重要的 package pattern。

## 目录约定

### 默认目录

SPM 约定俗成的结构：

```text
Package.swift
Sources/
Tests/
```

进一步展开常见为：

```text
Sources/TaskCore/
Sources/TaskCLI/
Tests/TaskCoreTests/
```

理解重点：

- target 名通常对应 `Sources/<TargetName>`
- test target 名通常对应 `Tests/<TargetName>Tests`

### 为什么本教程重视默认约定

因为默认结构本身就在帮你减少“工程噪音”。初学阶段别急着为目录美学自定义 `path:`，除非你真的有清楚理由。

## 常见 target 模式

### 模式 1：单 executable

适合：

- Part 1
- 小 CLI
- 教学早期最小程序

特点：

- 简单直接
- 适合先建立语言直觉
- 不适合长期堆核心模型和多边界责任

### 模式 2：library + executable

适合：

- Part 3 到 Part 4
- 想把核心规则和命令行入口分离

特点：

- `TaskCore` 可测试
- `TaskCLI` 保持为翻译层
- 为后续多个客户端复用做准备

### 模式 3：shared core + multiple clients

适合：

- 后期系统设计
- CLI + SwiftUI 并存

要点：

- 共享的是 domain contract，不是所有实现细节
- 客户端差异应停留在客户端，不要反向污染 core

## 常见 manifest 片段

### 添加平台限制

```swift
platforms: [
    .macOS(.v14),
    .iOS(.v17)
]
```

什么时候值得写：

- 你的 package 真的依赖某些平台能力
- SwiftUI / Apple SDK 路径需要明确部署目标

### 添加本地 target 依赖

```swift
.executableTarget(
    name: "TaskCLI",
    dependencies: ["TaskCore"]
)
```

### 添加外部 package 依赖

```swift
dependencies: [
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.0.0")
]
```

对应 target 里引用：

```swift
.executableTarget(
    name: "TaskCLI",
    dependencies: [
        .product(name: "ArgumentParser", package: "swift-argument-parser")
    ]
)
```

本教程默认不急着这样做。先自己把 CLI 表面讲清楚，再决定要不要依赖外部库。

### 添加资源

```swift
.target(
    name: "TaskCore",
    resources: [
        .process("Resources")
    ]
)
```

适用场景：

- 测试样例数据
- SwiftUI 资源
- 需要随 target 一起打包的文件

## 日常工作流建议

### CLI / Core 开发

最常见循环：

```bash
swift test
swift build
swift run TaskCLI list
```

顺序意义：

- 先看行为 contract
- 再看是否能构建
- 最后看入口表面是否仍然成立

### 改 `Package.swift` 后

优先做：

```bash
swift package resolve
swift build
swift test
```

### 怀疑边界放错时

先用：

```bash
swift package describe
```

这会逼你重新看当前 products、targets、dependencies 的形状，而不是盯着某个文件局部发呆。

## 本教程最重要的 SPM 判断

1. package 是工程边界工具，不是“构建能过就行”的脚手架。
2. target 应按责任拆，不按“看起来专业”拆。
3. test target 的价值在于锁 contract，不在于让目录显得完整。
4. 依赖越多不等于工程越成熟；边界越清楚才更成熟。

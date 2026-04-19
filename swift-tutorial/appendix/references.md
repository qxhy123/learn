# 参考资料

这份参考资料分三类：

- 官方 Swift 文档：优先建立语言、SPM、安装与语言演进的基线
- Apple 官方文档：用于 Xcode、XCTest、SwiftUI 与 Apple 平台工作流
- 选读外部资料：用来补强心智模型或补充工程视角

原则很简单：**先官方，后外部；先概念与 contract，后技巧与花样。**

## 官方 Swift 文档

### 1. Install Swift

链接：https://www.swift.org/install/

为什么值得看：

- 这是 Swift 官方安装入口
- 涵盖 macOS、Linux、Windows 的安装路径
- 适合在做本教程环境准备前先确认 toolchain 获取方式

### 2. Getting Started

链接：https://www.swift.org/getting-started/

为什么值得看：

- 官方入门入口页
- 适合确认安装完成后的起步路径
- 对刚进入 Swift 生态的读者有导航价值

### 3. The Swift Programming Language: Language Guide Index

链接：https://docs.swift.org/swift-book/LanguageGuide/

为什么值得看：

- 这是 Swift 语言指南索引
- 适合按主题回查 The Basics、Control Flow、Protocols、Generics、Concurrency 等核心章节
- 在本教程里，最适合做“概念回查”，不适合替代系统学习顺序

### 4. The Basics

链接：https://docs.swift.org/swift-book/LanguageGuide/TheBasics.html

为什么值得看：

- 回查 Part 1 的值、类型、`Optional`、基础语法时很方便
- 当你想确认“语言原义”而不是教程改写时，这是最直接的入口

### 5. Concurrency

链接：https://docs.swift.org/swift-book/LanguageGuide/Concurrency.html

为什么值得看：

- 回查 `async` / `await`、task、actor 等语言级并发能力
- 适合作为 Part 4 与 Part 7 的官方语义底座

### 6. Swift Package Manager Index

链接：https://docs.swift.org/package-manager/index.html

为什么值得看：

- SPM 官方文档入口
- 适合在 Part 3 回查 package、manifest、target、dependency 等主题

### 7. PackageDescription API

链接：https://docs.swift.org/package-manager/PackageDescription/index.html

为什么值得看：

- 查 `Package.swift` API 时最直接
- 写 manifest 时，比翻博客更可靠

## Apple 官方文档

### 8. Xcode Documentation

链接：https://developer.apple.com/documentation/xcode

为什么值得看：

- Xcode、Simulator、build/test/debug 文档总入口
- 做 Part 5 以后时，几乎一定会回到这里查工具行为

### 9. XCTest Documentation

链接：https://developer.apple.com/documentation/xctest/

为什么值得看：

- XCTest 的官方入口
- 用来回查断言、异步测试、性能测试、测试组织方式
- 即使 Apple 现在也在推进 Swift Testing，XCTest 仍是大量项目的现实表面

### 10. Asynchronous Tests and Expectations

链接：https://developer.apple.com/documentation/xctest/asynchronous_tests_and_expectations

为什么值得看：

- Part 4 以后做异步测试时很常用
- 能帮助你把“并发逻辑会跑”升级为“并发 contract 可验证”

### 11. Swift Concurrency Collection

链接：https://developer.apple.com/documentation/swift/concurrency

为什么值得看：

- Apple 对 Swift 并发 API 的官方集合页
- 当你已经知道概念名，想快速回查具体类型或协议时很好用

### 12. SwiftUI Documentation

链接：https://developer.apple.com/documentation/SwiftUI

为什么值得看：

- SwiftUI 文档主入口
- 适合回查 `View`、list、form、navigation、modifier、数据流 API

### 13. SwiftUI Apps Technology Overview

链接：https://developer.apple.com/documentation/technologyoverviews/swiftui

为什么值得看：

- 比 API 参考更偏“应用结构”视角
- 对 Part 5 与 Part 6 的 app 入口、scene、初始化和交互有帮助

### 14. SwiftUI Pathway / Get Started

链接：https://developer.apple.com/pathways/swiftui/

为什么值得看：

- 适合第一次进入 SwiftUI 官方学习路径时做导航
- 可以作为 Part 5 之后的补充练习入口

## 选读外部资料

### 15. Thinking in SwiftUI · objc.io

链接：https://www.objc.io/books/thinking-in-swiftui

为什么值得看：

- 它强调 SwiftUI 心智模型，而不是 API 穷举
- 对已经会别的语言、但想真正理解 SwiftUI 更新与布局机制的程序员特别有帮助
- 很适合作为 Part 5、Part 6 的补充材料

### 16. Point-Free Concurrency Collection

链接：https://www.pointfree.co/collections/concurrency

为什么值得看：

- 更偏工程语境地讨论并发设计、时序与可测试性
- 如果你做完 Part 4 还想继续深挖并发与测试的交叉问题，这组材料很有价值

### 17. Creating a command line tool using the Swift Package Manager · SwiftLee

链接：https://www.avanderlee.com/swift/command-line-tool-package-manager/

为什么值得看：

- 对 CLI + SPM 的最小落地路径讲得直白
- 适合作为 Part 1 和 Part 3 之间的辅助阅读
- 重点不是完全照做，而是对照本教程观察我们为何更强调边界与 contract

### 18. Debugging SwiftUI views: what caused that change? · SwiftLee

链接：https://www.avanderlee.com/swiftui/debugging-swiftui-views/

为什么值得看：

- 专门针对 SwiftUI 刷新与状态变化的调试思路
- 如果你在 Part 5 或 Part 6 频繁遇到“为什么它又重绘 / 不重绘”，这篇文章很有帮助

## 如何使用这些参考资料

更推荐这样用：

1. 先完成教程当前章节与对应 lab。
2. 遇到概念模糊，再回官方文档查语言原义或 API 定义。
3. 遇到心智模型卡住，再读外部资料补视角。

不推荐这样用：

- 一上来在官方文档里随机跳章节
- 把博客当成语言规范
- 看到新 API 就立刻改写当前项目主线

## 结尾提醒

真正高价值的参考资料，不是让你“知道更多链接”，而是让你在需要验证概念、API 或设计判断时，知道应该先去哪里查、为什么先查那里。

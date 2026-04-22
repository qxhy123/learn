# 环境准备

## 开发环境

推荐使用较新的 macOS 和 Xcode，保证 `Swift 6`、`SwiftUI`、`Observation`、`SwiftData` 和 `Swift Testing` 都可以直接使用。教程默认你能打开 Xcode，也能在终端里运行 `swift`、`swift build`、`swift test`。

## Swift 6 与 Xcode 要求

- Xcode：建议使用支持 `Swift 6` 的稳定版本
- 命令行工具：确保 `xcode-select -p` 指向正确的开发者目录
- 验证命令：

```bash
swift --version
xcodebuild -version
```

## 命令行工具

虽然教程是 App-first，但命令行工具仍然是工程面的一部分。建议提前熟悉：

- `swift build`
- `swift test`
- `git status`
- `rg`

这些命令会贯穿 `FocusCore`、`focusctl` 和验证脚本的使用过程。

# FocusList Starter：起始工程

## 这个 starter 包含什么

starter 包里放的是教程早期会一直使用的最小工程面：

- `FocusListApp`：SwiftUI 应用壳
- `FocusCore`：最小领域模型和状态存储
- `focusctl`：基于共享核心的轻量命令行入口

## 如何验证

```bash
cd swift-tutorial/projects/focuslist/starter
swift test
swift build --product FocusListApp
swift build --product focusctl
```

## 当前阶段期待

在教程前两部分，这个 starter 的目的不是“功能已经完整”，而是“结构足够清楚，能承接后续产品和工程上的增长”。

# SwiftUI 速查

## View 与状态

- `@State`：局部、短生命周期的界面状态
- `@Bindable`：把可观察模型暴露给 `TextField`、`Toggle` 等控件
- `@Environment`：注入上层共享上下文，但不应用来偷渡任意业务依赖

## 容器与导航

- `VStack` / `HStack` / `ZStack`：最基础的结构容器
- `List`：数据驱动的滚动列表
- `Form`：结构化输入面
- `NavigationStack` / `NavigationSplitView`：页面流和分栏导航

## 数据流

判断状态放哪的顺序建议是：

1. 这是不是只影响当前 View 的局部状态？
2. 这是不是多个屏幕共享的产品状态？
3. 这是不是已经应该进入 `FocusCore` 的领域规则？

如果一开始就想上 ViewModel、Coordinator、Repository 全家桶，通常是时机不对。

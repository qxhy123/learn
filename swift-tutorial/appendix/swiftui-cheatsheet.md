# SwiftUI 速查

## 本教程里最常用的容器

| API | 什么时候用 | 在 FocusList 里通常负责什么 |
| --- | --- | --- |
| `NavigationSplitView` | 需要稳定侧栏与详情关系时 | 建立 macOS 或大屏产品骨架 |
| `NavigationStack` | 需要推进式导航时 | 更适合 iOS 的逐层进入体验 |
| `List` | 展示动态集合时 | 任务列表、项目列表 |
| `Form` | 组织结构化输入时 | 设置页、任务编辑器 |
| `VStack` / `HStack` | 组织局部布局时 | 任务行、工具栏、输入区 |

## 本教程里最重要的状态工具

| 工具 | 什么时候用 | 常见误用 |
| --- | --- | --- |
| `@State` | 当前视图拥有的临时交互状态 | 把共享产品数据也塞进去 |
| `@Bindable` | 视图要与可观察模型协作时 | 把它误当成“任何状态都能双绑” |
| `@Environment` | 需要读取上层注入上下文时 | 让依赖来源变得过于隐蔽 |
| `.searchable` | 页面需要原生搜索入口时 | 搜索结果与搜索词都变成全局状态 |
| `.task(id:)` | 输入变化或页面出现时启动异步工作 | 忘记任务会被替换或取消 |

## 一条快速判断规则

- 只对当前页面临时有效：先考虑 `@State`
- 多个页面都会依赖：先考虑共享模型
- 还需要跨会话保存：继续问是否该进入持久化层

## 一个最小示例

```swift
@State private var searchText = ""

var filteredTasks: [FocusTask] {
    store.inboxTasks.filter { task in
        searchText.isEmpty || task.title.localizedCaseInsensitiveContains(searchText)
    }
}
```

这段代码里，`searchText` 是局部状态，`filteredTasks` 是派生结果，底层任务数据仍然属于共享状态。

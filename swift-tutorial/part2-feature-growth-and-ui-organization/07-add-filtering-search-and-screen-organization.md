# 第 7 章：加入筛选、搜索与界面组织

## 当功能变多，真正的难点是状态层级

给应用加一个搜索框并不难，难的是它一出现，系统里立刻多出一批新状态：

- 搜索词
- 当前筛选条件
- 选中的导航入口
- 派生出来的结果集

如果这些东西全塞进 `FocusStore`，产品会被局部交互噪音污染；如果又全留在页面里，跨页面视角会变得很难保留。所以这一章的核心不是 API，而是状态层级判断。

## 先把“筛选条件”和“结果集”分开

一个非常稳的起点，是让页面持有当前查询条件，再从共享数据推导结果：

```swift
enum InboxFilter: String, CaseIterable, Identifiable {
    case all
    case openOnly
    case doneOnly

    var id: Self { self }
}

@State private var searchText = ""
@State private var filter: InboxFilter = .all

var filteredTasks: [FocusTask] {
    store.inboxTasks.filter { task in
        let matchesText = searchText.isEmpty || task.title.localizedCaseInsensitiveContains(searchText)
        let matchesFilter = switch filter {
        case .all: true
        case .openOnly: !task.isDone
        case .doneOnly: task.isDone
        }
        return matchesText && matchesFilter
    }
}
```

这里有两条关键判断：

1. `searchText` 和 `filter` 是页面级交互状态。
2. `filteredTasks` 是从共享产品数据推导出来的结果，不需要额外存一份。

只要你把“条件”和“结果”分清，大部分搜索和筛选代码都会自然稳定下来。

## 把搜索接回 SwiftUI，而不是自创一套输入流

如果当前目标是 `Inbox`，最自然的做法通常是直接把搜索与页面绑定：

```swift
List(filteredTasks) { task in
    TaskRow(task: task, projectName: projectName(for: task))
}
.searchable(text: $searchText, prompt: "Search tasks")
.toolbar {
    Picker("Filter", selection: $filter) {
        ForEach(InboxFilter.allCases) { value in
            Text(value.rawValue.capitalized).tag(value)
        }
    }
}
```

现在你做的不是“多加一个控件”，而是给当前页面增加一个新的阅读视角。搜索框负责描述用户正在找什么，筛选器负责描述用户想保留哪一类任务。

## 界面组织必须跟着变化

当搜索和筛选出现后，侧栏已经不再只是“页面目录”。它开始承担产品地图的职责。一个稳妥的组织方式通常是：

- 侧栏负责切换大的产品视角，例如 `Inbox`、`Today`、`Projects`。
- 页面内部负责控制当前视角下的临时搜索与筛选。
- 如果某个筛选条件需要跨页面持续生效，再考虑把它上提到更高层。

这条规则非常重要，因为它能阻止一个常见坏味道：为了图省事，把所有视图状态都塞进 `FocusStore`。

## 什么时候应该把筛选升高

只有在下面这种场景里，你才该认真考虑让筛选条件脱离单页：

- 用户切换到别的入口后，返回时希望保留原筛选。
- 多个页面共享同一类视角，例如“只看高优先级任务”。
- 该筛选已经不是页面局部交互，而是产品级工作模式。

如果只是当前页面临时搜索标题，局部 `@State` 通常就是正确答案。

## 一次完整检查，确认不是“加了输入框就完事”

做完这一章后，手动走一遍：

1. 在 `Inbox` 里输入搜索词，结果应即时变化。
2. 切换筛选条件，确认只影响当前结果集，不直接改动底层数据。
3. 进入 `Projects` 再返回 `Inbox`，观察哪些状态应该保留、哪些应该清空。
4. 问自己：当前的侧栏是否已经能表达“用户正在切到哪种产品视角”？

只要第 4 点你说不清，说明界面组织还没跟上功能增长。

## 本章小结

筛选和搜索会暴露所有权问题。只要你能稳定区分局部交互状态、共享产品状态和派生结果，`FocusList` 的页面结构就会开始像真正的产品，而不是一堆功能碎片。

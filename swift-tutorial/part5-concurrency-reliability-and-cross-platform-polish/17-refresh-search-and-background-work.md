# 第 17 章：刷新、搜索与后台工作

## 并发首先会把“谁拥有状态”这个问题放大

一说到并发，很多人先想到 `async/await` 语法，真正让产品变难的通常却是另一些问题：

- 用户刚输入新搜索词，上一轮搜索结果回来怎么办？
- 页面离开时，后台刷新任务还要不要继续？
- 刷新中、完成后、失败时，界面分别显示什么？

这些都不是语法问题，而是所有权问题。只要说不清某个异步任务由谁启动、谁取消、谁消费结果，它迟早会变成产品异常。

## 先给异步状态一个明确宿主

对 `Inbox` 这类页面，一个稳妥做法是由页面或页面模型持有异步过程状态：

```swift
@State private var searchText = ""
@State private var isRefreshing = false
@State private var refreshError: String?
@State private var searchResults: [FocusTask] = []
```

这几个状态虽然都和异步有关，但职责不一样：

- `searchText` 是用户输入。
- `isRefreshing` 表示当前有没有任务在跑。
- `refreshError` 表示最近一次失败。
- `searchResults` 是异步任务产出的页面结果。

## 用任务 ID 或输入值绑定刷新生命周期

对搜索来说，一个很好用的模式是让任务跟输入值绑定：

```swift
.task(id: searchText) {
    guard !searchText.isEmpty else {
        searchResults = store.inboxTasks
        return
    }

    isRefreshing = true
    defer { isRefreshing = false }

    do {
        searchResults = try await searchService.search(text: searchText)
        refreshError = nil
    } catch is CancellationError {
        // 用户继续输入时，这次搜索被新的任务替代
    } catch {
        refreshError = "Search failed. Try again."
    }
}
```

这种写法的核心价值是：输入一变，旧任务就自动失效。你不需要手动维护一堆“当前是否还是最新请求”的旗子。

## 后台工作别直接撞进共享状态

另一个常见坏味道是：异步任务一拿到结果，就立即把共享状态全部重写。这样做很容易让页面闪烁、选择丢失、滚动位置跳变。更稳妥的节奏通常是：

1. 后台任务先得到结果。
2. 决定这个结果是否仍然有效。
3. 再把它合并回当前页面或共享核心。

这个“先判断结果是否仍然有效”的步骤，就是并发里最容易被漏掉的产品判断。

## 做一次刷新体验检查

完成这一章后，至少手动检查：

1. 用户连续输入搜索词时，旧结果不会覆盖新结果。
2. 页面离开时，不会留下还在乱改状态的后台任务。
3. 刷新中、失败、完成三种状态的界面语义不同。

只要第三点没做到，用户看到的就仍然只是一块“偶尔会转圈的列表”。

## 本章小结

并发不会自动把产品变高级，它只会放大你原本对状态所有权的判断。真正重要的不是你会不会写 `Task`，而是你能不能让异步工作进入已有数据流，而不是把已有结构冲散。

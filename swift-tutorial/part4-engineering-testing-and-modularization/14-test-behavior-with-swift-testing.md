# 第 14 章：用 Swift Testing 锁定行为

## 从这一章开始，测试不再是附赠品

没有测试时，你写出来的是“现在看起来没问题”的代码；有测试并且测试点选得对，你写出来的才是“未来重构时仍然敢动”的代码。Part 4 的第一步，就是把前面几部分积累的产品判断，翻译成稳定可验证的行为。

## 先测共享核心，而不是先追 UI

当前 starter 的测试已经给了你一个正确方向：

```swift
@Test func addTaskStoresItInInbox() {
    let store = FocusStore()
    store.addTask(title: "Write first SwiftUI screen")
    #expect(store.inboxTasks.count == 1)
    #expect(store.inboxTasks[0].title == "Write first SwiftUI screen")
}
```

这条测试之所以值钱，不是因为它有两个断言，而是因为它锁定了一个产品事实：新增任务后，任务确实进入收件箱。换成别的内部实现，这条事实也应该不变。

## 先写“最容易被改坏的行为”

围绕 `FocusStore`，你应该优先覆盖下面几类测试：

- 新增任务时的正常路径
- 空白输入是否被拒绝
- 切换完成状态是否只影响目标任务
- 未知任务 ID 是否保持系统稳定

例如，空白输入这个边界非常值得锁住：

```swift
@Test func blankTaskTitleIsIgnored() {
    let store = FocusStore()

    store.addTask(title: "   ")

    #expect(store.inboxTasks.isEmpty)
}
```

这种测试的价值很高，因为它正好保护了一条未来很容易在重构时被破坏的规则。

## 失败先行，比“写完顺手补测”更稳

如果你准备给 `FocusStore` 加一个 `updateTask` 或 `tasks(matching:)` 方法，先写失败测试再写实现，往往会让边界更清楚。一个简单流程是：

1. 先写出你期待的行为。
2. 运行测试，确认它失败得合理。
3. 写最小实现让测试通过。
4. 再补一个异常输入测试，确认边界也被锁住。

这不是教条，而是让你避免一种常见坏味道：实现已经写很大，测试只能被动围着实现细节打补丁。

## 测试名应该说产品，不要说代码结构

下面这种名字通常更有价值：

- `addTaskStoresItInInbox`
- `blankTaskTitleIsIgnored`
- `togglingUnknownTaskDoesNothing`

而下面这种通常太贴实现：

- `focusStoreArrayAppendWorks`
- `toggleCompletionCallsIndexLookup`

测试名越贴近产品行为，未来重构时价值越大。

## 别把实现细节也一起焊死

测试该保护的是“系统对外承诺了什么”，不是“你今天刚好怎么写”。例如你应该断言“新增后任务数量变成 1”，而不是断言“内部必须调用某个私有 helper”。后者会让测试变成重构阻力，而不是重构保险。

## 一次测试清单

做完这一章，确保至少有这类测试：

1. 正常输入会产生正确结果。
2. 非法输入不会污染状态。
3. 未知对象不会让系统崩坏。
4. 关键切换操作不会影响无关数据。

如果你已经能围绕这四类思路补测试，说明你不是在“给代码做装饰”，而是在给产品规则上锁。

## 本章小结

测试真正训练你的，是把模糊感觉翻译成明确承诺。只要 `FocusCore` 的关键行为被锁定，后面做 CLI、并发和收尾重构时，你就会明显更敢动代码。

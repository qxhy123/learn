# Swift Testing 速查

## 基础断言

```swift
import Testing

@Test func titleIsStored() {
    let store = FocusStore()
    store.addTask(title: "Draft release notes")
    #expect(store.inboxTasks.count == 1)
}
```

## 参数化测试

```swift
@Test(arguments: ["Inbox", "Projects", "Anytime"])
func acceptsCommonBuckets(_ name: String) {
    #expect(!name.isEmpty)
}
```

## 异步测试

```swift
@Test func asyncRefreshCompletes() async throws {
    let value = try await Task.sleep(for: .milliseconds(1))
    #expect(value == ())
}
```

在本教程里，测试不是“写完代码再补一点”，而是帮助你锁定边界和失败面的主要工具。

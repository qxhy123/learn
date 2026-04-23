# Swift Testing 速查

## 最小结构

```swift
import Testing

@Test func example() {
    #expect(true)
}
```

## 本教程里最值得优先测试什么

1. 共享核心的关键行为
2. 容易在重构时被破坏的边界输入
3. 未知对象或失败路径是否保持系统稳定

## 常用断言思路

| 目标 | 示例 |
| --- | --- |
| 验证结果正确 | `#expect(store.inboxTasks.count == 1)` |
| 验证边界输入被拒绝 | `#expect(store.inboxTasks.isEmpty)` |
| 验证异常输入不破坏状态 | `#expect(store.inboxTasks == before)` |

## 一个值得保留的模式

```swift
@Test func togglingUnknownTaskDoesNothing() {
    let store = FocusStore()
    let before = store.inboxTasks

    store.toggleCompletion(UUID())

    #expect(store.inboxTasks == before)
}
```

这类测试的价值，在于它保护系统面对异常输入时仍然稳定。

## 不要怎么测

- 不要只测最表面的正常路径
- 不要把私有实现细节当成公开承诺
- 不要因为 UI 难测，就完全放弃对共享规则的保护

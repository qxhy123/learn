# 第8章：Binding 与状态拥有关系

## 为什么现在必须讲 `Binding`

到这里，`BoardFlow` 已经不只是“显示一些内容”的页面了。它已经开始具有：

- 列表
- 选择
- 输入
- 多栏结构

一旦这些同时出现，一个典型问题就会立刻冒出来：

**父视图持有的状态，子视图能不能改？如果能，怎么改才不把状态边界搞乱？**

这就是 `Binding` 的位置。

## `Binding` 不是第二份状态，而是对已有状态的可写借用

很多人第一次学 `@Binding` 时，脑子里会默认成：“子 view 也有这份状态，只不过同步一下。”  
这正是最危险的误解。

更强的理解是：

**`Binding` 不是复制状态，而是把父层拥有的状态，以可写借用的方式交给子 view。**

例如：

```swift
struct BoardCreationForm: View {
    @Binding var draftTitle: String
    @Binding var template: BoardTemplate
    @Binding var showDotGrid: Bool
}
```

这里真正拥有这些状态的不是 `BoardCreationForm`，而是更上层的 screen model 或父 view。子 view 拿到的只是“我可以改，但我不拥有”。

## 从一个较弱起点开始：子视图偷偷再开一份 `@State`

最常见的错误长这样：

```swift
struct BoardCreationForm: View {
    @State private var draftTitle = ""
}
```

如果父层本来已经有创建草稿，这样做就会立刻出现双事实问题：

- 父层一份标题
- 子层一份标题
- 看起来都叫 `draftTitle`
- 实际却可能已经分叉

这类问题在 UI 还小的时候不明显，一旦进入：

- sidebar 选择
- inspector 编辑
- detail 区同步
- 保存和取消

它就会变成灾难。因为你再也说不清“哪一份才是真的”。

## 先问归属，再决定要不要传 `Binding`

关于 `Binding`，一个特别稳的判断顺序是：

1. 这份状态是谁创建的？
2. 这份状态是谁拥有的？
3. 子视图是否真的需要修改它？
4. 子视图修改时，是直接改真源，还是应该先有编辑缓冲？

这第四点特别重要。不是所有“子视图想编辑”的场景都应该直接传 `Binding`。例如：

- 临时草稿编辑
- 需要取消 / 确认的面板
- 表单提交前的校验

这些场景里，你往往更需要的是一份局部编辑缓冲，而不是对子系统真源的直接双向写入。

## `BoardFlow` 的状态边界应该怎样看

在 `BoardFlow v1` 桌面骨架里，至少会开始出现下面几类状态：

- `boards`：当前已有白板集合
- `selection`：当前选中的白板
- `draftTitle`：新建白板草稿标题
- `draftTemplate`：新建白板模板选择
- `showDotGrid`：局部配置开关

其中：

- `boards`、`selection` 更像屏幕级或 feature 级状态
- 创建表单草稿可能先由父层或 feature model 拥有
- 子表单组件通过 `Binding` 编辑它们

这正是 `Binding` 在工作台 UI 里的典型位置：  
**父拥有，子编辑。**

## 为什么本教程强调“状态拥有关系”而不是只讲 `@Binding` 语法

因为真正难的地方从来不是写出 `$draftTitle`，而是判断：

- 什么应该作为单一事实源存在
- 什么只是视图局部状态
- 什么是派生值
- 什么是编辑缓冲

如果这些不分清，你写出来的 UI 可能一开始很“灵活”，但几章之后就会开始：

- 一个地方改了，另一个地方没跟上
- detail 区和 sidebar 选择不同步
- 取消编辑时状态回不去
- 预览和运行时逻辑分叉

## 双语关键词

- binding：绑定 / 可写借用
- owner：拥有者
- draft state：草稿状态
- edit buffer：编辑缓冲
- single source of truth：单一事实源
- parent-owned child-edited：父拥有、子编辑

## 常见错误

### 1. 子 view 拿到父状态后又开一份同名 `@State`

这几乎一定会制造双事实。

### 2. 任何可编辑值都直接传 `Binding`

有些场景需要的是缓冲，而不是立即改真源。

### 3. 把 `Binding` 理解成双向同步魔法

它只是对已有状态的借用，不会帮你自动设计好状态边界。

### 4. 不先想拥有者，就先想怎么把值传下去

这会让参数传递越来越多，但系统越来越不清楚。

## English Recap

`Binding` is not a second copy of state. It is a writable borrow of state that is still owned somewhere else. In `BoardFlow`, the key discipline is to separate true screen state from local drafts and only pass bindings when a child really needs to edit parent-owned state.

## Drills

1. 为什么说 `Binding` 的关键不是语法，而是“谁拥有状态”？
2. 举一个 `BoardFlow` 里你觉得更适合“编辑缓冲”而不是直接 `Binding` 真源的场景。
3. 解释父拥有、子编辑这条规则为什么适合工作台表单。

## Project Handoff

你现在已经知道 `Binding` 应该出现在哪里，但还差最后一块：当状态不再只是某个父 view 的局部小块时，谁来协调整个 screen？这就是下一章的 observable model。

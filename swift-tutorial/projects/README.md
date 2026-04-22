# Projects：FocusList 项目主线

## FocusList 是什么

`projects/` 不是正文之外的附赠代码区，而是整套教程的工程主线。`FocusList` 是一个跨 `iOS + macOS` 的任务与计划应用。正文解释为什么某个能力现在必须出现，项目目录则负责展示这些能力如何落到产品结构上。

## starter / checkpoints / final 如何配合

- `starter/`：每个教学阶段的实际起点，故意保持简单，让问题暴露得足够明显。
- `checkpoints/`：阶段说明文档，告诉读者这一阶段新增了什么能力、改变了什么边界。
- `final/`：课程最终成品的总结参照，帮助读者理解整个项目是如何从最小应用走到可维护产品的。

## FocusCore 与 focusctl 何时出现

`FocusList` 一开始不会立刻拆模块。到 Part 3，随着模型、持久化和失败路径变复杂，我们才抽出 `FocusCore`。到 Part 4，再在其上补一个轻量 CLI `focusctl`，专门用来讲清共享逻辑、测试和工程复用。它们出现的时机是教学设计的一部分，不是目录上的装饰。

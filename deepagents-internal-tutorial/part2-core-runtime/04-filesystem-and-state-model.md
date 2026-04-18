# 第4章：Filesystem 与 Pregel State Model

## 本章回答什么

这一章不再把 filesystem 当成“先有文件，再顺便有状态”的主题，而是把它收回到 Pregel runtime 的主线里：channel 如何承载状态、task 如何产生 writes、reducer 如何在 barrier 后合并、backend 又如何决定真实介质。filesystem 仍然是本章的材料，但它的作用是把 Pregel state model 讲清楚。

## 在整套系统中的位置

这一章是 Part 2 里解释 runtime state spine 的一章：先把 Pregel 如何定义 channel snapshot、pending writes、barrier 与 checkpoint 说清楚，再用 `files` channel 把这些概念落地。若你要追执行路径、tool surface 或 runtime carrier 的调用链，请转到 [第5章：Tools 作为 Runtime Surface](./05-tools-as-runtime-surface.md)；若你要追更远处的传播、分层存储策略与 backend 设计，则继续看后续章节与 Part 4。

## 静态结构

### 为什么这一章先讲 Pregel state model

如果先从 `write_file`、`root_dir` 或宿主机磁盘讲起，维护者很容易把问题误判成“工具有没有写进去”。Chapter 4 更应该先回答 runtime 怎么定义“状态已经存在”。state 不是普通 dict，而是 channel + reducer 的组合可见面。

在 Deep Agents 这条链路里，tool 只是 task 的一个执行表面；真正决定什么能被后续节点稳定读取的，是 Pregel 的 state model。filesystem 在这里重要，不是因为它比别的 capability 更特殊，而是因为文件读写最容易暴露“写了但还没跨过 barrier”的时序差异。

### Pregel 的最小执行对象：channel、task、pending writes、reducer

把本章里最小的一组运行时对象摆清楚，后面的 filesystem 语义就不容易混淆：

- channel：状态槽位，不是任意 dict key，而是有命名、有归并规则的 Pregel state surface。
- task：当前 superstep 内执行的节点或工具调用，它读取的是这一轮已经稳定的 channel snapshot。
- pending writes：task 在本轮里发出的更新，它们先排队，尚未变成全局可见事实。
- reducer：step boundary 到来时负责合并 writes，并决定下一轮 channel snapshot 如何形成。

task 在当前 superstep 中产生 writes，但 writes 需要经过 step boundary 才成为下一轮稳定可见状态。把这句话记牢，比背某个工具函数签名更重要，因为它决定了你该在 node、barrier、还是 checkpoint 边界排查问题。

### `FilesystemState.files` 为什么仍然重要

`FilesystemState.files` 仍然是这一章的核心例子，因为它不是普通“附件字段”，而是一个带 reducer 的 state channel。`files` channel 是理解 Pregel state model 的最佳例子，因为它同时展示了 reducer、step boundary 与 backend 分层。

它之所以关键，有三个原因：

1. 它让“文件状态”能和 thread、checkpoint、superstep 一起进入统一的 Pregel 可见性模型。
2. 它能直观看到 pending writes 如何在 barrier 之后才进入下一轮 snapshot。
3. 它逼着我们区分 graph state 与真实介质，不会把宿主机写盘误认为 Pregel state 已提交。

### backend 与 graph state 不是一回事

backend 决定文件最终落到哪里，graph state 决定 Pregel 在 step 边界上认什么为稳定状态。这两层必须拆开看，否则会把“介质上存在”误判成“图状态已提交”。

1. `StateBackend`：把文件写入 Pregel state path。
2. `FilesystemBackend`：把文件写入宿主机文件系统。
3. `CompositeBackend`：统一读写视图，但不制造单一介质真相。

所以 `files` channel 可以是 canonical Pregel state surface，但它并不自动等于所有 backend 的唯一真相；反过来，磁盘上已经有字节，也不自动代表 graph state 这一轮已经完成 reducer 归并。

## 运行时链路

### superstep 与 barrier：什么时候状态才进入下一轮可见面

Pregel runtime 不是边执行边把每次写入立即广播成全局事实。一个 superstep 开始时，task 看到的是上一轮已经稳定下来的 channel snapshot；task 执行过程中发出的只是 writes；barrier 到来后，runtime 才会统一归并这些 writes，并生成下一轮可见状态。

这意味着“本轮已经调用过 `write_file`”和“下一处读取已经看见新文件”之间，天然隔着一个 step boundary。把 barrier 想成可见面的翻页点，而不是某个 callback 的副作用时刻，会更接近 Pregel 的真实语义。

### checkpoint 记录什么，不记录什么

checkpoint 的职责不是记录整个执行故事，而是记录 Pregel 状态边界。checkpoint 记录的是 channel snapshot / pending writes 边界，不是 callback 流。

因此 checkpoint 回放能回答的是：

- 某个 step 结束后，各 channel 的稳定快照是什么。
- 某个边界上是否还有待归并或待消费的 writes 信息。

它不能直接替代的是 callback、streaming event、trace span 这些运行时观测面。后者能告诉你“发生过什么事件”，但不能单独证明 reducer 已经把状态推进到下一轮。

### `files` channel 为什么是理解 Pregel state model 的最佳例子

很多 channel 太抽象，讲 reducer 和 barrier 时容易停留在定义层；`files` 则能把抽象语义直接变成维护者能触摸的现象。你会看到：同一轮 task 已经返回写入结果，但下一轮之前读取视图仍可能保持旧值；你也会看到：换 backend 后，介质真相和 graph state 真相可能分层存在。

因此 `files` channel 同时承担两个教学角色：一是展示 reducer 如何决定 channel 的合并结果，二是展示 backend 只是存储策略层，不会抹平 Pregel 的 step boundary 语义。

### `StateBackend` 把写入排进 Pregel writes，而不是立刻变成全局事实

默认路径下，filesystem 工具经由 `ToolRuntime` 解析到 `StateBackend`。这时写文件不是“直接把一个全局 dict 改掉”，而是把更新送进 Pregel 的 writes path，等待 barrier 后进入 reducer。

这也是为什么默认 filesystem 能天然跟 thread state、checkpoint、superstep 对齐：因为它遵守的是 Pregel 的提交模型，而不是绕过 graph runtime 自己偷偷维护一个立即生效的 side store。

### `FilesystemBackend` 与 `CompositeBackend` 分别改变哪一层语义

`FilesystemBackend` 改变的是介质层语义：文件字节落到宿主机文件系统，路径解析与宿主机目录行为开始变得关键。但它不会改写 Pregel 对 step boundary、task、pending writes 的定义。

`CompositeBackend` 改变的是路由层语义：同一个统一 filesystem 视图下，不同路径可以落到不同 backend。它把多个介质组织成一个读写入口，却不会凭空制造“所有路径已经共享同一个底层真相”的假设。

## 传播 / 可见性 / 拦截点

### 维护者最容易误判的四种“状态已经提交”

- node / tool 返回了值，不等于下一处读取已经拿到 reducer 后状态。
- 外层 consumer 看到了 update，不等于 checkpoint 已经落盘。
- callback / tracing 记录到事件，不等于 state 已经跨过 barrier。
- backend 看到了写入，不等于 graph state 和宿主机介质已经统一。

这些误判之所以常见，是因为不同观测面回答的是不同问题：task return 回答“本轮执行做了什么”，consumer update 回答“外层收到了什么信号”，callback / tracing 回答“系统发出过什么事件”，backend 回答“某个介质层看见了什么写入”。只有跨过 barrier 并完成 reducer 之后，某个 channel 才真正成为下一轮稳定可见状态。

## 扩展接口

这一章相关的扩展面只讨论 state model 本身，不讨论 Chapter 5 的执行路径细节：

- 定义或修改 channel：决定某块状态是不是 Pregel state surface，以及它的命名边界。
- 定义或修改 reducer：决定 pending writes 在 barrier 后如何归并成下一轮可见 snapshot。
- 选择或组合 backend：决定某条路径的真实介质，但不改变 Pregel 对 superstep 和 checkpoint 的边界定义。

## 常见问题与排障入口

- 你现在看到的是 task 返回值、consumer update、callback event，还是已经跨过 barrier 的 channel snapshot？
- 你要确认的是“写入发生过”，还是“reducer 已经把 writes 合并进下一轮可见状态”？
- 你要排查的是 checkpoint 为什么没有记录某个状态边界，还是某个事件流为什么出现过但没有形成稳定状态？
- 你面对的是 `files` channel 的 Pregel 可见性问题，还是 backend 介质层已经写入但尚未与 graph state 对齐的问题？

## 本章结论

- 先看 Pregel state model，再看 filesystem，才能正确区分 channel snapshot、pending writes、reducer、backend 介质。
- `FilesystemState.files` 不是附属细节，而是把抽象 state model 变得可观察的最佳工作样本。
- backend 决定写入落点，Pregel barrier 决定状态何时稳定可见；这两个判断层级必须始终分开。

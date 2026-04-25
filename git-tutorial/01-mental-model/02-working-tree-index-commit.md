# 工作区、暂存区与提交

## 本章位置

- 前置章节：`01-mental-model/01-see-the-repo-state.md`（先会看 `git status --short --branch`）。
- 后续章节：`01-mental-model/03-read-history-with-confidence.md`（再学习如何阅读提交历史）。
- 本章 Lab id：`LAB-MODEL-INDEX-01`。

## 场景

你正在修一个小问题，同时顺手改了文档、加了调试输出。准备提交时，你发现这些修改不应该全部进入同一个提交：真正要交付的是代码修复，文档可以稍后提交，调试输出不该提交。

这时最重要的问题不是“要不要执行 `git add .`”，而是先回答三件事：

1. 哪些内容只存在于工作区？
2. 哪些内容已经进入暂存区？
3. 下一次 `git commit` 到底会记录哪个快照？

Git 的日常提交能力，建立在对 **working tree / index / commit** 三层状态的判断上。

## 学习目标

学完本章后，你应该能够：

- 解释工作区、暂存区（index/staging area）和提交（commit）三层之间的关系。
- 判断 `git add`、`git restore --staged`、`git commit` 分别改变哪一层。
- 使用 `git diff`、`git diff --cached` 和 `git diff HEAD` 对比不同层。
- 识别同一文件同时有 staged 与 unstaged 修改的 `MM` 状态。
- 在提交前确认“将被提交的内容”和“仍留在本地的内容”。

## 心智模型

把 Git 的本地状态想成三层快照：

| 层 | 你在观察什么 | 常见命令 | 是否会被下一次提交记录 |
|---|---|---|---|
| 工作区 working tree | 文件系统里的当前内容 | `git diff` | 不一定 |
| 暂存区 index / staging area | 下一次提交的候选快照 | `git diff --cached` | 会 |
| 当前提交 HEAD | 当前分支指向的历史快照 | `git show HEAD` | 已经记录 |

关键结论：**`git commit` 记录的是暂存区，不是工作区的全部当前内容。**

`git add <path>` 不是“标记这个文件以后都自动提交”，而是“把这个路径此刻的内容复制一份到暂存区”。如果你 `git add` 之后又继续修改同一个文件，这个文件会同时拥有已暂存和未暂存的变化。

## 观察点

提交前至少做一次“三向观察”：

```bash
git status --short --branch
git diff          # 工作区 vs 暂存区：还没放进下一次提交的修改
git diff --cached # 暂存区 vs HEAD：下一次提交将记录的修改
git diff HEAD     # 工作区整体 vs HEAD：本地总变化
```

观察时逐项回答：

- `git status --short` 是否出现 ` M`、`M `、`MM` 或 `??`？
- `git diff --cached` 是否为空？如果为空，下一次普通 `git commit` 没有可提交内容。
- `git diff` 是否仍有输出？如果有，说明还有修改留在工作区，不会被这次提交记录。
- 是否存在不该提交的调试语句、临时文件、密钥或大文件？

常见短状态：

| 输出 | 工作区 | 暂存区 | 含义 |
|---|---|---|---|
| ` M file` | 有修改 | 无修改 | 改了但未暂存。 |
| `M  file` | 与暂存区一致 | 有修改 | 修改已进入下一次提交。 |
| `MM file` | 有新修改 | 有旧修改 | 同一文件一部分已暂存，之后又继续修改。 |
| `A  file` | 与暂存区一致 | 新文件 | 新文件已暂存。 |
| `?? file` | 未跟踪 | 不在暂存区 | Git 还没有管理它。 |

## 命令与解释

```bash
git add path/to/file
```

把指定路径当前内容复制进暂存区。它不会创建提交，也不会让后续新增修改自动进入暂存区。提交前仍需要用 `git diff --cached` review 暂存区。

```bash
git add -p path/to/file
```

按补丁块选择要暂存的内容。它适合把同一个文件中的“真正修复”和“临时调试”拆开，但初学时要认真阅读每个 hunk，不要机械输入 `y`。

```bash
git restore --staged path/to/file
```

把路径从暂存区移回未暂存状态。它不应该删除工作区中的文件内容，适合恢复“误暂存”。

```bash
git diff
```

查看工作区相对暂存区的差异。它回答：“还有哪些修改没有进入下一次提交？”

```bash
git diff --cached
```

查看暂存区相对 HEAD 的差异。它回答：“下一次提交会记录什么？”

```bash
git commit -m "message"
```

把暂存区快照记录成一个新提交，并移动当前分支引用。它不会自动包含所有工作区修改。

## 实验

**Lab id：`LAB-MODEL-INDEX-01`**

目标：让同一文件同时出现 staged 与 unstaged 修改，并解释每个 diff 视图。

### 准备

```bash
mkdir index-lab
cd index-lab
git init
printf "line 1\n" > story.txt
git add story.txt
git commit -m "seed story"
```

### 执行

```bash
printf "line 2 staged\n" >> story.txt
git add story.txt
printf "line 3 unstaged\n" >> story.txt

git status --short --branch
git diff
git diff --cached
git diff HEAD
```

### 观察

预期现象：

- `git status --short --branch` 显示 `MM story.txt`。
- `git diff --cached` 只显示已经暂存的 `line 2 staged`。
- `git diff` 只显示尚未暂存的 `line 3 unstaged`。
- `git diff HEAD` 同时显示 `line 2 staged` 和 `line 3 unstaged`，因为它观察的是工作区整体相对 HEAD 的变化。

### 提交验证

```bash
git commit -m "add staged story line"
git status --short
git show --stat HEAD
git diff
```

你应该看到：

- 新提交只记录 `line 2 staged`。
- `story.txt` 仍然是 modified，因为 `line 3 unstaged` 还留在工作区。
- `git diff` 仍能看到未提交的 `line 3 unstaged`。

### 清理

实验仓库只是练习目录。完成后回到上级目录并删除它即可：

```bash
cd ..
rm -rf index-lab
```

## 常见错误

- **以为 `git commit` 会提交所有修改**：它只提交暂存区。工作区里未暂存的修改会继续留在本地。
- **长期依赖 `git add .`**：容易把无关文件、调试输出、生成物或秘密一起放入提交。
- **只看 `git diff` 不看 `git diff --cached`**：提交前真正要 review 的是暂存区差异。
- **把 `MM` 当成 Git 坏了**：`MM file` 只是说明同一个文件有两层变化；先看 `git diff --cached`，再看 `git diff`。
- **误以为 `git restore --staged` 会删除代码**：它只取消暂存；若不加 `--staged`，语义会变成恢复工作区内容，需要更谨慎。

## 危险命令与恢复路径

> **危险命令：`git reset --hard`**
>
> 它会把暂存区和工作区一起重置到目标提交，未提交内容可能直接丢失。学习本章时不要用它来“清理状态”。

> **危险命令：`git checkout -- <path>` 或 `git restore <path>`**
>
> 不带 `--staged` 时，这类命令会改写工作区文件内容。执行前必须确认 `git diff <path>` 中的内容确实可以丢弃。

优先恢复路径：

1. 只是误暂存：执行 `git restore --staged <path>`，再用 `git status --short` 确认从 `M ` 回到 ` M`。
2. 暂存了过多内容：用 `git restore --staged <path>` 全部移出暂存区，再用 `git add -p <path>` 只选择需要的 hunk。
3. 误改工作区但还没丢弃：先用 `git diff <path>` 保存需要的片段，再决定是否 `git restore <path>`。
4. 已误删或误覆盖：立即停止继续重置，检查编辑器本地历史、IDE undo、文件系统备份；如果内容曾进入提交，再用 `git log` / `git reflog` 找回。

## 验收

请在不运行破坏性命令的前提下回答：

1. `git diff` 为空但 `git diff --cached` 不为空，表示什么？
2. `git status --short` 中 `MM story.txt` 为什么不是错误？
3. 如果你误把文件加入暂存区，应该先尝试哪条非破坏性命令？
4. `git add story.txt` 之后又修改 `story.txt`，下一次提交会包含后一次修改吗？为什么？
5. 提交前想 review “真正会进入提交的内容”，应该看 `git diff` 还是 `git diff --cached`？

参考答案要点：

- 暂存区有内容、工作区相对暂存区没有额外变化；下一次提交会记录暂存区。
- `MM` 表示同一文件既有 staged 修改也有 unstaged 修改。
- 优先使用 `git restore --staged <path>`。
- 不会自动包含；`git add` 只复制当时的内容进暂存区。
- 看 `git diff --cached`。

## 术语需求

- index / staging area：本教程统一译为“暂存区”，首次出现时可标注英文。
- working tree：本教程统一译为“工作区”。
- HEAD：当前检出的提交或当前分支尖端；后续章节继续展开。

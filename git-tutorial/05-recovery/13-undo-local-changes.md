# 13 Undo Local Changes

## 本章导航

- 前置章节：[12 Review 与团队约定](../04-collaboration/12-review-and-team-conventions.md)
- 后续章节：[14 Fix a Bad Commit](14-fix-a-bad-commit.md)
- 本章 Lab ID：`LAB-RECOVERY-UNDO-01`
- 所属模块：`05-recovery`

## 场景

你改了几个文件后发现方向错了；或者已经 `git add` 了一部分内容，却发现这次提交边界不对；再或者同事提醒你某个错误已经进入共享分支，不能用本地习惯直接“抹掉”。撤销本地改动的难点不在命令数量，而在判断：你要改变的是工作区、暂存区、当前分支指针，还是已经进入团队历史的提交效果。

本章训练一个安全顺序：先观察三层状态，再选择最小影响命令，最后为危险命令准备恢复路径。你应该把 `git reset --hard` 当成最后手段，而不是清理现场的第一反应。

## 学习目标

完成本章后，你应该能够：

1. 区分工作区（working tree）、暂存区（index）和 `HEAD` / 当前分支指针三层状态。
2. 使用 `git restore <path>` 丢弃已跟踪文件的工作区改动。
3. 使用 `git restore --staged <path>` 取消暂存，同时保留文件内容。
4. 解释 `git reset --soft`、`git reset --mixed`、`git reset --hard` 分别会移动或覆盖哪些层。
5. 判断错误已经共享时为什么应优先使用 `git revert`，而不是改写历史。
6. 在运行危险命令前建立补丁、救援分支或 reflog 恢复入口。

## 观察点

撤销前先不要急着输入命令。每一次恢复都从同一组观察开始：

```bash
git status -sb
git diff --stat
git diff
git diff --cached --stat
git diff --cached
git log --oneline --graph --decorate --all -n 8
```

观察时回答六个问题：

- `git status -sb` 中是 `M`、`A`、`??`，还是同时出现在暂存区和工作区？
- `git diff` 是否还有尚未备份的工作区内容？
- `git diff --cached` 是否有已经进入下一次提交的内容？
- 你要撤销的是某个文件、下一次提交的暂存状态，还是最近一次提交？
- 当前分支是否已经推送，`git log --decorate` 中目标提交是否可能被别人基于它继续开发？
- 是否存在未跟踪文件？`restore` 和多数 `reset` 命令不会自动保护或删除它们。

一个重要原则：如果你不能用观察输出说清“命令将改变哪一层”，就先不要执行破坏性撤销。

## 命令与判断

| 目标 | 首选命令 | 会影响什么 | 不会影响什么 | 典型场景 |
|---|---|---|---|---|
| 丢弃某个已跟踪文件的工作区改动 | `git restore <path>` | 工作区文件内容 | 暂存区、提交历史 | 改了方向错的文件，尚未暂存 |
| 取消暂存但保留文件内容 | `git restore --staged <path>` | 暂存区条目 | 工作区内容、提交历史 | `git add` 过早，想重新分组提交 |
| 同时从暂存区恢复某文件到 `HEAD` | `git restore --staged --worktree <path>` | 暂存区和工作区的该文件 | 其他文件、历史 | 确认某个文件的改动全都不要 |
| 撤回最近本地提交但保留 staged 内容 | `git reset --soft HEAD~1` | 当前分支指针 | 暂存区、工作区内容 | 提交说明或边界要重做，仍想保留暂存 |
| 撤回最近本地提交并取消暂存 | `git reset --mixed HEAD~1` | 当前分支指针、暂存区 | 工作区内容 | 想把最近提交拆开重新 `add` |
| 撤回提交并丢弃已跟踪文件状态 | `git reset --hard HEAD~1` | 当前分支指针、暂存区、已跟踪工作区 | 未跟踪文件通常不处理 | 只在确认没有要保留内容时使用 |
| 撤销已共享提交的效果 | `git revert <commit>` | 新增一个反向提交 | 不重写原历史 | 错误已经进入主干、远端或 PR 依赖链 |

### 工作区改动：优先局部恢复

只撤销一个文件时，不要为了“干净”重置整个仓库：

```bash
git diff -- path/to/file
git restore path/to/file
git status -sb
```

如果只想取消暂存：

```bash
git diff --cached -- path/to/file
git restore --staged path/to/file
git status -sb
```

取消暂存后，文件内容仍留在工作区。接下来你可以继续编辑、重新拆分 `git add -p`，或者确认后再 `git restore <path>` 丢弃内容。

### 本地提交：先确认是否共享

最近一次提交只在本地时，可以移动当前分支指针：

```bash
git branch -vv
git log --oneline --decorate -n 5
git reset --mixed HEAD~1
```

`--mixed` 是默认模式：提交被撤回，改动回到工作区，暂存区被清空。它适合把一个本地提交拆成更清晰的多次提交。

如果你已经推送或打开 PR，需要先确认团队约定。进入共享语境后，优先保留历史，用新提交表达撤销：

```bash
git revert <bad-commit>
```

`revert` 不是删除旧提交，而是新增一个反向补丁。它让主干历史仍然可审计，也让协作者不需要处理被改写的提交身份。

## 危险命令

以下命令会丢弃内容、移动分支指针，或改写别人可能已经看到的历史。执行前必须先完成“观察点”中的检查，并准备恢复路径。

```bash
git reset --hard <target>
git restore --staged --worktree <path>
git checkout -- <path>
git clean -fd
git push --force
git push --force-with-lease
```

危险点：

- `git reset --hard <target>` 会让当前分支、暂存区和已跟踪工作区回到目标提交；未备份的已跟踪改动会消失。
- `git restore --staged --worktree <path>` 会同时清掉某个路径在暂存区和工作区的改动，适合确认全不要，不适合犹豫时使用。
- `git checkout -- <path>` 是旧写法，效果接近丢弃工作区改动；在新教程中优先使用语义更清楚的 `git restore`。
- `git clean -fd` 会删除未跟踪文件和目录；这些内容通常不在 Git 历史里，恢复难度更高。
- `git push --force` 和 `git push --force-with-lease` 会影响远端分支历史；后者更安全，但仍需要团队约定。

## 恢复路径

撤销前先建立“能回头”的证据。根据风险级别选择一种或多种保护动作。

### 保护工作区改动

```bash
git status -sb
git diff > ../before-restore-worktree.patch
git diff --cached > ../before-restore-index.patch
```

如果误删了工作区内容，可以尝试把补丁应用回去：

```bash
git apply ../before-restore-worktree.patch
```

### 保护当前提交位置

在移动分支指针前创建救援分支：

```bash
git branch rescue-before-reset
git log --oneline --decorate -n 5
```

如果 reset 后发现目标错了，先不要继续猜。查看 reflog，再从旧位置创建分支：

```bash
git reflog --date=relative -n 20
git switch -c rescue-from-reflog HEAD@{1}
```

确认救援分支内容正确后，再决定把补丁合回、cherry-pick 回来，还是让正式分支回到旧位置。

### 保护未跟踪文件

`reset --hard` 通常不会删除未跟踪文件，但 `git clean -fd` 会。执行 clean 前必须先 dry-run：

```bash
git clean -nd
```

如果输出中有任何可能要保留的文件，先移动到仓库外安全目录，或把它们加入临时分支提交，再考虑清理。

## 实验

Lab ID：`LAB-RECOVERY-UNDO-01`

实验目标：在一个练习仓库中制造四类状态，并为每类状态选择最小影响的撤销动作。Labs agent 后续可据此落地脚本；本章只声明实验需求，不修改 `labs/**`。

1. **丢弃工作区改动**
   - 修改一个已跟踪文件但不暂存。
   - 运行 `git status -sb`、`git diff` 记录状态。
   - 执行 `git restore <path>`。
   - 验证 `git diff` 清空，`git log` 不变。
2. **取消暂存但保留内容**
   - 修改并 `git add` 一个文件。
   - 运行 `git diff --cached` 确认内容在暂存区。
   - 执行 `git restore --staged <path>`。
   - 验证内容回到工作区，仍可继续编辑或重新分组。
3. **比较三种 reset 模式**
   - 在三个临时分支上分别执行 `git reset --soft HEAD~1`、`git reset --mixed HEAD~1`、`git reset --hard HEAD~1`。
   - 每次都记录 `git status -sb`、`git diff --cached --stat`、`git diff --stat`、`git log --oneline -n 3`。
   - 用观察输出说明三种模式分别改变了哪些层。
4. **撤销共享错误**
   - 模拟一个已经进入主干的坏提交。
   - 不使用 reset；执行 `git revert <commit>`。
   - 验证历史中保留原提交和反向提交，并说明为什么团队更容易协作。

预期结果：学习者能从状态出发选择命令，而不是背“撤销 = reset”。

## 常见错误

- **没看 `git diff` 就运行 `git reset --hard`。** 这会把尚未提交的已跟踪改动直接丢掉；至少先导出补丁或创建救援分支。
- **把取消暂存误认为必须使用 `git reset`。** 新版 Git 中 `git restore --staged <path>` 更清楚地表达“只改暂存区”。
- **对已经推送的主干提交使用 reset。** 本地看起来干净了，协作者却会遇到分叉和强推风险。
- **误以为 `git revert` 会删除旧提交。** 它实际新增反向提交，保留审计线索。
- **忘记未跟踪文件的边界。** `restore` 和 `reset --hard` 主要处理已跟踪内容；`clean` 才会删除未跟踪文件，而且风险更高。
- **连续试错恢复。** 事故发生后反复 reset 会增加 reflog 噪音；先命名救援分支，再做判断。

## 验收

你应该能回答并演示以下问题：

1. “我只是不想把这个文件放进下一次提交，但文件内容还要保留。”应该用什么命令？它改变哪一层？
2. “我想丢弃某个已跟踪文件的工作区改动，但不确定其中有没有重要内容。”执行前应保存哪些证据？
3. `reset --soft`、`reset --mixed`、`reset --hard` 对工作区、暂存区、当前分支指针分别有什么影响？
4. “最近一次提交只在本地，我想拆开重新提交。”为什么 `git reset --mixed HEAD~1` 比 `--hard` 更安全？
5. “错误提交已经进入团队主干。”为什么优先 `git revert`，而不是 reset 后 force push？
6. 如果 reset 后发现改错了，如何用 `git reflog` 和救援分支保护旧位置？

完成验收时，必须能展示 `git status -sb`、`git diff` / `git diff --cached`、`git log --graph --decorate` 的输出，并用这些输出证明每条命令改变了哪一层。

## 术语需求

请后续 Appendix agent 确认或补充以下术语：

- working tree / 工作区
- index / staging area / 暂存区
- `HEAD` 与当前分支指针
- shared history / 共享历史
- rescue branch / 救援分支
- reflog / 引用日志

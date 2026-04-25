# 课程导向：怎样用状态观察学习 Git

## 场景

你接手一个仓库，准备修一个小问题。你知道可能会用到 `git add`、`git commit`、`git pull`、`git push`，但不确定现在仓库是否干净，也不知道操作错了能不能恢复。

本教程从这个时刻开始：**先观察状态，再做操作**。

如果你已经知道自己要解决哪类问题，可以把本章当作入口页：

- 第一次系统学习：读完本章后进入 [课程地图](./README.md#课程地图)，按 01 到 07 顺序推进。
- 只想练习：使用 [实验总览](./labs/README.md) 找到本章或后续章节的 lab id。
- 遇到危险命令：先查 [危险区](./appendix/danger-zone.md)，再回到对应章节执行。

## 学习目标

读完本章，你应该能够：

- 解释本教程为什么按任务组织，而不是按命令字母表组织。
- 使用统一的状态观察面板描述仓库当前状态。
- 识别哪些命令只是查看，哪些命令会移动内容，哪些命令会移动引用或改写历史。
- 在遇到危险命令前，先确认风险和恢复路径。

## 观察点

每次动手前，先运行：

```bash
git status --short --branch
git diff
git diff --cached
git log --oneline --graph --decorate --max-count=8
```

如果仓库还没有提交，最后一条 `git log` 可能失败，这是正常现象。此时你只需要记录“尚无提交历史”。

观察结果按这个格式写下来：

```text
工作区：是否有未暂存修改？
暂存区：是否已有下一次提交的候选内容？
HEAD：当前指向哪个分支或提交？
历史：最近提交是线性、分叉还是汇合？
远程：本地分支是否领先、落后或分叉？
```

## 命令分层

Git 命令可以先粗略分成三类。

| 类型 | 例子 | 默认风险 | 你要问的问题 |
|---|---|---|---|
| 查看状态 | `status`、`diff`、`log`、`show` | 低 | 它回答哪个问题？ |
| 移动内容 | `add`、`restore`、`commit` | 中 | 它改变工作区、暂存区还是提交历史？ |
| 移动引用/改写历史 | `switch`、`reset`、`merge`、`rebase`、`push --force-with-lease` | 中到高 | 它会不会影响已共享历史？失败后从哪里恢复？ |

本教程不会要求你一开始记住全部命令，而是要求你每次能判断“这条命令改变哪一层”。

## 实验

**Lab id：`LAB-ORIENT-STATUS-01`**

目标：初始化一个练习仓库，并记录第一次状态观察。完整实验说明见 [labs/scenarios/LAB-ORIENT-STATUS-01.md](./labs/scenarios/LAB-ORIENT-STATUS-01.md)。

```bash
mkdir git-practice
cd git-practice
git init
git status --short --branch
echo "hello git" > note.txt
git status --short --branch
git diff
```

预期观察：

- 初始化后有一个 Git 仓库，但还没有提交历史。
- 新建 `note.txt` 后，`git status --short` 会显示未跟踪文件。
- 对未跟踪文件，`git diff` 默认不会显示内容差异，因为它还不在 Git 的跟踪集合中。

## 危险命令提示格式

后续章节遇到危险命令时，会使用固定格式：

> **风险提示**：这条命令可能丢弃工作区内容、移动分支引用、改写提交历史或影响远程共享历史。执行前先确认 `git status --short --branch`、`git log --oneline --graph --decorate`，必要时创建备份分支。
>
> **恢复路径**：优先从 `git reflog`、备份分支、未关闭的编辑器、远程仓库或 teammate 的 clone 中恢复；不要在慌乱中连续执行更多破坏性命令。

## 常见错误

- **只看命令，不看状态**：同一条命令在不同状态下后果不同。
- **把暂存区当成临时文件夹**：暂存区是下一次提交的候选快照，不是随手缓存。
- **把远程当成实时状态**：`origin/main` 是你上次 fetch 到的远程状态，不会自动更新。
- **出错后连续试命令**：恢复的第一步通常是停下、观察、记录，而不是继续改。

## 验收

请用自己的话回答：

1. `git diff` 和 `git diff --cached` 分别比较哪两层？
2. 为什么危险命令前要看 `git status --short --branch`？
3. 为什么“本地没有报错”不等于“远程协作安全”？

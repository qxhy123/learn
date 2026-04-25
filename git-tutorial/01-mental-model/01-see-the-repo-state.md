# 看见仓库状态

## 场景

你打开一个目录，想知道它是不是 Git 仓库、现在有没有未提交修改、当前在哪个分支、是否可以安全开始工作。你不应该先猜，也不应该直接 `pull` 或 `reset`，而应该先把状态看出来。

## 学习目标

- 区分普通目录、Git 仓库和工作区状态。
- 使用 `git status --short --branch` 快速判断当前分支与文件状态。
- 理解 untracked、modified、staged 三种最常见状态。
- 在命令前后记录“状态变化”，而不是只记录“命令执行成功”。

## 观察点

运行：

```bash
git status --short --branch
git rev-parse --show-toplevel
git branch --show-current
```

分别回答：

- 当前目录是否位于 Git 仓库内部？
- 仓库根目录在哪里？
- 当前分支名是什么？如果没有分支名，是否处于 detached HEAD？
- 文件状态中是否有 `??`、` M`、`M ` 或 `MM`？

常见短状态含义：

| 输出 | 含义 |
|---|---|
| `?? file` | Git 还没有跟踪这个文件。 |
| ` M file` | 工作区有修改，但暂存区没有。 |
| `M  file` | 修改已经进入暂存区。 |
| `MM file` | 同一个文件既有已暂存修改，也有未暂存修改。 |

## 命令与解释

```bash
git status --short --branch
```

这条命令不改变仓库，只显示当前分支和文件状态。它应该成为你最频繁使用的命令之一。

```bash
git rev-parse --show-toplevel
```

这条命令显示仓库根目录。多人协作时，确认根目录能避免在错误目录里创建文件或运行脚本。

```bash
git status
```

完整输出更适合初学阶段阅读，因为它会给出下一步提示。但真正动手时，短格式更便于对比前后状态。

## 实验

**Lab id：`LAB-MODEL-STATE-01`**

目标：制造三种基础状态，并观察它们如何显示。

```bash
mkdir model-state-lab
cd model-state-lab
git init
printf "alpha\n" > tracked.txt
git add tracked.txt
git commit -m "seed tracked file"

printf "draft\n" > new.txt
printf "beta\n" >> tracked.txt
git status --short --branch

git add tracked.txt
git status --short --branch
```

预期观察：

- `new.txt` 显示为 `??`，因为它未被跟踪。
- `tracked.txt` 第一次显示为 ` M`，因为修改还在工作区。
- `git add tracked.txt` 后，`tracked.txt` 显示为 `M `，因为修改进入暂存区。

## 常见错误

- **把 `git status` 当成“可选步骤”**：不看状态就执行修改命令，是很多事故的起点。
- **看到 clean 就以为远程也同步**：工作区 clean 只说明本地文件层干净，不说明远程状态最新。
- **忽略 `??` 文件**：未跟踪文件不会自动进入提交，也不会被 `git diff` 默认展示。

## 验收

给定下面输出，说明每个文件处在哪一层：

```text
## main
 M app.py
M  README.md
MM config.yml
?? notes.txt
```

你应该能解释：`app.py` 只在工作区改了；`README.md` 已暂存；`config.yml` 同时有暂存和未暂存修改；`notes.txt` 还未被 Git 跟踪。

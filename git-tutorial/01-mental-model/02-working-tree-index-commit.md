# 工作区、暂存区与提交

## 场景

你改了几个文件，只想把其中一部分提交出去。此时最重要的问题不是“用不用 `git add .`”，而是：哪些内容还在工作区，哪些内容已经进入暂存区，下一次提交到底会记录什么。

## 学习目标

- 解释工作区、暂存区、提交三层之间的关系。
- 判断 `git add`、`git commit`、`git restore --staged` 分别改变哪一层。
- 使用 `git diff`、`git diff --cached` 和 `git diff HEAD` 对比不同层。
- 处理同一文件同时有 staged 与 unstaged 修改的情况。

## 观察点

提交前至少观察三次差异：

```bash
git diff          # 工作区 vs 暂存区
git diff --cached # 暂存区 vs HEAD
git diff HEAD     # 工作区整体 vs HEAD
```

如果 `git diff --cached` 为空，下一次提交不会记录任何内容。即使工作区有修改，也不会因为 `git commit` 自动进入提交。

## 命令与解释

```bash
git add path/to/file
```

把指定路径当前内容复制进暂存区。它不会创建提交，也不会保证之后的新增修改自动进入暂存区。

```bash
git restore --staged path/to/file
```

把文件从暂存区移回“未暂存”状态。它不应该删除工作区中的文件内容。

```bash
git commit -m "message"
```

把暂存区快照记录成一个新提交，并移动当前分支引用。它记录的是暂存区，不是工作区所有文件。

## 实验

**Lab id：`LAB-MODEL-INDEX-01`**

目标：让同一文件同时出现 staged 与 unstaged 修改。

```bash
mkdir index-lab
cd index-lab
git init
printf "line 1\n" > story.txt
git add story.txt
git commit -m "seed story"

printf "line 2 staged\n" >> story.txt
git add story.txt
printf "line 3 unstaged\n" >> story.txt

git status --short
git diff
git diff --cached
```

预期观察：

- `git status --short` 显示 `MM story.txt`。
- `git diff --cached` 只显示已经暂存的 `line 2 staged`。
- `git diff` 只显示尚未暂存的 `line 3 unstaged`。

## 风险与恢复路径

> **风险提示**：`git reset --hard` 会同时重置暂存区和工作区，未提交内容可能直接丢失。学习本章时不要用它来“清理状态”。
>
> **恢复路径**：如果只是误暂存，优先使用 `git restore --staged <path>`；如果误删工作区内容，先检查编辑器本地历史、IDE undo、备份分支和 `git reflog`，不要连续执行更多重置命令。

## 常见错误

- **以为 `git commit` 会提交所有修改**：它只提交暂存区。
- **长期依赖 `git add .`**：容易把无关文件、调试输出或秘密一起放入提交。
- **只看 `git diff` 不看 `git diff --cached`**：提交前真正要 review 的是暂存区差异。

## 验收

请回答：

1. `git diff` 为空但 `git diff --cached` 不为空，表示什么？
2. `git status --short` 中 `MM file` 为什么不是错误？
3. 如果你误把文件加入暂存区，应该先尝试哪条非破坏性命令？

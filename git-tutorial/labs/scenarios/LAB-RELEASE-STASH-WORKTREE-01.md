# LAB-RELEASE-STASH-WORKTREE-01: 处理中断、stash 与 worktree 并行修复

## 目标

处理中断、stash 与 worktree 并行修复。本场景对应 `06 Release and Debugging`，用于把章节中的 lab id 落地为可重复执行的本地 Git 练习。

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-RELEASE-STASH-WORKTREE-01 --force
cd workspaces/release-stash-worktree/interrupt-lab
```

## 执行

1. 观察 feature 分支上的中断状态。
2. 用 stash 或 worktree 隔离当前任务与 hotfix。
3. 解释 stash 和 worktree 哪个更适合长期并行。

## 观察

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=12
```

按需要追加：

```bash
git diff
git diff --cached
git reflog --date=relative
git show-ref --heads --tags
git config --show-origin --list | sed -n '1,40p'
```

## 恢复

- 不确定前先创建备份分支：`git branch backup/$(date +%Y%m%d-%H%M%S)`。
- 对未共享历史优先在本地修正；对已共享历史优先使用新增提交或 `git revert` 修复。
- 合并或 rebase 过程中不确定时，优先使用 `git merge --abort` 或 `git rebase --abort` 回到操作前状态。
- 如实验仓库混乱，回到 `git-tutorial/labs` 重新运行准备命令。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/release-stash-worktree
```

## 预期结果

你能根据默认观察面板说明当前 working tree、index、HEAD、branch、remote/ref 的状态，并解释本 lab 的安全恢复路径。

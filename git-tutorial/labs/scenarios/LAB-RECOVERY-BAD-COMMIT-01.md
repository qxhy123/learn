# LAB-RECOVERY-BAD-COMMIT-01: 修正坏提交、漏提交文件和已共享错误

## 目标

修正坏提交、漏提交文件和已共享错误。本场景对应 `05 Recovery`，用于把章节中的 lab id 落地为可重复执行的本地 Git 练习。

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-RECOVERY-BAD-COMMIT-01 --force
cd workspaces/recovery-bad-commit/bad-commit-lab
```

## 执行

1. 修正本地坏提交说明或漏提交文件。
2. 识别已 push 的错误提交。
3. 用新增修复提交或 revert 处理已共享错误。

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
rm -rf workspaces/recovery-bad-commit
```

## 预期结果

你能根据默认观察面板说明当前 working tree、index、HEAD、branch、remote/ref 的状态，并解释本 lab 的安全恢复路径。

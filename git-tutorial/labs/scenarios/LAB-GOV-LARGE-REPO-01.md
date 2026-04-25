# LAB-GOV-LARGE-REPO-01: 大文件、忽略规则和 LFS 决策

## 目标

大文件、忽略规则和 LFS 决策。本场景对应 `07 Scale and Governance`，用于把章节中的 lab id 落地为可重复执行的本地 Git 练习。

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-LARGE-REPO-01 --force
cd workspaces/gov-large-repo/large-repo-lab
```

## 执行

1. 找出已跟踪的构建产物和大文件候选。
2. 更新 `.gitignore` 和 `LFS_DECISION.md`。
3. 区分 ignore、LFS、制品库和历史清理四类动作。

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
rm -rf workspaces/gov-large-repo
```

## 预期结果

你能根据默认观察面板说明当前 working tree、index、HEAD、branch、remote/ref 的状态，并解释本 lab 的安全恢复路径。

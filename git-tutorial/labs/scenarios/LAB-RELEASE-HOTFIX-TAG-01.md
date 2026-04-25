# LAB-RELEASE-HOTFIX-TAG-01: 从发布标签切出 hotfix 并打 patch 标签

## 目标

从发布标签切出 hotfix 并打 patch 标签。本场景对应 `06 Release and Debugging`，用于把章节中的 lab id 落地为可重复执行的本地 Git 练习。

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-RELEASE-HOTFIX-TAG-01 --force
cd workspaces/release-hotfix-tag/release-lab
```

## 执行

1. 查看 `v1.0.0` 与 `v1.1.0` 标签。
2. 确认 hotfix 分支从正确标签切出。
3. 完成修复并说明 patch tag 应指向哪里。

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
rm -rf workspaces/release-hotfix-tag
```

## 预期结果

你能根据默认观察面板说明当前 working tree、index、HEAD、branch、remote/ref 的状态，并解释本 lab 的安全恢复路径。

# LAB-DAILY-IGNORE-01: 配置 .gitignore 并解释 tracked 文件边界

## 目标

配置 .gitignore 并解释 tracked 文件边界。本场景对应 `02 Daily Workflow`，用于把章节中的 lab id 落地为可重复执行的本地 Git 练习。

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-DAILY-IGNORE-01 --force
cd workspaces/daily-ignore/ignore-lab
```

## 执行

1. 查看 `.gitignore` 对 untracked 构建文件的影响。
2. 证明已跟踪的 `build/output.log` 不会因为新增 ignore 规则自动消失。
3. 写出停止跟踪构建产物的安全步骤。

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
rm -rf workspaces/daily-ignore
```

## 预期结果

你能根据默认观察面板说明当前 working tree、index、HEAD、branch、remote/ref 的状态，并解释本 lab 的安全恢复路径。

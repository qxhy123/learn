# 附录A：Git 速查表

本速查表按“你现在想回答什么问题”组织，而不是按命令字母排序。

---

## 1. 默认观察面板

如果你一时不确定现在发生了什么，优先执行：

```bash
git status -sb
git log --oneline --graph --decorate --all
git diff
git diff --cached
```

这四条命令通常足以把“混乱”先变成“可观察”。

---

## 2. 查看状态与历史

| 你想知道什么 | 命令 |
|--------------|------|
| 当前分支和工作区状态 | `git status -sb` |
| 当前整体历史图 | `git log --oneline --graph --decorate --all` |
| 某次提交改了什么 | `git show <commit>` |
| 最近提交影响了哪些文件 | `git show --name-status HEAD` |
| 某个文件怎么一路演化过来 | `git log -- path/to/file` |

---

## 3. 比较不同层的差异

| 问题 | 命令 |
|------|------|
| 工作区相对索引改了什么 | `git diff` |
| 下一次提交会包含什么 | `git diff --cached` |
| 当前所有未提交变化相对 HEAD 是什么 | `git diff HEAD` |
| 两个提交之间差了什么 | `git diff A B` |
| 两个版本之间某个文件差了什么 | `git diff A B -- path/to/file` |

---

## 4. 提交与整理

| 任务 | 命令 |
|------|------|
| 暂存文件 | `git add <file>` |
| 交互式暂存 | `git add -p` |
| 创建提交 | `git commit -m "message"` |
| 修改最近提交 | `git commit --amend` |
| 整理最近 3 次提交 | `git rebase -i HEAD~3` |

---

## 5. 分支、合并与历史整理

| 任务 | 命令 |
|------|------|
| 查看分支 | `git branch` |
| 创建并切换分支 | `git switch -c feature/x` |
| 切回主分支 | `git switch main` |
| 合并分支 | `git merge feature/x` |
| 基于主干重放当前分支 | `git rebase main` |
| 进入 detached HEAD | `git switch --detach <commit>` |

---

## 6. 远程协作

| 任务 | 命令 |
|------|------|
| 查看远程 | `git remote -v` |
| 拉取远程状态 | `git fetch` |
| 查看跟踪关系 | `git branch -vv` |
| 合并式同步 | `git pull` |
| 重放式同步 | `git pull --rebase` |
| 推送当前分支 | `git push` |
| 安全强推 | `git push --force-with-lease` |

---

## 7. 撤销与恢复

| 任务 | 命令 |
|------|------|
| 丢弃工作区改动 | `git restore <file>` |
| 取消暂存 | `git restore --staged <file>` |
| 回退最近提交但保留改动 | `git reset --soft HEAD~1` |
| 回退最近提交并重置索引 | `git reset --mixed HEAD~1` |
| 彻底丢弃最近提交和改动 | `git reset --hard HEAD~1` |
| 反向撤销某次共享提交 | `git revert <commit>` |
| 查看引用移动记录 | `git reflog` |

---

## 8. 中间状态与排障

| 任务 | 命令 |
|------|------|
| 暂存未完成工作 | `git stash push -m "wip"` |
| 查看 stash | `git stash list` |
| 恢复 stash | `git stash pop` |
| 删除未跟踪文件前预演 | `git clean -nd` |
| 真正删除未跟踪文件 | `git clean -fd` |
| 查某行最近修改来源 | `git blame <file>` |
| 搜索仓库内容 | `git grep "keyword"` |
| 搜索提交说明 | `git log --grep="keyword"` |
| 二分定位坏提交 | `git bisect start` |

---

## 9. 发布与高阶操作

| 任务 | 命令 |
|------|------|
| 创建附注标签 | `git tag -a v1.0.0 -m "Release v1.0.0"` |
| 推送标签 | `git push origin v1.0.0` |
| 拣选一条提交 | `git cherry-pick <commit>` |
| 新建 worktree | `git worktree add ../repo-hotfix hotfix/x` |
| 查看配置来源 | `git config --show-origin --list` |
| 查看索引条目 | `git ls-files --stage` |

---

## 10. 高风险命令前的三问

在执行 `reset --hard`、`clean -fd`、`push --force`、`rebase -i` 之前，先问自己：

1. 这条命令会改哪一层？
2. 它会不会影响共享历史？
3. 如果结果不对，我的恢复入口是什么？

如果答不出来，先停。

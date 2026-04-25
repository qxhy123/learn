# 附录：危险区与恢复路线

危险命令不是不能用，而是必须满足三个条件：你知道它会改哪一层状态；你知道是否影响共享历史；你有恢复入口。

## 通用安全流程

1. 观察：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=20
```

2. 备份：

```bash
git switch -c rescue/$(date +%Y%m%d-%H%M%S)
# 或至少记录当前提交
git rev-parse HEAD
```

3. 判断：改动是否已经 push 到别人会基于它继续工作的分支？
4. 执行：优先选择可逆、可审查的命令。
5. 验证：重新跑观察面板和必要测试。

## `git reset --hard`

影响：移动当前分支到目标提交，并让 index、working tree 与目标一致。

风险：未提交改动会从工作区消失；如果随后强推共享分支，会覆盖团队历史。

安全替代：

- 只取消暂存：`git restore --staged <path>`。
- 只丢弃文件改动：`git restore <path>`。
- 撤销共享提交：`git revert <sha>`。

恢复：

```bash
git reflog --date=relative
git switch -c rescue/reset-before <old-sha>
```

## `git clean -fd`

影响：删除 untracked 文件和目录。

风险：Git 没跟踪的文件通常无法通过 Git 找回。

安全流程：

```bash
git clean -nd   # 只预览
git clean -fd   # 确认后执行
```

恢复：只能依赖编辑器、本机备份或系统回收站；Git 通常帮不上忙。

## `git rebase`

影响：把提交复制到新基线，生成新的提交 ID。

风险：对已共享分支 rebase 后强推，会让他人历史分叉。

安全流程：

```bash
git fetch
git status -sb
git rebase origin/main
# 不确定时
git rebase --abort
```

恢复：

```bash
git reflog
git switch -c rescue/rebase-before <old-sha>
```

## `git push --force-with-lease`

影响：更新远程分支，即使不是快进；但会先检查远端仍是你上次看到的值。

风险：仍可能改写共享历史，只是比 `--force` 多一层保护。

使用边界：

- 只在自己独占的 PR 分支使用。
- 使用前 `git fetch`。
- 团队约定允许历史整理。
- 不用于 `main`、发布分支或多人共用分支。

恢复：找平台审计、同事本地 reflog 或镜像备份，恢复最后好提交。

## 删除分支或标签

影响：移除引用；对象可能仍在一段时间内可通过 reflog 或其他引用找回。

风险：删除远程分支/移动公开标签会影响 CI、发布系统和协作者。

安全流程：

```bash
git show-ref --heads --tags
git tag -n
git branch -vv
```

恢复：如果知道旧提交：

```bash
git branch rescue/deleted <sha>
git tag corrected-tag <sha>
```

公开标签打错时，优先发布修正标签并公告；不要静默移动。

## 历史清理工具

影响：批量重写提交历史。

风险：所有提交 ID 改变，fork、clone、CI 缓存、未合并分支都会受影响。

最低要求：

1. 镜像备份。
2. 冻结窗口。
3. 全员迁移说明。
4. 平台保护规则临时调整计划。
5. 回滚负责人。

没有这些条件，不要在共享仓库执行历史清理。

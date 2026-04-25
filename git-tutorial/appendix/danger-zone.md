# 附录：危险区与恢复路线

危险命令不是不能用，而是必须满足三个条件：你知道它会改哪一层状态；你知道是否影响共享历史；你有恢复入口。

## 通用安全流程

1. 观察：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=20
```

2. 备份或命名当前状态：

```bash
git rev-parse HEAD
git switch -c rescue/$(date +%Y%m%d-%H%M%S)
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

## `git reset --mixed` / `git reset --soft`

影响：移动当前分支；`--soft` 保留 index 和 working tree，`--mixed` 重置 index 但保留 working tree。

风险：提交 ID 会从当前分支历史中消失；若这些提交已经共享，再强推会改写别人依赖的历史。

使用边界：

- 适合拆分或重做本地未共享提交。
- 不适合作为修复共享错误提交的默认手段。

恢复：使用 reflog 找回 reset 前的提交，再创建救援分支。

## `git commit --amend`

影响：用新提交替换当前提交；即使只改提交说明，也会生成新的提交 ID。

风险：如果原提交已经 push，amend 后再推送通常需要改写远程分支。

安全边界：

- 未共享提交：可用于补漏文件、改说明。
- 已共享提交：优先新提交修复；不要为了美化历史破坏协作者基线。

恢复：reflog 中通常能看到 amend 前后的提交位置。

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
git reflog --date=relative
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

## `git cherry-pick`

影响：把某个提交的补丁复制到当前分支，生成新的提交 ID。

风险：可能重复引入同一修复、遗漏上下文，或在维护分支制造隐藏冲突。

安全流程：

```bash
git status -sb
git show --stat <sha>
git cherry-pick <sha>
# 不确定时
git cherry-pick --abort
```

使用边界：适合把小修复移植到维护分支；不适合搬运一串依赖复杂的功能提交。

## 删除分支或标签

影响：移除引用；对象可能仍在一段时间内可通过 reflog 或其他引用找回。

风险：删除远程分支、移动公开标签会影响 CI、发布系统和协作者。

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

## `git stash pop`

影响：应用 stash 并在成功时删除该 stash 条目。

风险：冲突或误用时，学习者容易同时丢失“补丁来源”和当前工作区上下文。

安全替代：

```bash
git stash show -p stash@{0}
git stash apply stash@{0}
# 验证完成后再删除
git stash drop stash@{0}
```

## Hooks 与配置改动

影响：改变本机或仓库的默认行为，如提交检查、提交信息格式、push 前检查。

风险：规则过严会阻断正常工作；规则不可复现会导致“我机器上能提交”。

安全边界：

- 先用只读提示或 CI 检查验证规则。
- 仓库级 hooks 路径要写清安装方式。
- 全局配置不要假设所有团队成员都拥有同样设置。

## 历史清理工具

影响：批量重写提交历史，例如移除大文件或敏感信息。

风险：所有提交 ID 改变，fork、clone、CI 缓存、未合并分支都会受影响。

典型命令包括：`git filter-repo`、`git filter-branch`、BFG、`git lfs migrate import`、`git push --mirror`。

最低要求：

1. 镜像备份。
2. 冻结窗口。
3. 临时远端演练。
4. 全员迁移说明。
5. 平台保护规则临时调整计划。
6. 明确回滚负责人。

没有这些条件，不要在共享仓库执行历史清理。

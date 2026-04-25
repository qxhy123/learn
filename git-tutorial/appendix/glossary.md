# 附录：术语表

## 工作区（working tree）

当前目录中真实可编辑的文件状态。`git restore <path>` 会从 index 或指定来源恢复工作区文件。

## 暂存区 / 索引（index / staging area）

下一次提交的候选快照。`git add` 更新 index，`git commit` 把 index 写入历史。

## 提交（commit）

一次项目快照，包含 tree、父提交、作者、提交者和说明。提交 ID 与内容相关，重写历史会生成新 ID。

## HEAD

当前检出位置。通常指向当前分支；detached HEAD 时直接指向某个提交。

## 分支（branch）

指向某个提交的可移动引用。新提交会让当前分支向前移动。

## 引用（ref）

Git 给对象起的名字，如 `refs/heads/main`、`refs/tags/v1.0.0`。

## 远程（remote）

本地仓库记录的同步端点，如 `origin`、`upstream`。

## 远程跟踪分支（remote-tracking branch）

本地记录的远程分支状态，如 `origin/main`。`git fetch` 更新它。

## upstream tracking branch

本地分支默认跟踪的上游分支，用于简化 `pull`、`push` 和 `branch -vv` 观察。

## 快进（fast-forward）

目标分支没有分叉时，只移动分支指针即可完成的合并。

## 三方合并（three-way merge）

基于共同祖先，把两条分支的改动合并成一个结果。

## Rebase

把一组提交重新播放到新基线之上，会生成新的提交 ID。

## Revert

创建一个反向提交，用新的历史记录撤销旧提交的效果，适合共享历史。

## Reset

移动当前分支引用，并可按模式影响 index 和 working tree。

## Restore

恢复工作区或 index 中的文件状态，适合处理未提交或 staged 的文件级问题。

## Reflog

本地记录引用移动的日志。常用于找回 reset、rebase、误删分支前的提交。

## Stash

临时保存未提交改动的栈。适合短期切换任务，不适合长期任务管理。

## Worktree

同一仓库的多个工作目录，适合并行处理长期分支或 hotfix。

## Tag

给对象的稳定标记，常用于发布版本。公开发布标签不应随意移动。

## Hook

Git 在特定生命周期触发的脚本，如 `pre-commit`、`commit-msg`、`pre-push`。

## Git LFS

用于管理大文件的扩展。Git 历史保存指针，大文件内容保存在 LFS 存储中。

## Monorepo

多个项目或模块放在一个仓库中，强调统一历史和跨模块原子变更。

## Submodule

在一个仓库中嵌入另一个仓库的固定提交引用。边界清晰，但日常操作成本较高。

## 主干保护（branch protection）

平台侧规则，用于限制谁能改主干、是否必须 PR、review、CI 通过。

## `--force-with-lease`

比 `--force` 更安全的强推形式，会检查远端分支仍是本地上次观察到的值；仍只应在允许重写的独占分支使用。

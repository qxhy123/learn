# 附录C：术语表

## 工作区（Working Tree）

当前目录里真实可见、可编辑的文件内容。

## 暂存区 / 索引（Index / Staging Area）

下一次提交的候选快照。`git add` 更新它，`git commit` 把它写进历史。

## 提交（Commit）

一次被 Git 记录下来的历史快照，带有父提交关系、作者信息和说明。

## 分支（Branch）

指向某个提交的可移动引用。

## HEAD

当前检出位置。通常指向当前分支，也可能直接指向某个提交。

## 远程仓库（Remote）

团队共享的同步入口，如 `origin` 或 `upstream`。

## 远程跟踪分支（Remote-tracking Branch）

本地记录的远程状态，例如 `origin/main`。

## 快进合并（Fast-forward）

目标分支没有分叉时，直接把分支指针向前移动的合并方式。

## 三方合并（Three-way Merge）

基于共同祖先，对两条分支改动进行汇合的合并方式。

## Rebase

把一组提交重新播放到新的基线之上，会重写提交 ID。

## Revert

通过创建一个反向提交来撤销旧提交的效果。

## Reset

移动当前分支引用，并可根据模式进一步影响索引和工作区。

## Restore

恢复工作区或索引中的文件状态。

## Reflog

本地记录的引用移动历史，可用于恢复误操作。

## Detached HEAD

HEAD 直接指向某个提交，而不是某个分支。

## Tag

给某个对象（通常是提交）打上的稳定标记，常用于发布。

## Upstream Tracking Branch

本地分支默认跟踪的远程分支，用于简化 pull / push 等操作。

## Worktree

同一仓库的多个并行工作目录。

## Submodule

把另一个 Git 仓库以固定提交形式嵌入当前仓库的机制。

## Git LFS

通过指针方式管理大文件内容的扩展方案，避免把大二进制直接塞进普通 Git 历史。

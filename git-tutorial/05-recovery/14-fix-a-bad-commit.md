# 14 Fix a Bad Commit

## 场景

你已经提交了，但马上发现提交说明写错、漏加文件、把调试代码带进提交，或者错误已经进入远端分支。修复坏提交的关键不是“找一个能改掉它的命令”，而是先判断这段历史是否已经共享。

## 学习目标

- 使用 `git commit --amend` 修正最近一次未共享提交。
- 用 `git revert` 安全撤销共享历史中的错误提交。
- 判断什么时候用 `git reset` 拆本地提交，什么时候用 `git cherry-pick` 搬运补丁。
- 理解提交 SHA 改变对 PR、review 和协作者同步的影响。

## 观察点

修提交前先检查：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all -n 12
```

重点观察：

- 当前分支是否有 upstream，是否显示 ahead/behind。
- 坏提交是否已经出现在远端跟踪分支或 PR 中。
- 是否有人可能基于这个提交继续开发。
- 工作区是否干净；不干净时先提交、stash 或另建临时分支保护现场。

## 命令与判断

### 修最近一次本地提交

```bash
git add <missing-file>
git commit --amend
```

`amend` 会生成一个新的提交对象，因此提交 SHA 会改变。它适合 PR 前的本地整理，不适合已经被团队依赖的稳定历史。

### 拆开本地提交

```bash
git reset --mixed HEAD~1
git add <first-set>
git commit -m "..."
git add <second-set>
git commit -m "..."
```

这个流程适合“提交尚未共享，但内容边界不清晰”的情况。先用 `--mixed` 保留工作区内容，再重新分组。

### 撤销共享错误

```bash
git revert <bad-commit>
```

`revert` 用新提交撤销旧提交效果。它保留历史上下文，适合主干、发布分支和已经被 review 的分支。

### 搬运一个修复

```bash
git switch release/1.2
git cherry-pick <fix-commit>
```

`cherry-pick` 是把某个补丁复制到当前分支，不是合并整条分支。它适合 hotfix/backport，但重复拣选会增加追踪成本。

## 风险提示

`commit --amend`、`reset` 和交互式 rebase 都会改写提交身份。只要提交已经共享，改写历史就会让别人的分支、PR 评论锚点和自动化记录变复杂。

恢复路径：

```bash
git reflog --date=relative -n 20
git switch -c rescue-before-history-edit HEAD@{1}
```

如果 amend 或 reset 后发现改错了，优先从 reflog 找到编辑前的位置，先创建救援分支，再决定是否把正式分支移回去。

## 实验

Lab：`LAB-RECOVERY-BAD-COMMIT-01`

1. 做一次提交，故意漏掉一个文件；用 `commit --amend` 补入并观察 SHA 改变。
2. 做一个包含两类改动的本地提交；用 `reset --mixed HEAD~1` 拆成两个提交。
3. 在模拟远端主干上制造错误提交；用 `git revert` 撤销，而不是 reset。
4. 从主干挑一条修复提交到 release 分支；用 `git cherry-pick` 后比较两边提交 SHA。

## 常见错误

- 认为 amend 只是“改文字”，不会影响提交身份。
- 在共享分支上 reset 后再 force push，迫使协作者处理分叉。
- 用 cherry-pick 长期替代正常合并，导致同一补丁在多条分支上难以追踪。
- 修复坏提交前忘记确认工作区是否干净，导致问题混在一起。

## 验收

给定一个坏提交案例，你应该能先问“是否共享”，再选择：

- 未共享且只修最近一次：`commit --amend`。
- 未共享且要拆分：`reset --mixed` 后重新提交。
- 已共享且要撤销效果：`revert`。
- 修复需要进入另一条维护分支：`cherry-pick`，并记录来源提交。

完成验收时，能展示 `git log --graph --decorate --all` 说明历史为什么仍然可读。

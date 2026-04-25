# 附录：命令决策树

本附录不是命令大全，而是“先判断，再执行”的入口。每个决策树都默认先观察：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=12
```

如果命令可能改变历史或工作区，先阅读 [危险区](danger-zone.md)。术语不确定时查 [术语表](glossary.md)，高频命令查 [速查表](cheatsheet.md)。

## 我有本地改动，想撤销

1. 改动是否已经提交？
   - 没有提交：继续第 2 步。
   - 已提交且未共享：可以考虑 `git reset`、`git commit --amend`，但先确认是否要保留改动。
   - 已提交且已共享：优先 `git revert`。
2. 改动是否已经 staged？
   - 只想取消暂存：`git restore --staged <path>`。
   - 想丢弃工作区改动：`git restore <path>`。
   - 想丢弃所有未提交改动：先 `git status -sb`，再谨慎使用 `git restore .`。
3. 是否有 untracked 文件？
   - 先 `git clean -nd` 预览。
   - 确认后才 `git clean -fd`。

## 我提交错了

1. 是否已经 push 到共享分支？
   - 没有：可以 `git commit --amend` 或 `git reset --soft HEAD~1`。
   - 已经：优先新提交修复或 `git revert <sha>`。
2. 错误是否只是提交说明？
   - 未 push：`git commit --amend`。
   - 已 push：通常不要为说明重写共享历史。
3. 是否需要拆分提交？
   - 未 push：`git reset --soft <base>` 后重新 stage/commit。
   - 已 push：用后续提交修正，或在 PR 平台 squash。

## 我 push 被拒绝

1. 先观察远端：

```bash
git fetch
git branch -vv
git log --oneline --graph --decorate --all --max-count=20
```

2. 如果远端有新提交：
   - 团队约定 merge：`git merge origin/main`。
   - 团队约定 rebase：`git rebase origin/main`。
3. 解决冲突后再 push。
4. 不要用普通 `--force` 覆盖他人历史；自己独占分支也优先 `--force-with-lease`。

## 我要同步主干

1. 当前工作区是否干净？
   - 不干净：先提交、stash 或使用 worktree。
2. 当前分支是否共享？
   - 共享分支：避免 rebase 后强推。
   - 个人分支：可按团队规则 rebase 主干。
3. 同步后是否需要跑测试？
   - 至少运行与改动相关的测试或检查。

## 我找不到提交了

1. 本地找：`git reflog --date=relative`。
2. 全局找：`git log --oneline --all --decorate --graph`。
3. 找到后先救援：`git switch -c rescue/<name> <sha>`。
4. 确认内容后再决定 merge、cherry-pick 或重置原分支。

## 团队事故入口

1. 主干坏了：优先 `revert` 或修复 PR，不重写共享历史。
2. 远程误强推：暂停相关分支 push，找最后好提交，创建救援分支，恢复引用。
3. 标签打错：确认发布系统是否消费过标签；公开标签不要静默移动。
4. 大文件进历史：先新增未来规则，再规划历史清理窗口。

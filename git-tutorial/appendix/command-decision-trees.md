# 附录：命令决策树

本附录不是命令大全，而是“先判断，再执行”的入口。每个决策树都默认先观察：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=12
```

如果命令可能改变历史、删除文件或更新远程引用，先阅读 [危险区](danger-zone.md)。术语不确定时查 [术语表](glossary.md)，高频命令查 [速查表](cheatsheet.md)。

## 我不知道当前仓库状态是否安全

1. `git status -sb` 是否能看懂？
   - 不能：先识别 `??`、`M`、`A`、`D`、ahead/behind，再决定。
   - 能：继续第 2 步。
2. 是否有未提交改动？
   - 有：先用 `git diff` / `git diff --cached` 看内容；切换分支、pull、rebase 前先保护。
   - 没有：继续第 3 步。
3. 当前分支是否跟踪上游？
   - 没有：需要显式指定 push/pull 目标，或设置 upstream。
   - 有：用 `git branch -vv` 判断 ahead/behind/diverged。

## 我要把改动变成一个干净提交

1. 本次改动是否只有一个意图？
   - 是：`git add <path>` 或 `git add -p`。
   - 否：拆成多个提交，先暂存最小相关路径。
2. 暂存区是否正确？
   - 用 `git diff --cached` 验证。
   - 误暂存：`git restore --staged <path>`。
3. 提交后是否还干净？
   - `git status -sb` 确认。
   - 若还有改动，判断它们是否属于下一个提交或应恢复。

## 我有本地改动想撤销

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
   - 未 push：`git reset --mixed <base>` 后重新 stage/commit。
   - 已 push：用后续提交修正，或在 PR 平台 squash。

## 我要为任务开分支或同步分支

1. 当前工作区是否干净？
   - 不干净：先提交、stash 或使用 worktree。
   - 干净：继续第 2 步。
2. 任务是否从最新主干开始？
   - 需要最新：`git fetch` 后从 `origin/main` 或本地更新后的 `main` 开始。
   - 不需要：记录当前基线提交。
3. 分支是否已经共享？
   - 未共享：可按团队规则 rebase 整理。
   - 已共享：优先 merge 或后续修复提交，避免随意改写。

## 我遇到合并或 rebase 冲突

1. 先停止连续输入命令，保存观察结果：

```bash
git status -sb
git diff
git log --oneline --graph --decorate --all --max-count=12
```

1. 能否明确冲突文件里的 ours/theirs 含义？
   - 能：编辑冲突文件，运行检查，`git add <file>`，再继续 merge/rebase。
   - 不能：优先 `git merge --abort` 或 `git rebase --abort` 回到操作前。
2. 冲突解决后是否验证业务结果？
   - 没有验证不要提交或继续 rebase。

## 我 push 被拒绝

1. 先观察远端：

```bash
git fetch
git branch -vv
git log --oneline --graph --decorate --all --max-count=20
```

1. 如果远端有新提交：
   - 团队约定 merge：`git merge origin/main`。
   - 团队约定 rebase：`git rebase origin/main`。
2. 解决冲突并验证后再 push。
3. 不要用普通 `--force` 覆盖他人历史；自己独占分支也优先 `--force-with-lease`。

## 我要同步主干

1. 当前工作区是否干净？
   - 不干净：先提交、stash 或使用 worktree。
2. 当前分支是否共享？
   - 共享分支：避免 rebase 后强推。
   - 个人分支：可按团队规则 rebase 主干。
3. 同步后是否需要跑测试？
   - 至少运行与改动相关的测试或检查。

## 我被紧急任务打断了

1. 当前改动预计几分钟内回来继续吗？
   - 是：可用带说明的 `git stash push -m "reason" -- <path>`。
   - 否：优先提交到临时分支，或使用 worktree。
2. 紧急任务是否需要从干净主干开始？
   - 是：`git worktree add ../repo-hotfix hotfix/name` 可避免污染当前工作区。
3. 恢复 stash 前是否会冲突？
   - 先 `git stash show -p`，再 `git stash apply`；确认无误后再 `drop`。

## 我要发布或处理 hotfix

1. 发布点是否已经验证？
   - 未验证：先跑测试和检查，不先打标签。
2. 标签是否公开？
   - 未公开且本地错误：可删除重建。
   - 已公开：不要静默移动，优先发布修正标签并公告。
3. hotfix 从哪里切？
   - 从发布标签或维护分支切出，修复后再按策略回流主干。

## 我要定位历史中的问题

1. 能否定义可重复的“好/坏”检查？
   - 不能：先用 `log`、`show`、`blame` 缩小范围。
   - 能：选择 good/bad 边界后使用 `bisect`。
2. `blame` 找到的是责任人还是上下文？
   - 只把它当上下文入口，继续看 PR、相邻提交和需求背景。
3. `bisect` 后是否形成结论？
   - 记录首个坏提交、证据、影响范围和修复策略。

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
5. 需要历史清理：先镜像备份、冻结窗口、演练和迁移说明，再考虑执行。

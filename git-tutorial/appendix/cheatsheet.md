# 附录：Git 速查表

> 速查表只解决“我已经知道目标命令，想确认参数”的问题。还没有判断清楚时，先走 [命令决策树](command-decision-trees.md)；命令可能丢数据或改写共享历史时，先读 [危险区](danger-zone.md)。

## 状态观察

```bash
git status -sb
git diff
git diff --cached
git log --oneline --graph --decorate --all --max-count=20
git branch -vv
git remote -v
git show --stat --summary HEAD
```

## 日常提交

```bash
git add <path>
git add -p
git restore --staged <path>
git diff --cached
git commit -m "type: intent"
git commit --amend
```

提交前自检：

- `git diff --cached` 已读过。
- 提交只表达一个意图。
- 提交信息说明为什么，而不是重复文件名。
- 没有把日志、生成物、密钥、本地配置放进暂存区。

## 忽略与仓库卫生

```bash
git status --ignored -sb
git check-ignore -v <path>
git ls-files <path>
git rm --cached <path>
git clean -nd
git clean -fd
```

`git clean -fd` 前必须先运行 `git clean -nd` 预览。

## 分支与历史整理

```bash
git switch -c task/name
git switch main
git branch -vv
git merge <branch>
git merge --abort
git rebase origin/main
git rebase --continue
git rebase --abort
```

共享分支默认不要 rebase 后强推；个人 PR 分支也要遵守团队约定。

## 远程协作

```bash
git remote -v
git fetch
git branch -vv
git pull --ff-only
git push -u origin task/name
git push --force-with-lease origin task/name
```

push 被拒绝时的默认顺序：`git fetch` → 看历史图 → 按团队规则 merge 或 rebase → 再 push。

## 撤销与恢复

```bash
git restore <path>
git restore --staged <path>
git reset --soft HEAD~1
git reset --mixed HEAD~1
git revert <sha>
git reflog --date=relative
git switch -c rescue/name <sha>
```

恢复底线：先保护现场，再移动分支；共享历史优先 `revert`。

## Stash 与 Worktree

```bash
git stash push -m "reason" -- <path>
git stash list
git stash show -p stash@{0}
git stash apply stash@{0}
git stash drop stash@{0}
git worktree list
git worktree add ../repo-hotfix hotfix/name
```

`stash` 适合短期中断；长期并行任务优先分支或 worktree。

## 发布与排障

```bash
git tag -a v1.0.0 -m "v1.0.0"
git tag --list --sort=-creatordate
git show v1.0.0
git blame <path>
git show <commit>
git bisect start
git bisect good <sha>
git bisect bad <sha>
git bisect reset
```

正式发布优先 annotated tag；公开标签不要静默移动。

## 规模与治理

```bash
git config --show-origin --list
git config --global alias.st 'status -sb'
git config core.hooksPath .githooks
git count-objects -vH
git ls-files
git show-ref --heads --tags
git worktree list
git lfs env 2>/dev/null || true
```

治理命令通常影响团队默认行为；先在临时仓库或试点分支验证。

## 危险命令提醒

先读 [危险区](danger-zone.md)，再考虑：

```bash
git reset --hard <sha>
git clean -fd
git push --force-with-lease
git tag -f <tag> <sha>
git push origin :branch-name
git filter-repo ...
git push --mirror
```

共享分支优先使用 `revert` 和修复提交；独占分支才考虑历史整理。

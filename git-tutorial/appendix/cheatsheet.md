# 附录：Git 速查表

## 状态观察

```bash
git status -sb
git diff
git diff --cached
git log --oneline --graph --decorate --all --max-count=20
git branch -vv
git show --stat --summary HEAD
```

## 日常提交

```bash
git add <path>
git add -p
git restore --staged <path>
git commit -m "type: intent"
git commit --amend
```

## 分支

```bash
git switch -c task/name
git switch main
git merge <branch>
git rebase origin/main
git merge --abort
git rebase --abort
```

## 远程协作

```bash
git remote -v
git fetch
git pull --ff-only
git push -u origin task/name
git push --force-with-lease origin task/name
```

## 恢复

```bash
git restore <path>
git restore --staged <path>
git reset --soft HEAD~1
git revert <sha>
git reflog --date=relative
git switch -c rescue/name <sha>
git clean -nd
```

## 发布与排障

```bash
git tag -a v1.0.0 -m "v1.0.0"
git show v1.0.0
git blame <path>
git bisect start
git bisect good <sha>
git bisect bad <sha>
git bisect reset
```

## 规模与治理

```bash
git config --show-origin --list
git config --global alias.st 'status -sb'
git config core.hooksPath .githooks
git count-objects -vH
git ls-files
git show-ref --heads --tags
git worktree list
```

## 危险命令提醒

先读 [危险区](danger-zone.md)，再考虑：

```bash
git reset --hard <sha>
git clean -fd
git push --force-with-lease
git tag -f <tag> <sha>
git push origin :branch-name
```

共享分支优先使用 `revert` 和修复提交；独占分支才考虑历史整理。

# 06 Release and Debugging：中断处理、发布与历史排障

本模块面向真实开发节奏：任务被打断、版本需要发布、线上问题需要从历史中定位。学习重点是把 Git 当成协作和排障工具，而不只是本地保存工具。

## 学习路径

1. [16 Stash, Worktree and Interruptions](./16-stash-worktree-and-interruptions.md)：安全处理中断和临时工作。
2. [17 Tags, Releases and Hotfixes](./17-tags-releases-and-hotfixes.md)：用标签和维护分支表达发布点与热修流程。
3. [18 Blame, Bisect and History Debugging](./18-blame-bisect-and-history-debugging.md)：从症状回溯到提交，用历史缩小排障范围。

## 本模块统一观察面板

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all -n 16
git tag --list --sort=-creatordate | head
```

排障时追加：

```bash
git blame <path>
git show <commit>
git bisect status
```

## 决策底线

- `stash` 是短期收纳，不是长期任务管理；长期并行优先考虑分支或 worktree。
- 正式发布标签应稳定、可追溯；错误标签若已推送，需要团队同步处理。
- `blame` 用来找上下文，不用来甩锅；`bisect` 用来把“感觉最近坏了”变成可验证的坏提交。

## Lab IDs

- `LAB-RELEASE-STASH-WORKTREE-01`：处理中断、恢复 stash 冲突、比较 stash 与 worktree。
- `LAB-RELEASE-HOTFIX-TAG-01`：从发布标签切出 hotfix，修复后打 patch 标签。
- `LAB-DEBUG-BISECT-01`：构造回归并用 bisect 定位首个坏提交。

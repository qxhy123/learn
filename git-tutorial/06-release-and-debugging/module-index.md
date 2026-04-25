# 06 发布与历史排障模块导览

## 模块目标

本模块面向真实开发节奏：任务被打断、版本需要发布、线上问题需要从历史中定位。学习重点是把 Git 当成协作和排障工具，而不只是本地保存工具。

## 学习路径

| 顺序 | 章节 | 你要学会的判断 | Lab id |
|---|---|---|---|
| 16 | [Stash、Worktree 与中断处理](./16-stash-worktree-and-interruptions.md) | 区分短期收纳、上下文切换和长期并行任务。 | `LAB-RELEASE-STASH-WORKTREE-01` |
| 17 | [Tags、Releases 与 Hotfixes](./17-tags-releases-and-hotfixes.md) | 用标签和维护分支表达发布点与热修流程。 | `LAB-RELEASE-HOTFIX-TAG-01` |
| 18 | [Blame、Bisect 与历史排障](./18-blame-bisect-and-history-debugging.md) | 从症状回溯到提交，用历史缩小排障范围。 | `LAB-DEBUG-BISECT-01` |

## 发布与排障观察面板

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=16
git tag --list --sort=-creatordate | head
```

排障时追加：

```bash
git blame <path>
git show <commit>
git bisect log
```

## 决策底线

- `stash` 是短期收纳，不是长期任务管理；长期并行优先考虑分支或 worktree。
- 正式发布标签应稳定、可追溯；公开标签打错时需要公告和修正方案。
- `blame` 用来找上下文，不用来甩锅；`bisect` 用来把“感觉最近坏了”变成可验证的坏提交。
- 自动 bisect 只适合检查命令稳定、无副作用且能清晰返回成功/失败的场景。

## 模块验收

- 能为“紧急 hotfix 打断当前任务”选择 stash、提交临时分支或 worktree。
- 能解释 annotated tag 与 lightweight tag 的协作差异。
- 能从发布标签切出 hotfix 分支，并说明修复如何回流主干。
- 能设计一个可靠的 bisect good/bad 边界和检查命令。

## 相关附录

- 中断处理入口：见 [命令决策树](../appendix/command-decision-trees.md#我被紧急任务打断了)。
- 标签风险：见 [危险区](../appendix/danger-zone.md#删除分支或标签)。
- 发布与排障命令：见 [速查表](../appendix/cheatsheet.md#发布与排障)。

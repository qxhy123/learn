# 04 远程协作模块导览

## 模块目标

远程协作的核心不是“把代码传上去”，而是在本地历史、远程跟踪分支、团队主干和 Pull Request 之间保持可解释的同步关系。
本模块覆盖 clone/fetch/pull/push、同步主干、处理 push rejected、打开 PR、review 与团队约定。

## 学习路径

| 顺序 | 章节 | 你要学会的判断 | Lab id |
| --- | --- | --- | --- |
| 10 | [clone、fetch、pull 与 push](./10-clone-fetch-pull-push.md) | 区分远程名、远程跟踪分支、上游分支和本地分支。 | `LAB-COLLAB-REMOTE-01` |
| 11 | [同步主干并打开 PR](./11-sync-with-main-and-open-pr.md) | 处理本地领先、远程领先、双方都前进和 push rejected。 | `LAB-COLLAB-PUSH-REJECTED-01` |
| 12 | [Review 与团队约定](./12-review-and-team-conventions.md) | 把 PR 当作团队历史治理接口，而不只是代码审查页面。 | `LAB-COLLAB-PR-01` |

## 协作观察面板

```bash
git status -sb
git branch -vv
git remote -v
git log --oneline --graph --decorate --all --max-count=16
```

观察顺序固定为：

1. 当前分支跟踪哪个上游分支。
2. 本地分支相对上游是 ahead、behind，还是 diverged。
3. `origin/main` 是否刚刚 fetch 过，是否代表最新远程快照。
4. 即将执行的命令只更新远程跟踪分支，还是会改动当前分支或远程分支。

## 模块验收

- 能解释 `fetch` 与 `pull` 的区别。
- 能在 push rejected 后先 `fetch` 和看图，再按团队规则 merge 或 rebase。
- 能写出 PR 自检清单：范围、测试、风险、回滚方式。
- 能说明 squash、merge commit、rebase merge 三种策略对历史的影响。

## 相关附录

- push rejected 入口：见 [命令决策树](../appendix/command-decision-trees.md#我-push-被拒绝)。
- 强推边界：见 [危险区](../appendix/danger-zone.md#git-push---force-with-lease)。
- 协作命令：见 [速查表](../appendix/cheatsheet.md#远程协作)。

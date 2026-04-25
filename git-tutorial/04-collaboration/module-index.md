# 04 远程协作模块导览

远程协作的核心不是“把代码传上去”，而是在本地历史、远程跟踪分支、团队主干和 Pull Request 之间保持可解释的同步关系。本模块围绕 clone/fetch/pull/push、同步主干并打开 PR、review 与团队约定展开。

## 学习路径

1. [10 clone、fetch、pull 与 push](./10-clone-fetch-pull-push.md)：先建立远程跟踪分支的心智模型，再执行同步命令。
2. [11 同步主干并打开 PR](./11-sync-with-main-and-open-pr.md)：处理本地领先、远程领先、双方都前进和 push rejected。
3. [12 Review 与团队约定](./12-review-and-team-conventions.md)：把 PR 视为团队历史治理接口，而不只是代码审查页面。

## 本模块统一观察面板

协作命令前后固定记录：

```bash
git status -sb
git branch -vv
git remote -v
git log --oneline --graph --decorate --all -n 16
```

观察顺序固定为：

1. 当前分支跟踪哪个远程分支。
2. 本地分支相对上游是 ahead、behind，还是 diverged。
3. `origin/main` 是否刚刚 fetch 过，是否代表最新远程快照。
4. 即将执行的命令会只更新远程跟踪分支，还是会改动当前分支或远程分支。

## Lab ID

- `LAB-COLLAB-REMOTE-01`：clone 仓库、fetch 更新并观察远程跟踪分支。
- `LAB-COLLAB-PUSH-REJECTED-01`：模拟双人提交导致的 non-fast-forward push rejected。
- `LAB-COLLAB-PR-01`：准备一个可 review 的任务分支和 PR 自检清单。

## 安全约定

- push 被拒绝时，默认先 `git fetch` 和看图，不直接 force push。
- `origin/main` 是本地保存的远程状态快照；不 fetch 就可能过期。
- PR 打开后是否允许改写历史，按团队约定执行，不把个人偏好当默认规则。

# 11 同步主干并打开 PR

## 章节导航

- 前置章节：`04-collaboration/10-clone-fetch-pull-push.md`
- 后续章节：`04-collaboration/12-review-and-team-conventions.md`
- 本章 Lab ID：`LAB-COLLAB-PUSH-REJECTED-01`
- 核心危险命令：`git push --force`
- 推荐恢复入口：`git fetch` 后重新观察历史图；同步失败时用
  `git merge --abort` 或 `git rebase --abort` 回到操作前现场。

## 场景

你在 `feature/login-copy` 上完成了改动，准备推送并打开 PR。
与此同时，`origin/main` 可能已经前进，远程同名任务分支也可能被
CI 修复提交、协作者提交或你另一台电脑上的提交更新过。

本章的目标不是让你记住“push 失败就 pull 一下”，而是训练协作前
的判断流程：先观察本地分支、远程跟踪分支、主干和 PR diff 的关系，
再决定是直接推送、merge 主干、rebase 主干，还是先暂停找队友确认。

## 学习目标

完成本章后，你应该能够：

1. 识别本地领先、远程领先、双方都前进和远程历史被改写这几类协作状态。
2. 在 non-fast-forward / push rejected 后解释 Git 正在保护远程已有提交。
3. 根据团队规则选择 merge 同步、rebase 同步或停止并沟通。
4. 打开 PR 前用自检清单确认变更范围、验证结果、风险和回滚说明。
5. 区分 `git push --force` 与 `git push --force-with-lease` 的安全边界。

## 观察点

准备推送或打开 PR 前，固定执行这组观察命令：

```bash
git status -sb
git fetch origin
git branch -vv
git log --oneline --graph --decorate --all -n 16
git diff --stat origin/main...HEAD
```

你要能说清楚：

- 当前工作区是否干净，是否还有未提交或未暂存改动。
- `origin/main` 是否因为刚刚 `fetch` 而前进。
- 当前任务分支相对上游是 `ahead`、`behind`、`ahead/behind` 都存在，还是没有上游。
- 历史图里你的任务分支是否基于最新主干。
- `origin/main...HEAD` 展示的是否正是本次 PR 应该被 review 的范围。

一个常见判断表：

| 观察结果 | 含义 | 安全下一步 |
| --- | --- | --- |
| 工作区干净，当前分支只 ahead | 本地有新提交，远程没有额外更新 | 可 push 或打开 PR |
| `origin/main` 前进，但任务分支仍可清楚比较 | 主干变了，PR 需要确认新基线 | 按团队规则 merge 或 rebase 主干 |
| push rejected | 远程分支已有你本地没有的提交 | 先 fetch + 看历史图 |
| 任务分支和远程同名分支 diverged | 双方都新增了提交 | 判断远程提交是谁的，再决定整合方式 |
| PR diff 混入无关文件 | 分支边界不干净 | 拆分或整理提交后再开 PR |

## 操作流程

### 1. 推送前先建立上游关系

如果这是第一次推送任务分支，且观察确认当前分支只是在本地领先：

```bash
git push -u origin feature/login-copy
```

`-u` 会让本地分支记住默认上游。之后 `git branch -vv` 能直接显示 ahead/behind，`git push` 也知道默认目标。

### 2. push 被拒绝时先停止自动反应

看到类似 `non-fast-forward`、`fetch first`、`rejected` 时，不要立刻执行
`git push --force`，也不要把错误理解成“Git 不让我工作”。先更新观察基线：

```bash
git fetch origin
git log --oneline --graph --decorate --all -n 16
git branch -vv
```

此时要回答两个问题：

1. 远程多出来的提交是谁产生的？是队友、CI、你另一台电脑，还是平台自动提交？
2. 这些提交是否已经被别人依赖，是否允许你改写远程任务分支历史？

### 3. 按团队规则同步主干

如果团队要求保留真实汇合历史，使用 merge：

```bash
git merge origin/main
```

它会在任务分支中保留“主干在此时汇入”的事实。适合多人协作分支、需要保留同步节点的团队。

如果这是个人任务分支，且团队允许线性整理，使用 rebase：

```bash
git rebase origin/main
```

它会把你的提交移到最新主干之后，让 PR 历史更线性。rebase 改写的是本地任务分支提交身份，因此在已经推送过的分支上使用前必须确认团队约定。

### 4. rebase 后更新远程个人分支

rebase 后如果需要更新远程个人任务分支，优先使用：

```bash
git push --force-with-lease
```

`--force-with-lease` 会检查远程分支仍然是你上次观察到的状态。它比 `--force` 安全，但仍是改写远程历史的动作，只能用于允许改写的个人/任务分支，不能把它当成“万能修复按钮”。

## PR 前自检

发起 PR 前，至少确认：

1. `git status -sb` 显示工作区干净。
2. `git diff origin/main...HEAD` 只包含本次任务范围。
3. `git log --oneline origin/main..HEAD` 中每个提交都有清楚意图。
4. 没有调试文件、临时日志、无关格式化或顺手重构。
5. 已执行与改动相关的自动测试或手工验证。
6. PR 描述包含背景、方案、验证、风险和回滚思路。

一个可直接套用的 PR 描述骨架：

```text
背景：为什么需要这个改动
方案：主要改了什么，为什么这样改
验证：运行了哪些检查；如未运行，原因是什么
风险：可能影响哪些路径或用户
回滚：如果出问题，如何撤回或关闭开关
```

## 实验

Lab ID：`LAB-COLLAB-PUSH-REJECTED-01`

实验目标：用两个本地副本模拟两名协作者，制造 push rejected，并分别观察 merge 同步与 rebase 同步的差异。

实验步骤：

1. 准备一个 bare 远程仓库，clone 成 `alice` 与 `bob` 两份副本。
2. `alice` 和 `bob` 都从同一个 `main` 提交开始，并各自创建任务分支。
3. `alice` 在远程同名任务分支上提交并 push。
4. `bob` 在旧基线上提交后尝试 push，观察 non-fast-forward rejected。
5. `bob` 执行 `git fetch origin`，再用历史图命令画出双方提交。
6. 在一个变体中执行 `git merge origin/main`；在另一个变体中执行 `git rebase origin/main`。
7. 比较两种历史图，并写下哪种策略符合你的团队规则。

预期结果：学习者能解释 push rejected 是在保护远程已有提交；能在打开 PR 前确认 diff 范围和同步策略，而不是盲目 force push。

## 常见错误

- **push rejected 后立刻 `--force`。** 这可能覆盖别人已经共享或依赖的提交。
- **不 fetch 就判断冲突。** 本地 `origin/*` 可能是旧快照，判断会基于过期信息。
- **把 `pull` 当成唯一同步方案。** 它可能触发 merge 或 rebase；协作场景中要知道自己选择了哪种整合方式。
- **PR 范围混入无关改动。** reviewer 会被迫替你拆分任务边界，风险也会扩大。
- **rebase 后不重新验证。** 新基线可能让原本通过的改动出现语义冲突。
- **PR 描述只写“如题”。** 这会隐藏风险、验证和回滚信息，让 review 变成猜谜。

## 危险提示与恢复路径

危险动作：`git push --force` 会无条件尝试改写远程分支，不检查远程是否已经被他人更新。它可能让队友的提交从远程分支上消失。

更安全的替代：

```bash
git push --force-with-lease
```

但这个替代也有边界：只有当这是允许改写的个人任务分支，并且你刚刚 `fetch`、确认没有协作者依赖旧历史时才使用。

恢复路径：

1. 同步过程中发现方向不对，先用 `git merge --abort` 或 `git rebase --abort` 保留现场。
2. 误强推个人分支时，先停止继续 push，使用本地 `git reflog`、远程平台记录或同事 clone 找回被覆盖提交。
3. 误强推共享分支时，立即通知团队，不要独自继续修；由维护者根据保护规则、reflog、平台审计记录或备份恢复。
4. 如果 PR diff 混乱，优先新建干净分支并 cherry-pick 必要提交，而不是在混乱分支上继续叠加修复。

## 验收

你应该能回答：

1. non-fast-forward rejected 保护的是什么？
2. 本地 `ahead`、`behind`、`diverged` 分别意味着什么？
3. 什么时候可以在个人分支上使用 `--force-with-lease`？
4. merge 同步主干和 rebase 同步主干在历史图上有什么差异？
5. 为什么 PR 前要看 `origin/main...HEAD`，而不是只看当前文件列表？
6. 一个合格 PR 描述至少应该包含哪些验证和回滚信息？

## 交付自查

- 模板字段：已包含场景、学习目标、观察点、实验、常见错误、验收。
- Lab ID：`LAB-COLLAB-PUSH-REJECTED-01`。
- 前置/后续章节：前置 `10-clone-fetch-pull-push.md`，后续 `12-review-and-team-conventions.md`。
- 危险命令：`git push --force`；替代和边界为 `git push --force-with-lease`。
- 恢复路径：`git merge --abort`、`git rebase --abort`、reflog、远程平台记录、同事 clone、干净分支 cherry-pick。
- 需要后置集成处理：Labs agent 需为本章 lab 落地 alice/bob 双副本实验；
  README/module-index 集成 agent 需接入本章导航。

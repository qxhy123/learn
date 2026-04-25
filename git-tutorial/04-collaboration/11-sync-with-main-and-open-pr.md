# 11 同步主干并打开 PR

## 场景

你已经在任务分支上完成改动，准备推送并打开 PR。但远程主干可能已经前进，甚至有人先推送了与你相关的改动。你需要先判断本地、远程和主干的相对位置，再选择 merge、rebase 或直接 push。

## 学习目标

完成本章后，你应该能够：

1. 识别本地领先、远程领先、双方都前进和远程历史被改写的状态。
2. 在 push rejected 后解释 Git 正在保护什么。
3. 根据团队规则选择 merge 同步或 rebase 同步。
4. 打开 PR 前用自检清单确认变更范围、验证结果和风险说明。

## 观察点

准备 PR 前固定执行：

```bash
git status -sb
git fetch origin
git branch -vv
git log --oneline --graph --decorate --all -n 16
git diff --stat origin/main...HEAD
```

你要判断：

- 当前分支是否干净。
- `origin/main` 是否刚更新。
- 当前分支相对上游是 ahead、behind 还是 diverged。
- `origin/main...HEAD` 展示的是否正是本次 PR 范围。

## 命令

如果任务分支只是在本地领先，可以推送：

```bash
git push -u origin feature/login-copy
```

如果 push 被拒绝，先不要 force：

```bash
git fetch origin
git log --oneline --graph --decorate --all -n 16
```

如果团队要求保留真实汇合历史：

```bash
git merge origin/main
```

如果这是个人任务分支，且团队允许线性整理：

```bash
git rebase origin/main
```

rebase 后需要更新远程个人分支时，优先使用：

```bash
git push --force-with-lease
```

`--force-with-lease` 会检查远程是否仍是你上次观察到的状态。它比 `--force` 更安全，但仍然是改写远程历史的动作，必须只用于允许改写的分支。

## PR 前自检

发起 PR 前，至少确认：

1. `git diff origin/main...HEAD` 只包含本次任务。
2. 无调试文件、临时日志、无关格式化。
3. 提交说明能表达每一步意图。
4. 已执行与改动相关的测试或手工验证。
5. PR 描述写清背景、方案、风险、验证和回滚思路。

一个简短 PR 描述可以包含：

```text
背景：为什么需要这个改动
方案：主要改了什么
验证：运行了哪些检查
风险：可能影响哪些路径
回滚：如果出问题，如何撤回
```

## 实验

Lab ID：`LAB-COLLAB-PUSH-REJECTED-01`

实验步骤：

1. 准备一个 bare 远程仓库，clone 成 `alice` 与 `bob` 两份副本。
2. `alice` 和 `bob` 都从同一主干提交开始。
3. `alice` 提交并 push。
4. `bob` 在旧基线上提交后尝试 push，观察 non-fast-forward rejected。
5. `bob` 执行 fetch，看历史图，再分别尝试 merge 同步和 rebase 同步的变体。
6. 记录哪种策略符合团队规则。

预期结果：学习者能解释 push rejected 是在保护远程已有提交，而不是 Git 阻止你工作。

## 常见错误

- **push rejected 后立刻 `--force`。** 这可能覆盖别人已经共享的历史。
- **不 fetch 就判断冲突。** 你的远程跟踪分支可能过期，判断基于旧信息。
- **PR 范围混入无关改动。** reviewer 会被迫替你拆分任务边界。
- **rebase 后不重新验证。** 新基线可能让原本通过的改动出现语义冲突。

## 危险提示与恢复路径

危险动作：`git push --force` 会无条件尝试改写远程分支。它不检查远程是否已经被他人更新。

恢复路径：

1. 需要改写个人分支时，用 `--force-with-lease` 替代 `--force`，并确认没有协作者依赖旧历史。
2. 如果误强推共享分支，立即停止后续 push，通知团队，使用远程平台记录、同事本地 clone 或 reflog 寻找被覆盖提交。
3. 如果只是本地同步失败，先 `git merge --abort` 或 `git rebase --abort` 保留现场，再重新观察图形。

## 验收

你应该能回答：

1. non-fast-forward rejected 保护的是什么？
2. 本地 ahead、behind、diverged 分别意味着什么？
3. 什么时候可以在个人分支上使用 `--force-with-lease`？
4. 为什么 PR 前要看 `origin/main...HEAD` 而不是只看当前文件列表？

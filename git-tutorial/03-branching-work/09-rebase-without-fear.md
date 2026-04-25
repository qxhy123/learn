# 09 不恐惧地使用 rebase

## 场景

你的任务分支还没有合并，但主干已经前进。你希望让自己的提交基于最新主干重新验证，减少同步型 merge commit。rebase 可以做到这一点，但它会重写提交身份，所以必须先判断边界：这是本地个人分支，还是别人已经依赖的共享历史？

## 学习目标

完成本章后，你应该能够：

1. 用“把一组本地提交重新播放到新基线”解释 rebase。
2. 区分 rebase 与 merge 对历史图的影响。
3. 在本地未共享分支上执行 rebase，并处理或中止冲突。
4. 识别不应该 rebase 的共享历史场景。

## 状态观察

rebase 前先观察：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all -n 16
```

你要确认：

- 工作区干净。
- 当前分支是个人任务分支，而不是受保护主干。
- 当前分支上的提交是否已经 push 给他人使用。
- 主干前进了哪些提交，你准备把哪些本地提交重放过去。

## 命令

在任务分支上同步主干：

```bash
git fetch origin
git switch feature/login-copy
git rebase origin/main
```

如果没有冲突，历史图会从分叉汇合变成线性重放。注意：原提交 ID 会变化，新提交不是原提交本身。

冲突发生时：

```bash
git status
# 编辑冲突文件
git add <resolved-file>
git rebase --continue
```

无法判断时：

```bash
git rebase --abort
```

只有在你明确当前提交已经不需要时，才考虑：

```bash
git rebase --skip
```

`--skip` 会丢掉正在重放的那次提交所代表的改动，不应作为默认修复动作。

## merge 与 rebase 的选择

- 需要保留真实分叉和汇合语义：倾向 merge。
- 整理本地个人分支、准备发起 review：可以 rebase。
- 主干、发布分支、多人共用分支：通常不要 rebase。
- PR 已经打开且 reviewer 正在按提交阅读：rebase 前先看团队约定。

关键不是“哪种更高级”，而是你是否接受改写提交身份带来的协作成本。

## 实验

Lab ID：`LAB-BRANCH-REBASE-01`

实验步骤：

1. 从 `main` 创建 `feature/rebase-demo`，连续做两次小提交。
2. 回到 `main`，再做一次独立提交。
3. 复制当前历史图作为 rebase 前证据。
4. 切回任务分支，执行 `git rebase main`。
5. 再次复制历史图，比较提交 ID、父子关系和图形变化。
6. 重做同样场景，用 `git merge main` 对比结果。

预期结果：学习者能解释 rebase 为什么改写提交身份，以及何时应选择 merge 而不是 rebase。

## 常见错误

- **把 rebase 当成美化历史的无风险命令。** 它会生成新提交，改变提交 ID 与父子关系。
- **在共享分支上随手 rebase。** 这会让其他人的本地历史难以同步。
- **冲突时盲目 `--skip`。** 这可能丢掉一条仍然需要的业务改动。
- **rebase 后忘记重新运行验证。** 新基线可能改变语义，即使命令成功也要测试。

## 危险提示与恢复路径

危险动作：对已共享分支 rebase 后再强推。它可能让远程历史不再包含协作者基于的提交。

恢复路径：

1. rebase 过程中不确定：执行 `git rebase --abort`。
2. rebase 已完成但尚未 push：用 `git reflog` 找到 rebase 前的分支位置，再在恢复模块中学习安全回到该位置。
3. 已经推送并影响他人：停止继续强推，通知团队，按协作模块的规则判断是否用 `--force-with-lease` 或创建修复提交。

## 验收

你应该能回答：

1. rebase 后提交内容相似，为什么提交 ID 仍会改变？
2. `git rebase --continue`、`--skip`、`--abort` 的语义分别是什么？
3. 为什么本地个人分支适合 rebase，而共享主干通常不适合？
4. rebase 成功后为什么仍要重新运行测试或至少检查关键 diff？

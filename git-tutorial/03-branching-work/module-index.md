# 03 分支工作模块导览

## 模块目标

分支不是“复制一份项目”，而是把一段任务工作放在可观察、可合并、可恢复的轨道上。
本模块围绕三类真实任务展开：为任务开分支、用 playbook 处理冲突、在安全边界内 rebase。

## 学习路径

| 顺序 | 章节 | 你要学会的判断 | Lab id |
| --- | --- | --- | --- |
| 07 | [为一个任务创建分支](./07-branch-for-a-task.md) | 从干净基线创建短生命周期任务分支，并保持可切换。 | `LAB-BRANCH-TASK-01` |
| 08 | [用 playbook 解决合并冲突](./08-merge-conflicts-with-a-playbook.md) | 把冲突当作历史汇合中的人工判断点，按步骤验证结果。 | `LAB-BRANCH-CONFLICT-01` |
| 09 | [不恐惧地使用 rebase](./09-rebase-without-fear.md) | 只在适合的边界内重放本地提交，并能随时中止恢复。 | `LAB-BRANCH-REBASE-01` |

## 本模块统一观察面板

```bash
git status -sb
git branch --show-current
git branch -vv
git log --oneline --graph --decorate --all --max-count=12
```

观察顺序固定为：

1. 工作区是否干净，是否能安全切换分支。
2. 当前分支的基线是什么，是否跟踪上游。
3. 当前分支相对主干是领先、落后，还是已经分叉。
4. 操作后历史图多了 merge commit、线性重放提交，还是只移动了分支引用。

## 模块验收

- 能解释任务分支和基线分支的关系。
- 能在冲突中分清 ours、theirs 和共同祖先。
- 能说明 merge 与 rebase 对提交 ID 和历史图的影响。
- 能在 merge/rebase 不确定时优先使用 `--abort`，而不是继续扩大事故面。

## 相关附录

- 分支选择入口：见 [命令决策树](../appendix/command-decision-trees.md#我要为任务开分支或同步分支)。
- 历史改写边界：见 [危险区](../appendix/danger-zone.md#git-rebase)。
- 分支命令：见 [速查表](../appendix/cheatsheet.md#分支与历史整理)。

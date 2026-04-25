# 03 分支工作模块导览

分支不是“复制一份项目”，而是把一段任务工作放在可观察、可合并、可恢复的轨道上。本模块围绕三类真实任务展开：为任务开分支、用 playbook 处理冲突、在安全边界内 rebase。

## 学习路径

1. [07 为一个任务创建分支](./07-branch-for-a-task.md)：从主干状态出发，创建短生命周期任务分支，并保持工作区可切换。
2. [08 用 playbook 解决合并冲突](./08-merge-conflicts-with-a-playbook.md)：把冲突当作历史汇合中的人工判断点，而不是工具故障。
3. [09 不恐惧地使用 rebase](./09-rebase-without-fear.md)：只在适合的边界内重放本地提交，并能随时中止恢复。

## 本模块统一观察面板

每次执行分支、合并、rebase 相关命令前后，至少记录：

```bash
git status -sb
git branch --show-current
git log --oneline --graph --decorate --all -n 12
```

观察顺序固定为：

1. 当前工作区是否干净。
2. HEAD 附着在哪个分支或提交。
3. 当前分支相对主干是领先、落后，还是已经分叉。
4. 操作后历史图多了 merge commit、线性重放提交，还是只移动了分支引用。

## Lab ID

- `LAB-BRANCH-TASK-01`：从干净主干创建任务分支并完成小步提交。
- `LAB-BRANCH-CONFLICT-01`：故意制造同一行冲突并按 playbook 解决。
- `LAB-BRANCH-REBASE-01`：对本地未共享分支 rebase，并对比 merge 历史图。

## 安全约定

- 对已共享分支执行会改写历史的操作前，先停下并确认团队规则。
- 不确定是否能继续合并或 rebase 时，优先保留现场：复制 `git status` 与历史图，再选择 `--abort`。
- 不用 `git push --force` 处理本模块问题；远程改写边界放在协作模块中讨论。

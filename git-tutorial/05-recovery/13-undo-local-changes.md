# 13 Undo Local Changes

## 场景

你改了几个文件后发现方向错了，或者已经 `git add` 了一部分内容但暂时不想提交。最危险的反应是直接运行 `git reset --hard`，因为你还没有确认要撤销的是工作区、暂存区，还是提交历史。本节训练你先观察，再选择最小影响的撤销动作。

## 学习目标

- 区分工作区、暂存区、当前分支引用三层状态。
- 用 `git restore` 丢弃工作区改动或取消暂存。
- 理解 `git reset --soft`、`--mixed`、`--hard` 分别影响哪些层。
- 在已共享历史中优先选择 `git revert` 而不是重写历史。

## 观察点

操作前先回答四个问题：

1. `git status -sb` 显示的是 modified、staged 还是 untracked？
2. `git diff` 里是否有尚未备份的工作区内容？
3. `git diff --cached` 里是否有已经进入暂存区的内容？
4. `git log --oneline --decorate -n 5` 中要移动的是不是已经推送或被别人使用的提交？

建议观察命令：

```bash
git status -sb
git diff
git diff --cached
git log --oneline --graph --decorate --all -n 8
```

## 命令与判断

| 目标 | 首选命令 | 会影响什么 | 不会影响什么 |
|---|---|---|---|
| 丢弃某个已跟踪文件的工作区改动 | `git restore <path>` | 工作区文件 | 暂存区、提交历史 |
| 取消暂存但保留文件内容 | `git restore --staged <path>` | 暂存区 | 工作区内容、提交历史 |
| 撤回最近本地提交但保留 staged 内容 | `git reset --soft HEAD~1` | 当前分支引用 | 工作区、暂存区内容 |
| 撤回最近本地提交并取消暂存 | `git reset --mixed HEAD~1` | 当前分支引用、暂存区 | 工作区内容 |
| 撤回提交并丢弃对应文件状态 | `git reset --hard HEAD~1` | 当前分支引用、暂存区、工作区 | 未跟踪文件通常不处理 |
| 撤销已共享提交的效果 | `git revert <commit>` | 新增一个反向提交 | 不重写原历史 |

## 风险提示

`git reset --hard` 是本节最高风险命令。它会把当前分支、暂存区和已跟踪工作区一起重置到目标提交；如果你还没有确认差异内容，可能直接丢掉未提交修改。

安全替代路径：

```bash
git status -sb
git diff > ../before-reset.patch
git branch rescue-before-reset
```

确认补丁和救援分支都存在后，再决定是否执行破坏性重置。对已共享提交，不要用 `reset --hard` 试图“抹掉问题”，优先用 `git revert` 留下可审计的修正记录。

## 实验

Lab：`LAB-RECOVERY-UNDO-01`

准备一个练习仓库，按顺序制造三种状态：

1. 修改文件但不暂存，运行 `git restore <path>`，观察 `git diff` 清空。
2. 修改并暂存文件，运行 `git restore --staged <path>`，观察内容回到工作区但不在暂存区。
3. 连续做两个本地提交，分别在临时分支上试 `reset --soft`、`reset --mixed`、`reset --hard`，记录 status、diff、log 的差异。
4. 模拟“错误提交已共享”：不要 reset 主干，使用 `git revert <commit>` 生成反向提交。

## 常见错误

- 没看 `git diff` 就运行 `reset --hard`。
- 把取消暂存误认为必须使用 `git reset`。
- 对已经推送的主干提交使用 reset，导致协作者历史分叉。
- 误以为 `revert` 会删除旧提交；它实际是新增反向提交。

## 验收

你应该能根据下列表述选择命令并说明后果：

- “我只是不想把这个文件放进下一次提交，但文件内容还要保留。”
- “最近一次提交只在本地，我想拆开重新提交。”
- “这个错误提交已经进了团队主干，需要保留审计记录。”
- “我想丢弃工作区改动，但不确定其中有没有重要内容。”

能说清上述答案，并能用观察面板证明每条命令改变了哪一层，就完成本节。

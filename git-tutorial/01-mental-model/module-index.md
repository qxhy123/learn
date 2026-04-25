# 01 心智模型模块导览

## 模块目标

本模块帮助你把 Git 看成一组可观察的状态层：工作区、暂存区、提交历史、引用和远程跟踪分支。学完后，你应该能读懂 `status`、`diff`、`log` 的输出，并知道下一步该改哪一层。

## 学习路径

1. [看见仓库状态](./01-see-the-repo-state.md)
2. [工作区、暂存区与提交](./02-working-tree-index-commit.md)
3. [有信心地阅读历史](./03-read-history-with-confidence.md)

## 本模块 lab id

- `LAB-MODEL-STATE-01`：制造 untracked、modified、staged 三种状态。
- `LAB-MODEL-INDEX-01`：同一文件同时存在 staged 与 unstaged 修改。
- `LAB-MODEL-HISTORY-01`：用三次提交读懂线性历史与差异。

## 术语约定

- **工作区**：你正在编辑的真实文件。
- **暂存区**：下一次提交的候选快照。
- **提交**：已经记录下来的项目快照和元数据。
- **引用**：指向提交的名字，例如分支、tag、`HEAD`。

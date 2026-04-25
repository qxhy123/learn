# 02 日常工作流模块导览

## 模块目标

本模块把心智模型落到每天都会发生的闭环：修改文件、观察状态、设计提交、review diff、提交、保持仓库卫生。

## 学习路径

1. [第一次干净提交](./04-first-change-to-clean-commit.md)
2. [提交设计与 diff review](./05-commit-design-and-diff-review.md)
3. [忽略文件与仓库卫生](./06-ignore-files-and-repo-hygiene.md)

## 本模块 lab id

- `LAB-DAILY-CLEAN-COMMIT-01`：完成一次从修改到干净提交的闭环。
- `LAB-DAILY-DIFF-REVIEW-01`：把混在一起的修改拆成两个提交。
- `LAB-DAILY-IGNORE-01`：配置 `.gitignore` 并解释 tracked 文件不受影响。

## 日常检查清单

提交前固定检查：

```bash
git status --short --branch
git diff
git diff --cached
git log --oneline --max-count=3
```

确认：

- 本次提交只包含一个意图。
- 暂存区 diff 已经读过。
- 提交信息说明“为什么”，不是重复文件名。
- 没有把生成物、日志、秘密或本地配置放进提交。

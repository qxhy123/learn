# 02 日常工作流模块导览

## 模块目标

本模块把心智模型落到每天都会发生的闭环：修改文件、观察状态、设计提交、review diff、提交、保持仓库卫生。
重点不是“尽快 commit”，而是让每个提交都可解释、可 review、可回滚。

## 学习路径

| 顺序 | 章节 | 你要学会的判断 | Lab id |
| --- | --- | --- | --- |
| 04 | [第一次干净提交](./04-first-change-to-clean-commit.md) | 从修改到暂存、提交、验证，完成一次小步闭环。 | `LAB-DAILY-CLEAN-COMMIT-01` |
| 05 | [提交设计与 diff review](./05-commit-design-and-diff-review.md) | 拆分混杂修改，确认提交只表达一个意图。 | `LAB-DAILY-DIFF-REVIEW-01` |
| 06 | [忽略文件与仓库卫生](./06-ignore-files-and-repo-hygiene.md) | 判断哪些文件应入库、应忽略、应从跟踪集合移除。 | `LAB-DAILY-IGNORE-01` |

## 提交前观察面板

```bash
git status -sb
git diff
git diff --cached
git log --oneline --max-count=3
```

提交前逐项确认：

1. 工作区没有混入与本任务无关的文件。
2. 暂存区 diff 已读过，且能用一句话解释“为什么要提交”。
3. 生成物、日志、秘密、本机配置没有进入暂存区。
4. 如果使用 `.gitignore`，已确认目标文件是否已经被 Git 跟踪。

## 模块验收

- 能用 `git add -p` 或分路径暂存拆出小提交。
- 能解释 `commit --amend` 只适合未共享提交。
- 能用 `git check-ignore -v` 追踪忽略规则来源。
- 能在清理 untracked 文件前先运行 `git clean -nd` 预览。

## 相关附录

- 提交流程入口：见 [命令决策树](../appendix/command-decision-trees.md#我要把改动变成一个干净提交)。
- 危险操作：见 [危险区](../appendix/danger-zone.md#git-clean--fd)。
- 高频命令：见 [速查表](../appendix/cheatsheet.md#日常提交)。

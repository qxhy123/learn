# 01 心智模型模块导览

## 模块目标

本模块先建立 Git 的“可观察状态”语言：工作区、暂存区、提交历史、引用、远程跟踪分支。学习者完成本模块后，应能在执行命令前说清楚当前改的是哪一层、命令会移动哪一个引用、恢复入口在哪里。

## 学习路径

| 顺序 | 章节 | 你要学会的判断 | Lab id |
| --- | --- | --- | --- |
| 01 | [看见仓库状态](./01-see-the-repo-state.md) | 用 `status`、分支名、短状态判断当前仓库是否安全可操作。 | `LAB-MODEL-STATE-01` |
| 02 | [工作区、暂存区与提交](./02-working-tree-index-commit.md) | 区分 working tree、index、HEAD，并用 diff 验证下一次提交内容。 | `LAB-MODEL-INDEX-01` |
| 03 | [有信心地阅读历史](./03-read-history-with-confidence.md) | 用 `log`、`show`、`diff` 和范围语法解释提交之间的关系。 | `LAB-MODEL-HISTORY-01` |

## 本模块统一观察面板

```bash
git status -sb
git diff
git diff --cached
git log --oneline --graph --decorate --all --max-count=12
```

观察顺序固定为：

1. 是否在 Git 仓库中，当前工作区是否有未提交变化。
2. 哪些文件是 untracked、modified、staged。
3. `HEAD` 指向哪里，当前分支是否与上游有 ahead/behind。
4. 历史图是否能解释“当前提交从哪里来、下一步会到哪里去”。

## 模块验收

- 能解释 `git status -sb` 的两列短状态。
- 能说明 `git diff` 与 `git diff --cached` 分别比较哪两层。
- 能从一段 `git log --graph --decorate` 判断当前分支尖端、tag 和远程跟踪分支位置。
- 遇到 detached HEAD、untracked 文件或误 staged 文件时，先观察再选择恢复命令。

## 相关附录

- 术语：见 [术语表](../appendix/glossary.md#核心状态层)。
- 命令入口：见 [命令决策树](../appendix/command-decision-trees.md#我不知道当前仓库状态是否安全)。
- 速查：见 [速查表](../appendix/cheatsheet.md#状态观察)。

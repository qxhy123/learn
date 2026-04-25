# 05 撤销、回退与找回模块导览

## 模块目标

本模块把“Git 出错了怎么办”拆成三类任务：撤销未提交变化、修正错误提交、找回看似丢失的工作。学习重点不是背命令，而是在操作前判断当前状态、历史是否已共享，以及哪一种恢复路线最少破坏上下文。

## 学习路径

| 顺序 | 章节 | 你要学会的判断 | Lab id |
| --- | --- | --- | --- |
| 13 | [撤销本地改动](./13-undo-local-changes.md) | 从工作区、暂存区和本地提交三层判断该撤哪一层。 | `LAB-RECOVERY-UNDO-01` |
| 14 | [修正错误提交](./14-fix-a-bad-commit.md) | 区分 amend、revert、reset 与 cherry-pick 的协作边界。 | `LAB-RECOVERY-BAD-COMMIT-01` |
| 15 | [用 reflog 找回工作](./15-recover-lost-work-with-reflog.md) | 用 reflog 和救援分支找回 reset、游离 HEAD、误删分支后的提交。 | `LAB-RECOVERY-REFLOG-01` |

## 恢复前观察面板

```bash
git status -sb
git diff
git diff --cached
git log --oneline --graph --decorate --all --max-count=12
git reflog --date=relative --max-count=20
```

## 决策底线

- 未提交内容先保护，再清理；不确定时先复制补丁、提交 WIP 到临时分支，或创建救援分支。
- 已经推送、被 PR 引用或被他人基于其继续工作的历史，默认当作共享历史。
- 共享历史优先用 `git revert` 增量修正；本地未共享历史才考虑 `reset`、`commit --amend` 或交互式 rebase。
- 恢复时先创建救援引用，再移动正式分支。

## 模块验收

- 能选择 `restore`、`restore --staged`、`reset`、`revert` 的适用边界。
- 能解释为什么 `reset --hard` 前必须确认未提交内容是否已保护。
- 能从 reflog 中找出事故前位置并创建 `rescue/...` 分支。
- 能说明共享历史中优先使用反向提交，而不是默认强推改写。

## 相关附录

- 撤销入口：见 [命令决策树](../appendix/command-decision-trees.md#我有本地改动想撤销)。
- 找回入口：见 [命令决策树](../appendix/command-decision-trees.md#我找不到提交了)。
- 高风险恢复命令：见 [危险区](../appendix/danger-zone.md)。

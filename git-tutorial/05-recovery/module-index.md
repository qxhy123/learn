# 05 Recovery：撤销、回退与找回

本模块把“Git 出错了怎么办”拆成三类任务：撤销未提交变化、修正错误提交、找回看似丢失的工作。学习重点不是背命令，而是在操作前判断当前状态、历史是否已共享，以及哪一种恢复路线最少破坏上下文。

## 学习路径

1. [13 Undo Local Changes](./13-undo-local-changes.md)：从工作区、暂存区和本地提交三层判断该撤哪一层。
2. [14 Fix a Bad Commit](./14-fix-a-bad-commit.md)：区分 amend、revert、reset 与 cherry-pick 的协作边界。
3. [15 Recover Lost Work with Reflog](./15-recover-lost-work-with-reflog.md)：用 reflog 和救援分支找回 reset、游离 HEAD、误删分支后的提交。

## 本模块统一观察面板

每个恢复动作前先运行最小观察组：

```bash
git status -sb
git diff
git diff --cached
git log --oneline --graph --decorate --all -n 12
```

如果涉及误操作或历史移动，再补充：

```bash
git reflog --date=relative -n 20
```

## 决策底线

- 未提交内容先保护，再清理；不确定时先新建临时分支或复制补丁。
- 已经推送、被 PR 引用或被他人基于其继续工作的历史，默认当作共享历史。
- 共享历史优先用 `git revert` 增量修正；本地未共享历史才考虑 `reset`、`commit --amend` 或交互式 rebase。
- 恢复时先创建救援引用，再移动正式分支。

## Lab IDs

- `LAB-RECOVERY-UNDO-01`：比较 restore、reset、revert 对工作区、暂存区和提交历史的影响。
- `LAB-RECOVERY-BAD-COMMIT-01`：修正提交说明、漏提交文件、错误提交已共享三类事故。
- `LAB-RECOVERY-REFLOG-01`：从误 reset、游离 HEAD 提交和误删分支中找回工作。

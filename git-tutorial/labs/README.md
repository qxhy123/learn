# Git Tutorial Labs

这些 labs 是新教程的可重复练习环境。每个场景都用稳定 lab id 标识，章节正文只引用 id；本目录负责落地准备、执行、观察、恢复与清理步骤。

## 快速开始

```bash
cd git-tutorial/labs
./bin/git-lab.sh --list
./bin/git-lab.sh LAB-MODEL-STATE-01 --force
```

工作区会生成在 `git-tutorial/labs/workspaces/`，该目录被 `.gitignore` 排除。

## 场景清单

| Lab id | 对应模块 | 目标 | 场景文件 |
|---|---|---|---|
| `LAB-ORIENT-STATUS-01` | 00 Orientation | 初始化练习仓库并运行状态观察面板 | [scenarios/LAB-ORIENT-STATUS-01.md](scenarios/LAB-ORIENT-STATUS-01.md) |
| `LAB-MODEL-STATE-01` | 01 Mental Model | 制造 untracked、modified、staged 三种状态 | [scenarios/LAB-MODEL-STATE-01.md](scenarios/LAB-MODEL-STATE-01.md) |
| `LAB-MODEL-INDEX-01` | 01 Mental Model | 观察同一文件的 staged 与 unstaged 差异 | [scenarios/LAB-MODEL-INDEX-01.md](scenarios/LAB-MODEL-INDEX-01.md) |
| `LAB-MODEL-HISTORY-01` | 01 Mental Model | 用线性历史、分支和标签读懂提交图 | [scenarios/LAB-MODEL-HISTORY-01.md](scenarios/LAB-MODEL-HISTORY-01.md) |
| `LAB-DAILY-CLEAN-COMMIT-01` | 02 Daily Workflow | 从修改到干净提交的闭环 | [scenarios/LAB-DAILY-CLEAN-COMMIT-01.md](scenarios/LAB-DAILY-CLEAN-COMMIT-01.md) |
| `LAB-DAILY-DIFF-REVIEW-01` | 02 Daily Workflow | 把混在一起的修改拆成可 review 的提交 | [scenarios/LAB-DAILY-DIFF-REVIEW-01.md](scenarios/LAB-DAILY-DIFF-REVIEW-01.md) |
| `LAB-DAILY-IGNORE-01` | 02 Daily Workflow | 配置 .gitignore 并解释 tracked 文件边界 | [scenarios/LAB-DAILY-IGNORE-01.md](scenarios/LAB-DAILY-IGNORE-01.md) |
| `LAB-BRANCH-TASK-01` | 03 Branching Work | 从干净主干创建任务分支并完成小步提交 | [scenarios/LAB-BRANCH-TASK-01.md](scenarios/LAB-BRANCH-TASK-01.md) |
| `LAB-BRANCH-CONFLICT-01` | 03 Branching Work | 故意制造同一行冲突并按 playbook 解决 | [scenarios/LAB-BRANCH-CONFLICT-01.md](scenarios/LAB-BRANCH-CONFLICT-01.md) |
| `LAB-BRANCH-REBASE-01` | 03 Branching Work | 对本地未共享分支 rebase 并观察历史图变化 | [scenarios/LAB-BRANCH-REBASE-01.md](scenarios/LAB-BRANCH-REBASE-01.md) |
| `LAB-COLLAB-REMOTE-01` | 04 Collaboration | clone、fetch、远程跟踪分支观察 | [scenarios/LAB-COLLAB-REMOTE-01.md](scenarios/LAB-COLLAB-REMOTE-01.md) |
| `LAB-COLLAB-PUSH-REJECTED-01` | 04 Collaboration | 模拟双人提交导致的 non-fast-forward push rejected | [scenarios/LAB-COLLAB-PUSH-REJECTED-01.md](scenarios/LAB-COLLAB-PUSH-REJECTED-01.md) |
| `LAB-COLLAB-PR-01` | 04 Collaboration | 准备可 review 的任务分支和 PR 自检清单 | [scenarios/LAB-COLLAB-PR-01.md](scenarios/LAB-COLLAB-PR-01.md) |
| `LAB-RECOVERY-UNDO-01` | 05 Recovery | 比较 restore、reset、revert 对状态和历史的影响 | [scenarios/LAB-RECOVERY-UNDO-01.md](scenarios/LAB-RECOVERY-UNDO-01.md) |
| `LAB-RECOVERY-BAD-COMMIT-01` | 05 Recovery | 修正坏提交、漏提交文件和已共享错误 | [scenarios/LAB-RECOVERY-BAD-COMMIT-01.md](scenarios/LAB-RECOVERY-BAD-COMMIT-01.md) |
| `LAB-RECOVERY-REFLOG-01` | 05 Recovery | 从误 reset、游离 HEAD 和误删分支中找回工作 | [scenarios/LAB-RECOVERY-REFLOG-01.md](scenarios/LAB-RECOVERY-REFLOG-01.md) |
| `LAB-RELEASE-STASH-WORKTREE-01` | 06 Release and Debugging | 处理中断、stash 与 worktree 并行修复 | [scenarios/LAB-RELEASE-STASH-WORKTREE-01.md](scenarios/LAB-RELEASE-STASH-WORKTREE-01.md) |
| `LAB-RELEASE-HOTFIX-TAG-01` | 06 Release and Debugging | 从发布标签切出 hotfix 并打 patch 标签 | [scenarios/LAB-RELEASE-HOTFIX-TAG-01.md](scenarios/LAB-RELEASE-HOTFIX-TAG-01.md) |
| `LAB-DEBUG-BISECT-01` | 06 Release and Debugging | 构造回归并用 bisect 定位首个坏提交 | [scenarios/LAB-DEBUG-BISECT-01.md](scenarios/LAB-DEBUG-BISECT-01.md) |
| `LAB-GOV-HOOKS-01` | 07 Scale and Governance | 本地 hooks、别名和配置层级 | [scenarios/LAB-GOV-HOOKS-01.md](scenarios/LAB-GOV-HOOKS-01.md) |
| `LAB-GOV-LARGE-REPO-01` | 07 Scale and Governance | 大文件、忽略规则和 LFS 决策 | [scenarios/LAB-GOV-LARGE-REPO-01.md](scenarios/LAB-GOV-LARGE-REPO-01.md) |
| `LAB-GOV-DISASTER-01` | 07 Scale and Governance | 分支策略和灾难恢复卡片 | [scenarios/LAB-GOV-DISASTER-01.md](scenarios/LAB-GOV-DISASTER-01.md) |

## 默认观察面板

进入任意生成仓库后先执行：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=12
```

按主题追加：

```bash
git diff
git diff --cached
git ls-files --stage
git show-ref --heads --tags
git config --show-origin --list | sed -n '1,40p'
```

## 安全与清理

- 脚本只创建或删除 `labs/workspaces/<lab-id>/` 下的目录。
- `--force` 只重建指定 lab 的工作区。
- `--smoke --force` 会在 `labs/workspaces/_smoke/` 下生成全部 lab，执行最小 Git 观察命令，然后清理 `_smoke`。
- 每个生成仓库会设置局部 `user.name` 和 `user.email`，不会污染全局 Git 配置。
- 清理单个 lab：`rm -rf workspaces/<lab-id>`。
- 清理全部 lab：`rm -rf workspaces`。

## 脚本覆盖范围

`labs/bin/git-lab.sh` 覆盖当前章节和模块索引中引用的全部 lab id。每个场景文件包含准备、执行、观察、恢复、清理和预期结果；复杂协作场景会生成本地 bare remote 与多个克隆，便于无网络演练。

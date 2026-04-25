# Git Tutorial Labs

这些 labs 是新教程的可重复练习环境。每个场景都用稳定 lab id 标识，章节正文只引用 id；本目录负责落地准备、执行、观察、恢复与清理步骤。

## 快速开始

```bash
cd git-tutorial/labs
./bin/git-lab.sh --list
./bin/git-lab.sh LAB-GOV-HOOKS-01 --force
```

工作区会生成在 `git-tutorial/labs/workspaces/`，该目录被 `.gitignore` 排除。

## 场景清单

| Lab id | 对应模块 | 目标 | 场景文件 |
|---|---|---|---|
| `LAB-BRANCH-CONFLICT-01` | 03 | branch conflict 01 | [scenarios/LAB-BRANCH-CONFLICT-01.md](scenarios/LAB-BRANCH-CONFLICT-01.md) |
| `LAB-BRANCH-REBASE-01` | 03 | branch rebase 01 | [scenarios/LAB-BRANCH-REBASE-01.md](scenarios/LAB-BRANCH-REBASE-01.md) |
| `LAB-BRANCH-TASK-01` | 03 | branch task 01 | [scenarios/LAB-BRANCH-TASK-01.md](scenarios/LAB-BRANCH-TASK-01.md) |
| `LAB-COLLAB-PR-01` | 04 | colpr 01 | [scenarios/LAB-COLLAB-PR-01.md](scenarios/LAB-COLLAB-PR-01.md) |
| `LAB-COLLAB-PUSH-REJECTED-01` | 04 | colpush rejected 01 | [scenarios/LAB-COLLAB-PUSH-REJECTED-01.md](scenarios/LAB-COLLAB-PUSH-REJECTED-01.md) |
| `LAB-COLLAB-REMOTE-01` | 04 | colremote 01 | [scenarios/LAB-COLLAB-REMOTE-01.md](scenarios/LAB-COLLAB-REMOTE-01.md) |
| `LAB-DAILY-CLEAN-COMMIT-01` | 02 | daily clean commit 01 | [scenarios/LAB-DAILY-CLEAN-COMMIT-01.md](scenarios/LAB-DAILY-CLEAN-COMMIT-01.md) |
| `LAB-DAILY-COMMIT-01` | 02 | daily commit 01 | [scenarios/LAB-DAILY-COMMIT-01.md](scenarios/LAB-DAILY-COMMIT-01.md) |
| `LAB-DAILY-DIFF-REVIEW-01` | 02 | daily diff review 01 | [scenarios/LAB-DAILY-DIFF-REVIEW-01.md](scenarios/LAB-DAILY-DIFF-REVIEW-01.md) |
| `LAB-DAILY-IGNORE-01` | 02 | daily ignore 01 | [scenarios/LAB-DAILY-IGNORE-01.md](scenarios/LAB-DAILY-IGNORE-01.md) |
| `LAB-DEBUG-BISECT-01` | 06 | debug bisect 01 | [scenarios/LAB-DEBUG-BISECT-01.md](scenarios/LAB-DEBUG-BISECT-01.md) |
| `LAB-GOV-DISASTER-01` | 07 | gov disaster 01 | [scenarios/LAB-GOV-DISASTER-01.md](scenarios/LAB-GOV-DISASTER-01.md) |
| `LAB-GOV-HOOKS-01` | 07 | gov hooks 01 | [scenarios/LAB-GOV-HOOKS-01.md](scenarios/LAB-GOV-HOOKS-01.md) |
| `LAB-GOV-LARGE-REPO-01` | 07 | gov large repo 01 | [scenarios/LAB-GOV-LARGE-REPO-01.md](scenarios/LAB-GOV-LARGE-REPO-01.md) |
| `LAB-MODEL-HISTORY-01` | 01 | model history 01 | [scenarios/LAB-MODEL-HISTORY-01.md](scenarios/LAB-MODEL-HISTORY-01.md) |
| `LAB-MODEL-INDEX-01` | 01 | model index 01 | [scenarios/LAB-MODEL-INDEX-01.md](scenarios/LAB-MODEL-INDEX-01.md) |
| `LAB-MODEL-STATE-01` | 01 | model state 01 | [scenarios/LAB-MODEL-STATE-01.md](scenarios/LAB-MODEL-STATE-01.md) |
| `LAB-ORIENT-STATUS-01` | 00/01 | orient status 01 | [scenarios/LAB-ORIENT-STATUS-01.md](scenarios/LAB-ORIENT-STATUS-01.md) |
| `LAB-RECOVERY-BAD-COMMIT-01` | 05 | recovery bad commit 01 | [scenarios/LAB-RECOVERY-BAD-COMMIT-01.md](scenarios/LAB-RECOVERY-BAD-COMMIT-01.md) |
| `LAB-RECOVERY-REFLOG-01` | 05 | recovery reflog 01 | [scenarios/LAB-RECOVERY-REFLOG-01.md](scenarios/LAB-RECOVERY-REFLOG-01.md) |
| `LAB-RECOVERY-RESET-01` | 05 | recovery reset 01 | [scenarios/LAB-RECOVERY-RESET-01.md](scenarios/LAB-RECOVERY-RESET-01.md) |
| `LAB-RECOVERY-UNDO-01` | 05 | recovery undo 01 | [scenarios/LAB-RECOVERY-UNDO-01.md](scenarios/LAB-RECOVERY-UNDO-01.md) |
| `LAB-RELEASE-BISECT-01` | 06 | release bisect 01 | [scenarios/LAB-RELEASE-BISECT-01.md](scenarios/LAB-RELEASE-BISECT-01.md) |
| `LAB-RELEASE-HOTFIX-TAG-01` | 06 | release hotfix tag 01 | [scenarios/LAB-RELEASE-HOTFIX-TAG-01.md](scenarios/LAB-RELEASE-HOTFIX-TAG-01.md) |
| `LAB-RELEASE-STASH-WORKTREE-01` | 06 | release stash worktree 01 | [scenarios/LAB-RELEASE-STASH-WORKTREE-01.md](scenarios/LAB-RELEASE-STASH-WORKTREE-01.md) |
| `LAB-SETUP-STATE-01` | 00/01 | setup state 01 | [scenarios/LAB-SETUP-STATE-01.md](scenarios/LAB-SETUP-STATE-01.md) |

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
- 每个生成仓库会设置局部 `user.name` 和 `user.email`，不会污染全局 Git 配置。
- 清理单个 lab：`rm -rf workspaces/<lab-id>`。
- 清理全部 lab：`rm -rf workspaces`。

## 脚本覆盖范围

`labs/bin/git-lab.sh` 负责生成最小可练习仓库。不是每个场景都需要复杂脚本；治理类 lab 还会生成 Markdown 决策卡，要求学习者填写策略和恢复步骤。

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
| `LAB-SETUP-STATE-01` | 00/01 | 初始化仓库并观察 working tree/index/HEAD | [scenarios/LAB-SETUP-STATE-01.md](scenarios/LAB-SETUP-STATE-01.md) |
| `LAB-DAILY-COMMIT-01` | 02 | 拆分提交、审查 diff、保持干净工作区 | [scenarios/LAB-DAILY-COMMIT-01.md](scenarios/LAB-DAILY-COMMIT-01.md) |
| `LAB-BRANCH-CONFLICT-01` | 03 | 创建任务分支并解决一次冲突 | [scenarios/LAB-BRANCH-CONFLICT-01.md](scenarios/LAB-BRANCH-CONFLICT-01.md) |
| `LAB-COLLAB-PUSH-REJECTED-01` | 04 | 模拟双人协作和 push rejected | [scenarios/LAB-COLLAB-PUSH-REJECTED-01.md](scenarios/LAB-COLLAB-PUSH-REJECTED-01.md) |
| `LAB-RECOVERY-RESET-01` | 05 | 练习 restore/reset/revert/reflog 选择边界 | [scenarios/LAB-RECOVERY-RESET-01.md](scenarios/LAB-RECOVERY-RESET-01.md) |
| `LAB-RELEASE-BISECT-01` | 06 | 标签、hotfix、bisect 定位坏提交 | [scenarios/LAB-RELEASE-BISECT-01.md](scenarios/LAB-RELEASE-BISECT-01.md) |
| `LAB-GOV-HOOKS-01` | 07 | 本地 hooks、别名、配置层级 | [scenarios/LAB-GOV-HOOKS-01.md](scenarios/LAB-GOV-HOOKS-01.md) |
| `LAB-GOV-LARGE-REPO-01` | 07 | 大文件、忽略规则、LFS 决策 | [scenarios/LAB-GOV-LARGE-REPO-01.md](scenarios/LAB-GOV-LARGE-REPO-01.md) |
| `LAB-GOV-DISASTER-01` | 07 | 分支策略和灾难恢复卡片 | [scenarios/LAB-GOV-DISASTER-01.md](scenarios/LAB-GOV-DISASTER-01.md) |

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

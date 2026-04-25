# 07 Scale and Governance：规模、规则与事故预案

这个模块把 Git 从“个人熟练度”推进到“团队系统”。你将不再只问“这个命令怎么写”，而是问：哪些规则应该自动化？哪些仓库形态会放大协作成本？事故发生时谁能做什么、先观察什么、如何恢复？

## 学习路径

1. [19 Hooks、配置与别名](19-hooks-config-and-aliases.md)：把高频观察和低风险规则自动化。
2. [20 Monorepo、LFS 与大仓库](20-monorepo-lfs-and-large-repos.md)：识别大仓库成本，并选择合适的治理手段。
3. [21 分支策略与灾难手册](21-branching-policy-and-disaster-playbook.md)：为团队制定最低协作规则和事故分流卡片。

## 模块统一观察面板

在本模块所有实验前，先跑：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=12
git config --show-origin --list | sed -n '1,40p'
```

如果涉及标签、引用或大文件，再补充：

```bash
git show-ref --heads --tags
git count-objects -vH
git lfs env 2>/dev/null || true
```

## 本模块 lab id

- `LAB-GOV-HOOKS-01`：本地 hooks、别名和配置层级。
- `LAB-GOV-LARGE-REPO-01`：大文件、忽略规则、LFS 决策演练。
- `LAB-GOV-DISASTER-01`：分支策略和灾难恢复卡片。

> 安全约定：本模块会讨论 hooks、`git config`、`git filter-repo`/历史清理、`--force-with-lease` 等高影响操作。教程中的危险命令只在 lab 临时仓库或只读设计题里出现；对真实团队仓库执行前必须先完成观察、备份和审批。

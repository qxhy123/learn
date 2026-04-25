# 07 规模与治理模块导览

## 模块目标

本模块把 Git 从“个人熟练度”推进到“团队系统”。你将不再只问“这个命令怎么写”，而是问：哪些规则应该自动化？哪些仓库形态会放大协作成本？事故发生时谁能做什么、先观察什么、如何恢复？

## 学习路径

| 顺序 | 章节 | 你要学会的判断 | Lab id |
|---|---|---|---|
| 19 | [Hooks、配置与别名](./19-hooks-config-and-aliases.md) | 把高频观察和低风险规则自动化，同时保留可解释性。 | `LAB-GOV-HOOKS-01` |
| 20 | [Monorepo、LFS 与大仓库](./20-monorepo-lfs-and-large-repos.md) | 识别大仓库成本，并选择合适的治理手段。 | `LAB-GOV-LARGE-REPO-01` |
| 21 | [分支策略与灾难手册](./21-branching-policy-and-disaster-playbook.md) | 为团队制定最低协作规则和事故分流卡片。 | `LAB-GOV-DISASTER-01` |

## 治理观察面板

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=12
git config --show-origin --list | sed -n '1,40p'
```

涉及标签、引用或大文件时追加：

```bash
git show-ref --heads --tags
git count-objects -vH
git lfs env 2>/dev/null || true
```

## 决策底线

- 自动化规则先从只读提示开始，确认稳定后再阻断提交或推送。
- 大仓库治理先减少未来伤害，再规划历史清理；不要把历史重写当作第一反应。
- 分支策略必须能被新人执行、能被 CI 验证、能在事故中快速降级。
- 任何影响共享历史、远程引用或发布标签的动作都需要备份、窗口、负责人和回滚路径。

## 模块验收

- 能说明配置层级、hook 触发点、alias 的收益与风险。
- 能区分 monorepo、多仓库、LFS、制品库、稀疏检出、部分克隆的适用边界。
- 能写出团队最小分支策略：主干保护、PR 要求、同步方式、强推规则、事故入口。
- 能为误强推、标签打错、大文件进历史这类事故设计先观察再恢复的手册。

## 相关附录

- 团队事故入口：见 [命令决策树](../appendix/command-decision-trees.md#团队事故入口)。
- 历史清理风险：见 [危险区](../appendix/danger-zone.md#历史清理工具)。
- 治理命令：见 [速查表](../appendix/cheatsheet.md#规模与治理)。

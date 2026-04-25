# Git 任务驱动实战教程

这套教程把 Git 当作一种“可观察、可恢复、可协作”的工程工具来学习。你不会只背命令，而是会先判断仓库状态，再选择动作，并能解释动作会改变工作区、暂存区、提交历史、分支引用还是远程跟踪状态。

## 适合谁

- 会敲 `add/commit/pull/push`，但遇到冲突、回退、同步就心慌的开发者。
- Git 零基础，但希望从一开始就建立正确心智模型的学习者。
- 需要在 Pull Request、代码评审、发布、事故恢复中安全协作的团队成员。

## 学习方式

每个章节都按同一节奏组织：

1. **场景**：先说明真实工作问题，而不是先列命令。
2. **学习目标**：明确学完后应该能判断什么。
3. **观察点**：每次操作前后都观察 `git status`、`git diff`、`git log`、分支和远程状态。
4. **命令与解释**：命令只作为改变状态的手段，重点说明它改了哪一层。
5. **实验**：用稳定 lab id 引用可复现实验。
6. **常见错误与恢复路径**：危险动作必须说明风险和补救办法。
7. **验收**：用问题或操作证明你真的理解了。

推荐先读 [00-orientation.md](./00-orientation.md)，再从“零基础到干净提交”路径开始。

## 状态观察面板

后续每章都默认使用这组观察命令：

```bash
git status --short --branch
git diff
git diff --cached
git log --oneline --graph --decorate --max-count=8
```

观察时固定回答四个问题：

- 工作区是否有未暂存修改？
- 暂存区是否已经准备好下一次提交？
- `HEAD` 指向哪个提交或分支？
- 本地分支和远程跟踪分支是否分叉？

## 共享写作契约

为保证多模块一致，所有章节遵守以下约定：

| 项目 | 约定 |
|---|---|
| 术语 | 使用“工作区、暂存区、提交、引用、远程跟踪分支、reflog”。第一次出现英文术语时可附英文。 |
| lab id | 使用 `LAB-<MODULE>-<SCENARIO>-<NN>`，例如 `LAB-MODEL-STATE-01`。章节只引用 id，实验目录最终统一落地。 |
| 危险提示 | 涉及 `reset --hard`、`clean -fd`、强推、rebase 已共享历史等动作时，必须写“风险”和“恢复路径”。 |
| 命令块 | 命令前说明目的；命令后说明预期观察结果。 |
| 链接 | 模块内部使用相对链接；跨模块链接由最终集成阶段统一修正。 |

## 课程地图

### 起点

- [课程导向：怎样用状态观察学习 Git](./00-orientation.md)：建立“先观察、再操作、能恢复”的学习节奏，并完成 `LAB-ORIENT-STATUS-01`。

### 01 心智模型：先看懂仓库状态

| 文件 | 你将学会 |
|---|---|
| [模块导览](./01-mental-model/module-index.md) | 本模块学习路径、观察习惯和 lab id。 |
| [看见仓库状态](./01-mental-model/01-see-the-repo-state.md) | 区分普通目录、Git 仓库、工作区状态和引用状态。 |
| [工作区、暂存区与提交](./01-mental-model/02-working-tree-index-commit.md) | 理解 `add`、`commit`、`restore --staged` 改变哪一层。 |
| [有信心地阅读历史](./01-mental-model/03-read-history-with-confidence.md) | 用 `log`、`show`、`diff` 读懂线性、分叉和汇合历史。 |

对应实验：`LAB-MODEL-STATE-01`、`LAB-MODEL-INDEX-01`、`LAB-MODEL-HISTORY-01`。

### 02 日常工作流：从修改到可 review 提交

| 文件 | 你将学会 |
|---|---|
| [模块导览](./02-daily-workflow/module-index.md) | 日常提交闭环、提交拆分和仓库卫生。 |
| [第一次干净提交](./02-daily-workflow/04-first-change-to-clean-commit.md) | 从修改、观察、暂存到提交的完整安全流程。 |
| [提交设计与 diff review](./02-daily-workflow/05-commit-design-and-diff-review.md) | 让提交变小、可读、可回滚、可审查。 |
| [忽略文件与仓库卫生](./02-daily-workflow/06-ignore-files-and-repo-hygiene.md) | 正确使用 `.gitignore`，避免把生成物和秘密带进历史。 |

对应实验：`LAB-DAILY-CLEAN-COMMIT-01`、`LAB-DAILY-DIFF-REVIEW-01`、`LAB-DAILY-IGNORE-01`。

### 03 分支工作：任务分支、冲突与 rebase

| 文件 | 你将学会 |
|---|---|
| [模块导览](./03-branching-work/module-index.md) | 分支、冲突和 rebase 前后的观察面板。 |
| [为一个任务创建分支](./03-branching-work/07-branch-for-a-task.md) | 从干净主干创建短生命周期任务分支，并小步提交。 |
| [用 playbook 解决合并冲突](./03-branching-work/08-merge-conflicts-with-a-playbook.md) | 把冲突当成需要人工判断的历史汇合点。 |
| [不恐惧地使用 rebase](./03-branching-work/09-rebase-without-fear.md) | 只在未共享边界内重放提交，并知道何时中止。 |

对应实验：`LAB-BRANCH-TASK-01`、`LAB-BRANCH-CONFLICT-01`、`LAB-BRANCH-REBASE-01`。

### 04 远程协作：同步、推送与 Pull Request

| 文件 | 你将学会 |
|---|---|
| [模块导览](./04-collaboration/module-index.md) | 本地分支、远程跟踪分支和 PR 之间的同步关系。 |
| [clone、fetch、pull 与 push](./04-collaboration/10-clone-fetch-pull-push.md) | 区分 fetch 更新远程跟踪分支与 pull/push 改动当前协作状态。 |
| [同步主干并打开 PR](./04-collaboration/11-sync-with-main-and-open-pr.md) | 处理 ahead、behind、diverged 和 push rejected。 |
| [Review 与团队约定](./04-collaboration/12-review-and-team-conventions.md) | 用 PR 自检和团队规则保护共享历史。 |

对应实验：`LAB-COLLAB-REMOTE-01`、`LAB-COLLAB-PUSH-REJECTED-01`、`LAB-COLLAB-PR-01`。

### 05 恢复：撤销、回退与找回

| 文件 | 你将学会 |
|---|---|
| [模块导览](./05-recovery/module-index.md) | 从工作区、暂存区、提交历史和共享边界选择恢复路线。 |
| [Undo Local Changes](./05-recovery/13-undo-local-changes.md) | 撤销未提交改动、取消暂存，并避免误删 untracked 文件。 |
| [Fix a Bad Commit](./05-recovery/14-fix-a-bad-commit.md) | 区分 amend、reset、revert、cherry-pick 的协作边界。 |
| [Recover Lost Work with Reflog](./05-recovery/15-recover-lost-work-with-reflog.md) | 用 reflog 和救援分支找回误移动引用后的工作。 |

对应实验：`LAB-RECOVERY-UNDO-01`、`LAB-RECOVERY-BAD-COMMIT-01`、`LAB-RECOVERY-REFLOG-01`。

### 06 发布与排障：中断、标签与历史定位

| 文件 | 你将学会 |
|---|---|
| [模块导览](./06-release-and-debugging/module-index.md) | 用 Git 支撑中断处理、发布表达和历史排障。 |
| [Stash, Worktree and Interruptions](./06-release-and-debugging/16-stash-worktree-and-interruptions.md) | 在任务切换时选择提交、stash、分支或 worktree。 |
| [Tags, Releases and Hotfixes](./06-release-and-debugging/17-tags-releases-and-hotfixes.md) | 用标签和维护分支表达发布点与热修流程。 |
| [Blame、Bisect 与历史排障](./06-release-and-debugging/18-blame-bisect-and-history-debugging.md) | 从症状回溯到提交，用 bisect 缩小回归范围。 |

对应实验：`LAB-RELEASE-STASH-WORKTREE-01`、`LAB-RELEASE-HOTFIX-TAG-01`、`LAB-DEBUG-BISECT-01`。

### 07 规模与治理：规则自动化、大仓库与灾难预案

| 文件 | 你将学会 |
|---|---|
| [模块导览](./07-scale-and-governance/module-index.md) | 把个人 Git 熟练度扩展成团队系统。 |
| [Hooks、配置与别名](./07-scale-and-governance/19-hooks-config-and-aliases.md) | 用 hooks、别名和配置层级自动化低风险规则。 |
| [Monorepo、LFS 与大仓库](./07-scale-and-governance/20-monorepo-lfs-and-large-repos.md) | 识别大仓库成本，并选择忽略、拆分或 LFS 等治理手段。 |
| [分支策略与灾难手册](./07-scale-and-governance/21-branching-policy-and-disaster-playbook.md) | 为团队写出最低分支策略和事故分流卡片。 |

对应实验：`LAB-GOV-HOOKS-01`、`LAB-GOV-LARGE-REPO-01`、`LAB-GOV-DISASTER-01`。

### Labs 与附录

| 文件 | 用途 |
|---|---|
| [实验总览](./labs/README.md) | 按 lab id 查找准备、执行、观察、恢复和清理步骤。 |
| [实验脚本](./labs/bin/git-lab.sh) | 在临时目录创建可重复实验仓库。 |
| [速查表](./appendix/cheatsheet.md) | 按状态观察、提交、分支、协作、恢复、发布和治理快速查命令。 |
| [命令决策树](./appendix/command-decision-trees.md) | 从“我要撤销/同步/找回”等问题反推安全命令。 |
| [危险区](./appendix/danger-zone.md) | 查看高风险命令的影响、边界和恢复入口。 |
| [术语表](./appendix/glossary.md) | 统一 working tree、index、ref、reflog、worktree、LFS 等术语。 |

## 推荐学习路径

### 路径 A：零基础到第一次稳定提交

1. [00-orientation.md](./00-orientation.md)
2. [01-see-the-repo-state.md](./01-mental-model/01-see-the-repo-state.md)
3. [02-working-tree-index-commit.md](./01-mental-model/02-working-tree-index-commit.md)
4. [04-first-change-to-clean-commit.md](./02-daily-workflow/04-first-change-to-clean-commit.md)
5. [06-ignore-files-and-repo-hygiene.md](./02-daily-workflow/06-ignore-files-and-repo-hygiene.md)

完成后你应该能解释：为什么文件改了不等于会进入提交，为什么提交前要看 diff，以及如何避免把无关文件提交进去。

### 路径 B：已有经验但想不再心慌

1. [状态观察面板](./00-orientation.md#状态观察面板)
2. [工作区、暂存区与提交](./01-mental-model/02-working-tree-index-commit.md)
3. [有信心地阅读历史](./01-mental-model/03-read-history-with-confidence.md)
4. [提交设计与 diff review](./02-daily-workflow/05-commit-design-and-diff-review.md)
5. [用 playbook 解决合并冲突](./03-branching-work/08-merge-conflicts-with-a-playbook.md)
6. [Fix a Bad Commit](./05-recovery/14-fix-a-bad-commit.md)
7. [Recover Lost Work with Reflog](./05-recovery/15-recover-lost-work-with-reflog.md)

### 路径 C：团队协作准备

1. 先完成 [01 心智模型](./01-mental-model/module-index.md) 和 [02 日常工作流](./02-daily-workflow/module-index.md)，保证个人提交质量。
2. 进入 [03 分支工作](./03-branching-work/module-index.md) 与 [04 远程协作](./04-collaboration/module-index.md)，练习冲突、同步和 PR。
3. 补齐 [06 发布与排障](./06-release-and-debugging/module-index.md)，理解标签、hotfix 与 bisect。
4. 最后用 [07 规模与治理](./07-scale-and-governance/module-index.md) 制定团队最低协作规则。

### 路径 D：只想查命令或恢复事故

1. 先看 [命令决策树](./appendix/command-decision-trees.md)，按问题定位安全路线。
2. 涉及 `reset --hard`、`clean -fd`、rebase、强推、删除分支或标签时，先读 [危险区](./appendix/danger-zone.md)。
3. 不确定术语时查 [术语表](./appendix/glossary.md)，需要命令短表时查 [速查表](./appendix/cheatsheet.md)。

## 最小验收

读完 01 和 02 后，请在一个练习仓库中完成：

- 修改两个文件，只暂存其中一个文件的一部分。
- 用 `git diff` 和 `git diff --cached` 解释哪些内容会进入下一次提交。
- 写出一条能被 reviewer 理解的提交信息。
- 加入一条 `.gitignore` 规则，并解释为什么它不会自动移除已经被跟踪的文件。

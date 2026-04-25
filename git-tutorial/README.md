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

- [课程导向：怎样用状态观察学习 Git](./00-orientation.md)

### 01 心智模型：先看懂仓库状态

| 文件 | 你将学会 |
|---|---|
| [模块导览](./01-mental-model/module-index.md) | 本模块学习路径、观察习惯和 lab id。 |
| [看见仓库状态](./01-mental-model/01-see-the-repo-state.md) | 区分普通目录、Git 仓库、工作区状态和引用状态。 |
| [工作区、暂存区与提交](./01-mental-model/02-working-tree-index-commit.md) | 理解 `add`、`commit`、`restore --staged` 改变哪一层。 |
| [有信心地阅读历史](./01-mental-model/03-read-history-with-confidence.md) | 用 `log`、`show`、`diff` 读懂线性、分叉和汇合历史。 |

### 02 日常工作流：从修改到可 review 提交

| 文件 | 你将学会 |
|---|---|
| [模块导览](./02-daily-workflow/module-index.md) | 日常提交闭环、提交拆分和仓库卫生。 |
| [第一次干净提交](./02-daily-workflow/04-first-change-to-clean-commit.md) | 从修改、观察、暂存到提交的完整安全流程。 |
| [提交设计与 diff review](./02-daily-workflow/05-commit-design-and-diff-review.md) | 让提交变小、可读、可回滚、可审查。 |
| [忽略文件与仓库卫生](./02-daily-workflow/06-ignore-files-and-repo-hygiene.md) | 正确使用 `.gitignore`，避免把生成物和秘密带进历史。 |

### 后续模块（由其他并行工作包交付）

- `03-branching-work/`：任务分支、冲突处理、rebase 边界。
- `04-collaboration/`：clone/fetch/pull/push、PR、团队约定。
- `05-recovery/`：restore/reset/revert/reflog 的恢复决策。
- `06-release-and-debugging/`：stash、worktree、标签、hotfix、bisect。
- `07-scale-and-governance/`：hooks、config、LFS、大仓库、治理和灾难手册。
- `labs/`：统一实验清单、准备/执行/观察/恢复/清理步骤。
- `appendix/`：命令决策树、危险区、术语表、速查表。

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
5. 后续进入 `05-recovery/` 与 `04-collaboration/`。

### 路径 C：团队协作准备

1. 先完成 01 和 02，保证个人提交质量。
2. 再进入 `03-branching-work/` 与 `04-collaboration/`。
3. 最后用 `07-scale-and-governance/` 制定团队最低协作规则。

## 最小验收

读完 01 和 02 后，请在一个练习仓库中完成：

- 修改两个文件，只暂存其中一个文件的一部分。
- 用 `git diff` 和 `git diff --cached` 解释哪些内容会进入下一次提交。
- 写出一条能被 reviewer 理解的提交信息。
- 加入一条 `.gitignore` 规则，并解释为什么它不会自动移除已经被跟踪的文件。

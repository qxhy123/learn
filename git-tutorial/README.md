# 从零到高阶的 Git 教程

## 项目简介

Git 是现代软件开发中最常见、也最容易“会敲命令却不真懂”的工具之一。

很多人会背这些命令：

- `git add`
- `git commit`
- `git pull`
- `git push`

但一遇到下面这些场景就会明显不稳：

- 文件明明改了，为什么提交里没有？
- 为什么 `pull` 有时是快进，有时是 merge，有时又会冲突？
- 为什么 `reset --hard` 一下去，问题可能更大？
- 为什么误删分支、离开 detached HEAD 后，提交有时还能找回来？
- 为什么团队里对 `merge`、`rebase`、`squash merge` 会有完全不同的偏好？

这套教程的目标不是把命令表再堆一遍，而是帮你建立一套稳定的 Git 判断框架：

1. Git 记录的是**项目状态的演化**，不只是“文件备份”
2. Git 操作的对象分布在**工作区、暂存区、提交历史、引用、远程跟踪状态**几个层次
3. 真正决定你会不会出事的，不是“记不记得命令”，而是**看到状态后能不能判断该动哪一层**

如果你能持续回答这三个问题，Git 会从“偶尔能用、经常心慌”变成“可解释、可恢复、可协作”的工具。

---

## Git 基础认知地图

学习 Git 时，建议从一开始就把对象分层：

| 层次 | 你看到的是什么 | 常见命令 | 典型问题 |
|------|----------------|----------|----------|
| 文件层 | 工作区里的真实文件 | `status` `diff` `restore` | 文件改了为什么没进提交 |
| 暂存层 | 下一次提交的候选快照 | `add` `restore --staged` | 为什么同一文件能同时 staged 和 unstaged |
| 提交层 | 已记录的历史快照和提交图 | `commit` `log` `show` `revert` | 怎样看懂历史、怎样安全回滚 |
| 引用层 | 分支、HEAD、tag、remote-tracking refs | `switch` `branch` `reset` `rebase` `fetch` | 分支到底是什么、哪些动作会改历史 |

本教程前 6 章主要解决“看清前 3 层”，第 7 章以后逐步把“引用层”和“团队协作层”补齐。

---

## 目标受众

- 完全没有版本控制经验、希望系统学习 Git 的初学者
- 会写代码，但只掌握基础命令、缺少协作与排障能力的开发者
- 希望理解分支模型、历史整理、签名、CI、仓库治理的进阶工程师
- 想把“日常操作”与“Git 内部原理”连起来的学习者

---

## 章节导航目录

### 开始之前

- [前言：如何学习 Git，如何避免“命令背了还是不会用”](./00-preface.md)

### 第一部分：Git 基础心智模型

| 章节 | 标题 | 主要内容 | 与后续关系 |
|------|------|----------|------------|
| 第1章 | [为什么需要 Git 与版本控制](./part1-foundations/01-why-git-and-version-control.md) | Git 解决的问题、版本控制的价值、分布式模型 | 为后续“快照、分支、协作”建立总框架 |
| 第2章 | [安装、配置与第一个仓库](./part1-foundations/02-install-config-and-first-repo.md) | 安装 Git、配置身份、初始化仓库、第一次提交 | 为后续实验建立稳定练习环境 |
| 第3章 | [工作区、暂存区、HEAD 与提交](./part1-foundations/03-working-tree-index-head-and-commit.md) | Git 三层模型、快照、提交对象、HEAD 的定位 | 决定你后面是否真的能理解 `add/commit/reset/rebase` |

### 第二部分：本地提交与历史阅读

| 章节 | 标题 | 主要内容 | 与后续关系 |
|------|------|----------|------------|
| 第4章 | [status、add、commit 与 diff](./part2-local-history/04-status-add-commit-and-diff.md) | 日常提交流程、差异查看、分块暂存 | 形成稳定的本地操作习惯 |
| 第5章 | [log、show 与历史阅读](./part2-local-history/05-log-show-and-history-reading.md) | 阅读提交图、定位变更、比较版本 | 为分支、合并、排障打基础 |
| 第6章 | [`.gitignore` 与提交卫生](./part2-local-history/06-gitignore-and-commit-hygiene.md) | 忽略规则、提交粒度、提交信息规范 | 决定后续 review、回滚、blame 的质量 |

### 第三部分：分支、合并与历史整理

| 章节 | 标题 | 主要内容 | 与后续关系 |
|------|------|----------|------------|
| 第7章 | [分支、HEAD 与 switch](./part3-branching-and-merging/07-branches-head-and-switch.md) | 分支是什么、HEAD 如何移动、切换与创建分支 | 为 merge/rebase/remote 奠定“引用视角” |
| 第8章 | [merge 与冲突解决](./part3-branching-and-merging/08-merge-and-conflict-resolution.md) | 快进合并、三方合并、冲突识别与处理 | 为团队协作中的分叉与汇合建立直觉 |
| 第9章 | [rebase 与历史整形](./part3-branching-and-merging/09-rebase-and-history-shaping.md) | 线性历史、交互式 rebase、何时不用 rebase | 直接影响第 11-12 章中的同步与 PR 策略 |

### 第四部分：远程协作与团队工作流

| 章节 | 标题 | 主要内容 | 与后续关系 |
|------|------|----------|------------|
| 第10章 | [远程仓库、跟踪分支与 fetch](./part4-remotes-and-collaboration/10-remote-tracking-and-fetch.md) | `origin`、远程跟踪分支、同步远程状态 | 建立“远程状态不是自动最新”的意识 |
| 第11章 | [pull、push 与同步策略](./part4-remotes-and-collaboration/11-pull-push-and-sync-strategies.md) | 本地领先/远程领先、push 被拒绝、同步策略 | 决定协作时的默认安全动作 |
| 第12章 | [Pull Request、Code Review 与团队约定](./part4-remotes-and-collaboration/12-pull-request-code-review-and-team-conventions.md) | PR 工作流、审查习惯、团队约定 | 把个人操作提升到团队治理层面 |

### 第五部分：撤销、回退与恢复

| 章节 | 标题 | 主要内容 | 与后续关系 |
|------|------|----------|------------|
| 第13章 | [restore、reset 与 revert](./part5-undo-and-recovery/13-restore-reset-and-revert.md) | 不同撤销命令的边界、风险和适用场景 | 决定你出事时是否能先选对方向 |
| 第14章 | [amend、cherry-pick 与历史编辑](./part5-undo-and-recovery/14-amend-cherry-pick-and-history-editing.md) | 修改最近提交、提取补丁、局部搬运历史 | 连接历史整理与补丁搬运 |
| 第15章 | [reflog 与丢失提交恢复](./part5-undo-and-recovery/15-reflog-and-lost-commit-recovery.md) | 找回误删分支、恢复错误 reset、救回 detached HEAD | 是第 24 章事故处理的关键前置 |

### 第六部分：效率工具与发布实践

| 章节 | 标题 | 主要内容 | 与后续关系 |
|------|------|----------|------------|
| 第16章 | [stash、clean 与进行中的工作](./part6-productivity-and-release/16-stash-clean-and-work-in-progress.md) | 暂存未完成工作、清理工作区、快速切换任务 | 连接到 worktree、多任务并行和工作区管理 |
| 第17章 | [标签、发布与版本语义](./part6-productivity-and-release/17-tags-release-and-versioning.md) | 标签、版本发布、热修与回滚 | 连接到签名、CI、灾难恢复 |
| 第18章 | [blame、grep 与历史排障](./part6-productivity-and-release/18-blame-grep-and-history-debugging.md) | 追责不是目的，定位变更才是目的 | 为 bisect 和事故调查做准备 |

### 第七部分：Git 内部原理与自动化

| 章节 | 标题 | 主要内容 | 与后续关系 |
|------|------|----------|------------|
| 第19章 | [objects、refs 与 packfiles](./part7-internals-and-automation/19-objects-refs-and-packfiles.md) | blob/tree/commit/tag、引用、压缩存储 | 为后续理解 rebase、gc、tag、branch 提供底层解释 |
| 第20章 | [hooks、aliases 与配置层级](./part7-internals-and-automation/20-hooks-aliases-and-config-layering.md) | 本地自动化、别名、系统/全局/仓库配置 | 为团队约束和个人效率提供机制基础 |
| 第21章 | [worktree、submodule 与仓库组合](./part7-internals-and-automation/21-worktree-submodule-and-repo-composition.md) | 多工作树、子模块、仓库拆分与组合 | 连接到大仓库治理和协作边界 |

### 第八部分：大型仓库与组织级实践

| 章节 | 标题 | 主要内容 | 与后续关系 |
|------|------|----------|------------|
| 第22章 | [分支模型与团队治理](./part8-advanced-practice/22-branching-models-and-team-governance.md) | GitFlow、Trunk-Based、发布节奏、治理边界 | 从团队规范进入组织级取舍 |
| 第23章 | [Monorepo、LFS 与大仓库性能](./part8-advanced-practice/23-monorepo-lfs-and-large-repository-performance.md) | 大仓库常见问题、Git LFS、性能习惯 | 解释“Git 好不好用”常常取决于仓库治理 |
| 第24章 | [签名、CI 与灾难恢复手册](./part8-advanced-practice/24-signing-ci-and-disaster-playbook.md) | 提交签名、CI 集成、误操作与灾难演练 | 收束全书，形成高阶协作与事故处理闭环 |

### 附录

| 附录 | 标题 | 说明 |
|------|------|------|
| 附录A | [Git 速查表](./appendix/git-cheatsheet.md) | 按任务而不是按字母排序的命令速查 |
| 附录B | [命令安全矩阵](./appendix/command-safety-matrix.md) | 哪些命令可逆、哪些高风险、用前应确认什么 |
| 附录C | [术语表](./appendix/glossary.md) | 工作区、暂存区、HEAD、远程跟踪分支等术语 |
| 附录D | [练习答案索引](./appendix/answers.md) | 每章练习的关键答案、观察点和误区提醒 |

---

## 路径选择器

如果你不想从头到尾线性阅读，可以先按自己的目标选路径。

### 路径一：零基础闭环

适合谁：

- 第一次真正系统学 Git 的读者
- 以前主要靠 GUI 或复制目录“管理版本”的读者

先读哪些章：

1. 第 1-6 章
2. 第 13 章
3. 附录 A、B

完成后应具备的能力：

- 能解释工作区、暂存区、提交历史的区别
- 能独立完成一次本地开发和多次小提交
- 能在不慌的前提下撤销错误暂存或错误修改

### 路径二：团队协作上手

适合谁：

- 已经会基本提交，但一碰远程、分支和 PR 就不稳的开发者

先读哪些章：

1. 快速复习第 3-6 章
2. 重点读第 7-12 章
3. 结合第 13、15 章补恢复能力

完成后应具备的能力：

- 能安全创建功能分支、同步主干、解决冲突
- 能解释 `pull`、`fetch`、`push` 在协作里的边界
- 能参与一次完整的 PR 工作流

### 路径三：恢复与排障专项

适合谁：

- 经常遇到“搞乱历史、丢提交、冲突不会解”的开发者

先读哪些章：

1. 第 13-18 章
2. 第 24 章

完成后应具备的能力：

- 能区分 `restore`、`reset`、`revert`、`reflog` 的使用边界
- 能根据症状选择观察命令和恢复路径
- 能把“出事先看什么”变成稳定习惯

### 路径四：高阶工程实践

适合谁：

- 已经熟练使用 Git，希望理解治理、底层与组织级实践的读者

先读哪些章：

1. 第 19-24 章
2. 复查附录 B、D

完成后应具备的能力：

- 能解释对象模型、引用、配置层级、hooks 和仓库组合方式
- 能为团队设计合适的分支模型、发布策略和事故响应规则
- 能在“可用”和“可治理”之间做出工程取舍

---

## 实验系统总览

这套教程建议你不要只按章读，而是复用 6 个标准实验包反复练。

配套可执行脚本见：[labs/README.md](./labs/README.md)

| 实验包 | 适用章节 | 核心目标 |
|--------|----------|----------|
| 实验 A：最小仓库 | 第 1-6 章 | 建立工作区、暂存区、提交历史的观察习惯 |
| 实验 B：双分支冲突 | 第 7-9 章 | 练分支切换、merge、rebase 与冲突处理 |
| 实验 C：双仓库协作 | 第 10-12 章 | 练 `fetch/pull/push`、领先/落后状态、PR 前整理 |
| 实验 D：撤销与恢复 | 第 13-16 章 | 练 `restore/reset/revert/reflog/stash/clean` |
| 实验 E：发布与排障 | 第 17-18、24 章 | 练 tag、release、blame、grep、bisect、回滚 |
| 实验 F：治理与高阶实践 | 第 19-24 章 | 练对象观察、hooks、worktree、策略选择 |

建议你至少完成 A、B、C、D 四个实验包，再去读“高阶工程实践”路线。

---

## 学习路径建议

### Git 入门闭环（3-5 天）

1. 阅读第 1-6 章，建立工作区/暂存区/历史的核心心智模型
2. 在本地完成至少 10 次小提交，反复观察 `status`、`diff`、`diff --cached`
3. 完成一次“同一文件同时 staged 和 unstaged”的实验
4. 验收标准：你应能解释为什么 `git commit` 记录的是索引而不是工作区

### 团队协作上手（3-4 天）

1. 快速复习第 3-6 章
2. 重点学习第 7-12 章
3. 模拟一次功能分支开发、同步主干、解决冲突、提交 PR 的流程
4. 验收标准：你应能在 push 被拒绝时先判断原因，而不是本能地 force push

### 恢复与排障专项（2-3 天）

1. 重点学习第 13-18 章
2. 人为制造一次错误 `reset --hard`、一次误删分支、一次 stash 冲突
3. 用 `reflog`、`cherry-pick`、`blame`、`grep` 练习恢复与定位
4. 验收标准：你应能把“症状 -> 观察命令 -> 安全动作 -> 禁止动作”说完整

### 高阶工程实践（4-6 天）

1. 学习第 19-24 章
2. 结合附录 B 建立团队命令安全规范
3. 设计一套适合自己团队的分支模型、CI 规则和灾难恢复演练
4. 验收标准：你应能解释某项策略为什么适合你的团队，而不是只会背术语

---

## 前置要求

- **必需**：基本命令行操作能力
- **推荐**：有一个文本编辑器（VS Code、Vim、Cursor、Zed 等均可）
- **推荐**：有最基础的软件开发经验，知道文件、目录、脚本是什么

如果你完全不写代码，也依然可以学习前 1-12 章，只是第 18 章以后会更适合开发团队场景。

---

## 环境配置

本教程建议在命令行中练习，推荐环境如下：

```bash
# 检查 Git 是否安装
git --version

# 推荐先设置身份
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# 可选但推荐的默认配置
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.editor "code --wait"
```

你可以在 macOS、Linux、Windows（PowerShell / Git Bash / WSL）中完成大多数实验。

---

## 环境与练习约定

为了让实验可重复，建议你固定采用下面这套约定：

1. 所有基础实验放在 `git-lab/`
2. 远程协作实验用两份目录模拟，例如 `git-remote-origin/` 与 `git-remote-alice/`
3. 每次实验都保留两个观察面板：

```bash
git status -sb
git log --oneline --graph --decorate --all
```

4. 在需要看索引时，再额外运行：

```bash
git diff
git diff --cached
git ls-files --stage
```

5. 在执行高风险命令前，先看附录 B

如果你把这几条习惯养成，后面很多复杂命令都会自然变得“可解释”。

如果你希望直接使用可重复练习仓库，而不是手动搭建，优先使用 [labs/README.md](./labs/README.md) 中的场景脚本。

---

## 如何使用本教程

1. **先理解模型，再背命令。**
   如果不清楚工作区、暂存区、提交历史、引用这几层的关系，命令会越学越乱。

2. **每学一个命令，都问自己三个问题。**
   它改的是哪一层？能不能撤销？共享历史里什么时候不该用？

3. **用固定观察面板复盘。**
   除了记命令，也要观察状态、提交图和差异视图。

4. **故意制造错误，再练恢复。**
   只练“正确路径”只能让你会用，练过恢复才能让你真正不慌。

5. **把“先看状态”变成肌肉记忆。**
   心里没底时，先看 `git status -sb` 和历史图，而不是先试命令。

---

## 毕业能力清单

如果你读完整套教程，至少应该能独立完成下面这些动作：

- 在新目录里初始化仓库并建立稳定配置
- 独立完成一次本地开发、分块暂存和小步提交
- 看懂提交图、比较两个提交和两个分支的差异
- 创建功能分支、同步主干、解决 merge 或 rebase 冲突
- 参与一次完整的 PR 和 review 流程
- 在误用 `reset`、误删分支、进入 detached HEAD 后完成恢复
- 做一次版本发布演练，并知道 tag、CI、签名和回滚的关系
- 为团队写一份最小 Git 治理和事故处理规则

如果这些事情里有一半你还做不到，说明你更需要回到实验，而不是继续背更多命令。

---

## 教程特色

- **从零到高阶完整链路**：从初始化仓库到团队治理和灾难恢复
- **强调状态视角**：持续区分工作区、暂存区、提交历史、引用和远程状态
- **误区导向**：不只讲“怎么做”，还讲“为什么会错、错了会怎样”
- **实验驱动**：练习不仅要求会敲命令，还要求观察状态和历史图变化
- **面向真实协作**：覆盖 PR、Code Review、同步策略、热修、版本发布
- **中文编写**：术语统一，方便连续阅读与复习

---

## 许可证

本项目采用 MIT 许可证开源。

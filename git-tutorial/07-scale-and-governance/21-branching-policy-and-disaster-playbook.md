# 21 分支策略与灾难手册：把“出事怎么办”写在出事之前

- 前置章节：[20 Monorepo、LFS 与大仓库](20-monorepo-lfs-and-large-repos.md)：你已经能识别仓库规模、ownership、历史重写和共享引用的治理风险。
- 后续章节：本章是正文最后一章；后续应进入 README、模块索引、labs 和 appendix 的集成复盘。
- 本章 Lab id：`LAB-GOV-DISASTER-01`
- 核心危险命令：`git push --force`、`git push --mirror`、删除远程分支、移动公开 tag、直接修改受保护主干、对共享分支执行 rebase 后强推。
- 恢复路径：先冻结写入、记录当前引用和 CI/发布状态、创建 `rescue/*` 备份分支或镜像备份，再选择 `revert`、修复提交、恢复引用或重新发布。

## 场景

团队成员越来越多后，主干偶尔被坏提交破坏，发布分支上有人直接改历史，紧急 hotfix 时不知道该从哪里切。每个人都懂一些 Git 命令，但事故时缺少共同剧本：谁能暂停合并、谁能恢复分支、什么时候 revert、什么时候重新发版、什么时候绝不能强推。

本章的目标不是替所有团队规定同一种流程，而是把“最低协作规则”和“灾难恢复卡片”写在事故之前。好的分支策略应当让日常开发更快，也让事故发生时的第一动作更安全。

## 学习目标

完成本章后，你应该能够：

1. 根据团队发布节奏选择 trunk-based、release branch 或 GitFlow-like 策略。
2. 定义主干保护、PR 审查、合并方式、force push 边界。
3. 写出四类事故的观察入口和第一安全动作。
4. 区分 `revert`、hotfix、回滚部署、历史重写的适用条件。
5. 建立小团队 Git 协作最低规则，并把事故经验反馈到保护规则、CI 和文档中。

## 前置与后续章节

- 前置：[20 Monorepo、LFS 与大仓库](20-monorepo-lfs-and-large-repos.md)。大仓库治理已经告诉你：共享历史和共享引用一旦被改写，会影响 clone、PR、CI、发布标签和同事本地仓库。
- 后续：进入全教程集成阶段。README、模块索引、实验索引和附录会汇总本章的 lab id、危险命令、术语和事故卡片。

## 观察点

先看当前团队真实行为，不要先改保护规则：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all --max-count=30
git for-each-ref --format='%(refname:short) %(objectname:short) %(committerdate:relative)' refs/heads refs/remotes refs/tags | sort
git remote show origin 2>/dev/null || true
```

如果怀疑引用被误改，先保存证据：

```bash
git fetch --all --prune
git show-ref --heads --tags > /tmp/git-refs-before-recovery.txt
git reflog --date=iso > /tmp/git-reflog-before-recovery.txt
```

在平台侧还要观察：

- 主干是否禁止直接 push。
- PR 是否要求至少一名 reviewer。
- CI、测试、扫描是否是必需检查。
- 是否允许 merge commit、squash merge、rebase merge。
- 谁有删除分支、修改保护规则、force push、移动 tag 的权限。
- 发布系统使用的是分支、tag、commit SHA 还是制品版本号作为部署锚点。

观察的输出要能回答三个问题：当前默认分支在哪里、哪些引用是发布锚点、哪些人或自动化有权改变共享状态。

## 命令与决策

### 分支策略选择

| 团队条件 | 更适合 | 关键前提 |
|---|---|---|
| 持续交付、CI 快、feature flag 成熟 | Trunk-based，短分支，高频合并 | 主干保护严格，坏提交可快速 revert 或关闭 flag |
| 定期发布、需要维护多个版本线 | 主干 + release 分支 + hotfix 分支 | release owner 清晰，补丁能回合主干 |
| 重流程、多个长期环境分支 | GitFlow-like，但要控制长期分叉成本 | 明确环境分支含义，避免把分支当部署状态数据库 |

策略不是口号，而是一组可执行规则：

- 从哪里切功能分支？从 `main`、release 分支，还是特定 tag？
- 功能分支多久必须同步主干？超过几天需要拆小？
- 合并时保留 merge commit、squash，还是 rebase merge？谁决定例外？
- 发布标签由谁打？标签指向 commit 还是制品版本？
- 主干坏了先 revert、关闭 feature flag、回滚部署，还是新提交修复？
- 哪些分支允许 `--force-with-lease`？哪些分支永远不允许强推？

### 最低团队规则模板

可以把下面模板放进团队文档，再按团队规模裁剪：

```text
主干：main，必须可构建、可部署或至少可运行核心测试。
功能分支：task/<ticket>-short-name，生命周期建议小于 3 天。
同步方式：合并前 fetch + rebase/merge main，按团队约定二选一。
合并方式：默认 squash merge；需要保留分支上下文时使用 merge commit。
保护：main 禁止直接 push，必须 1 review + 必需 CI 通过。
历史重写：已推送共享分支禁止 rebase 后强推；例外必须用 --force-with-lease 且提前通知。
发布：标签由 release owner 创建；hotfix 从最新发布标签或 release 分支切出。
事故：主干坏了优先 revert 坏提交、关闭开关或提交修复，不立即重写共享历史。
复盘：每次事故后补一条保护规则、CI 检查或文档卡片。
```

规则越少，越要明确“谁有权例外”。比如：谁能临时解除分支保护、谁能移动 tag、谁能强推 release 分支。没有 owner 的例外规则，事故时会变成多人同时操作。

### 合并方式的取舍

| 合并方式 | 优点 | 成本 | 适用场景 |
|---|---|---|---|
| Squash merge | 主干线性、一个 PR 一个提交、回滚简单 | 丢失分支内小提交上下文 | 多数产品功能、修复和文档改动 |
| Merge commit | 保留分支拓扑和完整上下文 | 历史图更复杂 | 大型集成、长期分支、需要审计分支边界 |
| Rebase merge | 主干线性且保留多个提交 | 对新手心智负担高，冲突可能重复处理 | 提交切分质量高、团队熟悉 rebase 的仓库 |

选择后要写清楚默认值和例外，而不是让每个 PR 作者临时决定。事故恢复时，“一个 PR 一个 squash 提交”通常更容易 revert；保留 merge commit 则更容易看出某条功能分支带来的整体影响。

### 灾难恢复四入口

| 症状 | 第一观察 | 第一安全动作 | 常用恢复 |
|---|---|---|---|
| 本地提交丢了 | `git reflog`、`git fsck --lost-found`（最后手段） | 创建救援分支：`git switch -c rescue/<name> <sha>` | cherry-pick、merge 或重新提交 |
| 主干合入坏提交 | `git log --oneline --graph`、CI 链接、部署状态 | 暂停继续合并，确认坏提交范围 | `git revert <bad-sha>`、关闭 feature flag、修复 PR |
| 远程分支被错误强推 | 平台审计、同事本地 reflog、镜像备份 | 冻结 push，保存当前引用 | 从最后好 SHA 恢复分支引用并通知全员重新同步 |
| 发布标签打错 | `git show <tag>`、发布系统状态、制品哈希 | 暂停基于该 tag 的发布 | 新建修正标签或按平台规则撤销，避免静默移动公开标签 |

### `revert`、回滚部署、hotfix 与历史重写

| 目标 | 更安全的动作 | 说明 |
|---|---|---|
| 撤回主干中的坏代码 | `git revert <bad-sha>` 或 revert PR | 保留审计历史，适合共享主干 |
| 线上版本有问题但代码修复还没准备好 | 回滚部署到上一个制品或 tag | 这是发布系统动作，不一定要先改 Git 历史 |
| 已发布版本需要小修 | 从 release 分支或发布 tag 切 `hotfix/*` | 修复后要回合主干，避免补丁漂移 |
| 私有功能分支整理提交 | `git rebase -i` 后 `git push --force-with-lease` | 仅限自己独占分支，先 fetch，提前通知 reviewer |
| 共享历史泄露密钥或大文件 | 团队级历史重写项目 | 需要冻结、镜像备份、全员迁移和审计，不是个人临场操作 |

事故时优先选择“可审计、可回退、影响面小”的动作。历史重写只有在保留历史本身会持续造成风险时才升级，例如密钥泄露或巨大敏感文件进入历史。

## 实验

- Lab id：`LAB-GOV-DISASTER-01`
- 场景文件：[../labs/scenarios/LAB-GOV-DISASTER-01.md](../labs/scenarios/LAB-GOV-DISASTER-01.md)
- 生成脚本：

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-DISASTER-01 --force
```

实验重点：为一个模拟团队填写分支策略表，并对“主干坏了”“远程被强推”“发布标签打错”三种事故写恢复卡片。

建议步骤：

1. 运行观察命令，记录分支、tag、远程和最近历史。
2. 打开 lab 生成的 `POLICY.md`，填写主干、功能分支、合并方式、保护规则和历史重写边界。
3. 打开 `INCIDENT-CARDS.md`，为每类事故写“第一观察、冻结动作、备份动作、恢复动作、通知对象”。
4. 对主干坏提交，优先设计 `git revert` 或修复 PR；不要把 `reset --hard && push --force` 写成默认方案。
5. 对远程误强推，写出从平台审计、镜像备份或同事 reflog 找回最后好 SHA 的路径。
6. 对 tag 错误，说明是新建修正 tag、撤销发布，还是在受控公告后移动 tag。

## 常见错误

1. **只规定分支名，不规定合并条件**：没有 CI、review 和同步要求，分支策略只是命名规范。
2. **事故时第一反应是 reset + force push**：共享历史优先考虑 `revert`、修复提交、部署回滚或恢复引用。
3. **发布标签可以随便移动**：公开标签是团队和发布系统的锚点，移动前必须有公告、审计和回滚计划。
4. **把权限给所有人**：权限越大越需要流程；主干保护是协作系统的一部分。
5. **把环境分支当真相来源**：`dev`、`test`、`staging` 分支如果只表示部署状态，容易和代码审计历史混在一起。
6. **hotfix 不回合主干**：发布分支修好了但主干没修，下一次发布会重新带回同一个问题。
7. **只写恢复命令，不写沟通对象**：共享事故需要冻结、通知、owner 和复盘，否则多个“修复”会互相覆盖。

## 危险命令与恢复路径

> **危险命令：`git push --force` / `git push --mirror`**
>
> **风险**：会改写远程共享引用，可能让 PR、CI、同事本地分支和发布锚点失效。
>
> **恢复路径**：先 `git fetch --all --prune`，保存 `git show-ref --heads --tags` 输出；误推后立即冻结 push，从镜像备份、平台审计或同事本地 reflog 找最后好 SHA，再由 owner 恢复引用。

> **危险命令：删除远程分支或移动公开 tag**
>
> **风险**：删除分支可能切断正在 review 的 PR；移动 tag 可能让同一个版本号对应不同代码。
>
> **恢复路径**：删除前记录分支/tag 指向；公开 tag 错误时优先新建修正 tag（如 `v1.2.1` 或 `v1.2.0+fix`），只有在团队公告和发布系统允许时才移动旧 tag。

> **危险动作：临时关闭主干保护或直接 push 主干**
>
> **风险**：绕过 review 和 CI，短期修复可能制造更大的事故。
>
> **恢复路径**：必须有明确 owner、时间窗口和恢复保护规则的检查项；事后用 PR 或审计记录补齐变更原因。

更安全的替代命令：

```bash
git push --force-with-lease origin your-branch   # 仅限自己独占分支，且先 fetch
git revert <bad-sha>                            # 共享主干优先反向提交
git switch -c rescue/<incident> <last-good-sha> # 先保留证据和恢复入口
git tag -a v1.2.1 <fixed-sha>                   # 用新标签修正公开发布锚点
```

事故恢复顺序：

1. **冻结**：暂停相关分支 push、合并和发布，指定 incident owner。
2. **观察**：记录当前 HEAD、远程引用、CI 状态、发布状态、相关人员本地 reflog。
3. **备份**：创建 `rescue/*` 分支、导出引用列表，必要时做 `git clone --mirror`。
4. **恢复**：选择 revert、修复提交、恢复引用、回滚部署或重新发布。
5. **同步**：告诉团队如何 `fetch`、如何处理本地分支、哪些 PR 需要重开或更新。
6. **复盘**：把触发条件写回分支策略、保护规则、CI、release checklist 或 lab 卡片。

## 验收

为你的团队写出一页 Git 协作最低规则，至少包含：

- 主干保护规则。
- 功能分支命名和最长生命周期。
- 合并方式和同步主干方式。
- 允许历史重写的边界。
- 发布标签责任人。
- 三条事故恢复卡片：本地丢提交、主干坏、远程误强推。

你还应该能回答：

1. 本章 lab id 是什么？它要求真实改写远程历史，还是写可执行的事故卡片？
2. 你的团队默认合并方式是什么？选择它对 revert 和审计有什么影响？
3. 哪些分支永远不允许强推？哪些个人分支可以在什么条件下 `--force-with-lease`？
4. 主干坏了时，`revert`、修复提交、关闭 feature flag、回滚部署分别解决什么问题？
5. 如果 release tag 打错，你如何避免“同一个 tag 指向两个版本”的审计问题？

## 术语需求

请 Appendix 集成时确认或补充以下术语：trunk-based development、release branch、GitFlow、主干保护、required checks、squash merge、merge commit、rebase merge、force-with-lease、revert、hotfix、发布标签、镜像备份、incident owner。

## 交付给集成阶段

- 本章引用的 lab id：`LAB-GOV-DISASTER-01`。
- 前置章节：`07-scale-and-governance/20-monorepo-lfs-and-large-repos.md`。
- 后续章节：无正文后续；进入 README、module-index、labs、appendix 集成。
- 危险命令：`git push --force`、`git push --mirror`、删除远程分支、移动公开 tag、直接修改受保护主干、对共享分支 rebase 后强推。
- 恢复路径：冻结写入，保存 refs/reflog/CI/发布状态，创建 `rescue/*` 或镜像备份；优先 `revert`、修复提交、恢复引用或重新发布，历史重写仅作为团队级项目。
- Labs 需求：场景应让学习者填写 `POLICY.md` 和 `INCIDENT-CARDS.md`，覆盖主干坏提交、远程误强推、发布 tag 错误三类事故；主路径要求写恢复卡片，不要求真实破坏远程历史。

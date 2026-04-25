# 附录：术语表

本术语表统一教程中的中英文叫法。第一次出现时可写“中文（English）”，后续优先使用中文名；命令、选项和引用名保持原样。

## 核心状态层

### 工作区（working tree）

当前目录中真实可编辑的文件状态。`git restore <path>` 会从 index 或指定来源恢复工作区文件。

### 暂存区 / 索引（index / staging area）

下一次提交的候选快照。`git add` 更新 index，`git commit` 把 index 写入历史。

### 提交（commit）

一次项目快照，包含 tree、父提交、作者、提交者和说明。提交 ID 与内容相关，重写历史会生成新 ID。

### HEAD

当前检出位置。通常指向当前分支；detached HEAD 时直接指向某个提交。

### 未跟踪文件（untracked file）

存在于工作区，但还没有进入 Git 跟踪集合的文件。很多恢复命令无法保护它。

### 已暂存 / 未暂存（staged / unstaged）

已暂存表示改动已经进入 index，未暂存表示改动只存在于 working tree。一个文件可以同时有 staged 和 unstaged 改动。

## 引用与历史

### 分支（branch）

指向某个提交的可移动引用。新提交会让当前分支向前移动。

### 任务分支（task branch）

为一个具体任务创建的短生命周期分支，通常从主干或约定基线切出。

### 基线分支（base branch）

任务分支打算合入或对齐的目标分支，例如 `main`。

### 引用（ref）

Git 给对象起的名字，如 `refs/heads/main`、`refs/tags/v1.0.0`。

### 分支尖端（branch tip）

某个分支当前指向的提交。

### 提交范围（revision range）

用 `A..B`、`A...B` 等语法描述的一组提交或差异问题。

### Merge commit

三方合并产生的提交，通常有两个或多个父提交。

### 快进（fast-forward）

目标分支没有分叉时，只移动分支指针即可完成的合并。

### 三方合并（three-way merge）

基于共同祖先，把两条分支的改动合并成一个结果。

### Detached HEAD

HEAD 直接指向某个提交，而不是指向分支名。此状态下的新提交需要尽快创建分支保护。

## 远程协作

### 远程（remote）

本地仓库记录的同步端点，如 `origin`、`upstream`。

### 远程名（remote name）

本地仓库中指向远程 URL 的别名，例如 `origin`、`upstream`。

### 远程跟踪分支（remote-tracking branch）

本地记录的远程分支状态，如 `origin/main`。`git fetch` 更新它。

### 上游分支（upstream branch）

本地分支默认跟踪的远程分支，用于简化 `pull`、`push` 和 `branch -vv` 观察。

### Push rejected / non-fast-forward

远程拒绝本地 push，通常因为远程分支包含本地没有的提交。默认先 fetch 和看图。

### Pull Request / Merge Request

把分支改动交给团队审查、验证、合并的平台工作流。

### 共享历史（shared history）

已经 push、被 PR 引用、被 CI 或协作者基于其继续工作的提交历史。共享历史默认不改写。

## 撤销与恢复

### Restore

恢复工作区或 index 中的文件状态，适合处理未提交或 staged 的文件级问题。

### Reset

移动当前分支引用，并可按模式影响 index 和 working tree。

### Revert

创建一个反向提交，用新的历史记录撤销旧提交的效果，适合共享历史。

### Rebase

把一组提交重新播放到新基线之上，会生成新的提交 ID。

### Reflog / 引用日志

本地记录引用移动的日志。常用于找回 reset、rebase、误删分支前的提交。

### 救援分支（rescue branch）

恢复时为候选提交创建的临时命名分支，用于先保护再判断。

### 可达 / 不可达对象（reachable / unreachable object）

可达对象能从分支、标签、HEAD 等引用追溯到；不可达对象可能在垃圾回收后消失。

### 反向提交

`git revert` 生成的新提交，用补丁反向抵消目标提交的效果。

## 中断、发布与排障

### Stash

本地临时补丁栈。适合短期保存未提交改动，不适合长期任务管理。

### Worktree

同一仓库对象库对应的额外工作目录，适合并行处理多个分支或 hotfix。

### Dry-run

只预览不执行的安全检查步骤，例如 `git clean -nd`。

### Tag

给对象的稳定标记，常用于发布版本。公开发布标签不应随意移动。

### Annotated tag

附注标签，包含标签对象、说明、作者和日期，适合正式发布审计。

### Lightweight tag

轻量标签，只是一个引用，适合临时本地标记，不适合作为团队正式发布默认方案。

### Hotfix branch

从已发布版本或维护分支切出的最小修复分支。

### Semantic version

常见版本号形式如 `MAJOR.MINOR.PATCH`；hotfix 通常递增 patch 位。

### Blame

按行查看最后修改提交的工具。用于找上下文，不用于甩锅。

### Bisect

二分搜索历史，用可重复检查定位首个坏提交。

## 规模与治理

### Hook

Git 在特定生命周期触发的脚本，如 `pre-commit`、`commit-msg`、`pre-push`。

### Git config

Git 配置系统，包含 system、global、local 等层级。排查时优先看 `git config --show-origin --list`。

### Alias

Git 命令别名。好 alias 应缩短高频、低风险、可解释的操作，不隐藏危险语义。

### Git LFS

用于管理大文件的扩展。Git 历史保存指针，大文件内容保存在 LFS 存储中。

### Monorepo

多个项目或模块放在一个仓库中，强调统一历史和跨模块原子变更。

### 多仓库（multi-repo）

多个项目拆在多个仓库中，强调边界隔离和独立发布。

### 制品库（artifact repository）

用于保存构建产物、二进制包或大文件发布物的系统，不应默认由 Git 历史承载。

### 稀疏检出（sparse checkout）

只检出仓库中部分路径，降低大仓库工作区成本。

### 部分克隆（partial clone）

延迟下载部分对象，降低大仓库克隆成本。

### 对象数据库（object database）

Git 保存 blob、tree、commit、tag 等对象的底层存储。

### 历史重写（history rewrite）

批量改变既有提交 ID 的操作，例如 rebase、filter-repo、LFS migrate。

### 镜像仓库（mirror repository）

包含所有引用的完整镜像，常用于迁移、备份和历史清理前保护。

### CODEOWNERS

平台约定文件，用路径规则指定审查负责人，常用于 monorepo 治理。

### Submodule

在一个仓库中嵌入另一个仓库的固定提交引用。边界清晰，但日常操作成本较高。

### 主干保护（branch protection）

平台侧规则，用于限制谁能改主干、是否必须 PR、review、CI 通过。

### `--force-with-lease`

比 `--force` 更安全的强推形式，会检查远端分支仍是本地上次观察到的值；仍只应在允许重写的独占分支使用。

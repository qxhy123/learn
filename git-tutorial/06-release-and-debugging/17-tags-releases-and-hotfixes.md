# 17 Tags, Releases and Hotfixes

## 本章契约

- Lab ID：`LAB-RELEASE-HOTFIX-TAG-01`
- 前置章节：[16 Stash, Worktree and Interruptions](16-stash-worktree-and-interruptions.md)
- 后续章节：[18 Blame, Bisect and History Debugging](18-blame-bisect-and-history-debugging.md)
- 重点能力：把“某次发布”固定成可审计坐标，并能从发布点切出 hotfix、验证、打补丁版本标签、再把修复带回主干。
- 危险命令：`git tag -f`、`git tag -d`、`git push --delete origin <tag>`、`git push --force origin <tag>`、在 detached HEAD 上直接提交后不建分支。
- 恢复路径：错误标签未推送时本地删除重打；已推送或已被 CI/CD 消费时优先发布新版本号并公告；detached HEAD 上产生的提交先用 `git switch -c rescue/<name>` 或 `git branch rescue/<name> <commit>` 固定；任何删除远端标签前先 `git ls-remote --tags origin <tag>` 和团队确认。

## 场景

你的团队准备发布 `v1.2.0`。一周后线上发现严重问题，需要从已发布版本切出最小修复，而不是把 `main` 上尚未发布的新功能一起带上线。事故复盘时，大家还需要回答：线上包到底来自哪个提交？谁创建了发布点？hotfix 是否已经合回主干？

如果只说“应该是 main 上某个位置”，发布就没有稳定坐标。分支会继续移动，提交哈希不便于人类沟通，而标签可以把一个提交命名为版本。hotfix 流程则把“从哪个版本修、修了什么、发布成哪个补丁版本、如何回流主干”串成可审计路径。

本章不是教你把标签当书签乱贴，而是建立发布前后的判断顺序：**先确认提交和工作区状态，再创建不可随意移动的发布标签；需要补丁时，从标签开分支，修复后打新版本标签，并把修复合回主干。**

## 学习目标

学完本章后，你应该能够：

1. 区分分支、轻量标签、附注标签在发布语境中的职责。
2. 在打标签前检查工作区、当前分支、目标提交、测试状态和同名标签。
3. 创建、查看、推送附注标签，并解释为什么正式发布优先使用附注标签。
4. 从发布标签切出 hotfix 分支，完成修复、验证、补丁版本标签和回流主干。
5. 判断错误标签是否已经推送或被自动化消费，并选择安全恢复路径。
6. 处理 detached HEAD 风险，避免在历史提交上做了工作却没有分支保存。

## 观察点

发布和 hotfix 都要先观察，不要先打标签、先删除或先强推。建议固定运行：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all -n 16
git tag --list --sort=version:refname
git show --stat HEAD
```

打标签前回答这些问题：

- **工作区是否干净？** 发布标签应该指向一个已经提交、已验证的状态，不应依赖本地未提交文件。
- **当前提交是否就是要发布的提交？** 用 `git log --decorate` 和 CI 记录核对，而不是凭印象。
- **当前分支是否符合团队发布策略？** 有的团队只允许从 `main`、`release/*` 或受保护分支发布。
- **同名标签是否已经存在？** `git tag --list v1.2.0` 命中时不要覆盖，先确认原因。
- **测试和构建是否已经通过？** 标签不运行测试，它只是给某个提交命名；质量证据必须来自发布流程。
- **标签是否已经推送或被流水线监听？** 远端标签可能触发构建、制品上传和通知。

hotfix 前再补充观察：

```bash
git show --stat v1.2.0
git branch --contains v1.2.0
git ls-remote --tags origin v1.2.0
```

这些命令帮助你确认发布标签存在、指向正确、远端可见，并判断从它切出的补丁线是否合理。

## 操作流程

### 1. 选择发布坐标：分支会移动，标签不应移动

分支名表示一条持续演进的开发线，例如 `main`、`release/1.2`、`hotfix/v1.2.1`。标签表示某个固定提交，例如 `v1.2.0`。发布记录应该绑定标签，而不是绑定“当时的 main”。

正式发布推荐附注标签：

```bash
git tag -a v1.2.0 -m "Release v1.2.0"
git show v1.2.0
```

附注标签是一个独立对象，带有标签作者、日期和说明，适合审计。轻量标签只是一个引用，更像本地书签；临时标记可以用，但正式发布不要优先用它。

### 2. 发布前检查：先确认，再创建标签

一个可复用的发布前检查清单：

```bash
git status -sb
git fetch --tags origin
git tag --list v1.2.0
git log --oneline --decorate -n 5
git show --stat HEAD
```

判断标准：

- `git status -sb` 没有未提交改动。
- `HEAD` 是你要发布的提交，并且对应 CI/测试已经通过。
- `v1.2.0` 还不存在，或现有标签经过确认就是目标版本。
- 本地标签列表已经通过 `git fetch --tags` 与远端同步，避免本地不知道远端已有同名标签。

确认后创建并推送：

```bash
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0
```

推送后再验证远端：

```bash
git ls-remote --tags origin v1.2.0
git show --stat v1.2.0
```

### 3. 从发布标签切 hotfix 分支

线上版本有问题时，先从已发布标签创建补丁分支，而不是直接从当前 `main` 开始：

```bash
git fetch --tags origin
git switch -c hotfix/v1.2.1 v1.2.0
```

这会让修复基于 `v1.2.0` 的真实发布内容。修复时保持改动最小：

```bash
# 编辑文件，添加或更新回归测试
git status -sb
git diff --stat
git add <fixed-files>
git commit -m "fix: patch login timeout in v1.2.1"
```

发布补丁版本：

```bash
git tag -a v1.2.1 -m "Release v1.2.1 hotfix"
git push origin hotfix/v1.2.1 v1.2.1
```

`v1.2.1` 是新的不可变发布点，不要试图把 `v1.2.0` 移到修复提交上。版本号递增比移动旧标签更容易被团队、制品仓库和客户理解。

### 4. 把 hotfix 回流主干

补丁上线后，工作还没有结束。你必须让主干也包含这个修复，否则下个常规版本可能重新引入问题。

常见路线：

```bash
git switch main
git pull --ff-only
git merge --no-ff hotfix/v1.2.1
# 或按团队策略 cherry-pick hotfix 提交
git log --oneline --decorate -n 8
git status -sb
```

如果 `main` 已经重构过，直接 merge 可能冲突；这时用 PR、代码评审和测试来完成回流，不要因为补丁已上线就跳过主干验证。

完成后记录四件事：

- 受影响版本：例如 `v1.2.0`。
- 补丁版本：例如 `v1.2.1`。
- 修复分支：例如 `hotfix/v1.2.1`。
- 回流方式：merge、cherry-pick 或后续重做。

### 5. 查看标签和发布历史

排查发布问题时常用：

```bash
git tag --list --sort=version:refname
git show --stat v1.2.0
git log --oneline --decorate v1.2.0 -n 5
git for-each-ref --sort=-creatordate --format='%(refname:short) %(creatordate:short) %(subject)' refs/tags
```

如果某个制品声称来自 `v1.2.0`，你应该能用 `git show v1.2.0` 找到标签对象和目标提交，再和构建系统记录对齐。

## 危险命令与恢复路径

| 命令/场景 | 风险 | 更安全的做法 | 事故后先做什么 |
|---|---|---|---|
| `git tag -f v1.2.0 <commit>` | 移动本地同名标签，可能与远端或他人机器不一致 | 正式发布不要覆盖版本；需要修复就打 `v1.2.1` | 先 `git show v1.2.0`、`git reflog show refs/tags/v1.2.0`（若可用），确认旧目标 |
| `git tag -d v1.2.0` | 删除本地标签；若未记录目标，容易失去发布坐标 | 删除前 `git show --stat v1.2.0` 并确认未推送 | 若只是本地删除，可 `git fetch --tags origin` 从远端取回 |
| `git push --delete origin v1.2.0` | 删除远端标签，影响 CI/CD、制品和其他开发者 | 已发布版本优先废弃并发布新版本号，不删除旧标签 | 立即公告冻结发布；若误删，按记录把同一提交重新推回并通知所有人同步 |
| `git push --force origin v1.2.0` | 远端同名版本指向变化，不同机器可能看到不同制品 | 避免强推标签；用新版本号表达新发布 | 停止流水线，记录旧/新提交，按团队事故流程处理 |
| detached HEAD 上直接提交 | 提交没有分支名保护，后续切走后容易找不到 | 从标签修复时直接 `git switch -c hotfix/... <tag>` | 立刻 `git switch -c rescue/hotfix-work` 或记录提交哈希后建分支 |

通用恢复策略：

1. **先判断影响范围。** 本地未推送、已推送、已触发自动化，是三种不同级别。
2. **保留证据。** 记录 `git show <tag>`、`git ls-remote --tags origin <tag>`、构建号、制品哈希和相关提交。
3. **优先新增版本，不移动已发布版本。** 对外发布过的 `v1.2.0` 出错时，通常用 `v1.2.1` 修复。
4. **需要改远端标签时先冻结自动化。** 通知团队停止消费该版本，确认所有下游处理后再操作。
5. **用分支保护工作。** 在 detached HEAD 或不确定提交上做了修复，先建 `rescue/*` 或 `hotfix/*` 分支，再继续整理。

## 实验

Lab：`LAB-RELEASE-HOTFIX-TAG-01`

实验目标：在本地练习仓库完成一次发布、一次 hotfix、一次错误标签恢复判断，并形成发布记录。

1. **准备发布提交**
   - 新建练习仓库或使用 Labs Agent 提供的场景仓库。
   - 在 `main` 上创建一个可发布提交，例如更新 `app.txt` 或 `CHANGELOG.md`。
   - 运行 `git status -sb`、`git log --oneline --decorate -n 5`，确认工作区干净且提交正确。
2. **创建附注标签**
   - 运行 `git tag -a v1.0.0 -m "Release v1.0.0"`。
   - 用 `git show v1.0.0` 观察标签对象、说明和目标提交。
   - 用 `git tag --list --sort=version:refname` 验证标签列表。
3. **从标签切 hotfix**
   - 运行 `git switch -c hotfix/v1.0.1 v1.0.0`。
   - 做一个最小修复并提交。
   - 创建 `git tag -a v1.0.1 -m "Release v1.0.1 hotfix"`。
   - 用 `git log --oneline --graph --decorate --all -n 12` 观察发布线。
4. **回流主干**
   - 切回 `main`，把 hotfix 合回主干或按实验要求 cherry-pick 修复提交。
   - 运行测试或最小检查命令，记录结果。
   - 用 `git status -sb` 确认工作区干净。
5. **模拟错误标签恢复**
   - 在本地创建一个错误标签 `v1.0.2` 指向错误提交。
   - 在“未推送”前提下，用 `git show v1.0.2` 记录错误，再 `git tag -d v1.0.2` 删除并重打。
   - 写下如果 `v1.0.2` 已经推送并触发发布，为什么不能只当成本地问题处理。

给 Labs Agent 的落地需求：场景文件需要包含本地 bare remote 或模拟 origin 的可选步骤，以便练习 `git push origin v1.0.0`、`git ls-remote --tags origin v1.0.0`；所有删除或覆盖标签步骤必须限定在本地练习仓库，并在执行前要求记录原始提交哈希。

## 常见错误

- 把标签当成会持续前进的分支，发布后又移动同名标签。
- 打标签前没有确认工作区干净、目标提交正确、测试已经通过。
- 没有先 `git fetch --tags`，导致本地不知道远端已有同名标签。
- 用轻量标签做正式发布，缺少标签说明、作者和日期等审计信息。
- 从当前 `main` 切 hotfix，把未发布的新功能一起带进补丁版本。
- 从标签 checkout 后在 detached HEAD 上修复，却忘记创建分支保存提交。
- 发布 hotfix 后忘记合回主干，导致下个版本丢失补丁。
- 已推送错误标签后直接删除或强推，没有冻结流水线、公告团队和记录下游影响。

## 验收

请用下面问题自查：

1. 你能说明为什么发布点适合用标签，而持续开发线适合用分支。
2. 你能解释轻量标签和附注标签的差异，并说明正式发布为什么推荐附注标签。
3. 你能在打标签前列出至少五个观察项：工作区、分支、提交、测试、同名标签、远端标签。
4. 你能从 `v1.0.0` 切出 `hotfix/v1.0.1`，提交修复，打 `v1.0.1` 标签，并把修复回流主干。
5. 你能说明错误标签在未推送、已推送、已触发自动化三种情况下的处理差异。
6. 你能说出 detached HEAD 上提交的风险，以及如何用救援分支保护它。

验收命令建议：

```bash
git status -sb
git tag --list --sort=version:refname
git log --oneline --graph --decorate --all -n 12
git show --stat v1.0.0
git show --stat v1.0.1
```

最终交付应满足：工作区干净；`v1.0.0` 和 `v1.0.1` 都是附注标签；hotfix 分支或主干中能看到补丁提交；发布记录写清楚标签、提交、测试结果和回流方式；错误标签恢复说明区分了本地和远端影响。

## 术语需求

- `tag`：指向某个提交的稳定名字，适合标记发布点。
- `annotated tag`：附注标签，包含标签对象、说明、作者和日期，适合正式发布审计。
- `lightweight tag`：轻量标签，只是一个引用，适合临时本地标记，不适合作为团队正式发布默认方案。
- `hotfix branch`：从已发布版本或维护分支切出的最小修复分支。
- `detached HEAD`：HEAD 直接指向提交而不是分支；在此状态提交需要尽快创建分支保护。
- `semantic version`：常见版本号形式如 `MAJOR.MINOR.PATCH`；hotfix 通常递增 patch 位。

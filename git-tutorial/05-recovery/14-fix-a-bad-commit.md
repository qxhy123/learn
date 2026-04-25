# 14 Fix a Bad Commit

## 章节导航

- 前置章节：`05-recovery/13-undo-local-changes.md`，先学会区分工作区、暂存区和本地提交层面的撤销。
- 后续章节：`05-recovery/15-recover-lost-work-with-reflog.md`，再学习历史改写后如何用 reflog 找回看不见的提交。
- Lab id：`LAB-RECOVERY-BAD-COMMIT-01`

## 场景

你刚提交完就发现问题：提交说明写错、漏加了测试文件、把调试代码提交进去了，或者更麻烦——这个坏提交已经 push 到远端并进入同事的 review。此时真正的决策点不是“哪个命令能撤销它”，而是：**坏提交是否已经共享**。

未共享提交可以整理，因为只有你的本地历史会改变；已共享提交要优先保留历史，用新的提交修正错误。把这条边界判断清楚，才能避免把个人修补变成团队事故。

## 学习目标

完成本章后，你应该能够：

- 判断坏提交是否已经共享到远端、PR 或他人的分支。
- 使用 `git commit --amend` 修正最近一次未共享提交。
- 使用 `git reset --mixed HEAD~1` 拆开尚未共享的本地提交。
- 使用 `git revert <commit>` 安全撤销已共享提交的效果。
- 使用 `git cherry-pick <commit>` 把修复补丁带到维护分支，并解释为什么新分支上的提交 SHA 不同。
- 在 amend、reset 或 cherry-pick 出错后，知道优先用 reflog 和救援分支恢复。

## 观察点

修坏提交前先建立状态面板：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all -n 12
```

逐项观察：

1. `git status -sb` 是否干净；如果还有未提交改动，先决定是否暂存、提交到临时分支或 stash。
2. `git branch -vv` 是否显示当前分支关联 upstream，以及当前分支相对远端是 ahead、behind 还是 diverged。
3. `git log --decorate --all` 中坏提交是否已经出现在 `origin/<branch>`、PR 目标分支或其他协作分支上。
4. 坏提交是否是最近一次提交；如果不是，`amend` 不能直接修它，可能需要 revert、交互式 rebase 或新增修复提交。
5. 是否有人已经基于这个提交 review、测试、部署或继续开发。

一个实用判断：

| 问题 | 如果答案是“是” | 首选方向 |
|---|---|---|
| 坏提交只在本地吗？ | 是 | 可以 amend、reset 或 rebase 整理 |
| 坏提交已经 push 了吗？ | 是 | 优先 revert 或追加修复提交 |
| 只是最近一次提交漏文件/说明错？ | 是且未共享 | `git commit --amend` |
| 一个本地提交混了两类改动？ | 是且未共享 | `git reset --mixed HEAD~1` 后重新分组 |
| 修复要进维护分支？ | 是 | 在维护分支 `cherry-pick` 修复提交 |

## 命令与判断

### 修最近一次未共享提交

适用：最近一次提交只在本地，问题是漏文件、提交说明错误或需要把小修补并入同一逻辑提交。

```bash
git status -sb
git add <missing-file>
git commit --amend
```

如果只改提交说明：

```bash
git commit --amend -m "更准确的提交说明"
```

观察结果：

```bash
git log --oneline --decorate -n 3
```

`amend` 会创建一个新的提交对象，所以提交 SHA 会改变。它不是“原地修改”，而是“用新提交替换分支指针指向的旧提交”。

### 拆开一个未共享本地提交

适用：最近一次本地提交包含两类或更多不该放在一起的改动，例如同时改业务逻辑和格式化。

```bash
git status -sb
git reset --mixed HEAD~1
git status -sb
git add <first-set>
git commit -m "提交第一类改动"
git add <second-set>
git commit -m "提交第二类改动"
```

`--mixed` 会把当前分支移回上一个提交，并把原提交内容留在工作区但取消暂存，因此适合重新分组。拆分后再看历史图：

```bash
git log --oneline --graph --decorate -n 6
```

### 撤销已经共享的错误提交

适用：坏提交已经 push、进入主干、被 review、被 CI/CD 使用，或不确定是否有人基于它继续工作。

```bash
git status -sb
git revert <bad-commit>
git log --oneline --graph --decorate --all -n 8
```

`revert` 会新增一个反向提交，旧提交仍在历史里。这样做的好处是：协作者不需要处理历史重写，审计记录也保留了“错误发生”和“如何修正”的上下文。

如果连续多个共享提交都要撤销，先确认范围，再考虑：

```bash
git revert <oldest-bad-commit>^..<newest-bad-commit>
```

范围 revert 更容易产生冲突。遇到冲突时，不要急着继续，先用 `git status -sb` 和 `git diff` 确认反向改动是否符合预期。

### 把修复搬到维护分支

适用：主干已经有修复提交，发布分支或 hotfix 分支也需要同样修复，但暂时不能合并整条主干。

```bash
git switch release/1.2
git cherry-pick <fix-commit>
git log --oneline --graph --decorate --all -n 12
```

`cherry-pick` 会把补丁复制到当前分支，通常生成一个新的提交 SHA。请在提交说明、PR 描述或发布记录中标明来源提交，避免将来难以追踪同一个修复为什么出现在多条分支上。

## 危险命令

本章涉及会改变历史形状或复制提交的命令，需要明确边界：

| 命令 | 危险点 | 安全使用条件 |
|---|---|---|
| `git commit --amend` | 改变最近一次提交的 SHA | 最近一次提交未共享，或团队明确允许重写该分支 |
| `git reset --mixed HEAD~1` | 移动当前分支指针，原提交不再被分支引用 | 仅处理未共享提交，并确认工作区状态可控 |
| `git reset --hard <target>` | 丢弃已跟踪工作区和暂存区内容 | 本章不作为常规修坏提交方案；执行前必须先备份 |
| `git push --force` | 用本地历史覆盖远端历史 | 只有在受保护的个人分支、团队同意且知道后果时才考虑 |
| `git cherry-pick <commit>` | 同一补丁在多条分支上出现不同 SHA | 记录来源提交，避免长期替代正常合并流程 |

原则：**已共享历史默认不可改写；如果不确定是否共享，就按已共享处理。**

## 恢复路径

如果 amend、reset 或 cherry-pick 后发现处理错了，先停止继续改写历史，用 reflog 找回操作前的位置。

推荐恢复步骤：

```bash
git status -sb
git reflog --date=relative -n 20
git switch -c rescue-before-history-edit HEAD@{1}
```

然后验证救援分支：

```bash
git log --oneline --decorate -n 6
git show --stat
```

常见恢复选择：

1. amend 后想回到旧提交：从 reflog 找到 amend 前的 `HEAD@{n}`，创建救援分支，再决定是否 `reset` 回去。
2. reset 拆分后发现漏了内容：先在当前状态建 `rescue-current`，再从 reflog 给 reset 前位置建 `rescue-before-reset`。
3. revert 生成了不想要的反向提交：如果 revert 尚未共享，可以重置本地分支；如果已经共享，通常再 revert 这个 revert。
4. cherry-pick 产生冲突且不想继续：使用 `git cherry-pick --abort` 回到 cherry-pick 前状态。

## 实验

Lab：`LAB-RECOVERY-BAD-COMMIT-01`

在练习仓库中完成四个小场景，每一步都记录 `status` 和 `log` 的变化。

### 场景 A：amend 补漏文件

1. 新建文件 `app.txt` 并提交。
2. 发现漏了 `test.txt`，运行 `git add test.txt && git commit --amend`。
3. 对比 amend 前后的 `git log --oneline -n 2`，说明 SHA 为什么改变。

### 场景 B：reset mixed 拆提交

1. 在同一个提交里同时修改 `feature.txt` 和 `docs.txt`。
2. 运行 `git reset --mixed HEAD~1`。
3. 分两次 `git add` 和 `git commit`，把功能改动与文档改动拆开。
4. 用 `git log --graph --decorate -n 6` 验证历史变成两个清晰提交。

### 场景 C：revert 共享错误

1. 建立模拟远端或把 `main` 当作共享分支。
2. 制造一个错误提交，并假设它已经被 push。
3. 使用 `git revert <bad-commit>` 撤销效果。
4. 解释为什么没有使用 `reset --hard`。

### 场景 D：cherry-pick 修复到维护分支

1. 在 `main` 上做一个修复提交。
2. 切到 `release/demo` 分支。
3. 运行 `git cherry-pick <fix-commit>`。
4. 比较两个分支上的提交 SHA，并记录来源关系。

## 常见错误

- 认为 `amend` 只是修改提交说明，不会改变提交身份。
- 在已经 push 的分支上 reset 后强推，导致协作者必须手动处理分叉。
- 没有确认工作区干净就开始 amend 或 reset，把未完成改动混入修复。
- 用 `cherry-pick` 长期替代正常合并，导致同一补丁在多条分支上反复出现。
- revert 后以为旧提交消失了；实际上旧提交仍在历史中，新增的是反向提交。
- cherry-pick 冲突时直接乱改文件，没有先理解当前分支缺少哪些上下文。

## 验收

给定一个坏提交案例，你应该能先问“是否共享”，再选择命令并说明后果：

- 未共享且只修最近一次：使用 `git commit --amend`，并说明 SHA 会改变。
- 未共享且要拆分：使用 `git reset --mixed HEAD~1`，再重新分组提交。
- 已共享且要撤销效果：使用 `git revert <bad-commit>`，保留可审计历史。
- 修复需要进入另一条维护分支：在维护分支 `git cherry-pick <fix-commit>`，并记录来源提交。
- 误操作后需要找回：先查 `git reflog`，再从候选位置创建救援分支。

完成验收时，请展示：

```bash
git status -sb
git log --oneline --graph --decorate --all -n 12
```

并能解释历史图中哪些提交是原始错误、哪些是修复、哪些操作改变了提交 SHA。

## 交付给后续集成的事项

- Lab id：`LAB-RECOVERY-BAD-COMMIT-01`。
- 需要 Labs agent 落地四个场景：amend 补漏、mixed reset 拆提交、revert 共享错误、cherry-pick 到维护分支。
- 术语需求：共享历史、提交身份/SHA、反向提交、维护分支、救援分支。
- 危险命令需纳入 appendix 危险区：`commit --amend`、`reset --mixed`、`reset --hard`、`push --force`、`cherry-pick`。

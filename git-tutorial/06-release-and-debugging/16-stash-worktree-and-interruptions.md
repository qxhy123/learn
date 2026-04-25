# 16 Stash, Worktree and Interruptions

## 本章契约

- Lab ID：`LAB-RELEASE-STASH-WORKTREE-01`
- 前置章节：[15 Recover Lost Work with Reflog](../05-recovery/15-recover-lost-work-with-reflog.md)
- 后续章节：[17 Tags, Releases and Hotfixes](17-tags-releases-and-hotfixes.md)
- 重点能力：在“手头工作没做完，但必须切换上下文”时，选择 stash、临时提交、临时分支或 worktree，并能安全恢复。
- 危险命令：`git stash pop`、`git stash drop`、`git stash clear`、`git clean -fd`、`git worktree remove --force`
- 恢复路径：优先 `git stash apply` 后人工确认；对重要现场先 `git stash push -u -m ...` 或临时提交；对清理操作先 `git clean -nd`；误删/误丢线索时先回到上一章的 reflog 救援流程。

## 场景

你正在开发一个功能，文件改到一半、测试还没跑完，线上突然需要处理 hotfix；或者同事让你立刻切到 `main` 验证一个问题，但当前工作区既不干净，也不适合提交到正式历史。

这时常见反应是直接运行 `git stash`，甚至在恢复时直接 `git stash pop`。这不一定错，但它只适合短期、低风险的中断。中断时间更长、任务需要并行编译测试、或者改动已经能解释清楚时，临时分支、临时提交和 `git worktree` 往往更安全。

本章的目标不是背“如何藏起来”，而是建立一个中断处理决策：**先观察状态，再选择隔离方式，最后用可验证的恢复路径回到原任务。**

## 学习目标

学完本章后，你应该能够：

1. 根据中断时长、改动完整度、是否需要并行运行测试，判断该用 stash、临时提交还是 worktree。
2. 用带说明的 `git stash push -m` 保存短期 WIP，并知道什么时候需要 `-u` 包含未跟踪文件。
3. 区分 `git stash apply` 与 `git stash pop`，并能处理应用 stash 时产生的冲突。
4. 使用 `git worktree add` 为 hotfix 或排障创建独立工作目录，避免反复收纳当前现场。
5. 理解 `git clean -fd` 删除未跟踪文件的风险，坚持先 dry-run，并知道误操作后的有限恢复手段。

## 观察点

处理中断前不要先藏、先切、先删；先看现场。建议固定运行：

```bash
git status -sb
git diff --stat
git diff --cached --stat
git stash list
```

观察时回答四个问题：

- **当前改动能否形成一个可解释的小提交？** 能提交就优先提交到任务分支或临时分支，不要为了“看起来干净”把长期工作塞进 stash。
- **中断预计持续多久？** 几分钟可以 stash；几小时以上应考虑临时分支；需要持续多天或并行测试时优先 worktree。
- **是否有 staged、unstaged、untracked 三类内容混在一起？** stash 默认保存已跟踪文件的改动；未跟踪文件需要显式 `-u`。
- **切换过去后是否需要两个目录同时存在？** 如果新任务也要跑服务、测试或构建，用 worktree 避免来回切分支。

一个简单判断表：

| 情况 | 推荐工具 | 原因 |
|---|---|---|
| 五分钟内切走看一眼，当前改动还不成形 | `git stash push -m` | 快速收纳，恢复成本低 |
| 当前改动已经能说明意图 | 临时提交到当前任务分支 | 提交比匿名 stash 更容易审查和找回 |
| 要处理一个独立 hotfix | 新分支或 worktree | hotfix 有自己的历史和测试路径 |
| 两条任务都要持续编译/运行服务 | `git worktree add` | 两个目录互不污染 |
| 只想清掉构建产物或临时文件 | `git clean -nd` 后再决定 | 未跟踪文件不一定能恢复 |

## 操作流程

### 1. 短期收纳：用带说明的 stash

```bash
git stash push -m "wip: pause checkout validation"
git stash list
git stash show --stat stash@{0}
```

如果现场包含新建但尚未跟踪的文件，例如临时笔记、测试 fixture 或新源码文件，要显式包含未跟踪文件：

```bash
git stash push -u -m "wip: include new validation fixture"
```

不建议只运行裸 `git stash`，因为几天后你很难从 `WIP on branch-name` 判断它是否还重要。

### 2. 切换上下文：先确认工作区已干净

```bash
git status -sb
git switch main
git switch -c hotfix/login-timeout
```

如果 `git switch` 失败，不要立刻加 `--discard-changes` 或其它强制选项。回到观察点，确认是否还有未保存内容、是否需要 `stash push -u`，或者当前改动是否应该先提交到任务分支。

### 3. 恢复工作：优先 apply，再手动 drop

初学阶段建议把“应用”和“删除 stash 记录”拆开：

```bash
git switch feature/current-task
git stash apply stash@{0}
git status -sb
# 确认文件、测试和 diff 都正确后再删除
git stash drop stash@{0}
```

`git stash pop` 等价于“apply 成功后尝试 drop”。它不是危险到不能用，但在冲突或不确定现场中，拆开操作能保留更多判断空间。

如果应用 stash 时发生冲突，处理方式和 merge 冲突类似：

```bash
git status -sb
# 编辑冲突文件
git diff
git add <resolved-file>
git status -sb
```

确认解决后再继续测试和提交。不要在冲突未解决时继续 `pop` 其它 stash。

### 4. 长期并行：用 worktree 创建另一个工作目录

当你需要当前任务继续保留在原目录，同时另一个任务也要运行测试或服务时，用 worktree：

```bash
git worktree list
git worktree add ../project-hotfix main
cd ../project-hotfix
git switch -c hotfix/login-timeout
```

这会在相邻目录创建一个独立工作区。它和原目录共享同一个仓库对象库，但每个 worktree 有自己的工作区和当前分支。适合：

- 当前目录跑着开发服务，不能停。
- hotfix 需要从 `main` 开新分支并独立测试。
- 你想比较两个分支的构建结果，而不是来回 stash。

清理 worktree 前先确认目录中没有未提交工作：

```bash
cd ../project-hotfix
git status -sb
cd -
git worktree remove ../project-hotfix
```

不要为了省事使用 `git worktree remove --force`，除非你已经确认里面没有需要保存的改动。

### 5. 清理未跟踪文件：clean 必须先 dry-run

构建产物、临时日志、实验目录常常是未跟踪文件。清理前先看 Git 将删除什么：

```bash
git clean -nd
```

确认每一项都可删除后，再执行：

```bash
git clean -fd
```

`git clean -fd` 删除的是未跟踪文件和目录，这些内容通常不在 Git 历史里。删除后是否能恢复取决于编辑器、本机文件系统或备份，不应把它当成可逆命令。

## 危险命令与恢复路径

| 命令 | 风险 | 更安全的做法 | 事故后先做什么 |
|---|---|---|---|
| `git stash pop` | 应用并可能删除 stash；冲突时容易误判现场 | 先 `git stash apply stash@{n}`，确认后 `git stash drop` | `git status -sb`，确认 stash 是否仍在 `git stash list` |
| `git stash drop stash@{n}` | 删除单条 stash 引用 | 删除前 `git stash show --stat`，重要内容导出 patch | 立即停止操作，尝试从 `git fsck`/reflog 线索救援，成功率有限 |
| `git stash clear` | 删除全部 stash | 几乎不要在学习或团队仓库中使用 | 停止写入，记录时间点，尝试对象级救援，预期不要太高 |
| `git clean -fd` | 永久删除未跟踪文件和目录 | 先 `git clean -nd`；重要文件先移动、提交或 stash `-u` | 查编辑器本地历史/系统回收站/备份；Git 通常帮不上忙 |
| `git worktree remove --force` | 强制移除含未保存改动的 worktree | 进入该 worktree 跑 `git status -sb` | 若目录还在，先复制；若已删，按文件系统备份处理 |

通用恢复策略：

1. **停止继续清理或切换。** 越多写入操作，线索越少。
2. **保存当前可见现场。** 对已跟踪改动可 `git diff > ../rescue.patch`；对未跟踪文件先复制到仓库外。
3. **导出 stash 内容。** 如果 stash 还在，运行 `git stash show -p stash@{n} > ../stash-rescue.patch`。
4. **用上一章 reflog 方法保护历史位置。** 对已提交或引用移动事故，先创建救援分支，而不是直接 reset。
5. **把恢复结果作为新提交合回。** 恢复完成后用清晰提交说明中断和修复原因。

## 实验

Lab：`LAB-RELEASE-STASH-WORKTREE-01`

实验目标：制造一次真实中断，分别体验 stash、冲突恢复、clean dry-run 和 worktree 并行处理。

1. **准备现场**
   - 在练习仓库新建分支 `feature/interruption-demo`。
   - 修改一个已跟踪文件，并新建一个未跟踪文件。
   - 运行 `git status -sb`、`git diff --stat` 记录状态。
2. **短期 stash**
   - 先运行 `git stash push -m "wip: tracked only demo"`，观察未跟踪文件是否还在。
   - 再恢复现场，使用 `git stash push -u -m "wip: include untracked demo"`，比较差异。
3. **冲突恢复**
   - 切到另一分支，修改同一行并提交。
   - 回到原分支执行 `git stash apply stash@{0}`，制造或观察冲突。
   - 用 `git status -sb`、编辑器和 `git add` 完成解决。
4. **clean dry-run**
   - 创建 `tmp-output/` 和日志文件。
   - 先运行 `git clean -nd`，写下将被删除的路径。
   - 只有确认它们都是实验产物后，才运行 `git clean -fd`。
5. **worktree 并行**
   - 从 `main` 创建 `../project-hotfix` worktree。
   - 在 worktree 中创建 hotfix 分支并做一个小提交。
   - 回到原目录，比较 `git worktree list` 与 `git status -sb` 的输出。

给 Labs Agent 的落地需求：场景文件需要包含准备、执行、观察、恢复、清理步骤；清理步骤必须覆盖 `git worktree remove` 和测试目录删除；所有 destructive 步骤都要先展示 dry-run 或替代备份。

## 常见错误

- 把 stash 当成长期任务列表，不写说明，也不定期清理。
- 认为 stash 一定不会冲突，恢复前不看当前分支和 diff。
- 不区分 `apply` 和 `pop`，在复杂现场中直接 `pop`。
- 忘记 stash 默认不一定保存未跟踪文件，导致新文件仍留在工作区或后续被 `clean` 删除。
- 为了切分支而反复 stash，其实更适合创建 worktree。
- 在 `git clean -fd` 前没有运行 `git clean -nd`。
- 用 `git worktree remove --force` 清掉还没提交的 hotfix 目录。

## 验收

请用下面问题自查：

1. 给出三个中断案例时，你能说明为什么选择 stash、临时提交或 worktree，而不是只背一个命令。
2. 你能解释 `git stash apply` 与 `git stash pop` 的差异，并说明为什么冲突场景优先 `apply`。
3. 你能演示 stash 包含和不包含未跟踪文件时的状态差异。
4. 你能在执行 `git clean -fd` 前用 `git clean -nd` 预测删除范围。
5. 你能创建、查看并安全移除一个 worktree。
6. 你能说出本章至少三条危险命令及其恢复路径。

验收命令建议：

```bash
git status -sb
git stash list
git worktree list
git clean -nd
```

最终交付应满足：工作区干净；实验中的 stash 已确认保留或删除；临时 worktree 已安全移除；重要改动以提交、分支或 patch 的形式保存，而不是只停留在匿名 stash 中。

## 术语需求

- `stash`：本地临时补丁栈，适合短期保存未提交改动。
- `worktree`：同一仓库对象库对应的额外工作目录，适合并行处理多个分支。
- `untracked file`：Git 尚未跟踪的文件；很多恢复命令无法保护它。
- `dry-run`：只预览不执行的安全检查步骤，例如 `git clean -nd`。

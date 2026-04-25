# 09 不恐惧地使用 rebase

## 场景

你在 `feature/login-copy` 上做了两次小提交，还没有发起 PR。与此同时，`main` 已经合入了同事的更新。你希望把自己的提交放到最新 `main` 之后重新验证，让历史看起来像“我基于最新主干继续开发”，而不是制造一次只用于同步的 merge commit。

`rebase` 可以做到这一点：它把当前分支相对旧基线多出来的提交，按顺序重新播放到新基线之上。困难不在命令本身，而在边界判断：这些提交是否只属于你本地？是否已经被别人基于其上继续工作？如果答案不清楚，先不要 rebase。

本章把 rebase 当成一个有风险边界的整理工具：先观察历史图和共享状态，再执行本地分支 rebase；遇到冲突时能继续、中止或谨慎跳过；完成后重新验证，因为提交身份已经改变。

## 前置与后续章节

- 前置章节：`08-merge-conflicts-with-a-playbook.md`。你应已经能识别冲突状态、阅读冲突文件，并在无法判断时安全中止一次合并。
- 后续章节：`../04-collaboration/10-clone-fetch-pull-push.md`。下一模块会进入远程协作，解释 fetch、pull、push 与远端跟踪分支如何影响 rebase 边界。
- 本章 Lab ID：`LAB-BRANCH-REBASE-01`。

## 学习目标

完成本章后，你应该能够：

1. 用“把本地提交复制成新提交并接到新基线之后”解释 rebase。
2. 区分 merge 与 rebase 对历史图、提交 ID 和协作成本的影响。
3. 在本地未共享任务分支上执行 rebase，并用历史图证明基线已经改变。
4. 在 rebase 冲突中按状态继续、跳过或中止，而不是盲目重试命令。
5. 判断哪些分支不应该 rebase，尤其是主干、发布分支和多人依赖的共享分支。
6. rebase 完成后重新运行验证，确认新基线没有改变业务语义。

## 观察点

rebase 前先固定观察面板：

```bash
git status -sb
git branch --show-current
git branch -vv
git log --oneline --graph --decorate --all -n 16
```

你要能回答五个问题：

- 工作区是否干净？如果有未提交改动，先提交、stash 或放弃，不要带着脏工作区 rebase。
- 当前分支是不是个人任务分支，而不是 `main`、`release/*` 或团队共用分支？
- `git branch -vv` 是否显示当前分支已经 push，或与远端分支存在领先/落后关系？
- 当前分支相对旧基线多出了哪些提交？这些提交是否只是你自己的工作？
- 新基线是什么，例如 `origin/main` 或本地最新 `main`？你是否已经通过 `fetch` 更新了它？

一个常见的 rebase 前历史图可能像这样：

```text
* c3c3c3c (HEAD -> feature/login-copy) docs: adjust button copy
* b2b2b2b docs: add login help text
| * d4d4d4d (origin/main, main) docs: update navigation
|/
* a1a1a1a initial guide
```

这表示任务分支和 `main` 已经分叉。rebase 的目标是把 `b2b2b2b`、`c3c3c3c` 代表的改动重新播放到 `d4d4d4d` 之后。

## 命令与解释

先更新远端信息，再切到任务分支：

```bash
git fetch origin
git switch feature/login-copy
git status -sb
```

确认干净后执行：

```bash
git rebase origin/main
```

如果没有冲突，Git 会为你的本地提交生成新的提交对象，并让 `feature/login-copy` 指向这些新提交。原提交内容可能相同，但提交 ID 会改变，因为提交 ID 由内容、父提交、作者、时间等信息共同决定；父提交变了，ID 也会变。

rebase 后再次观察：

```bash
git log --oneline --graph --decorate --all -n 16
git diff origin/main...HEAD
git status -sb
```

你应该看到任务分支现在线性接在 `origin/main` 之后：

```text
* e6e6e6e (HEAD -> feature/login-copy) docs: adjust button copy
* f5f5f5f docs: add login help text
* d4d4d4d (origin/main, main) docs: update navigation
* a1a1a1a initial guide
```

注意 `f5f5f5f`、`e6e6e6e` 是新提交，不是原来的 `b2b2b2b`、`c3c3c3c`。

## rebase 冲突 playbook

rebase 遇到冲突时，Git 会停在“正在重放某一个提交”的中间状态。不要连续敲命令，先观察：

```bash
git status
git diff --name-only --diff-filter=U
git diff
```

按这个顺序处理：

1. **确认正在 rebase。** `git status` 会提示当前处于 rebase 过程中，并说明正在处理哪个提交。
2. **阅读冲突文件。** 像处理 merge 冲突一样，找出当前新基线和正在重放提交之间的语义差异。
3. **编辑最终结果。** 删除冲突标记，保留符合新基线语义的内容。
4. **复查并暂存。**
   ```bash
   git diff
   git add <resolved-file>
   git status
   ```
5. **继续重放。**
   ```bash
   git rebase --continue
   ```
6. **重复直到完成。** 每个冲突提交都要重新判断，不要把第一次选择机械套用到后续提交。

如果发现自己无法判断最终语义，优先中止：

```bash
git rebase --abort
```

`--abort` 会回到 rebase 前的分支位置，是安全退出路线。等你补充上下文、同步主干或请团队确认后，再重新开始。

只有在你明确“当前正在重放的这一个提交已经不需要，而且丢掉它不会损害任务目标”时，才使用：

```bash
git rebase --skip
```

`--skip` 会丢掉当前提交代表的改动。它不是“跳过冲突继续保留改动”，而是“不要这个提交”。

## merge 与 rebase 的选择

两者都能把主干更新纳入任务分支，但它们留下的历史语义不同：

- **merge** 保留真实分叉和汇合。适合需要记录“这两个分支曾经并行开发并在此汇合”的场景，也适合共享分支和团队需要审计的历史。
- **rebase** 改写本地提交的父提交，让历史呈现为线性重放。适合整理尚未共享的个人任务分支，减少同步型 merge commit。

可以用下面的判断表：

| 场景 | 倾向 | 原因 |
|---|---|---|
| 个人本地分支，尚未 push 或无人依赖 | rebase | 可读性收益高，协作成本低 |
| 已打开 PR，但团队允许更新提交序列 | 谨慎 rebase | 先看团队约定，并通知 reviewer |
| `main`、`release/*`、多人共用分支 | merge 或修复提交 | 不应改写别人依赖的历史 |
| 需要保留集成时间点和双父提交 | merge | merge commit 本身就是有用记录 |
| 只想把主干最新修复纳入本地验证 | rebase 或 merge 均可 | 取决于团队历史策略和共享状态 |

关键不是“rebase 更高级”，而是你是否愿意承担改写提交身份带来的协作成本。

## 实验

**Lab ID：`LAB-BRANCH-REBASE-01`**

目标：制造一个本地任务分支落后主干的场景，对比 rebase 与 merge 的历史图，并解释为什么 rebase 后提交 ID 改变。

实验步骤：

1. 在练习仓库从 `main` 创建任务分支：

   ```bash
   git switch -c feature/rebase-demo
   printf "first task line\n" >> notes.md
   git add notes.md
   git commit -m "docs: add first task line"
   printf "second task line\n" >> notes.md
   git add notes.md
   git commit -m "docs: add second task line"
   ```

2. 回到 `main`，制造主干前进：

   ```bash
   git switch main
   printf "main line\n" >> main.md
   git add main.md
   git commit -m "docs: add main line"
   ```

3. 复制 rebase 前历史图作为证据：

   ```bash
   git log --oneline --graph --decorate --all -n 12
   ```

4. 切回任务分支并 rebase 到最新 `main`：

   ```bash
   git switch feature/rebase-demo
   git rebase main
   ```

5. 再次复制历史图，并记录任务提交的新 ID：

   ```bash
   git log --oneline --graph --decorate --all -n 12
   git diff main...HEAD
   ```

6. 重做同样初始场景，改用 `git merge main`，比较两种历史图：一个是线性重放，一个保留 merge commit 或分叉汇合。

预期结果：学习者能指出 rebase 前后的旧提交 ID 与新提交 ID，能解释父提交改变导致 ID 改变，并能说明何时应选择 merge 而不是 rebase。

## 常见错误

- **把 rebase 当成无风险美化命令。** rebase 会生成新提交，改变提交 ID、父子关系和别人引用这些提交的方式。
- **在共享分支上随手 rebase。** 如果同事已经基于旧提交继续工作，你改写历史会让他们必须额外修复本地分支。
- **冲突时盲目 `--skip`。** `--skip` 会丢掉当前提交的改动，不是保留改动的快捷键。
- **rebase 后忘记验证。** 命令成功只说明 Git 完成重放，不代表新基线下业务仍然正确。
- **没有先 `fetch` 就 rebase。** 你可能只是 rebase 到过期的本地 `main`，并没有基于远端最新主干。
- **rebase 完成后直接强推。** 如果远端分支已存在，推送策略必须遵守团队约定；不要用 `git push --force` 覆盖他人历史。

## 危险命令与恢复路径

危险命令：

```bash
git rebase <base>
git rebase --skip
git rebase --abort
git reset --hard ORIG_HEAD
git push --force
```

风险说明：

- `git rebase <base>` 会改写当前分支上被重放提交的身份；它适合本地个人分支，不适合共享历史。
- `git rebase --skip` 会放弃当前正在重放的提交。
- `git rebase --abort` 是退出 rebase 的安全命令，但它不会替你解决业务冲突。
- `git reset --hard ORIG_HEAD` 可能回到 rebase 前位置，但也会丢弃工作区和暂存区未保存内容，必须先确认状态。
- `git push --force` 会让远端分支指向你的本地历史，可能覆盖协作者的远端提交；协作模块会讨论更安全的 `--force-with-lease` 边界。

恢复路径：

1. **rebase 前发现工作区不干净。** 停止操作，先提交、stash 或丢弃当前改动，再重新观察。
2. **rebase 过程中不确定如何解决冲突。** 运行 `git rebase --abort` 回到操作前状态，保存 `git status` 和历史图后再求助。
3. **已经解决部分冲突但还没完成 rebase。** 继续用 `git status` 判断当前步骤；如果语义判断错误且不想继续，仍可 `git rebase --abort`。
4. **rebase 已完成但尚未 push。** 用 `git reflog` 找到 rebase 前的分支位置；需要恢复时，可创建临时分支保存当前结果，再按恢复模块学习安全回到旧位置。
5. **已经推送并影响他人。** 停止继续强推，通知团队，确认是否按团队规则使用 `--force-with-lease`、重新建分支，或通过修复提交恢复远端状态。

## 验收

你应该能完成以下检查：

1. 用自己的话解释 rebase 为什么会改变提交 ID，即使文件内容看起来相同。
2. 给定一个历史图，指出哪些提交会被重放到新基线之后。
3. 说明 `git rebase --continue`、`git rebase --skip`、`git rebase --abort` 的差异。
4. 判断一个分支是否适合 rebase：它是否是个人本地分支？是否已被他人依赖？是否属于受保护分支？
5. rebase 成功后，用 `git log --graph`、`git diff <base>...HEAD` 和项目验证命令证明结果仍然正确。
6. 说明为什么共享主干通常不应 rebase，以及已经错误推送后第一步为什么是停止并通知团队。

## 交付自查

- 唯一修改范围：`03-branching-work/09-rebase-without-fear.md`。
- Lab ID：`LAB-BRANCH-REBASE-01`。
- 前置章节：`03-branching-work/08-merge-conflicts-with-a-playbook.md`。
- 后续章节：`04-collaboration/10-clone-fetch-pull-push.md`。
- 危险命令：`git rebase <base>`、`git rebase --skip`、`git rebase --abort`、`git reset --hard ORIG_HEAD`、`git push --force`。
- 恢复路径：按“rebase 前脏工作区 / rebase 中冲突 / 已完成未推送 / 已推送影响他人”分阶段处理。
- 需要后置集成：Labs agent 为 `LAB-BRANCH-REBASE-01` 准备可重复练习仓库；Appendix agent 汇总术语 `rebase`、新基线、重放、共享历史、`ORIG_HEAD`；Module/README agent 统一维护导航链接。

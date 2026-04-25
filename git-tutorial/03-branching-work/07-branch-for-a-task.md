# 07 为一个任务创建分支

## 场景

你在 `main` 上收到一个小任务：调整登录页文案，并顺手补一段说明。这个任务不大，但它仍然需要被 review、能回滚、能和主干同步。如果直接在 `main` 上改，你很难把“正在尝试的改动”和“可交付的主干状态”分开；如果把整个仓库复制一份，又会失去 Git 分支轻量、可追踪的优势。

本章把任务分支当作一个短生命周期工作区间：从干净基线创建分支，在分支上提交一个小而清晰的改动，并用状态和历史图解释 `HEAD`、当前分支、`main` 分别指向哪里。

## 学习目标

完成本章后，你应该能够：

1. 用“会移动的提交引用”解释分支，而不是把分支理解成项目副本。
2. 从干净主干创建任务分支，并验证 `HEAD` 已附着到新分支。
3. 在任务分支上完成一次可 review 的提交，并解释为什么只有任务分支向前移动。
4. 在切换分支前判断工作区是否安全，避免覆盖或混淆未提交改动。
5. 用清晰的分支名表达任务边界，让后续 merge、PR 和删除分支都有据可依。

## 前置与后续章节

- 前置章节：`../02-daily-workflow/06-ignore-files-and-repo-hygiene.md`。你应已经能保持工作区干净，并知道哪些文件不该进入提交。
- 后续章节：`08-merge-conflicts-with-a-playbook.md`。任务分支合回主干时，可能遇到冲突，需要按 playbook 处理。

## 观察点

创建分支前先固定观察面板：

```bash
git status -sb
git branch --show-current
git log --oneline --graph --decorate --all -n 8
```

你要确认三件事：

- `git status -sb` 显示工作区干净；如果不干净，你能说清每个改动属于当前任务、旧任务，还是临时实验。
- 当前分支是团队约定的基线，通常是 `main` 或集成分支。
- 历史图中 `HEAD -> main` 与远端跟踪分支处在你预期的位置；如果本地落后，应先按团队流程同步。

一个干净起点通常像这样：

```text
## main
* a1b2c3d (HEAD -> main, origin/main) docs: update guide
```

这表示 `HEAD` 附着在 `main` 上，`main` 和 `origin/main` 指向同一个提交。

## 命令与解释

创建并切换到任务分支：

```bash
git switch -c feature/login-copy
```

`git switch -c` 做了两件事：创建分支名 `feature/login-copy`，并让 `HEAD` 附着到这个新分支。它不会复制整个仓库；新分支一开始只是指向当前提交的另一个名字。

再次观察：

```bash
git status -sb
git branch --show-current
git log --oneline --graph --decorate --all -n 8
```

此时你应该看到当前分支变成 `feature/login-copy`，而 `main` 仍停在原来的提交上。做一次小改动后，用提交前流程确认边界：

```bash
git diff
git add docs/login.md
git diff --cached
git commit -m "docs: clarify login copy"
```

提交后再看历史图：

```bash
git log --oneline --graph --decorate --all -n 8
```

重点观察：

- 新提交让 `feature/login-copy` 向前移动。
- `main` 没有自动跟着移动，因为你当前提交的是任务分支。
- `HEAD` 仍然附着在当前分支上，并通过当前分支间接指向最新提交。

如果要回到主干观察差异：

```bash
git switch main
git log --oneline --graph --decorate --all -n 8
```

你会看到历史图里仍有任务分支提交，但当前工作区回到了 `main` 指向的文件状态。

## 分支命名建议

分支名应该帮助别人判断任务范围，而不是记录情绪或临时想法。常见格式包括：

- `feature/login-copy`：新增或调整功能。
- `fix/session-timeout`：修复明确问题。
- `docs/branch-playbook`：文档改动。
- `chore/update-ci-cache`：维护性改动。

避免只用 `test`、`fix`、`new` 这类无法说明边界的名字。任务结束并合并后，短生命周期分支通常应删除，让分支列表保持可读。

## 实验

**Lab id：`LAB-BRANCH-TASK-01`**

目标：创建一个任务分支，提交一处文档改动，并证明 `main` 没有随任务提交移动。

实验步骤：

1. 在练习仓库确认 `main` 干净：

   ```bash
   git status -sb
   git branch --show-current
   ```

2. 从 `main` 创建任务分支：

   ```bash
   git switch -c feature/readme-note
   ```

3. 修改一个文档文件，例如：

   ```bash
   printf "\nBranch note: work happens on a task branch.\n" >> README.md
   git diff
   git add README.md
   git diff --cached
   git commit -m "docs: add branch workflow note"
   ```

4. 用历史图记录 `main` 与任务分支的位置：

   ```bash
   git log --oneline --graph --decorate --all -n 8
   ```

5. 切回 `main`，确认任务分支提交不会改变当前工作区：

   ```bash
   git switch main
   git status -sb
   git log --oneline --graph --decorate --all -n 8
   ```

预期结果：学习者能解释“分支只是一个会移动的名字”，并能说明为什么新提交没有改变 `main` 的指向。

## 常见错误

- **把分支当成完整目录副本。** 分支主要是提交引用，Git 复用对象，不会复制整个项目。
- **在脏工作区直接切分支。** 如果目标分支会覆盖当前改动，Git 可能拒绝；如果不拒绝，也会让你难以判断改动归属。
- **从错误基线创建分支。** 如果你从过期或临时分支切出任务分支，后续 PR 会混入无关提交。
- **长期保留任务分支不合并。** 短任务分支越久不同步，后续冲突和 review 成本越高。
- **分支名只写 `fix` 或 `test`。** 好的分支名应表达任务边界，例如 `fix/session-timeout`。

## 危险命令与恢复路径

危险命令：

```bash
git switch -f <branch>
git checkout -f <branch>
git branch -D <branch>
```

风险说明：

- `git switch -f` / `git checkout -f` 会强制切换分支，可能丢弃工作区中尚未提交的改动。
- `git branch -D` 会强制删除分支名；如果分支上的提交没有被其他引用保存，你可能只能依赖 `reflog` 找回。

恢复路径：

1. 如果只是切换被拒，先运行 `git status -sb`，不要加 `-f`。
2. 想保留改动：提交到当前分支，或在后续中断工作章节学习 `stash` / `worktree`。
3. 已经强制切换后，先检查 `git reflog` 和 `git status`，确认是否有提交或引用可恢复。
4. 已经强制删除分支但还记得最近提交：用 `git reflog` 找到提交哈希，再执行 `git branch <branch-name> <hash>` 重建分支名。
5. 如果改动从未进入暂存区或提交，Git 通常无法可靠恢复；这就是本章要求先观察的原因。

## 验收

你应该能回答：

1. 当前 `HEAD` 是直接指向提交，还是附着在某个分支上？你用哪条命令验证？
2. 任务分支提交后，`main` 为什么没有移动？
3. 切换分支前为什么要先看 `git status -sb`？
4. 如果任务做到一半需要临时切回主干，你会先检查什么？
5. 什么情况下可以删除任务分支？删除前要确认哪些引用还保留提交？

## 给后续集成 agent 的说明

- 本章引用的 lab id：`LAB-BRANCH-TASK-01`。
- Labs 需要提供一个含 `README.md` 的练习仓库，并能让学习者创建 `feature/readme-note` 后观察 `main` 与任务分支的指向差异。
- Appendix 可汇总术语：`HEAD`、任务分支、基线分支、远端跟踪分支、强制切换、强制删除分支。

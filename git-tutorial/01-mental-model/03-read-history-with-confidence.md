# 有信心地阅读历史

## 场景

你接手一个仓库，准备改代码之前先想回答几个问题：最近发生了什么？当前分支从哪里分出来？某个提交到底改了哪些文件？这时不要急着运行会改动历史或工作区的命令。先把 Git 历史当成一张由提交组成的地图：分支名只是指向某个提交的标签，`HEAD` 表示你当前站在哪里。

本章训练的是“读懂再行动”：用 `log` 看路径，用 `show` 看单个提交，用 `diff` 看两个点之间的内容差异。

## 学习目标

完成本章后，你应该能够：

- 使用 `git log --oneline --graph --decorate --all` 阅读提交图。
- 解释 `HEAD`、分支名、远程跟踪分支在历史图中的位置。
- 用 `git show --stat` / `git show --name-only` 查看单个提交的范围。
- 用 `git diff A..B` 比较两个提交快照的最终内容差异。
- 区分“提交路径问题”（用 `log` 回答）和“内容差异问题”（用 `diff` 回答）。
- 在看到 merge commit、detached HEAD 或看不懂的历史时，先观察而不是破坏性修复。

## 前置与后续

- 前置章节：[`02-working-tree-index-commit.md`](02-working-tree-index-commit.md)
- 后续章节：[`../02-daily-workflow/04-first-change-to-clean-commit.md`](../02-daily-workflow/04-first-change-to-clean-commit.md)
- 本章 Lab id：`LAB-MODEL-HISTORY-01`

## 观察点

每次阅读历史前，先确认自己没有未保存的本地修改：

```bash
git status --short --branch
```

然后使用这些只读命令观察历史：

```bash
git log --oneline --graph --decorate --all --max-count=12
git show --stat HEAD
git show --name-only HEAD
git diff HEAD~1..HEAD
```

观察时回答：

- `HEAD` 当前指向哪个提交？它是否同时被某个分支名指向？
- 当前分支尖端是哪一个提交？主线分支尖端是哪一个提交？
- 历史是线性的，还是出现了分叉和汇合？
- 最近一次提交的意图是什么？它改了哪些文件？
- `diff` 展示的是“中间经过哪些提交”，还是“两个提交快照的最终差异”？

## 命令与解释

### 用 `log` 看提交路径

```bash
git log --oneline --graph --decorate --all --max-count=20
```

- `--oneline`：每个提交压缩成一行，适合快速扫图。
- `--graph`：用 ASCII 线条显示分叉与汇合。
- `--decorate`：显示 `HEAD`、分支名、tag、远程跟踪分支等引用。
- `--all`：不仅看当前分支，也看本地所有引用可达的历史。

`log` 回答的是“历史路径是什么”：哪些提交在某条分支上、哪里分叉、哪里合并。

### 用 `show` 看单个提交

```bash
git show --stat HEAD
git show --name-only HEAD
```

`show` 聚焦一个提交。`--stat` 适合看改动规模，`--name-only` 适合快速确认文件范围。review 自己刚写的提交时，先用 `show` 看提交意图和范围，再决定是否需要补充提交或修正提交。

### 用 `diff` 看两个快照的最终差异

```bash
git diff main..feature
```

`diff` 比较两个引用指向的提交快照。它不关心中间路径如何演化，只回答“从 `main` 的内容到 `feature` 的内容，最终差异是什么”。

> 经验规则：想问“有哪些提交？”用 `git log`；想问“最终改了什么？”用 `git diff`。

### 用范围语法问清楚问题

```bash
git log main..feature --oneline
git diff main..feature
```

这两个命令看起来相似，但问题不同：

- `git log main..feature`：列出 `feature` 有、`main` 没有的提交。
- `git diff main..feature`：比较 `main` 与 `feature` 两个提交快照的内容差异。

阅读历史时先把问题说清楚，再选命令。

## 实验

**Lab id：`LAB-MODEL-HISTORY-01`**

目标：创建一段包含线性提交、任务分支和 merge commit 的历史，并用不同视图回答问题。

> 建议在临时目录执行，不要在真实项目里做练习提交。

```bash
mkdir history-lab
cd history-lab
git init -b main
git config user.name "Git Tutorial"
git config user.email "git-tutorial@example.com"

printf "v1\n" > app.txt
git add app.txt
git commit -m "create app"

printf "v2\n" >> app.txt
git add app.txt
git commit -m "extend app"

git switch -c feature/history-note
printf "history note\n" > history.txt
git add history.txt
git commit -m "add history note"

git switch main
printf "release note\n" > release.txt
git add release.txt
git commit -m "add release note"

git merge --no-ff feature/history-note -m "merge history note"
```

执行观察命令：

```bash
git status --short --branch
git log --oneline --graph --decorate --all --max-count=12
git show --stat HEAD
git show --name-only HEAD~1
git diff HEAD~2..HEAD
```

预期观察：

- `log --graph` 显示一个任务分支和一次汇合。
- `HEAD` 位于当前主线分支尖端，并指向 merge commit。
- `show --stat HEAD` 显示 merge commit 的提交信息；如果 merge 没有产生额外补丁，统计可能很小或为空，这是正常现象。
- `show --name-only HEAD~1` 能看到 merge 前主线最后一次提交涉及的文件。
- `diff HEAD~2..HEAD` 展示两个提交快照之间的最终内容差异，而不是逐个提交列表。

实验结束后清理临时目录：

```bash
cd ..
rm -rf history-lab
```

## 危险命令

本章的核心命令都是只读命令，但阅读历史时常见的“顺手修一下”可能有风险：

- `git reset --hard <commit>`：会移动当前分支并丢弃工作区/暂存区改动；不要用它来“只是回到某个提交看看”。
- `git checkout <commit>` 或 `git switch --detach <commit>`：会进入 detached HEAD 状态；这本身不危险，但如果在其中提交，提交可能因为没有分支名而被遗忘。
- `git push --force` / `git push --force-with-lease`：会改写远程分支历史；本章不需要使用。

安全替代：先用 `git show <commit>`、`git diff A..B`、`git log A..B` 读信息；需要临时查看旧版本时，优先新建分支或使用一次性临时目录。

## 恢复路径

如果阅读历史时误操作，按下面顺序处理：

1. 先停手并记录当前状态：

   ```bash
   git status --short --branch
   git log --oneline --decorate --max-count=5
   git reflog --date=local --max-count=10
   ```

2. 如果只是进入 detached HEAD，回到原分支（把 `main` 换成你的原分支名）：

   ```bash
   git switch main
   ```

3. 如果在 detached HEAD 中产生了想保留的提交，先给它一个分支名：

   ```bash
   git switch -c rescue/history-reading
   ```

4. 如果误用了 `reset --hard`，不要继续提交；用 `git reflog` 找到 reset 前的位置，再在确认后新建救援分支：

   ```bash
   git switch -c rescue/before-reset <reflog-entry>
   ```

5. 如果错误已经 push 到共享分支，停止本地修复，先和团队同步；不要直接 force push 覆盖他人历史。

## 常见错误

- **把 `log` 和 `diff` 混成一个问题**：`log` 回答“经过哪些提交”，`diff` 回答“最终内容差异是什么”。
- **看到 merge commit 就以为出错**：merge commit 只是记录两条历史汇合；是否有问题要结合提交意图、文件范围和测试结果判断。
- **忽略引用装饰**：看不懂 `HEAD -> main`、`origin/main`、tag，就无法判断自己和远程/发布点的关系。
- **用破坏性命令代替观察命令**：只是想看旧提交时，不要先 `reset --hard`。
- **在 detached HEAD 中提交后直接离开**：如果提交有价值，离开前先创建分支保存。

## 验收

请在完成实验后回答：

1. `git log --oneline --graph --decorate --all` 中，`HEAD`、当前分支名、任务分支名分别指向哪里？
2. `git log main..feature/history-note --oneline` 适合回答什么问题？
3. `git diff main..feature/history-note` 适合回答什么问题？它为什么不等同于提交列表？
4. 看到 merge commit 时，你会用哪两个命令确认它的文件范围和历史位置？
5. 如果误进入 detached HEAD 并产生了一个想保留的提交，应该先执行什么恢复动作？

参考答案要点：

- `log` 用于阅读提交路径和引用位置；`diff` 用于比较两个提交快照的最终内容。
- `show --stat` / `show --name-only` 可确认单个提交范围。
- detached HEAD 中的有价值提交应先用 `git switch -c <rescue-branch>` 命名保存。

## 术语需求

请后续 Appendix/Glossary 集成这些术语：`HEAD`、引用（ref）、分支尖端、远程跟踪分支、merge commit、detached HEAD、提交范围（revision range）、快照差异。

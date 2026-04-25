# 10 clone、fetch、pull 与 push

## 场景

你加入一个已有项目，收到的第一条指令往往是“把仓库 clone 下来，然后拉一下最新代码”。如果只把这句话理解成几条命令，很快就会在多人协作里迷路：`origin/main` 到底是不是服务器上的 `main`？`fetch` 为什么看起来“什么都没发生”？`pull` 为什么会突然产生 merge commit 或触发 rebase？`push -u` 又为什么会影响后续默认推送目标？

本章把远程协作拆成四层状态来观察：**远程仓库、远程跟踪分支、本地分支、工作区/暂存区**。你要学会先看状态，再决定是只更新本地的远程快照，还是把远程变化整合到当前分支，最后再安全推送自己的分支。

## 前置 / 后续

- 前置章节：[09 不恐惧地使用 rebase](../03-branching-work/09-rebase-without-fear.md) — 你已经能解释 merge 与 rebase 对历史图的影响，并知道改写历史的边界。
- 后续章节：[11 同步主干并打开 PR](./11-sync-with-main-and-open-pr.md) — 你将处理 push rejected、同步主干并准备 PR。

## 学习目标

完成本章后，你应该能够：

1. 区分远程仓库、远程名、本地分支和远程跟踪分支。
2. 解释为什么 `git fetch` 是最安全的远程同步起点。
3. 拆解 `git pull` 通常包含的两个动作，并说明 merge pull 与 rebase pull 的差异。
4. 在 `push` 前判断当前分支的上游关系和远程目标。
5. 在推送或同步出错时，先保留现场并选择可恢复路径。

## 观察点

克隆或准备同步前，固定运行这个观察面板：

```bash
git remote -v
git status -sb
git branch -vv
git log --oneline --graph --decorate --all -n 16
```

你要能从输出里回答：

- `origin` 指向哪个远程 URL？它是团队主仓库、个人 fork，还是本机模拟远程？
- 当前本地分支是否有上游分支？`git branch -vv` 中是否显示 `[origin/main]` 或 `[origin/feature/x]`？
- `origin/main` 是你本地保存的远程主干快照，还是服务器上的实时分支？
- 当前分支相对上游是 ahead、behind、diverged，还是干净同步？
- 工作区和暂存区是否干净，能否安全执行 pull、merge、rebase 等整合动作？

一个典型输出可能像这样：

```text
$ git branch -vv
* main 8f3a2c1 [origin/main: behind 1] docs: clarify setup
```

这表示本地 `main` 正在跟踪 `origin/main`，并且本地落后一个提交。它不表示工作区一定能安全合并；你还必须结合 `git status -sb` 判断本地是否有未提交改动。

## 命令与解释

### 1. clone：复制仓库并建立默认远程关系

```bash
git clone <url> project
cd project
git remote -v
git branch -vv
```

`clone` 不只是下载文件。它会：

1. 创建一个新的本地仓库。
2. 添加默认远程名，通常叫 `origin`。
3. 下载远程对象和分支信息。
4. 创建一个本地默认分支，并让它跟踪对应的远程跟踪分支。

克隆后不要急着改文件，先确认远程 URL 和默认分支是否符合预期。团队主仓库、个人 fork、练习用 bare 仓库看起来都可以叫 `origin`，但协作含义不同。

### 2. remote：看清“名字”指向哪里

```bash
git remote -v
```

`origin` 只是本地配置里的远程别名，不是 Git 的魔法关键字。一个仓库可以有多个远程，例如：

```text
origin   git@example.com:me/project.git (fetch)
origin   git@example.com:me/project.git (push)
upstream git@example.com:team/project.git (fetch)
upstream git@example.com:team/project.git (push)
```

在 fork 工作流中，`origin` 可能是你的 fork，`upstream` 才是团队主仓库。任何 `push` 之前都要先确认目标远程和分支。

### 3. fetch：安全更新远程跟踪分支

```bash
git fetch origin
git log --oneline --graph --decorate --all -n 16
```

`fetch` 会把远程仓库的新对象取回本地，并更新 `origin/main`、`origin/feature/x` 这类远程跟踪分支。它通常不会移动你当前检出的本地分支，也不会改工作区文件。

这就是 `fetch` 安全的原因：它让你先看到远程发生了什么，再决定后续动作。你可以在 fetch 后比较：

```bash
git log --oneline --graph --decorate main origin/main -n 16
git diff --stat main..origin/main
```

如果只是想“看看远程有没有新东西”，优先用 `fetch`，不要把 `pull` 当成只读观察命令。

### 4. pull：fetch 后再整合当前分支

```bash
git pull
```

默认情况下，`pull` 可以理解成两步：

```bash
git fetch
# 然后根据配置执行其中一种整合方式
git merge <upstream>
# 或
git rebase <upstream>
```

所以 `pull` 不是“下载最新代码”的单一动作。它会尝试把远程更新整合进当前分支，可能导致：

- 快进（fast-forward）：本地没有新提交，分支指针直接前移。
- merge commit：本地和远程都前进过，Git 通过合并提交保留汇合点。
- rebase：如果团队配置了 rebase pull，本地提交会被重新播放到新基线，提交 ID 会变化。
- 冲突：远程和本地改到同一区域，需要你手工解决或中止。

执行 `pull` 前先确认工作区干净：

```bash
git status -sb
```

如果输出里有 `M`、`A`、`??` 等未处理改动，先提交、暂存、stash 或明确放弃，避免本地未完成工作和远程整合冲突混在一起。

### 5. push：把本地提交发送到远程分支

```bash
git push -u origin feature/login-copy
```

`push` 会把本地提交发送到指定远程分支。`-u`（`--set-upstream`）会建立上游跟踪关系，让后续命令知道默认目标：

```bash
git branch -vv
git status -sb
git push
git pull
```

第一次推送任务分支时，明确写出远程和分支名比依赖默认值更安全。建立上游后，也仍要在关键操作前用 `git branch -vv` 复核目标。

## 判断流程

遇到远程同步任务时，按这个顺序处理：

1. `git status -sb`：确认工作区是否干净。
2. `git remote -v`：确认远程别名对应的 URL。
3. `git branch -vv`：确认当前分支和上游关系。
4. `git fetch origin`：先更新远程跟踪分支。
5. `git log --oneline --graph --decorate --all -n 16`：看清本地、远程和主干的相对位置。
6. 如果当前分支落后上游，按团队规则选择 merge 或 rebase 整合。
7. 如果本地领先且目标正确，再 `git push` 或第一次 `git push -u origin <branch>`。

这个流程的核心是：**先让本地视野变新，再决定是否移动当前分支，最后才改远程状态**。

## 实验

**Lab ID：`LAB-COLLAB-REMOTE-01`**

目标：模拟一个远程仓库和两位协作者，观察 `fetch` 与 `pull` 对不同状态层的影响。

```bash
mkdir remote-lab
cd remote-lab
git init --bare origin.git
git -C origin.git symbolic-ref HEAD refs/heads/main

git clone origin.git alice
```

在 `alice` 中创建初始提交并推送。练习仓库里可以使用本地身份配置，避免污染全局 Git 配置：

```bash
cd alice
git config user.name "Alice"
git config user.email "alice@example.com"
printf "hello\n" > app.txt
git add app.txt
git commit -m "seed project"
git push -u origin main
cd ..
```

现在再 clone 第二份工作副本，并让 `alice` 模拟远程上已经有人先提交：

```bash
git clone origin.git bob

cd alice
printf "alice update\n" >> app.txt
git add app.txt
git commit -m "add alice update"
git push
cd ..
```

让 `bob` 先观察本地旧视图，再只获取远程更新：

```bash
cd bob
git config user.name "Bob"
git config user.email "bob@example.com"
git remote -v
git status -sb
git branch -vv
git log --oneline --graph --decorate --all -n 16

git fetch origin
git log --oneline --graph --decorate --all -n 16
git status -sb
```

继续让 `bob` 整合远程更新：

```bash
git pull
git branch -vv
git log --oneline --graph --decorate --all -n 16
```

再模拟 `bob` 推送任务分支：

```bash
git switch -c feature/bob-note
printf "bob note\n" >> app.txt
git add app.txt
git commit -m "add bob note"
git push -u origin feature/bob-note
git branch -vv
```

预期结果：

- `fetch` 后，`origin/main` 更新，但 `bob` 当前本地分支和工作区不会自动变成最新内容。
- `pull` 后，当前分支才整合上游变化。
- 第一次 `push -u` 后，`feature/bob-note` 会显示对应上游分支。
- 学习者能用历史图指出每一步移动的是哪一层状态。

实验清理：离开 `remote-lab` 后删除整个练习目录即可；不要在真实项目中直接删除未确认的仓库目录。

## 常见错误

- **把 `origin/main` 当成服务器实时状态。** 它只是上次 fetch 后保存在本地的远程跟踪分支快照。
- **把 `pull` 当成只下载。** 它会在 fetch 后整合当前分支，可能 merge、rebase 或冲突。
- **不看上游关系就 push。** 多远程或 fork 场景下，可能推到错误仓库或错误分支。
- **在脏工作区中 pull。** 本地未完成改动和远程整合冲突混在一起，会显著增加恢复难度。
- **push rejected 后立刻强推。** 这可能覆盖远程已有提交；下一章会专门处理 push rejected。
- **以为远程名一定叫 origin。** 远程名是本地约定，真正重要的是 URL 和团队工作流。

## 危险命令与恢复路径

> **风险提示：在不知道当前分支和上游关系时执行 `git push`，可能把提交推到错误远程或错误分支。** 这在 fork、多远程、同名分支很多的仓库里尤其常见。
>
> **恢复路径：** 推送前运行 `git remote -v`、`git branch -vv` 和 `git log --oneline --decorate -n 8`。如果误推到个人分支且未影响共享主干，可以删除错误远程分支或重新开正确 PR；如果误推到受保护或共享分支，立即停止继续 push，通知团队，并按团队治理规则创建修复或回滚提交。

> **风险提示：`git pull` 可能移动当前分支并触发 merge/rebase。** 如果工作区不干净，冲突和本地未提交改动会混在一起，难以判断哪些变化来自远程。
>
> **恢复路径：** pull 前先让 `git status -sb` 干净。merge pull 冲突且想回到 pull 前，可在未完成合并时使用 `git merge --abort`；rebase pull 冲突且无法判断时使用 `git rebase --abort`。如果 pull 已完成但结果不符合预期，先查看 `git reflog` 找到 pull 前位置，再进入恢复模块选择安全做法。

> **风险提示：不要用 `git push --force` 解决普通 push 失败。** 它会尝试改写远程分支历史，不会保护他人的新提交。
>
> **恢复路径：** 普通 push rejected 先 `git fetch origin` 并观察历史图；如果这是个人任务分支且团队允许改写，优先使用 `git push --force-with-lease`，并确认远程仍是你刚观察到的状态。共享分支默认不要强推。

## 验收

你应该能回答：

1. `origin`、`origin/main`、`main` 和远程服务器上的 `main` 分别是什么？
2. 为什么 `git fetch` 后当前工作区通常没有变化？
3. `git pull` 在默认 merge 模式下包含哪两步？如果配置为 rebase pull，哪一步会改变？
4. `git push -u origin feature/x` 建立的上游关系会影响哪些后续命令？
5. 在 fork 或多远程场景中，push 前必须检查哪两个输出？
6. pull 冲突时，`git merge --abort` 和 `git rebase --abort` 分别对应什么场景？

## 术语需求

- **远程名（remote name）**：本地仓库中指向远程 URL 的别名，例如 `origin`、`upstream`。
- **远程跟踪分支（remote-tracking branch）**：本地保存的远程分支快照，例如 `origin/main`；它会在 fetch 时更新。
- **上游分支（upstream branch）**：当前本地分支默认 pull/push 对应的远程分支。
- **快进（fast-forward）**：本地分支没有独立提交时，分支指针直接移动到后代提交的整合方式。

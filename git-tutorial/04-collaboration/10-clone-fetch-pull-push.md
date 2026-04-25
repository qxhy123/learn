# 10 clone、fetch、pull 与 push

## 场景

你加入一个已有项目，需要从远程仓库开始工作。你会执行 `clone`、`fetch`、`pull`、`push`，但真正重要的是知道每个命令改动哪一层状态：本地工作区、当前分支、远程跟踪分支，还是远程仓库。

## 学习目标

完成本章后，你应该能够：

1. 解释远程仓库、远程名和远程跟踪分支的区别。
2. 说明 `git fetch` 为什么是最安全的同步起点。
3. 拆解 `git pull` 通常包含的两个动作。
4. 在 push 前判断本地分支与远程分支的关系。

## 观察点

克隆后先观察：

```bash
git remote -v
git branch -vv
git status -sb
git log --oneline --graph --decorate --all -n 12
```

你要能指出：

- `origin` 指向哪个远程地址。
- 当前本地分支跟踪哪个上游分支。
- `origin/main` 是本地记录的远程主干快照，不是服务器上的分支本身。
- 当前工作区是否干净，能否安全同步。

## 命令

克隆仓库：

```bash
git clone <url> project
cd project
```

查看远程：

```bash
git remote -v
```

只更新远程跟踪分支：

```bash
git fetch origin
```

整合远程更新到当前分支：

```bash
git pull
```

默认情况下，`pull` 可以理解成：

```bash
git fetch
git merge
```

如果团队配置了 rebase pull，则第二步会变成 rebase。执行前应先确认团队默认值，而不是把 `pull` 当作“下载最新代码”的单一动作。

推送当前分支：

```bash
git push -u origin feature/login-copy
```

`-u` 会建立上游跟踪关系，让后续 `git branch -vv`、`git pull`、`git push` 能更清楚地显示默认目标。

## 实验

Lab ID：`LAB-COLLAB-REMOTE-01`

实验步骤：

1. 准备一个 bare 仓库作为远程，再 clone 两份本地工作副本。
2. 在第一份副本执行 `git remote -v` 与 `git branch -vv`。
3. 在第二份副本提交并 push。
4. 回到第一份副本，先看旧的 `origin/main`，再执行 `git fetch`。
5. 比较 fetch 前后的历史图，确认当前本地分支没有自动移动。

预期结果：学习者能解释 fetch 只更新远程跟踪分支，pull 才会进一步整合当前分支。

## 常见错误

- **把 `origin/main` 当成远程服务器实时状态。** 它只是上次 fetch 后保存在本地的快照。
- **把 `pull` 当成只下载。** 它通常还会 merge 或 rebase 当前分支。
- **不看上游关系就 push。** 可能推到错误远程或错误分支。
- **在脏工作区中同步。** 冲突与本地未提交改动混在一起，会显著增加恢复难度。

## 危险提示与恢复路径

危险动作：在不知道当前分支跟踪关系时执行 `git push`，尤其是在多个远程或 fork 场景中。

恢复路径：

1. 先运行 `git remote -v` 与 `git branch -vv` 确认默认目标。
2. 如果推错了个人分支但未影响共享主干，创建说明清楚的修正 PR 或删除错误远程分支。
3. 如果推到了受保护或共享分支，停止继续操作，通知团队并按治理规则恢复。

## 验收

你应该能回答：

1. `origin/main` 与 `main` 分别是什么？
2. `fetch` 后为什么当前工作区通常没有变化？
3. `pull` 为什么可能引入 merge commit 或触发 rebase？
4. `push -u` 建立的上游关系对后续协作有什么帮助？

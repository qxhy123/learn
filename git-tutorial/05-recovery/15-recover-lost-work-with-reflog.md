# 15 Recover Lost Work with Reflog

## 章节导航

- 前置章节：[`14 Fix a Bad Commit`](14-fix-a-bad-commit.md)
- 后续章节：[`16 Stash, Worktree and Interruptions`](../06-release-and-debugging/16-stash-worktree-and-interruptions.md)
- 本章 Lab ID：`LAB-RECOVERY-REFLOG-01`
- 本章定位：当提交、分支或改动“看不见了”时，先用本地引用移动记录定位线索，再用救援分支恢复。

## 场景

你 reset 过头、误删了本地分支、在游离 HEAD 上提交后切走，或者 rebase/amend 后发现少了内容。此时不要立刻断定“工作丢了”：很多时候提交对象仍在本地对象库里，只是没有分支名、标签或远端引用继续指向它。

`git reflog` 是处理这类事故的第一入口。它不是一份远端备份，而是“本机上的引用移动轨迹”：HEAD 去过哪里、分支曾指向哪里、reset/rebase/amend/checkout 什么时候移动过引用。恢复的关键不是猜一个 `HEAD@{n}` 直接 reset，而是先把候选位置命名成救援分支，再比较、验证、合回。

## 学习目标

- 理解 `git log` 和 `git reflog` 的差异：一个看可达历史，一个看本地引用移动轨迹。
- 从 reset、switch/checkout、commit、amend、rebase 记录中定位事故前后位置。
- 用救援分支保护疑似正确提交，再选择 merge、cherry-pick 或 reset 回到主线。
- 识别 reflog 的边界：本地、有限保留、受垃圾回收影响，不能替代远端推送和备份。
- 在恢复前建立“停止写入 → 保存现场 → 命名候选 → 比较验证 → 合回目标”的安全流程。

## 观察点

事故发生后先减少写入性操作，避免继续移动引用或覆盖线索。进入恢复前运行：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all -n 12
git reflog --date=relative -n 30
```

重点观察：

- `git status -sb`：当前是否还有未提交改动；如果有，先保护现场。
- `git branch -vv`：当前分支是否跟踪远端，后续是否可能影响共享历史。
- `git log --all`：所有“仍有名字指向”的提交里是否已经能看到目标内容。
- `git reflog`：HEAD 或当前分支最近经历了哪些移动。

阅读 reflog 时常见线索：

- `reset: moving to ...`：reset 记录附近通常能找到 reset 前的位置。
- `commit:`：可能指向在游离 HEAD 或临时分支上做过的提交。
- `checkout:` / `switch:`：能显示你从哪条分支或哪个提交切走。
- `commit (amend):`：能帮助找到 amend 前后的两个提交身份。
- `rebase ...`：能帮助定位 rebase 开始前、进行中、完成后的不同位置。

> 判断原则：reflog 条目是线索，不是结论。先把候选位置命名，再通过 `git show`、`git diff`、测试结果确认。

## 命令与判断

### 1. 先保存当前现场

如果当前状态不清楚，先给当前位置一个名字：

```bash
git branch rescue-current
```

如果工作区还有未提交内容，可以先做一个临时保护提交或 stash。初学阶段更推荐临时分支上的保护提交，因为它更容易被 `git log --all` 看见：

```bash
git switch -c rescue-current-wip
git add -A
git commit -m "rescue: preserve current work before reflog recovery"
```

只有在确认这些改动不需要进入正式历史后，后续再清理救援分支。

### 2. 从 reflog 命名候选位置

找到可疑条目后，不要马上让正式分支 reset 到它。先创建救援分支：

```bash
git switch -c rescue-lost-work HEAD@{3}
```

也可以直接使用具体提交：

```bash
git switch -c rescue-lost-work <commit-sha>
```

确认救援分支内容：

```bash
git status -sb
git show --stat --oneline HEAD
git log --oneline --decorate -n 5
```

如果不确定哪个候选正确，就给每个候选取不同名字，例如 `rescue-before-reset`、`rescue-detached-commit`、`rescue-before-rebase`，再比较它们。

### 3. 再决定如何回到主线

如果救援分支包含一组应回到目标分支的提交，并且保留历史上下文最重要，可以 merge：

```bash
git switch <target-branch>
git merge rescue-lost-work
```

如果只需要找回某一条补丁，可以 cherry-pick：

```bash
git switch <target-branch>
git cherry-pick <rescued-commit>
```

只有在目标分支未共享，或团队明确同意改写该分支历史时，才考虑把分支指针移回候选提交：

```bash
git switch <target-branch>
git reset --hard <rescued-commit>
```

## 危险命令与恢复路径

| 命令/动作 | 危险点 | 更安全的前置动作 | 如果做错了怎么恢复 |
|---|---|---|---|
| `git reset --hard HEAD@{n}` | 同时移动分支并丢弃工作区/暂存区改动；如果猜错 `HEAD@{n}`，会制造更多 reflog 噪音 | 先 `git branch rescue-current`，再 `git switch -c rescue-candidate HEAD@{n}` | 再查 `git reflog --date=relative -n 50`，为 reset 前位置创建救援分支 |
| `git branch -D <branch>` | 删除分支名；若提交没有其他引用指向，会变成难以发现的对象 | 删除前记录 `git log --oneline <branch> -n 3` 或先打临时标签/救援分支 | 用 `git reflog show <branch>`（若仍可查）或 HEAD reflog 找到最后提交，再建分支 |
| `git rebase` / `git commit --amend` | 改写提交身份，旧提交从常规历史上消失 | rebase/amend 前确认未共享；必要时 `git branch rescue-before-history-edit` | 用 reflog 找到 rebase/amend 前的提交，创建 `rescue-before-history-edit` |
| `git gc --prune=now` | 主动清理不可达对象，可能缩短找回窗口 | 恢复未完成前不要运行；不要把它当“修复 Git”的命令 | 若对象已被清理，reflog 也可能无法恢复，只能找远端、同事副本、备份或补丁文件 |

安全恢复路径固定为：

1. 停止继续 reset、rebase、amend、clean、gc 等写入性或清理性操作。
2. 保存当前现场：`git branch rescue-current`；必要时在临时分支提交 WIP。
3. 读取 `git reflog --date=relative -n 30`，记录 2-3 个候选位置。
4. 对每个候选创建救援分支，而不是直接移动正式分支。
5. 用 `git diff`、`git show`、测试结果确认哪个候选正确。
6. 根据是否共享历史，选择 merge、cherry-pick，或在未共享前提下 reset。
7. 恢复完成后清理救援分支，并把真正重要的结果推送到远端。

## 实验

Lab：`LAB-RECOVERY-REFLOG-01`

实验目标：制造三种“看起来丢了”的本地工作，并用同一套救援流程找回。

1. **误 reset 找回提交**
   - 做一次提交，记下 `git log --oneline -n 3`。
   - 执行 `git reset --hard HEAD~1`，确认普通 `git log` 看不到刚才提交。
   - 用 `git reflog` 找到 reset 前位置，创建 `rescue-reset`。
   - 比较 `rescue-reset` 与当前分支，选择 merge 或 cherry-pick 找回内容。
2. **游离 HEAD 提交找回**
   - 切到历史提交形成 detached HEAD。
   - 做一次新提交，再切回主分支。
   - 用 reflog 找到那条 detached commit，创建 `rescue-detached`。
   - 解释为什么“没有分支名”不等于“提交立刻消失”。
3. **历史编辑前状态找回**
   - 在本地分支上做 amend 或 rebase。
   - 用 reflog 找到编辑前位置，创建 `rescue-before-edit`。
   - 比较编辑前后提交 SHA 和补丁差异。
4. **边界观察**
   - 删除一个本地分支，观察 reflog、`git log --all` 与可达性的关系。
   - 只讨论 `git fsck --lost-found` 的作用，不把它当首选恢复流程。

> Labs agent 需要为本章提供准备、执行、观察、恢复、清理步骤；章节 agent 不修改 `labs/**`。

## 常见错误

- 把 `git log` 看不到理解为提交已经永久删除；`git log` 默认只看当前可达历史。
- 直接在正式分支上多次 `reset --hard HEAD@{n}`，没有先建立救援分支。
- 忘记 reflog 主要是本地记录，不能指望同事机器、CI 或远端有同样轨迹。
- 事故后继续运行大量 rebase、amend、reset、gc，导致线索更难读或对象被清理。
- 把 `HEAD@{n}` 当成稳定编号写进文档；它会随着新的引用移动而变化。
- 找回内容后不推送、不打标签、不合回主线，导致下次清理时再次丢失上下文。

## 术语需求

- `reflog`：本地引用移动日志，不是项目共享历史。
- `reachable / unreachable object`：是否能从分支、标签、HEAD 等引用追溯到对象。
- `detached HEAD`：HEAD 直接指向提交而不是分支名的状态。
- `rescue branch`：恢复时为候选提交创建的临时命名分支。

## 验收

完成本章后，你应该能在不查命令速查表的情况下完成以下验收：

- 给定 `git log` 看不到某提交的案例，解释为什么应继续查看 `git reflog`。
- 从一段 reflog 中指出 reset、amend、switch/rebase 相关线索，并选择 1-2 个候选位置。
- 使用 `git switch -c rescue-lost-work HEAD@{n}` 或具体 SHA 创建救援分支。
- 比较救援分支与目标分支，并说明为什么先命名候选比直接 reset 更安全。
- 根据“是否已共享”选择 merge、cherry-pick 或 reset，而不是默认改写历史。
- 说出 reflog 的两个边界：本地记录、有限保留；重要工作仍要及时提交、推送或备份。

# 15 Recover Lost Work with Reflog

## 场景

你 reset 过头、删除了分支、在游离 HEAD 上提交后切走，或者 rebase 后发现少了内容。当前分支上看不见的提交不一定真正丢失；它可能只是没有名字指向它。`reflog` 是恢复这类事故的第一入口。

## 学习目标

- 理解 `git reflog` 记录的是本地引用移动轨迹。
- 从 reset、checkout/switch、amend、rebase 记录中定位事故前位置。
- 用救援分支保护疑似正确提交，再恢复正式分支。
- 知道 reflog 的边界：本地、有限保留、不能替代备份和远端协作。

## 观察点

事故发生后先减少动作，避免继续覆盖线索。运行：

```bash
git status -sb
git log --oneline --graph --decorate --all -n 12
git reflog --date=relative -n 30
```

阅读 reflog 时关注：

- `reset: moving to ...` 前一条通常是 reset 前位置。
- `commit:` 记录可能指向看似丢失的新提交。
- `checkout` 或 `switch` 记录能显示你从哪里切走。
- `rebase`、`amend` 记录能帮助找到编辑历史前的提交。

## 恢复流程

### 先建立救援分支

找到可疑位置后，不要马上 reset 正式分支。先保住它：

```bash
git switch -c rescue-lost-work <commit-or-HEAD@{n}>
```

确认内容：

```bash
git show --stat
git log --oneline --decorate -n 5
```

### 再决定如何回到主线

如果救援分支就是你要的状态，可以选择：

```bash
git switch <target-branch>
git merge rescue-lost-work
```

如果只是找回某一条补丁，可以选择：

```bash
git switch <target-branch>
git cherry-pick <rescued-commit>
```

只有在确认目标分支未共享或团队同意后，才考虑：

```bash
git reset --hard <rescued-commit>
```

## 风险提示

恢复时最危险的习惯是连续猜命令，例如反复 `reset --hard HEAD@{n}`。每一次移动分支都会新增 reflog 记录，也可能让你更难判断事故前位置。

更安全的恢复路径是：

1. 停止写入性操作。
2. 保存当前现场：`git branch rescue-current`。
3. 从 reflog 为每个候选位置创建命名分支。
4. 用 `git diff`、`git show`、测试结果判断哪个分支正确。
5. 最后再把正确内容合回目标分支。

## 实验

Lab：`LAB-RECOVERY-REFLOG-01`

1. 做一次提交后执行 `git reset --hard HEAD~1`，用 reflog 找回提交并创建 `rescue-reset`。
2. 切到历史提交形成游离 HEAD，做一次新提交，再切回主分支；用 reflog 找到该提交并创建分支。
3. 创建并删除一个本地分支；用 reflog 或 `git fsck --lost-found` 的提示理解引用丢失和对象保留的区别。
4. 对每次恢复都先建救援分支，再选择 merge、cherry-pick 或 reset。

## 常见错误

- 把 `git log` 看不到理解为提交已经永久删除。
- 直接在正式分支上 reset 到猜测位置，没有建立救援分支。
- 忘记 reflog 主要是本地记录，不能指望同事机器上也有同样轨迹。
- 事故后继续运行大量历史编辑命令，增加恢复难度。

## 验收

你应该能解释并演示：

- `git log` 和 `git reflog` 分别回答什么问题。
- 如何从 `HEAD@{n}` 创建救援分支。
- 为什么恢复时先命名候选位置比直接 reset 更安全。
- reflog 不能保证永久找回所有对象，重要工作仍要及时提交和推送。

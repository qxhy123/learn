# 16 Stash, Worktree and Interruptions

## 场景

你正在做一半功能，线上突然需要 hotfix；或者你要临时切到另一条分支验证问题，但当前工作区还不适合提交。`stash`、临时分支和 `worktree` 都能处理中断，但适用时间尺度不同。

## 学习目标

- 判断当前改动适合 stash、提交到临时分支，还是放进独立 worktree。
- 使用带说明的 `git stash push` 保存短期 WIP。
- 处理 `stash apply` 或 `stash pop` 产生的冲突。
- 理解 `git clean` 删除未跟踪文件的风险，并坚持 dry-run。

## 观察点

中断前先确认：

```bash
git status -sb
git diff --stat
git diff --cached --stat
git stash list
```

判断维度：

- 改动是否已经能形成一个可解释的小提交？能提交就不要 stash。
- 中断预计是几分钟、几小时还是多天？时间越长越不适合 stash。
- 是否需要两个任务同时打开、同时跑测试？是的话优先 worktree。
- 是否有未跟踪文件？stash 默认不一定包含它们。

## 命令与判断

### 短期收纳

```bash
git stash push -m "wip: describe current interruption"
git stash list
git stash show --stat stash@{0}
```

如果要连未跟踪文件一起收纳：

```bash
git stash push -u -m "wip: include untracked notes"
```

### 恢复工作

```bash
git stash apply stash@{0}
# 确认恢复无误后再删除
git stash drop stash@{0}
```

`git stash pop` 等于 apply 后尝试 drop。初学阶段建议先 apply，再手动 drop，避免冲突时误判状态。

### 长期并行

```bash
git worktree add ../project-hotfix main
```

worktree 适合“当前任务不能收起来，但还要同时处理另一条线”的场景。

### 清理未跟踪文件

```bash
git clean -nd
git clean -fd
```

永远先 dry-run。未跟踪文件通常没有进入 Git 历史，被 clean 删除后不一定能恢复。

## 风险提示

- `git stash pop` 在冲突时不会让问题自动消失；你仍需用 `git status` 解决冲突并确认 stash 是否保留。
- `git clean -fd` 会删除未跟踪文件和目录；对实验输出、临时脚本和本地配置尤其危险。
- 长期堆积 stash 会让上下文消失，最后变成一组难以恢复的匿名补丁。

恢复路径：

```bash
git stash list
git stash show -p stash@{0} > ../stash-rescue.patch
git branch rescue-before-clean
```

对 clean，只有 dry-run 输出确认无误后才执行；对重要未跟踪内容，先移动到安全目录或提交到临时分支。

## 实验

Lab：`LAB-RELEASE-STASH-WORKTREE-01`

1. 制造一组未提交改动，用 `stash push -m` 保存，切分支后再 apply。
2. 在目标分支修改同一行，应用 stash 制造冲突；用 `git status` 和编辑器解决。
3. 创建未跟踪目录，先运行 `git clean -nd`，记录将删除内容，再决定是否 `git clean -fd`。
4. 用 `git worktree add` 创建并行工作目录，比较它和 stash 在中断处理上的差异。

## 常见错误

- 把 stash 当成长期任务列表，不写说明也不清理。
- 认为 stash 一定不会冲突。
- 用 `git clean -fd` 清理前没有 dry-run。
- 明明改动已经完整，却为了“保持干净”放进 stash，而不是提交到临时分支。

## 验收

你应该能为三个中断案例选择工具：

- 五分钟内切走看一眼：stash。
- 一个完整小修复：提交到分支。
- 两个任务都要持续编译测试：worktree。

同时能解释 `stash apply`、`stash pop`、`clean -nd`、`clean -fd` 的风险差异。

# 08 用 playbook 解决合并冲突

## 场景

你在 `feature/profile-copy` 上完成了个人资料页文案，准备合并回 `main`。同时，同事已经在 `main` 上修改了同一段文案。Git 能判断两个分支从哪个共同祖先分开，也能指出同一位置出现了两套结果，但它不能替团队决定“最终业务语义应该是什么”。

本章把合并冲突当成一个可控的人工判断点，而不是仓库损坏。你会练习一套固定 playbook：先确认状态，再阅读双方意图，然后编辑最终结果，最后用暂存区和历史图证明合并真的完成。

## 前置与后续

- 前置章节：`03-branching-work/07-branch-for-a-task.md`，你应该已经能创建任务分支、保持主干干净，并用历史图解释分支从哪里分出。
- 后续章节：`03-branching-work/09-rebase-without-fear.md`，下一章会讨论何时用 rebase 整理本地历史，以及为什么不要在共享历史上随意 rebase。
- 本章 Lab ID：`LAB-BRANCH-CONFLICT-01`。

## 学习目标

完成本章后，你应该能够：

1. 区分快进合并、普通三方合并和冲突合并。
2. 在冲突发生时识别工作区、暂存区和合并状态分别发生了什么。
3. 读懂 `<<<<<<<`、`=======`、`>>>>>>>` 标记背后的“当前分支”和“被合并分支”。
4. 按固定顺序解决冲突，而不是凭感觉删除标记或整文件覆盖。
5. 在无法判断业务语义时安全中止合并，并知道已经提交错误合并后的恢复方向。

## 观察点

合并前先观察三件事：工作区是否干净、当前在哪个分支、两个分支的历史关系。

```bash
git status -sb
git branch --show-current
git log --oneline --graph --decorate --all -n 12
```

你要能回答：

- 目标分支是否干净？如果有未提交改动，先提交、暂存或放弃，不要带着脏工作区合并。
- 被合并分支从哪里分出？历史图是否显示双方都有新提交？
- 这次合并可能是快进、普通 merge commit，还是有较高冲突概率？

冲突发生后立即观察：

```bash
git status
git diff --name-only --diff-filter=U
git diff
```

重点看：

- `git status` 是否提示 “unmerged paths”。这表示仓库处于合并中间状态。
- `git diff --name-only --diff-filter=U` 列出的文件就是还没解决的冲突文件。
- `git diff` 展示的是你尚未暂存的最终编辑结果，而不是完整历史解释；解决前后都要看。

## 合并冲突 playbook

从主干合并任务分支的常见流程如下：

```bash
git switch main
git status -sb
git merge feature/profile-copy
```

如果 Git 报告冲突，按这个顺序处理：

1. **暂停操作。** 不要继续切分支、不要急着提交、不要运行清理命令。
2. **列出冲突文件。**
   ```bash
   git status
   git diff --name-only --diff-filter=U
   ```
3. **阅读冲突块。** 打开每个冲突文件，找到 `<<<<<<< HEAD`、`=======`、`>>>>>>> feature/profile-copy`。
4. **判断双方意图。** 不是问“保留哪边”，而是问“最终内容如何同时满足两个改动背后的需求”。
5. **编辑最终结果。** 删除冲突标记，保留一份能运行、能解释、符合业务语义的内容。
6. **复查 diff。**
   ```bash
   git diff
   ```
7. **暂存已解决文件。**
   ```bash
   git add <resolved-file>
   git status
   ```
8. **完成合并提交。** 如果所有冲突都解决，运行：
   ```bash
   git commit
   ```
9. **验证历史图。**
   ```bash
   git log --oneline --graph --decorate --all -n 12
   ```

完成后你应该能指出 merge commit 的两个父提交：一个来自当前 `main`，一个来自被合并的任务分支。

## 冲突文件如何阅读

典型冲突块如下：

```text
<<<<<<< HEAD
显示给团队内部用户的个人资料说明
=======
显示给新注册用户的个人资料引导
>>>>>>> feature/profile-copy
```

- `<<<<<<< HEAD` 到 `=======` 之间：当前分支，也就是你执行 `git merge` 时所在分支的内容。
- `=======` 到 `>>>>>>> feature/profile-copy` 之间：被合并分支的内容。
- 冲突标记本身不是代码或正文，最终文件里不能留下这些标记。

解决时用四个问题检查自己：

1. 当前分支为什么这样改？
2. 被合并分支为什么那样改？
3. 最终结果是否同时满足两个改动背后的需求？
4. 是否需要补充测试、截图、文档或产品确认来证明这个结果正确？

如果第 3 个问题无法回答，不要假装解决。合并冲突的核心是语义判断，不是编辑器技巧。

## 实验

Lab ID：`LAB-BRANCH-CONFLICT-01`

实验目标：制造一个同一文件同一行的冲突，按 playbook 完成解决，并解释合并提交的两个父提交。

建议步骤：

1. 按 `labs/scenarios/LAB-BRANCH-CONFLICT-01.md` 准备练习仓库。
2. 从同一个提交创建 `feature/a` 和 `feature/b`。
3. 让两个分支修改同一文件同一行，但表达不同业务意图。
4. 回到 `main`，先合并 `feature/a`，再合并 `feature/b`。
5. 冲突发生后，记录：
   ```bash
   git status
   git diff --name-only --diff-filter=U
   ```
6. 编辑成一个同时表达两边意图的最终结果。
7. 运行 `git diff` 复查，确认没有冲突标记。
8. `git add` 已解决文件，再次运行 `git status`。
9. 完成 `git commit`，用历史图解释 merge commit 的两个父提交。

预期结果：你能说清楚“冲突是人工判断点”，能识别合并未完成时 Git 的保护状态，并能展示最终文件没有冲突标记。

## 常见错误

- **把冲突当成仓库损坏。** 冲突只是 Git 无法自动决定业务结果，仓库仍处于可恢复状态。
- **只删除冲突标记。** 删除标记不等于语义正确，最终内容必须能运行、能解释。
- **没看 `git status` 就提交。** 可能还有未解决文件，或者你并没有真正暂存最终结果。
- **用整文件覆盖解决冲突。** 这很容易把另一边的有效改动一并删除。
- **在冲突中继续切换任务。** 合并中间状态会影响工作区判断，先完成或中止当前合并。
- **把工具按钮当判断。** 编辑器的“接受当前/接受传入/接受全部”只是编辑动作，不是业务决策。

## 危险命令

以下命令和按钮不是禁止使用，但必须先确认你理解后果：

```bash
git checkout --ours <file>
git checkout --theirs <file>
git restore --source=HEAD -- <file>
git merge --abort
git reset --hard
```

- `git checkout --ours <file>` 会把冲突文件恢复成当前分支一侧，可能丢掉被合并分支的有效改动。
- `git checkout --theirs <file>` 会把冲突文件恢复成被合并分支一侧，可能丢掉当前分支的有效改动。
- `git restore --source=HEAD -- <file>` 可能让你误以为“恢复了文件”，实际是绕过了合并语义。
- `git merge --abort` 用于放弃这次合并；它是安全退出，不是解决冲突。
- `git reset --hard` 会丢弃工作区和暂存区改动；除非你已经确认没有要保留的本地内容，否则不要用它处理冲突。

## 恢复路径

按当前阶段选择恢复方式：

1. **冲突刚发生，还没编辑。** 运行 `git merge --abort` 回到合并前状态，然后重新观察历史和工作区。
2. **已经编辑但还没 `git add`。** 继续编辑成正确结果；如果想重来，优先用 `git merge --abort`，不要直接 `reset --hard`。
3. **已经 `git add` 但尚未提交。** 继续编辑文件，重新 `git add <resolved-file>`，再用 `git status` 确认所有冲突都已解决。
4. **已经提交了错误合并，但还没共享。** 可以在确认团队规则后重做本地历史；下一章和恢复模块会讨论边界。
5. **已经把错误合并推送到共享主干。** 不要直接重写共享历史。后续恢复模块会用 `revert -m` 处理合并提交回滚。

## 验收

你应该能完成以下检查：

1. 解释快进合并为什么不会产生新的 merge commit。
2. 解释冲突发生时，工作区、暂存区和合并状态为什么会处于中间状态。
3. 给定一个冲突块，指出哪一段来自当前分支，哪一段来自被合并分支。
4. 说出 `git merge --abort` 适合在什么时机使用，以及它不适合替代什么判断。
5. 解决一个冲突后，用 `git status`、`git diff`、`git log --graph` 证明合并已经完成。
6. 说明为什么解决冲突的核心不是“选一边”，而是“构造正确最终结果”。

## 交付自查

- 唯一修改范围：`03-branching-work/08-merge-conflicts-with-a-playbook.md`。
- Lab ID：`LAB-BRANCH-CONFLICT-01`。
- 前置章节：`03-branching-work/07-branch-for-a-task.md`。
- 后续章节：`03-branching-work/09-rebase-without-fear.md`。
- 危险命令：`git checkout --ours`、`git checkout --theirs`、`git restore --source=HEAD -- <file>`、`git merge --abort`、`git reset --hard`。
- 恢复路径：按“未编辑 / 已编辑未暂存 / 已暂存未提交 / 已提交未共享 / 已推送共享”分阶段处理。
- 需要后置集成：Labs agent 确认 `LAB-BRANCH-CONFLICT-01` 场景覆盖同一行冲突、观察、恢复、清理；Module/README agent 统一维护导航链接。

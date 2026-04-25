# commit 设计与 diff review

## 导航与契约

- 前置章节：[第一次干净提交](04-first-change-to-clean-commit.md)
- 后续章节：[忽略文件与仓库卫生](06-ignore-files-and-repo-hygiene.md)
- 本章 Lab id：`LAB-DAILY-DIFF-REVIEW-01`
- 危险命令：`git reset --hard`、`git checkout -- <path>`、`git restore <path>`、`git commit --amend`、`git push --force`
- 恢复路径：先停止继续暂存或提交，保存 `git status --short --branch`、`git diff`、`git diff --cached`；误暂存用 `git restore --staged <path>`；拆错提交但未共享时优先用新分支或下一章的修正流程整理；已经共享的提交不要 amend/force push，优先追加修复提交或和团队确认。

## 场景

你准备提交一个看似简单的改动：修了一个配置 bug、顺手重命名变量、又让编辑器格式化了整个文件。一次性提交也许能通过测试，但 reviewer 需要在大量无关 diff 里寻找真正的行为变化；将来回滚时，也很难只撤销 bug 修复而保留重命名或格式化。

本章要练的不是“把提交切得越碎越好”，而是先读懂 diff，再用提交边界表达意图。好的提交应该让 reviewer 能快速回答：这次为什么改、改了什么行为、能否独立验证、如果出问题能否单独回滚。

## 学习目标

完成本章后，你应该能够：

- 用 `git diff --stat` 先判断变更范围，再用完整 diff 审查关键内容。
- 区分工作区 diff 与暂存区 diff，并把 `git diff --cached` 当作提交前最后 review。
- 判断一次提交是否只有一个主要意图，而不是机械按文件数量拆分。
- 使用按文件暂存、`git add -p`、`git restore --staged <path>` 调整暂存区。
- 在拆分失败、误暂存、误提交时选择保留工作的恢复路径。

## 观察点

拆分提交前先观察四个视角：

```bash
git status --short --branch
git diff --stat
git diff
git diff --cached --stat
```

你要能回答：

- 当前变更涉及哪些文件？是否出现临时文件、生成物或无关格式化？
- 每个文件变化服务于同一个意图，还是混合了多个意图？
- 暂存区目前是否为空？如果不为空，它是否只包含下一次提交要表达的内容？

提交前再观察暂存区本身：

```bash
git diff --cached
git status --short --branch
```

提交后复查历史：

```bash
git log --oneline --decorate --max-count=3
git show --stat HEAD
```

你要确认 `HEAD` 的最新提交只包含刚刚设计好的那一个意图，而剩余工作区改动仍然留在工作区或下一次暂存中。

## 命令与解释

最常见的提交设计流程如下：

```bash
git status --short --branch
git diff --stat
git diff

git add path/to/first-intent-file
git diff --cached
git commit -m "Fix parser fallback for empty input"

git status --short --branch
git add path/to/second-intent-file
git diff --cached
git commit -m "Rename parser option for clarity"
```

关键命令含义：

- `git diff --stat`：快速看文件范围和增删规模，适合先发现“这次是不是太大了”。
- `git diff`：查看工作区尚未暂存的细节，帮助你决定哪些内容属于同一个提交。
- `git add <path>`：按文件暂存；当一个文件只服务于一个意图时，这是最稳定、最容易解释的方式。
- `git add -p`：按 hunk 交互式暂存，适合同一文件里混合多个意图的情况。使用前必须读懂每个 hunk，不要为了拆分而盲选 `y` 或 `n`。
- `git restore --staged <path>`：把误放入暂存区的内容拿回工作区，不丢失文件内容。
- `git diff --cached`：提交前最终确认，它显示的就是下一次提交会记录的内容。

提交边界的判断可以用三个问题：

1. **意图是否单一？** 例如“修复空输入崩溃”是一个意图；“修复崩溃 + 全文件格式化 + 改 README”通常不是。
2. **能否独立验证？** 如果这个提交单独存在，测试或人工 review 是否能判断它正确？
3. **回滚是否合理？** 如果这个提交出问题，撤销它是否会同时撤掉不该撤的重命名、文档或格式化？

也不要走向另一个极端：一个功能改动可能必须同时修改代码、测试和文档。只要它们共同服务于同一个可解释意图，放在一次提交里通常比拆成三个无法独立工作的提交更好。

## 实验

**Lab id：`LAB-DAILY-DIFF-REVIEW-01`**

目标：在一次可丢弃的实验仓库里制造两个意图，先观察混合 diff，再拆成两个可 review 的提交。

### 准备

```bash
mkdir diff-review-lab
cd diff-review-lab
git init
git config user.name "Git Learner"
git config user.email "learner@example.com"
printf "name=demo\ncount=1\n" > app.conf
printf "# Demo\n" > README.md
git add app.conf README.md
git commit -m "Seed demo project"
```

### 执行

制造两个不同意图：一个是配置命名变化，一个是文档补充。

```bash
printf "name=demo-app\ncount=1\n" > app.conf
printf "# Demo\n\nLocal notes for reviewers.\n" > README.md

git status --short --branch
git diff --stat
git diff
```

先提交配置命名变化：

```bash
git add app.conf
git diff --cached
git commit -m "Rename demo app config"
```

再提交文档补充：

```bash
git status --short --branch
git add README.md
git diff --cached
git commit -m "Document reviewer notes"
```

复查历史：

```bash
git log --oneline --decorate --max-count=3
git show --stat HEAD~1
git show --stat HEAD
```

### 预期观察

- 初始 `git diff --stat` 同时显示 `app.conf` 和 `README.md`，说明当前工作区混合了两个意图。
- 第一次 `git diff --cached` 只包含 `app.conf` 的 `name=demo` 到 `name=demo-app` 变化。
- 第一次提交后，`git status --short --branch` 仍显示 `README.md` 有未暂存或未提交变化，因为文档意图还没提交。
- 第二次 `git diff --cached` 只包含 README 的 reviewer notes。
- `git log --oneline` 中两个提交的信息能分别解释配置命名和文档补充。

### 误暂存恢复练习

如果你不小心把两个文件都暂存了，不要用危险命令清空现场。运行：

```bash
git restore --staged README.md
git diff --cached
git diff README.md
```

预期结果：`README.md` 的内容仍在工作区，暂存区只剩 `app.conf`。

### 清理

确认你位于实验目录外层后删除可丢弃仓库：

```bash
cd ..
rm -rf diff-review-lab
```

如果你不确定当前位置，先运行 `pwd` 和 `git status --short --branch`，不要直接删除。

## 常见错误

- **只看 `git diff` 不看 `git diff --cached`**：工作区 diff 不等于下一次提交内容；暂存过的内容只会出现在 cached diff 中。
- **把“文件不同”误认为“意图不同”**：一个功能可能需要同时改代码、测试和文档；按意图拆分，不是按文件机械拆分。
- **把“同一文件”误认为“同一意图”**：同一个文件里可能同时有 bug 修复、重命名和格式化，这时可以考虑 `git add -p` 或先手动整理。
- **盲用 `git add -p`**：不理解 hunk 就选择，会制造无法编译或无法解释的中间提交。
- **提交信息只描述操作**：`Update files`、`Change config` 很难帮助 reviewer；优先写结果和原因，例如 `Rename demo app config`。
- **为了拆分而破坏可验证性**：如果一个提交单独 checkout 后无法构建、无法测试、无法解释，就需要重新考虑边界。

## 危险命令与恢复路径

本章的核心危险不是某个必用命令，而是在 diff 混乱时急着“清场”。以下命令不要作为拆分提交的默认工具：

- `git reset --hard`：会丢弃已跟踪文件在工作区和暂存区里的本地修改，可能把尚未提交的工作直接抹掉。
- `git checkout -- <path>` 或 `git restore <path>`：会丢弃指定文件的工作区修改；只有确认该文件内容可丢弃时才使用。
- `git commit --amend`：会重写最近一次提交；只适合未共享的本地提交修正。
- `git push --force`：会重写远端历史；在共享分支上可能覆盖队友工作。

安全恢复顺序：

1. 停止继续 `add`、`commit` 或清理，先运行 `git status --short --branch`。
2. 用 `git diff` 和 `git diff --cached` 分别确认内容在工作区还是暂存区。
3. 只是误暂存：运行 `git restore --staged <path>`，内容会留在工作区。
4. `git add -p` 拆错 hunk：先 `git restore --staged <path>` 撤回该文件暂存，再重新选择；如果工作区也被你改乱，先复制需要保留的片段。
5. 已经提交但还没有共享：不要立刻 `reset --hard`；可在后续章节学习用 amend、reset 或新分支整理，并先记录 `git log --oneline --decorate --max-count=5`。
6. 已经 push 或进入 PR：优先追加修复提交；若确实要重写历史，先和团队确认规则并使用更安全的 `--force-with-lease`，不要直接 `--force`。

## 验收

请用练习仓库或真实小改动交付以下证据：

1. 本章 Lab id：`LAB-DAILY-DIFF-REVIEW-01`。
2. 前置章节：[第一次干净提交](04-first-change-to-clean-commit.md)；后续章节：[忽略文件与仓库卫生](06-ignore-files-and-repo-hygiene.md)。
3. 提交前的 `git diff --stat`，说明当前变更为什么包含一个或多个意图。
4. 每次提交前的 `git diff --cached`，证明暂存区只包含该提交的意图。
5. `git log --oneline --decorate --max-count=3`，证明两个提交信息可区分、可 review。
6. 能说出本章危险命令：`git reset --hard`、`git checkout -- <path>`/`git restore <path>`、`git commit --amend`、`git push --force`。
7. 能演示或解释误暂存的恢复路径：`git restore --staged <path>` 保留工作区内容，只调整 index。

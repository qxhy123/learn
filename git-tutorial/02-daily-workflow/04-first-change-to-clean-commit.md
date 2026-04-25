# 第一次干净提交

## 导航与契约

- 前置章节：[读懂历史，不迷路](../01-mental-model/03-read-history-with-confidence.md)
- 后续章节：[commit 设计与 diff review](05-commit-design-and-diff-review.md)
- 本章 Lab id：`LAB-DAILY-CLEAN-COMMIT-01`
- 危险命令：`git reset --hard`、`git clean -fd`、`git commit --amend`、`git push --force`
- 恢复路径：先停止继续提交，保留 `git status --short --branch`、`git diff`、`git diff --cached` 输出；若只是暂存错了，用 `git restore --staged <path>`；若工作区改坏了，先复制需要保留的内容再 `git restore <path>`；若已经提交错了，在未共享时用下一章的拆分/修正方法，已共享时优先新增修复提交。

## 场景

你刚完成一个小改动：也许是改 README 的一句说明，也许是修正文档里的一个错字。现在你要把它变成一次可以被队友 review、可以被回滚、也可以被未来的自己读懂的提交。

本章的关键不是“把 `git commit` 跑成功”，而是练会一条稳定流程：先确认仓库状态，再只暂存同一个意图的文件，提交前 review 暂存区，最后用清晰的提交信息记录这次改变为什么存在。

## 学习目标

完成本章后，你应该能够：

- 按“观察 → 修改 → 暂存 → review → 提交 → 再观察”的顺序完成一次提交。
- 区分 `git diff` 与 `git diff --cached`：前者看工作区中尚未暂存的变化，后者看即将进入提交的变化。
- 写出说明意图的提交信息，而不是只写 `update`、`fix` 这类模糊词。
- 在发现改动混杂多个意图时停下来，拆成更小、更容易 review 的提交。
- 遇到误暂存、误修改、提交身份缺失等常见问题时，选择不会丢工作的恢复路径。

## 观察点

提交前先观察三件事：

```bash
git status --short --branch
git diff
git diff --cached
```

你要能回答：

- 当前在哪个分支？工作区是否 clean？
- 哪些变化还没有暂存？
- 暂存区里是否只包含这次提交想表达的一个意图？

提交后再观察三件事：

```bash
git status --short --branch
git log --oneline --decorate --max-count=3
git show --stat HEAD
```

你要确认：

- 工作区是否回到 clean。
- `HEAD` 是否指向刚创建的新提交。
- 新提交的文件列表和行数是否符合预期，没有混入调试文件、临时文件或无关格式化。

## 命令与解释

最小干净提交通常长这样：

```bash
git status --short --branch
git diff
git add README.md
git diff --cached
git commit -m "Clarify local setup steps"
git status --short --branch
git show --stat HEAD
```

逐步理解这些命令：

- `git status --short --branch`：用最紧凑的格式确认分支、工作区和暂存区状态。
- `git diff`：查看“还在工作区、尚未暂存”的改动。暂存前它应该显示你准备提交的内容；暂存后如果同一文件没有继续修改，它通常会变空。
- `git add README.md`：把当前版本的 `README.md` 放入暂存区。它不是“登记文件名”，而是把这一刻的文件内容快照放进 index。
- `git diff --cached`：提交前最重要的 review 视图，只显示即将进入下一次提交的内容。
- `git commit -m "..."`：把暂存区记录成一个新提交，并让当前分支前进到这个提交。
- `git show --stat HEAD`：用统计视角复查刚才提交了哪些文件。

提交信息建议使用能说明结果的动词短语，例如：

```text
Clarify local setup steps
```

它比下面的信息更适合给队友和未来的自己阅读：

```text
update README
```

如果 `git diff --cached` 里同时出现“修 bug”“重排格式”“改文档”三类变化，先不要提交。把它们拆开，或至少重新暂存成一个更清楚的提交边界。

## 实验

**Lab id：`LAB-DAILY-CLEAN-COMMIT-01`**

目标：在一次可丢弃的实验仓库里完成最小干净提交，并证明提交前后状态变化符合预期。

### 准备

```bash
mkdir clean-commit-lab
cd clean-commit-lab
git init
git config user.name "Git Learner"
git config user.email "learner@example.com"
printf "# Demo\n" > README.md
```

### 执行

```bash
git status --short --branch
git diff
git add README.md
git diff --cached
git commit -m "Create demo readme"
git status --short --branch
git log --oneline --decorate --max-count=1
git show --stat HEAD
```

### 预期观察

- `git status --short --branch` 在暂存前显示 `?? README.md`。
- `git diff` 对未跟踪文件通常不显示内容；如果你想看新文件内容，需要先暂存后看 `git diff --cached`。
- `git diff --cached` 显示 `README.md` 将被新增，并包含 `# Demo`。
- 提交后工作区回到 clean，`HEAD` 指向 `Create demo readme` 这个提交。
- `git show --stat HEAD` 只列出 `README.md`，没有其它临时文件。

### 清理

确认你位于实验目录外层后删除可丢弃仓库：

```bash
cd ..
rm -rf clean-commit-lab
```

如果你不确定当前位置，先运行 `pwd` 和 `git status --short --branch`，不要直接删除。

## 常见错误

- **先提交再 review**：提交前不看 `git diff --cached`，容易把调试代码、临时文件或无关格式化带入历史。
- **以为 `git add` 只记录文件名**：`git add` 记录的是当时的内容快照；暂存后继续修改同一文件，会出现同一文件既有 staged 又有 unstaged 的状态。
- **提交信息只写 `update`**：未来读历史时看不出意图，也很难判断能否安全回滚。
- **一次提交混多个意图**：例如同时修 bug、格式化全文件、改文档，会增加 review 和回滚成本。
- **在真实仓库里急着用危险命令清场**：`git reset --hard` 和 `git clean -fd` 都可能丢失未保存工作；使用前必须确认可丢弃，并保留必要备份。
- **提交身份未配置**：如果 `git commit` 提示缺少 `user.name` 或 `user.email`，先用仓库级配置补齐，再重新提交，不要改全局配置污染其它练习环境。

## 危险命令与恢复路径

本章不需要用危险命令完成实验，但你应该知道它们为什么危险：

- `git reset --hard`：丢弃已跟踪文件在工作区和暂存区里的本地变化。
- `git clean -fd`：删除未跟踪文件和目录，常见于误删新建文件。
- `git commit --amend`：重写最近一次提交；如果该提交已经共享，会影响协作者。
- `git push --force`：重写远端历史；除非团队规则明确允许，否则不要在共享分支使用。

安全恢复顺序：

1. 停止继续提交或清理，先运行 `git status --short --branch`。
2. 用 `git diff` 和 `git diff --cached` 判断内容分别在工作区还是暂存区。
3. 误暂存但内容正确：运行 `git restore --staged <path>`，只把内容从暂存区拿回工作区。
4. 工作区内容改坏但还没有提交：先复制需要保留的片段，再对单个文件运行 `git restore <path>`。
5. 已经提交错了：如果未共享，留到下一章学习如何修正；如果已共享，优先新增一个修复提交，避免重写别人可能已经基于其工作的历史。

## 验收

请完成一次干净提交，并能交付以下证据：

1. 提交前的 `git status --short --branch`，说明当前分支和文件状态。
2. 提交前的 `git diff --cached`，证明暂存区只包含一个意图。
3. 提交后的 `git show --stat HEAD`，证明新提交只包含预期文件。
4. 一句话解释提交信息为什么描述了“结果/意图”，而不是只描述“改了文件”。
5. 遇到误暂存时，能说出为什么优先使用 `git restore --staged <path>`，而不是 `git reset --hard`。

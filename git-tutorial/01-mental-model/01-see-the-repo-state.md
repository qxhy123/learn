# 看见仓库状态

## 导航与契约

- 所属模块：01 Mental Model（建立 Git 状态模型）
- 本章 lab id：`LAB-MODEL-STATE-01`
- 前置章节：`00-orientation.md`（环境约定、安全约定、观察面板）
- 后续章节：`01-mental-model/02-working-tree-index-commit.md`（工作区、暂存区、提交）
- 本章只使用只读观察命令和可丢弃练习仓库；不要在真实项目中直接试验破坏性命令。

## 场景

你接手一个目录，或者刚从编辑器里切到终端。你想回答四个问题：

1. 这里是不是 Git 仓库？
2. 当前工作目录对应的仓库根目录在哪里？
3. 当前分支、HEAD 和远程跟踪状态是什么？
4. 有没有未提交、已暂存、未跟踪的文件？

正确顺序不是先 `pull`、不是先 `reset`，也不是凭记忆猜“我应该在 main 上”。正确顺序是先观察，再决定下一步。

本章训练一种习惯：每次操作前后都能说清楚仓库状态，而不是只会说“命令成功了”。

## 学习目标

完成本章后，你应该能够：

- 判断普通目录、Git 仓库内部目录、仓库根目录之间的区别。
- 用 `git status --short --branch` 快速读取分支和文件状态。
- 区分 untracked、modified、staged、clean 四类常见状态。
- 解释 `??`、` M`、`M `、`MM`、`A ` 等短状态符号。
- 在任何写入命令之前先建立“状态基线”，并在写入命令之后对比状态变化。
- 遇到不确定状态时，选择安全的只读命令继续调查，而不是立刻执行破坏性命令。

## 心智模型：先看四层状态

本教程后续会反复使用同一个观察框架：

```text
working tree  <->  index/staging area  <->  HEAD commit  <->  branch/remote
正在编辑的文件      准备提交的快照             当前提交          分支与远程关系
```

本章先不要求你掌握所有内部细节，只要求你能把状态归类：

| 层次 | 你在观察什么 | 常用命令 |
|---|---|---|
| 当前目录 | 我是否在仓库里？仓库根目录在哪里？ | `git rev-parse --show-toplevel` |
| 分支/HEAD | 我在哪个分支？是否 detached HEAD？ | `git status --short --branch`、`git branch --show-current` |
| 工作区 | 文件内容是否改了但还没暂存？ | `git status --short` |
| 暂存区 | 哪些修改已经准备进入下一次提交？ | `git status --short`、下一章的 `git diff --cached` |

如果你只能记住一句话：**Git 操作前先看状态，Git 操作后再看状态。**

## 观察点

### 1. 是否在仓库里

```bash
git rev-parse --show-toplevel
```

可能结果：

- 输出一个路径：你在某个 Git 仓库内部；这个路径就是仓库根目录。
- 报错 `fatal: not a git repository`：当前目录不在 Git 仓库中。

观察问题：

- 我是不是在预期的仓库里？
- 我是否位于子目录，导致相对路径容易误判？
- 如果这是普通目录，我是否应该先 `git init`，还是应该切换到已有仓库？

### 2. 当前分支与 HEAD

```bash
git status --short --branch
git branch --show-current
```

`git status --short --branch` 的第一行通常类似：

```text
## main
```

也可能出现：

```text
## feature/login...origin/feature/login [ahead 1]
## main...origin/main [behind 2]
## HEAD (no branch)
```

解释：

- `ahead 1`：本地有 1 个提交还没有推送到上游。
- `behind 2`：上游有 2 个提交本地还没有取得或合入。
- `HEAD (no branch)`：你可能处于 detached HEAD，先不要提交新工作，除非你知道如何保留它。

### 3. 文件短状态

```bash
git status --short
```

短状态的两列很重要：

```text
XY path
```

- `X`：暂存区相对 `HEAD` 的状态。
- `Y`：工作区相对暂存区的状态。

常见输出：

| 输出 | 层次判断 | 含义 | 下一步常见选择 |
|---|---|---|---|
| `?? notes.txt` | 未跟踪 | Git 还没有跟踪这个文件 | `git add` 或加入 `.gitignore` |
| ` M app.py` | 工作区 | 已跟踪文件被修改，但未暂存 | 先 review，再决定是否 `git add` |
| `M  README.md` | 暂存区 | 修改已经暂存，准备进入提交 | 用下一章的 diff 命令确认后提交 |
| `MM config.yml` | 两层都有 | 同一文件既有已暂存修改，又有未暂存修改 | 分别检查 staged 和 unstaged diff |
| `A  new.py` | 暂存区 | 新文件已经暂存 | 确认文件是否应该被提交 |
| ` D old.txt` | 工作区 | 已跟踪文件在工作区被删除但未暂存 | 确认删除是否有意 |
| `D  old.txt` | 暂存区 | 删除已经暂存 | 提交前再次确认 |

## 安全命令清单

本章推荐把下面三条命令当作“观察面板”：

```bash
git status --short --branch
git rev-parse --show-toplevel
git branch --show-current
```

这些命令是只读的：它们不会修改文件、暂存区、提交历史或远程仓库。你可以频繁运行它们。

## 危险命令与禁止动作

本章不需要任何危险命令。遇到状态不确定时，尤其不要用下面命令“试试看”：

| 命令 | 风险 | 本章替代做法 |
|---|---|---|
| `git reset --hard` | 丢弃工作区和暂存区修改 | 先 `git status --short --branch`，再用恢复章节的决策树 |
| `git clean -fd` | 删除未跟踪文件，常误删草稿或生成物 | 先确认 `??` 文件价值；必要时备份到仓库外 |
| `git checkout .` 或 `git restore .` | 丢弃当前目录下已跟踪文件的未暂存修改 | 先确认每个 ` M` 文件是否可丢弃 |
| `git pull` | 可能引入合并、rebase 或冲突，掩盖你原本的本地状态 | 先观察本地是否 clean、当前分支是否正确 |

安全原则：**看状态不危险；在没看懂状态前执行写入命令才危险。**

## 实验

**Lab id：`LAB-MODEL-STATE-01`**

目标：在一个可丢弃仓库中制造 untracked、modified、staged 三种基础状态，并观察它们如何显示。

> 建议在临时目录中执行，例如 `~/tmp/git-tutorial-labs`。不要在真实工作项目里直接运行实验命令。

### 准备

```bash
mkdir model-state-lab
cd model-state-lab
git init
git status --short --branch
```

预期观察：

- 你已经位于一个新仓库中。
- 第一行通常显示 `## No commits yet on main` 或 `## No commits yet on master`，具体名称取决于本机 Git 配置。
- 没有文件状态输出，说明工作区暂时干净。

### 建立一个基线提交

```bash
printf "alpha\n" > tracked.txt
git status --short --branch
git add tracked.txt
git status --short --branch
git commit -m "seed tracked file"
git status --short --branch
```

观察重点：

- `tracked.txt` 刚创建时显示为 `?? tracked.txt`。
- `git add tracked.txt` 后显示为 `A  tracked.txt`。
- 提交后短状态没有文件行，说明工作区和暂存区相对 `HEAD` 干净。

如果 `git commit` 因用户名或邮箱未配置而失败，请按 Git 提示配置临时身份，或等待 labs 集成 agent 在 lab 文档中提供统一环境准备步骤。

### 制造三种常见状态

```bash
printf "draft\n" > new.txt
printf "beta\n" >> tracked.txt
git status --short --branch

git add tracked.txt
git status --short --branch

printf "gamma\n" >> tracked.txt
git status --short --branch
```

预期观察：

1. 第一次观察：

   ```text
    M tracked.txt
   ?? new.txt
   ```

   - `tracked.txt` 是已跟踪文件，修改还在工作区。
   - `new.txt` 是未跟踪文件。

2. `git add tracked.txt` 后：

   ```text
   M  tracked.txt
   ?? new.txt
   ```

   - `tracked.txt` 的当前修改进入暂存区。
   - `new.txt` 仍然未跟踪。

3. 再次修改 `tracked.txt` 后：

   ```text
   MM tracked.txt
   ?? new.txt
   ```

   - `tracked.txt` 同时有已暂存修改和未暂存修改。
   - 这类状态在提交前必须格外小心，因为一次 `git commit` 只会提交暂存区那一部分。

### 清理

如果这个目录只是实验仓库，可以回到上级目录后删除整个目录：

```bash
cd ..
rm -rf model-state-lab
```

只删除你刚刚创建的实验目录。不要把这条命令改成更宽泛的路径。

## 恢复路径

如果实验过程中状态不符合预期，先不要执行 `reset --hard`。按下面顺序恢复：

1. **确认位置**：运行 `pwd` 和 `git rev-parse --show-toplevel`，确认你在实验仓库里。
2. **记录状态**：运行 `git status --short --branch`，把输出保存下来。
3. **未提交实验仓库可重建**：如果确认目录只用于本章实验，可以退出目录后删除 `model-state-lab`，重新开始。
4. **误在真实仓库中操作**：停止继续写入；保存 `git status --short --branch` 输出；不要运行 `git clean`、`reset --hard`；转到恢复模块选择 `restore`、`reset`、`reflog` 等路径。
5. **commit 失败**：通常是身份未配置，不会丢失文件；先看状态，再按提示配置用户名邮箱或重新在准备好的 lab 环境运行。

## 常见错误

- **把 `git status` 当成可选步骤**：不看状态就执行修改命令，是很多事故的起点。
- **看到 clean 就以为远程也同步**：工作区 clean 只说明本地 working tree/index 相对 `HEAD` 干净，不说明远程状态最新。
- **忽略 `??` 文件**：未跟踪文件不会自动进入提交，也不会被普通 `git diff` 展示。
- **误读 `M ` 和 ` M`**：左列是暂存区，右列是工作区；空格的位置决定修改在哪一层。
- **在 detached HEAD 中继续开发**：如果第一行显示 `HEAD (no branch)`，先创建或切换分支，再开展长期修改。
- **用破坏性命令清理“看不懂”的状态**：看不懂时应该继续观察和备份，而不是立刻丢弃。

## 验收

### 验收题 1：解释短状态

给定下面输出，说明每个文件处在哪一层：

```text
## main
 M app.py
M  README.md
MM config.yml
?? notes.txt
A  src/new_feature.py
```

你应该能解释：

- `app.py`：只在工作区修改，尚未暂存。
- `README.md`：修改已暂存，下一次提交会包含它。
- `config.yml`：既有已暂存修改，也有新的未暂存修改。
- `notes.txt`：未跟踪文件，不会自动进入提交。
- `src/new_feature.py`：新文件已暂存。

### 验收题 2：选择下一步

你准备开始修改需求，但看到：

```text
## feature/cart...origin/feature/cart [behind 3]
 M package.json
?? debug-notes.md
```

回答：

1. 你是否应该立刻 `git pull`？为什么？
2. `package.json` 和 `debug-notes.md` 分别有什么风险？
3. 你会先运行哪三条只读命令来确认状态？

参考答案要点：

- 不应盲目 `git pull`，因为本地已有未提交修改，拉取可能引入冲突或混淆状态。
- `package.json` 是已跟踪文件的未暂存修改；`debug-notes.md` 是未跟踪文件，可能是草稿也可能应加入忽略。
- 先运行 `git status --short --branch`、`git rev-parse --show-toplevel`，必要时再运行后续章节会讲的 `git diff`。

## 术语需求

供后续 Appendix 汇总：

- working tree：当前文件系统中的工作副本。
- index / staging area：下一次提交的候选快照。
- HEAD：当前检出的提交。
- untracked：尚未被 Git 纳入版本管理的文件。
- detached HEAD：HEAD 指向具体提交而不是分支名的状态。

## 交付自查

- 模板字段：已包含场景、学习目标、观察点、实验、常见错误、验收。
- Lab id：`LAB-MODEL-STATE-01`。
- 前置章节：`00-orientation.md`。
- 后续章节：`01-mental-model/02-working-tree-index-commit.md`。
- 危险命令：已列出 `reset --hard`、`clean -fd`、`restore .`、`pull` 的风险与替代观察路径。
- 恢复路径：已给出实验仓库重建、真实仓库误操作时的停止与观察流程。
- 集成需求：Labs agent 需要为 `LAB-MODEL-STATE-01` 落地准备/执行/观察/恢复/清理步骤；Appendix agent 需要吸收本章术语。

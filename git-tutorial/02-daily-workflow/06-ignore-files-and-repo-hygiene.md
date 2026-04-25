# 06 忽略文件与仓库卫生

## 场景

你完成了一次功能修改，准备提交前运行 `git status -sb`，却看到日志、缓存、构建产物、编辑器配置和本地环境文件混在真正的代码改动旁边。你不确定哪些文件应该提交，哪些应该忽略，也担心一条清理命令把还没保存的工作删掉。

本章目标不是背 `.gitignore` 语法大全，而是建立一个可重复判断流程：**先分清文件是否已被 Git 跟踪，再决定忽略、保留、移出跟踪或安全清理**。仓库卫生的核心是让 `status` 保持低噪音，让每次提交只包含可 review、可复现、可协作的内容。

## 前置 / 后续

- 前置章节：[05 提交设计与 diff review](./05-commit-design-and-diff-review.md) — 你已经能把变更拆成可审查的小提交。
- 后续章节：[07 为一个任务创建分支](../03-branching-work/07-branch-for-a-task.md) — 干净工作区会让你更安全地切换任务分支。

## 学习目标

完成本章后，你应该能够：

1. 判断一个文件是未跟踪、已跟踪、已暂存，还是被忽略。
2. 区分团队共享的忽略规则、个人本地规则和临时清理动作。
3. 解释 `.gitignore` 只影响未跟踪文件，不能自动停止跟踪已经提交过的文件。
4. 使用 `git check-ignore -v` 和 `git ls-files` 排查“为什么这个文件出现/消失在 status 里”。
5. 在使用 `git rm --cached`、`git clean` 等危险命令前说明风险与恢复路径。

## 观察点

先固定观察面板：

```bash
git status -sb
git status --ignored -s
git diff -- .gitignore
git check-ignore -v <path> || true
git ls-files <path>
```

每个候选文件都回答四个问题：

- **它是否已经被跟踪？** `git ls-files <path>` 有输出，说明它已经在 Git 的跟踪集合中。
- **它是否只是未跟踪噪音？** `git status -sb` 中的 `??` 表示还没有进入 Git 历史。
- **它是否被某条规则忽略？** `git check-ignore -v <path>` 会显示匹配的 ignore 文件、行号和规则。
- **它是否应该被团队共享？** 生成物、日志、密钥、本机配置通常不提交；源码、模板、锁文件、文档和必要配置通常要提交。

一个常见状态可能像这样：

```text
## main
 M src/login.js
?? debug.log
?? build/
?? .env.local
?? .gitignore
```

此时真正需要 review 的可能只有 `src/login.js`；其余文件要先分类处理，不能用“全加”掩盖噪音。

## 命令与解释

### 1. 为团队噪音添加共享规则

目的：把确定不该进入仓库的未跟踪文件从普通 `status` 输出中移除。

```bash
printf "*.log\nbuild/\n.env.local\n" >> .gitignore
git diff -- .gitignore
git status -sb
```

预期观察：`debug.log`、`build/`、`.env.local` 不再作为普通未跟踪文件干扰状态，而 `.gitignore` 本身会显示为待提交文件。提交 `.gitignore` 是在提交团队规则，不是提交本机产物。

### 2. 排查某个文件为什么被忽略

目的：当文件没有出现在 `status` 中，先找出匹配它的具体规则。

```bash
git check-ignore -v debug.log build/output.txt .env.local
```

预期观察：输出包含规则所在文件、行号和匹配模式。例如：

```text
.gitignore:1:*.log debug.log
.gitignore:2:build/ build/output.txt
.gitignore:3:.env.local .env.local
```

如果没有输出，说明该路径没有被 ignore 规则匹配；你需要回到 `git status -sb` 和路径拼写继续排查。

### 3. 确认文件是否已经被 Git 跟踪

目的：避免误以为“写进 `.gitignore` 就能让已提交文件消失”。

```bash
git ls-files config/local.json
git check-ignore -v config/local.json || true
```

预期观察：只要 `git ls-files` 有输出，说明该文件已经被跟踪。即使后续添加 ignore 规则，Git 仍会继续报告它的内容变化，因为 ignore 规则默认只影响未跟踪文件。

### 4. 停止跟踪一个不该入库的文件，但保留本地副本

目的：把已经误提交的生成物或本地配置从下一次提交中移出，同时不删除工作区里的实际文件。

```bash
git rm --cached config/local.json
git status -sb
git diff --cached --name-status
```

预期观察：暂存区会出现对 `config/local.json` 的删除，工作区文件仍保留在磁盘上。你还需要添加合适的 `.gitignore` 规则，避免它再次作为未跟踪文件出现。

### 5. 清理未跟踪文件前先预演

目的：在真正删除文件前确认会被删除的清单。

```bash
git clean -fdn
```

预期观察：Git 只列出“如果执行 `git clean -fd` 会删除什么”，不会实际删除。只有当清单里没有你需要保留的文件时，才考虑执行真正清理命令。

## 判断流程

遇到陌生文件时，按这个顺序处理：

1. 运行 `git status -sb`，先看它是 `??`、`M`、`A` 还是不显示。
2. 运行 `git ls-files <path>`，确认它是否已经被跟踪。
3. 如果未跟踪且不该提交，添加或修正 `.gitignore`，再用 `git check-ignore -v <path>` 验证。
4. 如果已跟踪但不该继续入库，先确认团队同意，再用 `git rm --cached <path>` 配合 ignore 规则提交一次“停止跟踪”的变更。
5. 如果只是临时文件需要删除，先用 `git clean -fdn` 预演，再决定是否清理。

## 实验

**Lab ID：`LAB-DAILY-IGNORE-01`**

目标：添加忽略规则，并验证 `.gitignore` 不会自动停止跟踪已经提交过的文件。

```bash
mkdir ignore-lab
cd ignore-lab
git init
printf "keep\n" > app.txt
git add app.txt
git commit -m "seed app"

printf "debug\n" > debug.log
mkdir build
printf "artifact\n" > build/output.txt
git status -sb

printf "*.log\nbuild/\n" > .gitignore
git status -sb
git check-ignore -v debug.log build/output.txt
```

继续验证“已跟踪文件不受 ignore 自动影响”：

```bash
printf "local\n" > local.txt
git add local.txt
git commit -m "track local example"
printf "local.txt\n" >> .gitignore
printf "changed\n" > local.txt
git status -sb
git ls-files local.txt
```

预期结果：

- 添加 ignore 规则前，`debug.log` 和 `build/output.txt` 作为未跟踪文件出现。
- 添加规则后，它们不再作为普通未跟踪文件干扰 `status`。
- `.gitignore` 本身应该被提交，因为这是团队共享规则。
- `local.txt` 即使后来写进 `.gitignore`，仍然会被 Git 跟踪并报告修改。

实验清理：离开目录后删除整个 `ignore-lab` 练习仓库即可；不要在真实项目目录里直接套用清理命令。

## 常见错误

- **用 `git add .` 把噪音一起提交。** 提交前先看 `git diff --cached`，确认暂存区只包含本次任务需要 review 的内容。
- **以为 `.gitignore` 会删除文件。** 它只是告诉 Git 忽略未跟踪路径，不会删除磁盘文件，也不会自动改历史。
- **以为 `.gitignore` 会停止跟踪已提交文件。** 对已跟踪文件要用 `git rm --cached` 生成一次明确的“从版本控制移除”提交。
- **把秘密写进历史后再 ignore。** 一旦密钥进入提交历史，就要按安全流程轮换密钥并清理历史；ignore 不能撤销泄漏。
- **规则过宽。** 例如 `*.json` 可能误伤必须提交的配置模板；优先写更具体的路径，如 `config/local.json` 或 `.env.local`。
- **把个人偏好放进团队规则。** 编辑器缓存、系统文件可放进个人全局 ignore；项目必须复现的规则才写进仓库 `.gitignore`。

## 危险命令与恢复路径

> **风险提示：`git clean -fd` 会删除未跟踪文件和目录。** 这些文件没有进入 Git 对象库，误删后通常不能从 Git 恢复。执行前必须先运行 `git clean -fdn` 预演，并确认清单里没有新建源码、文档、配置模板或实验记录。
>
> **恢复路径：** 如果只是预演发现清单不对，停止执行并用 `.gitignore`、移动文件或提交保留内容来处理；如果已经误删，优先检查编辑器本地历史、系统废纸篓、备份、构建工具重新生成能力，Git 本身通常帮不上忙。

> **风险提示：`git rm --cached <path>` 会把已跟踪文件从下一次提交中删除。** 它保留工作区文件，但会改变仓库快照；如果该文件其实是项目运行必需文件，其他协作者拉取后会失去它。
>
> **恢复路径：** 提交前可用 `git restore --staged <path>` 撤销暂存删除；已经提交但尚未共享时可用后续恢复章节的方法修正提交；已经共享后，应新增一次恢复提交把必要文件加回，而不是擅自改写共享历史。

## 验收

你应该能回答：

1. `.gitignore` 为什么只影响未跟踪文件，而不会自动停止跟踪已经提交过的文件？
2. `git check-ignore -v <path>` 的输出能帮助你定位哪两类问题？
3. 什么时候应该提交 `.gitignore`，什么时候应该使用个人全局 ignore？
4. 运行 `git clean -fd` 前为什么必须先运行 `git clean -fdn`？
5. 如果一个本地配置文件已经被提交过，你如何在保留本地副本的同时让仓库停止跟踪它？

## 术语需求

- **未跟踪文件（untracked file）**：存在于工作区，但还没有进入 Git 跟踪集合的文件。
- **忽略规则（ignore pattern）**：让 Git 默认不报告、不添加某些未跟踪路径的匹配规则。
- **仓库卫生（repository hygiene）**：让提交历史只包含可复现、可协作、可审查内容的一组习惯。

# 18 Blame、Bisect 与历史排障：把“什么时候坏的”变成可验证证据

## 本章契约

- Lab ID：`LAB-DEBUG-BISECT-01`
- 前置章节：[17 Tags, Releases and Hotfixes](17-tags-releases-and-hotfixes.md)
- 后续章节：[19 Hooks, Config and Aliases](../07-scale-and-governance/19-hooks-config-and-aliases.md)
- 重点能力：用历史记录缩小故障范围，先建立可复现的好/坏边界，再用 `git blame`、`git log`、`git show` 和 `git bisect` 找到最可能的引入提交。
- 危险命令：`git bisect bad`、`git bisect good`、`git bisect run <cmd>`、`git bisect skip`、排障脚本里的 `git reset --hard`、`git clean -fd`
- 恢复路径：开始前保证工作区干净或另建 worktree；任何时候不确定就先 `git bisect reset`；误标 good/bad 时重置 bisect 并重新记录边界；清理命令必须先 dry-run 或在一次性练习仓库中执行。

## 场景

线上刚发布后出现回归：用户保存设置时偶尔失败。你知道 `v1.4.0` 没问题，当前 `main` 有问题，但两者之间有几十个提交。另一个同事指出某行超时配置“看起来可疑”，于是团队自然想运行 `git blame` 看是谁改的。

这类排障最容易走偏：把 `blame` 当成追责工具，或在没有稳定复现步骤时启动 `bisect`，最后得到一个并不可信的“坏提交”。本章的目标是建立一条更工程化的路径：**先把问题变成可重复检查，再用历史命令收集上下文，最后用 bisect 定位第一个引入回归的提交。**

历史排障不是为了证明“谁犯错”，而是回答四个问题：症状是什么、从哪里开始坏、哪次改动解释了变化、下一步修复或回滚怎么做。

## 学习目标

学完本章后，你应该能够：

1. 用 `git log`、`git grep`、`git blame` 和 `git show` 从关键词、文件、行号进入历史上下文。
2. 解释为什么 `git blame` 只能提供线索，不能单独证明根因或责任。
3. 在已知好版本和坏版本之间手动运行 `git bisect`，并正确标记 good/bad。
4. 为 `git bisect run` 编写退出码可靠的最小检查脚本。
5. 在 bisect 结束、误标或检查脚本不稳定时安全恢复工作区。
6. 写出一份包含症状、边界、首个坏提交、证据和修复路线的排障记录。

## 观察点

历史排障前先不要急着 blame，也不要直接 reset 到旧提交。建议固定观察：

```bash
git status -sb
git branch -vv
git log --oneline --graph --decorate --all -n 24
git tag --list --sort=-creatordate | head -20
```

观察时回答六个问题：

- **工作区是否干净？** `bisect` 会不断切换提交；未提交改动会让切换失败，甚至污染排障结果。
- **坏现象是否能稳定复现？** 如果复现步骤不稳定，先修复测试方法，不要让随机失败驱动 good/bad 标记。
- **已知好版本是什么？** 可以是标签、提交哈希、发布分支或某个 PR 合并前的提交。
- **已知坏版本是什么？** 通常是当前 `HEAD`、线上发布标签或能复现问题的提交。
- **线索来自哪里？** 关键词、报错日志、可疑文件、某个配置值、某条测试失败，都对应不同历史入口。
- **排障是否需要隔离目录？** 如果当前目录有未完成工作，优先用上一章的 `git worktree` 建一个干净排障目录。

常见入口选择：

| 已知线索 | 推荐命令 | 目的 |
|---|---|---|
| 报错信息或配置名 | `git grep "keyword"` | 找到当前代码里的位置 |
| 提交说明或 PR 关键词 | `git log --grep="keyword"` | 找到相关变更讨论入口 |
| 某个文件最近变了 | `git log --oneline -- <path>` | 看文件历史范围 |
| 某一行很可疑 | `git blame -L <start>,<end> <path>` | 找到最近改动该行的提交 |
| 知道好/坏边界 | `git bisect` | 二分定位首个坏提交 |

## 操作流程

### 1. 从关键词和文件建立上下文

先把“感觉可疑”变成可查看的历史证据：

```bash
git grep "timeout"
git log --oneline --decorate -- src/config.ts
git log --oneline --grep="timeout"
git show --stat <commit>
```

看提交时至少关注：

- 改动是否真的影响当前症状。
- 提交说明是否描述了业务背景或回滚风险。
- 同一提交是否改了测试、配置、迁移脚本或依赖版本。
- 后续是否又有修复提交覆盖了同一区域。

不要只看单个 diff。一个提交可能只是重命名、格式化或搬迁代码，真正的根因在更早的提交或外部配置里。

### 2. 正确使用 blame：找上下文，不找替罪羊

对可疑行使用范围化 blame，避免整屏噪音：

```bash
git blame -L 40,80 src/config.ts
git show <blame-commit>
git log --oneline --decorate <blame-commit>~5..<blame-commit> -- src/config.ts
```

如果文件经历过大规模格式化或移动，可以增加辅助选项：

```bash
git blame -w -L 40,80 src/config.ts
```

`-w` 会忽略纯空白差异，有助于穿过格式化提交。但它也可能隐藏“空白确实有语义”的语言或配置文件差异，所以只能作为辅助视角。

使用 blame 的沟通原则：

- 说“这行最后由某提交引入/修改”，不要说“某人导致事故”。
- 紧接着看 `git show`、测试和 PR 讨论，确认当时的上下文。
- 如果要联系作者，目标是补上下文和修复方案，不是追责。

### 3. 准备 bisect：定义可靠的 good/bad 边界

`git bisect` 需要一个坏提交和一个好提交。开始前先在当前坏版本验证症状：

```bash
./run-regression-check.sh
echo $?
```

约定退出码：

- `0`：检查通过，当前提交是 good。
- 非 `0`：检查失败，当前提交是 bad。
- 无法判断：手动使用 `git bisect skip`，不要硬标。

然后验证好版本确实好。为了避免污染主工作区，推荐在排障 worktree 中做：

```bash
git worktree add ../project-bisect main
cd ../project-bisect
git switch --detach v1.4.0
./run-regression-check.sh
```

确认好版本通过后，回到坏版本开始 bisect：

```bash
git switch main
git bisect start
git bisect bad HEAD
git bisect good v1.4.0
```

如果你的“好版本”其实也坏，bisect 结论会失去意义。此时应扩大搜索范围或重新定义症状。

### 4. 手动 bisect：每一步都记录证据

启动后，Git 会切到一个候选提交。每次只做两件事：运行检查、标记结果。

```bash
./run-regression-check.sh
# 通过：
git bisect good
# 失败：
git bisect bad
```

建议同时记录一个排障日志：

```text
symptom: saving settings times out after 3 seconds
bad: HEAD on main (abc1234)
good: v1.4.0 (def5678)
check: ./run-regression-check.sh
steps:
- 91ab222 bad, test_save_settings_timeout failed
- 73cd333 good, regression not reproduced
```

当 Git 输出首个坏提交后，先看它：

```bash
git show --stat --summary <first-bad-commit>
git show <first-bad-commit>
```

最后一定恢复：

```bash
git bisect reset
git status -sb
```

`git bisect reset` 会回到开始 bisect 前的位置。不要在 bisect 中间直接继续开发或提交修复。

### 5. 自动 bisect：只在检查命令可靠时使用

当复现步骤能写成脚本时，可以让 Git 自动二分：

```bash
git bisect start
git bisect bad HEAD
git bisect good v1.4.0
git bisect run ./run-regression-check.sh
git bisect reset
```

最小检查脚本应该：

- 自己构造输入，不依赖本机临时状态。
- 输出清晰失败原因，便于复盘。
- 用退出码表达 good/bad。
- 遇到缺少依赖、测试无法运行、历史提交不兼容时谨慎返回可识别状态，并考虑手动 `skip`。

一个示例：

```bash
#!/usr/bin/env bash
set -euo pipefail

npm test -- --runInBand settings-timeout
```

如果历史提交较老，测试命令在某些提交上不存在或依赖无法安装，自动 bisect 可能会把“环境坏”误判为“产品坏”。这时改用手动 bisect，并对无法判断的提交使用 `git bisect skip`。

### 6. 形成排障结论：首个坏提交不是修复方案本身

定位首个坏提交后，还要判断怎么处理：

| 情况 | 可能路线 | 判断依据 |
|---|---|---|
| 提交很小且只引入回归 | `git revert <commit>` | 回滚风险低，能快速恢复服务 |
| 提交包含多个业务改动 | hotfix 补丁 | 直接 revert 会撤掉太多有效变更 |
| 根因在配置或环境 | 修配置并补监控 | 代码提交只是暴露问题，不是唯一原因 |
| 检查脚本不稳定 | 先修复测试/复现 | 没有可信证据前不要合入修复 |

排障记录建议包含：

```text
症状：保存设置偶发超时，复现命令为 ./run-regression-check.sh
已知好版本：v1.4.0 (def5678)
已知坏版本：main@abc1234
首个坏提交：91ab222 change settings timeout default
证据：bisect 日志；git show 显示 timeout 从 10s 改为 3s；测试在该提交开始失败
处理路线：提交 hotfix 恢复默认值，并补一条回归测试
后续：检查部署配置是否覆盖默认值
```

## 危险命令与恢复路径

| 命令或动作 | 风险 | 更安全的做法 | 事故后先做什么 |
|---|---|---|---|
| `git bisect bad` / `git bisect good` | 标错会让最终坏提交不可信 | 标记前运行同一检查命令，并记录输出 | `git bisect reset`，重新从可信 good/bad 边界开始 |
| `git bisect run <cmd>` | 不稳定脚本会自动放大误判 | 先在已知 good 和 bad 上分别验证脚本退出码 | 停止自动 bisect，修脚本或改手动标记 |
| `git bisect skip` | 跳过太多提交会得到多个候选或无结论 | 只对确实无法判断的提交 skip，并记录原因 | 扩大边界、改善检查环境，必要时人工审查候选提交 |
| 在 bisect 中直接修改代码 | 排障现场和修复现场混在一起 | bisect 只定位；修复回到正常分支后进行 | `git diff` 保存必要临时记录，`git bisect reset` 后重新建修复分支 |
| 排障脚本里使用 `git reset --hard` | 删除已跟踪的未提交改动 | 只在一次性练习仓库或干净 worktree 中运行 | 立即停止，检查 reflog/编辑器本地历史；未提交改动可能无法完整恢复 |
| 排障脚本里使用 `git clean -fd` | 删除未跟踪文件和目录 | 先 `git clean -nd`，并限制在临时目录 | 查备份、回收站或编辑器本地历史；Git 通常没有记录未跟踪文件 |

通用恢复策略：

1. **任何不确定先退出 bisect。** 运行 `git bisect reset`，确认 `git status -sb` 回到预期分支。
2. **保留排障日志。** 误标或脚本失败时，记录已尝试的边界和失败原因，避免下一轮重复犯错。
3. **隔离排障目录。** 对真实项目优先创建 worktree；对破坏性脚本使用专门练习仓库。
4. **把修复放到新分支。** 定位完成后从最新目标分支创建修复分支，不在 detached HEAD 或 bisect 中间状态提交。
5. **用测试保护结论。** 如果 bisect 找到了回归提交，尽量补一条能在坏提交失败、修复后通过的回归测试。

## 实验

Lab：`LAB-DEBUG-BISECT-01`

实验目标：构造一个小型回归历史，分别用手动 bisect 和自动 bisect 定位首个坏提交，并写出排障记录。

1. **准备练习仓库**
   - 初始化或进入 Labs Agent 准备的练习仓库。
   - 创建一个简单脚本，例如 `calc.sh` 或 `settings-check.sh`，让它在正常情况下返回 `0`。
   - 连续提交 8 到 10 次小改动，其中某一次引入可检测回归。
2. **建立边界**
   - 在回归前的提交打标签 `known-good`。
   - 在当前 `HEAD` 运行检查脚本，确认它失败。
   - 记录 `git log --oneline --graph --decorate --all -n 12`。
3. **手动 bisect**
   - 运行 `git bisect start`、`git bisect bad HEAD`、`git bisect good known-good`。
   - 每次 Git 切换候选提交后运行检查脚本。
   - 根据结果执行 `git bisect good` 或 `git bisect bad`，并记录每一步。
4. **自动 bisect**
   - 运行 `git bisect reset` 回到起点。
   - 确认检查脚本在 `known-good` 上返回 `0`，在坏版本上返回非 `0`。
   - 重新开始 bisect，并执行 `git bisect run ./settings-check.sh`。
5. **复盘和恢复**
   - 用 `git show <first-bad-commit>` 解释首个坏提交为什么引入问题。
   - 运行 `git bisect reset` 和 `git status -sb`，确认工作区恢复。
   - 写一段排障记录，包含症状、good/bad 边界、首个坏提交、证据和修复路线。

给 Labs Agent 的落地需求：场景文件需要包含可复制的提交构造脚本、一个稳定失败的检查脚本、手动 bisect 记录模板、自动 `git bisect run` 步骤、误标 good/bad 后如何 `git bisect reset` 重来的恢复步骤，以及清理练习仓库的说明。

## 常见错误

- 把 `git blame` 当成甩锅工具，而不是历史上下文索引。
- 没有稳定复现步骤，就让 `git bisect run` 自动标记大量提交。
- 已知好版本没有先验证，导致 good/bad 边界本身错误。
- 在 bisect 中间状态继续开发、提交修复或切换无关分支。
- 测试脚本依赖本机缓存、外部网络或未记录的环境变量，导致结果不可重复。
- 遇到无法构建的历史提交时硬标 bad，而不是 `skip` 或改用人工审查。
- 找到首个坏提交后忘记 `git bisect reset`，继续停留在 detached HEAD 上。
- 只写“某提交坏了”，没有说明症状、证据和修复路线。

## 验收

请用下面问题自查：

1. 你能说明本章的 Lab ID、前置章节和后续章节分别是什么吗？
2. 给出一个报错关键词和可疑文件时，你能选择 `git grep`、`git log --grep`、`git blame` 或 `git show` 中合适的入口吗？
3. 你能解释为什么 `git blame` 的输出不能直接等同于事故责任吗？
4. 你能在干净 worktree 中运行一次手动 `git bisect`，并在每一步正确标记 good/bad 吗？
5. 你能写出一个退出码可靠的检查脚本，并先在已知 good 与 bad 上验证它吗？
6. 如果 bisect 过程中标错或脚本不稳定，你能用 `git bisect reset` 恢复并重新开始吗？
7. 你能列出本章危险命令，并说明每个命令对应的恢复路径吗？
8. 你能交付一段排障记录，包含症状、good/bad 边界、首个坏提交、`git show` 证据和修复路线吗？

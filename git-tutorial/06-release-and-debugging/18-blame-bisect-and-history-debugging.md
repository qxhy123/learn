# 18 Blame, Bisect and History Debugging

## 场景

线上出现回归，你知道现在是坏的，也知道某个旧版本是好的，但不知道哪次提交引入问题。或者某行代码看起来可疑，需要理解它为什么变成现在这样。本节把 Git 历史用于排障：先找线索，再用可重复检查定位首个坏提交。

## 学习目标

- 用 `git grep`、`git log --grep`、`git blame` 和 `git show` 建立历史上下文。
- 理解 `blame` 是线索入口，不是责任判定工具。
- 使用 `git bisect` 在已知好坏边界之间定位回归提交。
- 在排障后用可读记录说明症状、范围、首个坏提交和修复路线。

## 观察点

排障前先建立边界：

```bash
git status -sb
git log --oneline --graph --decorate --all -n 20
git tag --list --sort=-creatordate | head
```

然后根据线索选择入口：

- 知道关键词：`git grep "keyword"`。
- 知道提交说明关键词：`git log --grep="keyword"`。
- 知道可疑文件或行：`git blame <path>` 后接 `git show <commit>`。
- 知道一个好版本和一个坏版本：`git bisect`。

## 命令与判断

### 从代码线索进入历史

```bash
git grep "timeout"
git log --oneline -- src/config.ts
git blame src/config.ts
git show <commit>
```

`blame` 只能告诉你某行最近一次来自哪个提交。真正排障还要看提交上下文、测试、PR 或 issue，而不是把作者当成根因。

### 用 bisect 定位回归

```bash
git bisect start
git bisect bad HEAD
git bisect good v1.2.0
# 每次 Git 切到一个候选提交后运行检查
./run-regression-check.sh
git bisect good   # 或 git bisect bad
```

结束后清理状态：

```bash
git bisect reset
```

如果有自动检查命令：

```bash
git bisect run ./run-regression-check.sh
```

检查命令的退出码必须可靠：0 表示 good，非 0 表示 bad 或无法通过。

## 风险提示

`git bisect` 会不断切换工作区到历史提交。开始前工作区必须干净，否则你可能把未提交内容带入排障过程或被切换阻塞。

恢复路径：

```bash
git bisect reset
git status -sb
```

如果 bisect 过程中发现测试脚本本身不可靠，先 `git bisect reset`，修正检查方法后重新开始。不要把不稳定检查结果硬标为 good/bad。

## 实验

Lab：`LAB-DEBUG-BISECT-01`

1. 构造 8 到 10 个提交，其中某一提交引入一个可检测回归。
2. 标记一个已知好提交和当前坏提交。
3. 手动运行 `git bisect start`、`good`、`bad`，每一步记录 Git 选中的候选提交。
4. 编写一个最小检查脚本，用 `git bisect run` 自动定位首个坏提交。
5. 定位后运行 `git bisect reset`，用 `git show <bad-commit>` 写出根因摘要。

## 常见错误

- 把 `git blame` 当成甩锅工具，而不是上下文索引。
- 没有确定“已知好”和“已知坏”边界就开始 bisect。
- bisect 过程中测试不稳定，却继续标记 good/bad。
- 忘记 `git bisect reset`，导致自己停留在中间历史提交上。

## 验收

你应该能交付一段排障记录，包含：

- 症状是什么，如何复现。
- 已知好版本和坏版本分别是什么。
- bisect 找到的首个坏提交是什么。
- `git show` 中哪些改动解释了问题。
- 修复选择：revert、hotfix、后续补丁或进一步调查。

能够把这些信息写清楚，就说明你已经把 Git 历史用于工程排障，而不是只用它保存文件。

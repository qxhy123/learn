# 第一次干净提交

## 场景

你刚完成一个小改动，想把它变成一次可以被队友 review、可以被回滚、可以被历史阅读的提交。目标不是“赶紧提交成功”，而是让提交边界清楚。

## 学习目标

- 按“观察 → 修改 → 暂存 → review → 提交 → 再观察”的顺序完成一次提交。
- 解释 `git diff` 与 `git diff --cached` 在提交前后为什么会变化。
- 写出能说明意图的提交信息。
- 判断什么时候应该停止并拆分提交。

## 观察点

提交前：

```bash
git status --short --branch
git diff
git diff --cached
```

提交后：

```bash
git status --short --branch
git log --oneline --decorate --max-count=3
git show --stat HEAD
```

## 命令与解释

```bash
git add README.md
git diff --cached
git commit -m "Explain project setup path"
```

- `git add` 把当前文件内容放入暂存区。
- `git diff --cached` 是提交前 review 的核心视图。
- `git commit` 记录暂存区，并移动当前分支到新提交。

提交信息建议使用动词短语，说明这次变更带来的结果，例如：

```text
Clarify setup steps for local development
```

比下面这种信息更有价值：

```text
update README
```

## 实验

**Lab id：`LAB-DAILY-CLEAN-COMMIT-01`**

目标：完成一次最小干净提交。

```bash
mkdir clean-commit-lab
cd clean-commit-lab
git init
printf "# Demo\n" > README.md
git status --short --branch
git add README.md
git diff --cached
git commit -m "Create demo readme"
git status --short --branch
git show --stat HEAD
```

预期观察：

- 暂存前 `README.md` 是未跟踪文件。
- 暂存后 `git diff --cached` 显示将进入提交的内容。
- 提交后工作区应回到 clean，`HEAD` 指向新提交。

## 常见错误

- **先提交再 review**：提交前不看 `diff --cached`，容易把调试代码带入历史。
- **提交信息只写 “update”**：未来读历史时看不出意图。
- **一次提交混多个意图**：例如同时修 bug、格式化全文件、改文档，会增加 review 和回滚成本。

## 验收

请完成一次提交，并能展示：

1. 提交前的 `git diff --cached`。
2. 提交后的 `git show --stat HEAD`。
3. 一句话说明这次提交为什么是一个独立意图。

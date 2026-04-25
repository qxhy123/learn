# 提交设计与 diff review

## 场景

你修改了三个地方：修了一个 bug、顺手重命名变量、又格式化了文件。全部塞进一个提交也许能通过测试，但 reviewer 很难判断真正的行为变化。好的提交设计会让协作、回滚、排障都更简单。

## 学习目标

- 判断一次提交是否只有一个意图。
- 使用 `git add -p` 或按文件暂存来拆分提交。
- 在提交前 review 暂存区 diff。
- 识别不应该强行拆分的情况。

## 观察点

```bash
git status --short
git diff --stat
git diff
git diff --cached --stat
git diff --cached
```

`--stat` 先看范围，完整 diff 再看细节。提交前真正要确认的是 `git diff --cached`。

## 命令与解释

```bash
git add -p
```

交互式选择 diff hunk 进入暂存区。它适合把同一文件中的不同意图拆开，但不适合在你还没理解 diff 时机械操作。

```bash
git restore --staged <path>
```

撤回误暂存内容，回到工作区修改状态。它是拆分提交时的安全工具。

```bash
git commit -m "Fix null handling in parser"
```

提交信息说明本次变更意图。格式化、重命名、行为修复最好分开，让历史可读。

## 实验

**Lab id：`LAB-DAILY-DIFF-REVIEW-01`**

目标：把两个意图拆成两个提交。

```bash
mkdir diff-review-lab
cd diff-review-lab
git init
printf "name=demo\ncount=1\n" > app.conf
git add app.conf
git commit -m "seed config"

printf "name=demo-app\ncount=1\n" > app.conf
printf "notes\n" > README.md

git diff --stat
git add app.conf
git diff --cached
git commit -m "Rename demo app config"

git add README.md
git diff --cached
git commit -m "Document demo notes"
```

预期观察：

- 第一次提交只包含 `app.conf` 的配置命名变化。
- 第二次提交只新增 README 说明。
- `git log --oneline` 中两个提交的意图可分辨。

## 常见错误

- **把“文件不同”误认为“意图不同”**：一个功能可能需要同时改代码和文档，可以放在一次提交里。
- **为了拆分而拆分**：过度拆分会让 review 需要在多个提交之间来回跳。
- **不看 staged diff**：只看工作区 diff，无法确认下一次提交到底包含什么。

## 验收

拿一个真实或练习 diff，回答：

1. 这次修改有几个意图？
2. 每个意图是否可以独立构建或解释？
3. 如果要拆分，先暂存哪一部分，为什么？

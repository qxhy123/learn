# 第4章：status、add、commit 与 diff

## 学习目标

完成本章后，你应能：

1. 熟练使用 `git status` 作为默认状态检查工具
2. 读懂工作区、暂存区与历史之间的差异
3. 使用 `git add`、`git add -p`、`git commit` 组织提交
4. 用 `git diff` 系列命令确认“下一次提交到底会包含什么”

---

## 4.1 `git status` 应该成为你的默认动作

只要你不确定仓库现在是什么状态，先运行：

```bash
git status -sb
```

它能最快告诉你：

- 当前在哪个分支
- 工作区是否有未提交改动
- 是否存在 staged 与 unstaged 的分离
- 本地分支是否领先或落后远程

很多 Git 灾难都不是因为命令太难，而是因为用户在状态没看清时就贸然操作。

---

## 4.2 建议固定的状态阅读顺序

看到 `git status` 时，建议按这个顺序读：

1. **先看当前分支**
   先确认自己站在哪。

2. **再看 staged 区**
   下一次提交已经准备了什么。

3. **再看 unstaged 区**
   工作区还有哪些尚未进入索引的改动。

4. **最后看 untracked 文件**
   有没有新文件根本没被 Git 管起来。

这个顺序会直接决定你后面是否能稳地使用 `add`、`commit`、`restore`。

---

## 4.3 `git diff`、`git diff --cached`、`git diff HEAD` 分别回答什么问题

### `git diff`

工作区 vs 索引：

```bash
git diff
```

回答的是：“我改了但还没暂存的内容是什么？”

### `git diff --cached`

索引 vs HEAD：

```bash
git diff --cached
```

回答的是：“下一次提交将包含什么？”

### `git diff HEAD`

工作区 + 索引 整体 vs HEAD：

```bash
git diff HEAD
```

回答的是：“从最近一次提交到现在，我总共动了什么？”

如果你把这三条命令分不清，提交前就很容易误判。

---

## 4.4 一套更稳的本地提交流程

推荐你养成这套节奏：

```bash
git status -sb
git diff
git add <files>
git diff --cached
git commit -m "type: summary"
git status -sb
```

这套流程的价值不是形式，而是让你每一步都明确：

- 现状是什么
- 你准备提交什么
- 提交之后仓库是否回到干净状态

---

## 4.5 `git add -p`：分块暂存，而不是整文件一起带走

如果同一个文件里既有重构、又有功能修改、又有临时调试代码，理想方式不是一起提交，而是拆开。

`git add -p` 可以按代码块交互式选择：

```bash
git add -p app.js
```

常见交互选项：

- `y`：暂存这一块
- `n`：跳过这一块
- `s`：进一步拆分
- `q`：退出

### 什么时候值得拆块提交

- 一个文件中混了两类主题改动
- 你想先提交“安全的重构”，后提交“真正的行为变化”
- 你想让 review 和回滚边界更清晰

### 什么时候不要硬拆

- 这些改动在语义上根本不可分
- 拆太碎会让历史变得难读

也就是说，小步提交不是越碎越好，而是**每次提交只表达一件清晰的事**。

---

## 4.6 `git commit -am` 为什么不应该成为初学默认动作

`git commit -am "..."` 会把已跟踪文件的改动直接加入索引并提交。

它的优点是快，但有几个边界：

- 不会自动包含新文件
- 容易跳过“提交前审阅 staged diff”的习惯
- 容易让你从“明确挑选”退化成“顺手提交”

初学阶段，建议尽量坚持显式的 `add` + `diff --cached` + `commit`。

---

## 4.7 一个必须亲手做的实验：双重改动文件

在同一文件里做两类改动：

```bash
echo "a" > note.txt
git add note.txt
git commit -m "feat: add note"

echo "b" >> note.txt
echo "c" >> note.txt
```

然后：

```bash
git add -p note.txt
git status -sb
git diff
git diff --cached
git diff HEAD
```

你应该能回答：

- 哪些变化已经进入下一次提交
- 哪些变化还留在工作区
- 为什么 `nothing to commit` 和“没有改动”不是一回事

---

## 常见误区

- **误区 1：看到 `git status` 很长就跳过不看。**
  越复杂越应该先看状态。

- **误区 2：提交前不看 `git diff --cached`。**
  这会导致很多低级误提交。

- **误区 3：`git add .` 永远安全。**
  它可能把你根本没打算提交的文件一起带进去。

- **误区 4：`git commit -am` 等于更高级的提交方式。**
  对初学者来说，它更可能掩盖状态判断不足。

---

## 本章练习

1. 修改两个文件，只暂存其中一个，再观察 `git status -sb`。
2. 在同一文件里做两处不同类型的改动，尝试用 `git add -p` 只提交其中一部分。
3. 提交前运行 `git diff --cached`，说明你看到的内容是相对谁比较出来的。
4. 思考题：为什么“提交前看 staged diff”比“提交后再看 log”更重要？

---

## 本章小结

这一章的关键不是多会几条命令，而是形成一套稳定的阅读顺序。下一章我们继续学如何读历史图，因为如果你不会读历史，后面的分支、merge、rebase 就只能靠猜。

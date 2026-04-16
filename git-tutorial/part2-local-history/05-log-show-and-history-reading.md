# 第5章：log、show 与历史阅读

## 学习目标

完成本章后，你应能：

1. 使用 `git log` 阅读提交历史
2. 看懂最基本的提交图形结构
3. 用 `git show` 查看某次提交的具体内容
4. 比较两个提交、两个分支或某个文件的历史变化

---

## 5.1 `git log` 不是“看流水账”，而是读项目演化

最基础的历史查看命令是：

```bash
git log
```

但实际工作中，更常用的是这种更简洁、信息密度更高的形式：

```bash
git log --oneline --graph --decorate --all
```

它会同时展示：

- 精简提交 ID
- 提交说明
- 分支和标签指向
- 分叉和合并关系

这条命令的真正价值，不是“好看”，而是它让你第一次把 Git 历史当作**图**来读，而不是当作时间线来读。

---

## 5.2 先学会看三种历史形态

### 线性历史

```text
* c3 add tests
* b2 refactor parser
* a1 initialize project
```

适合回答：最近是怎么一步步演化过来的。

### 分叉历史

```text
* d4 add login form
| * c3 main hotfix
|/
* b2 init app
```

适合回答：哪些改动是在不同分支上并行发生的。

### 汇合历史

```text
*   e5 merge feature/login
|\
| * d4 add login form
| * c3 add auth service
|/
* b2 init app
```

适合回答：某个主题什么时候进入主干。

---

## 5.3 `git show`：看某次提交到底改了什么

如果你想查看某次提交的详细内容：

```bash
git show <commit>
```

它通常会显示：

- 提交作者和时间
- 提交说明
- 具体 diff

建议你养成一个习惯：看某条提交时，不要只看 message，也看 diff。因为真正决定这条提交质量的，是它是否真的只做了一件清晰的事。

---

## 5.4 按问题切换视图，而不是死记一个命令

### 我想看最近整体历史

```bash
git log --oneline --graph --decorate --all
```

### 我想看某个文件的演化

```bash
git log -- path/to/file
```

### 我想看某次提交改了哪些文件

```bash
git show --name-status <commit>
```

### 我想看两个提交之间的差异

```bash
git diff HEAD~2 HEAD
```

### 我想看某个文件在两个版本之间的差异

```bash
git diff HEAD~2 HEAD -- src/app.js
```

高质量历史阅读的关键，是先问自己：**我到底想看整体图、某次提交、某个文件，还是某段范围？**

---

## 5.5 读历史时最容易卡住的：范围和比较

Git 的很多比较其实是在回答不同问题：

- `git diff`
  工作区 vs 索引

- `git diff --cached`
  索引 vs HEAD

- `git diff A B`
  A 和 B 这两个提交之间的内容差异

- `git diff A B -- path`
  只看某个文件在两个版本之间的变化

这一章先不强行展开所有范围语法，但你需要先建立一个基本直觉：

> 比较命令不是越复杂越高级，而是“比较对象不同，回答的问题不同”。

---

## 5.6 一个很有价值的习惯：切换不同历史视图

建议你在同一个仓库上尝试这些命令：

```bash
git log --oneline --graph --decorate --all
git log --stat --oneline
git show --name-status HEAD
git diff --name-only HEAD~2 HEAD
```

你会发现：

- 图视图擅长看结构
- `--stat` 擅长看改动规模
- `show` 擅长看单次提交
- `diff --name-only` 擅长看影响范围

高阶读历史的人，不会只依赖一种视图。

---

## 5.7 为什么“只看当前文件内容”不够

如果你只看现在目录里的文件，你最多知道“现在是什么样”，却不知道：

- 它是怎么变成这样的
- 哪次提交引入了这个行为
- 某段改动是实验、重构还是紧急修复

而 Git 的真正价值，恰恰在于让你从“结果”回溯到“演化过程”。

---

## 常见误区

- **误区 1：`git log` 看起来像时间线，所以它本质就是时间线。**
  更准确地说，它展示的是提交图的一种遍历结果。

- **误区 2：只会 `git log` 就够了。**
  高质量历史阅读需要 `show`、`diff`、按文件查看等多种视图切换。

- **误区 3：只看当前文件内容就能排障。**
  很多问题只有放回历史图里才能说清楚。

---

## 本章练习

1. 在练习仓库里做 3 次提交，用 `git log --oneline --graph --decorate --all` 查看结果。
2. 使用 `git show HEAD` 查看最近一次提交的 diff。
3. 修改一个文件并提交两次，然后比较 `HEAD~1` 与 `HEAD`。
4. 思考题：为什么阅读历史图是理解分支、merge 和 rebase 的前提？

---

## 本章小结

会写历史是一回事，会读历史是另一回事。下一章我们继续处理一个很现实的问题：哪些文件不该进版本库，以及怎样让历史对未来的你和团队更友好。

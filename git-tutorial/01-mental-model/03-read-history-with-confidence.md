# 有信心地阅读历史

## 场景

你准备修改代码，但想先知道最近发生了什么：谁改过、某个提交包含哪些文件、当前分支是否从主线分叉。读历史不是看流水账，而是在提交图里定位项目状态如何演化。

## 学习目标

- 使用 `git log --oneline --graph --decorate` 阅读提交图。
- 用 `git show` 查看单个提交的意图和差异。
- 用 `git diff A..B` 比较两个提交之间的内容变化。
- 区分线性历史、分叉历史和汇合历史。

## 观察点

```bash
git log --oneline --graph --decorate --all --max-count=12
git show --stat HEAD
git show --name-only HEAD
git diff HEAD~1..HEAD
```

观察时回答：

- 当前分支尖端是哪一个提交？
- 是否有 merge commit？
- 一个提交改了哪些文件？
- 某两个提交之间的最终内容差异是什么？

## 命令与解释

```bash
git log --oneline --graph --decorate --all
```

`--graph` 让分叉和汇合可见；`--decorate` 显示分支、tag、`HEAD` 等引用；`--all` 显示所有本地引用可达的历史。

```bash
git show HEAD
```

展示最近一次提交的元数据和补丁。review 自己刚写的提交时，它比只看文件最终状态更可靠。

```bash
git diff main..feature
```

比较两个引用指向的提交内容差异。注意：`diff` 比较的是内容结果，`log` 展示的是提交路径；两者回答的问题不同。

## 实验

**Lab id：`LAB-MODEL-HISTORY-01`**

目标：创建三次提交，并用不同视图阅读历史。

```bash
mkdir history-lab
cd history-lab
git init
printf "v1\n" > app.txt
git add app.txt
git commit -m "create app"
printf "v2\n" >> app.txt
git add app.txt
git commit -m "extend app"
printf "notes\n" > notes.txt
git add notes.txt
git commit -m "add notes"

git log --oneline --graph --decorate
git show --stat HEAD
git diff HEAD~2..HEAD
```

预期观察：

- `log` 显示三次线性提交。
- `show --stat HEAD` 只关注最近一次提交。
- `diff HEAD~2..HEAD` 显示从第一版到当前版本的最终内容差异。

## 常见错误

- **把 `log` 和 `diff` 混成一个问题**：`log` 回答“经过哪些提交”，`diff` 回答“最终内容差异是什么”。
- **只看当前文件，不看提交意图**：文件最终状态看不出为什么这么改。
- **忽略引用装饰**：不知道 `HEAD -> main`、`origin/main` 的含义，就很难判断同步状态。

## 验收

请解释下面两个命令分别适合回答什么问题：

```bash
git log --oneline --graph --decorate main..feature
git diff main..feature
```

前者适合看 feature 上有哪些 main 没有的提交；后者适合看 feature 相对 main 的最终内容差异。

# Git Tutorial Labs

这套 labs 不是“额外样例代码”，而是配合 [README.md](../README.md) 中实验系统使用的可重复练习环境。

所有练习仓库都由脚本生成在：

`git-tutorial/labs/workspaces/`

这些工作区已经通过 `.gitignore` 排除，不会被当前教程仓库纳入版本控制。

---

## 使用方式

进入脚本目录后运行：

```bash
./bin/git-lab.sh --list
./bin/git-lab.sh basics
./bin/git-lab.sh collaboration
./bin/git-lab.sh recovery
./bin/git-lab.sh advanced
```

如果某个场景已经存在，想重新生成：

```bash
./bin/git-lab.sh basics --force
```

---

## 场景说明

### `basics`

对应教程：

- 第 1-6 章

生成内容：

- 一个最小本地仓库
- 已经准备好 staged + unstaged 的同一文件
- `.gitignore`、ignored 文件、untracked 文件并存的状态

适合练：

- `status`
- `diff`
- `diff --cached`
- `add -p`
- `ls-files --stage`
- `check-ignore -v`

### `collaboration`

对应教程：

- 第 7-12 章

生成内容：

- 一个 bare `origin.git`
- 两个克隆：`alice` 和 `bob`
- `alice` 本地分支与远端分支已制造出分叉状态
- `feature/login` 分支已准备好用于 merge / rebase 练习

适合练：

- `branch`
- `switch`
- `fetch`
- `pull`
- `push`
- `merge`
- `rebase`

### `recovery`

对应教程：

- 第 13-18 章

生成内容：

- 一个带 `known-good` / `known-bad` 标签的历史
- 一个可自动验证好坏的 `verify.sh`
- 一个 `release/1.0` 分支，适合练 `cherry-pick`

适合练：

- `restore`
- `reset`
- `revert`
- `reflog`
- `cherry-pick`
- `bisect`

### `advanced`

对应教程：

- 第 19-24 章

生成内容：

- 一个带 tag、release 分支和示例 hooks 的仓库
- 一个额外的 `worktree`
- 适合观察对象、引用、配置和 hooks 的环境

适合练：

- `cat-file`
- `show-ref`
- `config --show-origin --list`
- `worktree`
- `core.hooksPath`

---

## 推荐练习顺序

1. 先做 `basics`
2. 再做 `collaboration`
3. 之后做 `recovery`
4. 最后做 `advanced`

这样基本与教程章节顺序一致，也能让实验结果彼此复用。

---

## 默认观察面板

每次进入某个 lab 目录后，建议先跑：

```bash
git status -sb
git log --oneline --graph --decorate --all
```

如果当前练的是基础或恢复主题，再加：

```bash
git diff
git diff --cached
git ls-files --stage
```

如果当前练的是内部原理，再加：

```bash
git rev-parse HEAD
git cat-file -p HEAD
git show-ref --heads --tags
```

---

## 安全说明

- 脚本只会操作 `git-tutorial/labs/workspaces/` 下的目录
- `--force` 只会删除对应场景目录，不会删除其它位置
- 生成的仓库都使用独立的本地用户名和邮箱，不污染你的全局 Git 配置

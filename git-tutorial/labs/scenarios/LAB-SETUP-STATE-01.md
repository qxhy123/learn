# LAB-SETUP-STATE-01: 初始化仓库并观察三棵树

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-SETUP-STATE-01 --force
cd workspaces/setup-state/state-lab
```

## 执行

- 修改 `notes.md` 并分别制造 staged、unstaged、untracked 状态。
- 比较 `git diff` 与 `git diff --cached` 的输出。

## 观察

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=12
```

```bash
git diff
git diff --cached
git ls-files --stage
```

## 恢复

- 用 `git restore notes.md` 丢弃工作区未暂存改动。
- 用 `git restore --staged notes.md` 取消暂存。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/setup-state
```

## 预期结果

你能解释 working tree、index、HEAD 三者各自包含什么。

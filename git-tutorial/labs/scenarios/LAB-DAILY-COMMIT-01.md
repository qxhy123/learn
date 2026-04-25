# LAB-DAILY-COMMIT-01: 拆分提交和 diff review

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-DAILY-COMMIT-01 --force
cd workspaces/daily-commit/daily-lab
```

## 执行

- 使用 `git add -p` 把 `app.txt` 中两类改动拆开。
- 分两次提交，提交说明分别描述意图。

## 观察

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=12
```

```bash
git diff
git diff --cached
git show --stat --summary HEAD
```

## 恢复

- 如果 staged 内容不对，使用 `git restore --staged <file>` 回到未暂存。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/daily-commit
```

## 预期结果

最终历史中有两个小提交，工作区干净。

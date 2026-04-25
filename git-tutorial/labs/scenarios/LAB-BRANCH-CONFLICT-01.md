# LAB-BRANCH-CONFLICT-01: 任务分支和冲突 playbook

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-BRANCH-CONFLICT-01 --force
cd workspaces/branch-conflict/alice
```

## 执行

- 在 `alice` 中 fetch 后尝试合并 `origin/main` 到 `feature/profile`。
- 按文件中的冲突标记完成合并。

## 观察

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=12
```

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=20
```

## 恢复

- 合并未完成且想重来时使用 `git merge --abort`。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/branch-conflict
```

## 预期结果

你能指出冲突来自共同祖先、当前分支和目标分支的哪几行。

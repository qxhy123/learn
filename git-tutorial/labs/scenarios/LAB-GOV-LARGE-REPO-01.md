# LAB-GOV-LARGE-REPO-01: 大文件、忽略规则和 LFS 决策

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-LARGE-REPO-01 --force
cd workspaces/gov-large-repo/large-repo-lab
```

## 执行

- 找出已跟踪的构建产物和大文件候选。
- 更新 `.gitignore` 和 `LFS_DECISION.md`，写出处理建议。

## 观察

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=12
```

```bash
git count-objects -vH
git ls-files | grep -E "dist/|assets/|\.zip|\.mp4" || true
```

## 恢复

- 未提交前可 `git restore --staged . && git restore .` 重置本地演练。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/gov-large-repo
```

## 预期结果

你能区分忽略、LFS、制品库和历史清理四类动作。

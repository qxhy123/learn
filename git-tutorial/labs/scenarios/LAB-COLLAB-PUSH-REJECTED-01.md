# LAB-COLLAB-PUSH-REJECTED-01: 双人协作和 push rejected

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-COLLAB-PUSH-REJECTED-01 --force
cd workspaces/collab-push-rejected/alice
```

## 执行

- 在 `alice` 中直接 `git push`，观察 rejected。
- `git fetch` 后选择 merge 或 rebase 同步主干。

## 观察

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=12
```

```bash
git branch -vv
git log --oneline --graph --decorate --all --max-count=20
```

## 恢复

- 如果 rebase 过程中不确定，先 `git rebase --abort`，回到同步前状态。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/collab-push-rejected
```

## 预期结果

你能在不覆盖 Bob 历史的前提下推送 Alice 改动。

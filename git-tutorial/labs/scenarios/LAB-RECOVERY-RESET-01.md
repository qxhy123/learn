# LAB-RECOVERY-RESET-01: restore/reset/revert/reflog 选择边界

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-RECOVERY-RESET-01 --force
cd workspaces/recovery-reset/recovery-lab
```

## 执行

- 制造一次本地错误 reset。
- 用 `git reflog` 找到 reset 前提交并建立 `rescue/reflog` 分支。

## 观察

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=12
```

```bash
git reflog --date=relative
git log --oneline --graph --decorate --all --max-count=20
```

## 恢复

- 用 `git switch rescue/reflog` 检查救回内容；不要覆盖原分支直到确认。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/recovery-reset
```

## 预期结果

你能说明哪些错误适合 restore、reset、revert 或 reflog。

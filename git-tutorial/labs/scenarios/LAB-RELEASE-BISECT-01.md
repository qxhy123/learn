# LAB-RELEASE-BISECT-01: 标签、hotfix 和 bisect 定位

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-RELEASE-BISECT-01 --force
cd workspaces/release-bisect/release-lab
```

## 执行

- 查看 `v1.0.0` 与 `v1.1.0` 标签。
- 运行 `git bisect` 配合 `./verify.sh` 找到坏提交。

## 观察

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=12
```

```bash
git tag -n
git show --stat v1.1.0
```

## 恢复

- 使用 `git bisect reset` 退出定位过程。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/release-bisect
```

## 预期结果

你能从症状出发定位坏提交，并说明 hotfix 应从哪里切。

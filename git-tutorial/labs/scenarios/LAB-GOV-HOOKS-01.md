# LAB-GOV-HOOKS-01: hooks、别名和配置层级

## 准备

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-HOOKS-01 --force
cd workspaces/gov-hooks/governance-lab
```

## 执行

- 查看 `core.hooksPath` 来源。
- 尝试提交 `secret.pem`，观察 hook 拦截。
- 移除敏感文件后重新提交。

## 观察

```bash
git status -sb
git log --oneline --graph --decorate --all --max-count=12
```

```bash
git config --show-origin --get core.hooksPath
git config --show-origin --list | grep alias || true
```

## 恢复

- `git restore --staged secret.pem && rm secret.pem` 后重新提交。
- hook 故障时先 `git config --unset core.hooksPath`，不要默认 `--no-verify`。

## 清理

```bash
cd git-tutorial/labs
rm -rf workspaces/gov-hooks
```

## 预期结果

你能解释本地 hook 能提醒什么，哪些规则仍必须进入 CI。

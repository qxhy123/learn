# 19 Hooks、配置与别名：把团队规则变成可观察的自动化

## 场景

你加入一个小团队后发现：有人提交调试日志，有人忘记写提交说明，有人用不同的换行规则导致 diff 很吵。团队想用 Git 自动提醒，但又不希望本地 hook 变成“只有我机器上通过”的隐形门槛。

本章目标不是让你写复杂脚本，而是建立判断：哪些规则适合本地快速提醒，哪些必须放到 CI 或代码评审里统一执行。

## 学习目标

完成本章后，你应该能够：

1. 解释 `pre-commit`、`commit-msg`、`pre-push` 的触发位置。
2. 用 `git config --show-origin --list` 判断配置来自 system、global、local 还是 worktree。
3. 设计 3-5 个低风险 alias，降低观察成本而不是隐藏危险动作。
4. 区分“本地提醒”与“团队强制门禁”。
5. 为 hooks 准备失败后的恢复路径。

## 观察点

先观察当前仓库的规则来源：

```bash
git status -sb
git config --show-origin --list | grep -E 'alias\.|core\.hooksPath|user\.|pull\.|push\.' || true
git rev-parse --git-path hooks
git hook list 2>/dev/null || true
```

如果你设置了 `core.hooksPath`，默认 `.git/hooks/` 里的脚本可能不会触发；如果你只看 `git config --global --list`，可能漏掉仓库级覆盖。

## 命令与决策

### 配置层级

Git 配置常见来源从宽到窄大致是：

| 层级 | 适合放什么 | 风险 |
|---|---|---|
| system | 公司镜像或统一环境默认值 | 影响所有用户，个人不应随意改 |
| global | 个人身份、个人 alias、编辑器偏好 | 会影响所有本机仓库 |
| local | 某仓库协作规则，如 pull 策略、hooksPath | 只影响当前仓库，适合团队约定 |
| worktree | 同一仓库多工作树差异 | 容易被忽略，必须有说明 |

观察配置来源比记住优先级更重要：

```bash
git config --show-origin --get user.email
git config --show-origin --get pull.rebase
git config --show-origin --get core.hooksPath
```

### 好 alias 的标准

好 alias 应该让观察更快，而不是把危险操作伪装成无害快捷键。

推荐：

```bash
git config --global alias.st 'status -sb'
git config --global alias.lg 'log --oneline --graph --decorate --all'
git config --global alias.last 'show --stat --summary HEAD'
git config --global alias.unstage 'restore --staged'
```

谨慎：

```bash
git config --global alias.undo 'reset --hard HEAD~1'
```

上面这种 alias 把高风险命令隐藏起来，容易让人误操作。危险动作应该保留完整命令、配观察步骤和恢复路径。

### hooks 的边界

本地 hook 适合：

- 快速检查 staged 内容。
- 拦截明显错误，如密钥、调试日志、空提交说明。
- 给出可读的修复建议。

CI/服务端规则适合：

- 必须全员一致的测试、构建、合规检查。
- 需要标准环境或密钥的检查。
- 不能依赖个人机器安装状态的门禁。

一个最小 `pre-commit` 示例：

```bash
mkdir -p .githooks
cat > .githooks/pre-commit <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if git diff --cached --name-only | grep -E '\.(pem|key)$' >/dev/null; then
  echo 'Refusing to commit private-key-like files.' >&2
  echo 'Move the file out of the repository or add a safe placeholder.' >&2
  exit 1
fi
EOF
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
```

## 实验

- Lab id：`LAB-GOV-HOOKS-01`
- 场景文件：[../labs/scenarios/LAB-GOV-HOOKS-01.md](../labs/scenarios/LAB-GOV-HOOKS-01.md)
- 生成脚本：

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-HOOKS-01 --force
```

实验重点：观察 `core.hooksPath` 来源、触发一次失败的 `pre-commit`、修复 staged 内容后重新提交。

## 常见错误

1. **把 hook 当成唯一门禁**：本地 hook 可以被跳过，关键规则必须在 CI 或服务端保护里重复执行。
2. **alias 隐藏危险命令**：`reset --hard`、`push --force` 不应该被包装成短 alias。
3. **只改 global 配置**：团队规则应写在仓库文档中，必要时通过 `core.hooksPath` 或模板脚本显式安装。
4. **hook 输出不可读**：失败信息如果不告诉用户如何修复，会变成协作摩擦。

## 风险提示与恢复路径

> 危险动作：修改 hooks、`core.hooksPath` 或提交前检查会改变团队提交体验；错误 hook 可能阻塞所有本地提交。

恢复路线：

```bash
git config --show-origin --get core.hooksPath
git config --unset core.hooksPath     # 只取消当前仓库设置
chmod -x .githooks/pre-commit         # 临时停用单个 hook
SKIP=1 git commit                     # 仅当团队约定允许时使用脚本自定义跳过方式
```

不要把 `--no-verify` 当成常规流程。它只适合“hook 本身故障、且 CI 仍会兜底”的应急情况，并应在 PR 说明里写明原因。

## 验收

回答以下问题：

1. 当前仓库的 `core.hooksPath` 来自哪里？没有设置时 Git 会去哪里找 hooks？
2. 哪三个 alias 能提高你当前团队的观察效率？有没有隐藏危险操作？
3. 如果一个 hook 在你机器上失败但同事没有失败，你会先检查哪三个点？
4. 哪些规则必须移到 CI，而不能只依赖本地 hooks？

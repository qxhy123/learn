# 19 Hooks、配置与别名：把团队规则变成可观察的自动化

## 场景

你加入一个小团队后发现：有人提交调试日志，有人忘记写提交说明，有人用不同的换行规则导致 diff 很吵。团队想用 Git 自动提醒，但又不希望本地 hook 变成“只有我机器上通过”的隐形门槛。

本章目标不是让你写复杂脚本，而是建立判断：哪些规则适合本地快速提醒，哪些必须放到 CI、代码评审或服务端保护里统一执行。

## 学习目标

完成本章后，你应该能够：

1. 解释 `pre-commit`、`commit-msg`、`pre-push` 的触发位置和适用边界。
2. 用 `git config --show-origin --list` 判断配置来自 system、global、local 还是 worktree。
3. 设计 3 到 5 个低风险 alias，降低观察成本而不是隐藏危险动作。
4. 区分“本地提醒”与“团队强制门禁”。
5. 为 hooks、配置和 alias 准备失败后的恢复路径。

## 前置与后续章节

- 前置章节：[18 Blame, Bisect and History Debugging](../06-release-and-debugging/18-blame-bisect-and-history-debugging.md)：你已经能用历史定位问题，现在把可重复的低级错误前移到提交前发现。
- 后续章节：[20 Monorepo、LFS 与大仓库](20-monorepo-lfs-and-large-repos.md)：规模继续变大后，需要把本章的本地规则扩展成仓库体积、文件类型和 ownership 治理。

## 观察点

先观察当前仓库的规则来源，不要直接改配置：

```bash
git status -sb
git config --show-origin --list | grep -E 'alias\.|core\.hooksPath|user\.|pull\.|push\.' || true
git rev-parse --git-path hooks
ls -la "$(git rev-parse --git-path hooks)" 2>/dev/null || true
```

关注四个问题：

- `core.hooksPath` 有没有设置？如果设置了，默认 `.git/hooks/` 里的脚本可能不会触发。
- alias 来自 `--global` 还是当前仓库？个人习惯不应该悄悄变成团队约定。
- `user.email`、`pull.rebase`、`push.default` 是否被仓库级配置覆盖？
- hook 失败时有没有可读提示和恢复路线？

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

如果你要写入配置，先明确作用域：

```bash
git config --local pull.rebase false       # 当前仓库约定
git config --global core.editor "code --wait" # 个人偏好
git config --worktree feature.flag true    # 当前 worktree 特例
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

谨慎或避免：

```bash
git config --global alias.undo 'reset --hard HEAD~1'
git config --global alias.force 'push --force'
```

上面这类 alias 把高风险命令隐藏起来，容易让人误操作。危险动作应该保留完整命令、配观察步骤和恢复路径。

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
cat > .githooks/pre-commit <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
if git diff --cached --name-only | grep -E '\.(pem|key)$' >/dev/null; then
  echo 'Refusing to commit private-key-like files.' >&2
  echo 'Move the file out of the repository or add a safe placeholder.' >&2
  exit 1
fi
HOOK
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
```

这个 hook 只能保护“会运行它的人”。团队必须在 CI 或服务端保护里重复关键检查，避免有人用 `--no-verify`、旧客户端或未安装 hook 的环境绕过规则。

## 实验

- Lab id：`LAB-GOV-HOOKS-01`
- 场景文件：[../labs/scenarios/LAB-GOV-HOOKS-01.md](../labs/scenarios/LAB-GOV-HOOKS-01.md)
- 生成脚本：

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-HOOKS-01 --force
```

实验重点：观察 `core.hooksPath` 来源、触发一次失败的 `pre-commit`、修复 staged 内容后重新提交，并记录哪些规则需要搬到 CI。

建议记录表：

| 步骤 | 观察命令 | 你看到什么 | 决策 |
|---|---|---|---|
| 配置来源 | `git config --show-origin --get core.hooksPath` | 例如 `.git/config` | 是否是团队约定 |
| hook 触发 | `git commit` | 失败提示是否可读 | 修复 staged 内容还是调整 hook |
| 团队门禁 | CI 检查列表 | 哪些规则本地可跳过 | 是否需要服务端兜底 |

## 常见错误

1. **把 hook 当成唯一门禁**：本地 hook 可以被跳过，关键规则必须在 CI 或服务端保护里重复执行。
2. **alias 隐藏危险命令**：`reset --hard`、`push --force` 不应该被包装成短 alias。
3. **只改 global 配置**：团队规则应写在仓库文档中，必要时通过 `core.hooksPath` 或模板脚本显式安装。
4. **hook 输出不可读**：失败信息如果不告诉用户如何修复，会变成协作摩擦。
5. **跨平台假设过强**：只在 macOS 或某个 shell 可运行的 hook，不适合作为唯一门禁。

## 危险命令

以下命令不是禁止使用，而是必须先说明作用域、备份或恢复路线：

```bash
git config --global alias.undo 'reset --hard HEAD~1'
git config core.hooksPath .githooks
git commit --no-verify
git push --force
```

判断原则：

- 改 `--global` 前确认它不会影响其他仓库。
- 改 `core.hooksPath` 前确认团队知道 hook 的安装和更新方式。
- 使用 `--no-verify` 前确认 CI 或服务端保护仍会兜底，并在 PR 说明里写明原因。
- 任何 `push --force` 都应优先替换为 `--force-with-lease`，并确认目标分支不是共享保护分支。

## 恢复路径

> 危险动作：修改 hooks、`core.hooksPath`、global alias 或提交前检查会改变团队提交体验；错误 hook 可能阻塞所有本地提交。

恢复路线：

```bash
git config --show-origin --get core.hooksPath
git config --unset core.hooksPath              # 只取消当前仓库设置
git config --global --unset alias.undo         # 删除个人危险 alias
chmod -x .githooks/pre-commit                  # 临时停用单个 hook
git status -sb                                 # 确认工作区没有被额外修改
```

如果 hook 本身故障且团队允许临时绕过，可以应急使用：

```bash
git commit --no-verify
```

不要把 `--no-verify` 当成常规流程。它只适合“hook 本身故障、且 CI 仍会兜底”的应急情况，并应在 PR 说明里写明原因。

## 验收

回答以下问题：

1. 当前仓库的 `core.hooksPath` 来自哪里？没有设置时 Git 会去哪里找 hooks？
2. 哪三个 alias 能提高你当前团队的观察效率？有没有隐藏危险操作？
3. 如果一个 hook 在你机器上失败但同事没有失败，你会先检查哪三个点？
4. 哪些规则必须移到 CI，而不能只依赖本地 hooks？
5. 本章 lab id、前置章节、后续章节、危险命令和恢复路径分别是什么？

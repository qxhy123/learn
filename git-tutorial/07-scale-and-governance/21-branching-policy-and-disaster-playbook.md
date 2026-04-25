# 21 分支策略与灾难手册：把“出事怎么办”写在出事之前

## 场景

团队成员越来越多后，主干偶尔被坏提交破坏，发布分支上有人直接改历史，紧急 hotfix 时不知道该从哪里切。每个人都懂一些 Git 命令，但事故时缺少共同剧本。你需要制定最低分支规则和灾难恢复卡片。

## 学习目标

完成本章后，你应该能够：

1. 根据团队发布节奏选择 trunk-based、release branch 或 GitFlow-like 策略。
2. 定义主干保护、PR 审查、合并方式、force push 边界。
3. 写出四类事故的观察入口和第一安全动作。
4. 区分 `revert`、hotfix、回滚部署、历史重写的适用条件。
5. 建立小团队 Git 协作最低规则。

## 观察点

先看当前团队真实行为：

```bash
git branch -vv
git log --oneline --graph --decorate --all --max-count=30
git for-each-ref --format='%(refname:short) %(committerdate:relative)' refs/heads refs/remotes refs/tags | sort
git remote show origin 2>/dev/null || true
```

在平台侧还要观察：

- 主干是否禁止直接 push。
- PR 是否要求至少一名 reviewer。
- CI 是否是必需检查。
- 是否允许 merge commit、squash merge、rebase merge。
- 谁有删除分支、修改保护规则、force push 权限。

## 命令与决策

### 分支策略选择

| 团队条件 | 更适合 |
|---|---|
| 持续交付、CI 快、feature flag 成熟 | Trunk-based，短分支，高频合并 |
| 定期发布、需要维护多个版本线 | 主干 + release 分支 + hotfix 分支 |
| 重流程、多个长期环境分支 | GitFlow-like，但要控制长期分叉成本 |

策略不是口号，而是一组可执行规则：

- 从哪里切功能分支？
- 多久必须同步主干？
- 合并时保留 merge commit 还是 squash？
- 发布标签由谁打？
- 主干坏了先 revert 还是新提交修复？

### 最低团队规则模板

```text
主干：main，必须可构建。
功能分支：task/<ticket>-short-name，生命周期建议小于 3 天。
同步方式：合并前 fetch + rebase/merge main，按团队约定二选一。
合并方式：默认 squash merge；需要保留分支上下文时使用 merge commit。
保护：main 禁止直接 push，必须 1 review + CI 通过。
历史重写：已推送共享分支禁止 rebase 后强推；例外必须用 --force-with-lease 且提前通知。
发布：标签由 release owner 创建；hotfix 从最新发布标签或 release 分支切出。
事故：主干坏了优先 revert 坏提交或提交修复，不立即重写历史。
```

### 灾难恢复四入口

| 症状 | 第一观察 | 第一安全动作 |
|---|---|---|
| 本地提交丢了 | `git reflog` | 创建救援分支：`git switch -c rescue/<name> <sha>` |
| 主干合入坏提交 | `git log --oneline --graph`、CI 链接 | `git revert <bad-sha>` 或提交修复 PR |
| 远程分支被错误强推 | 平台审计、同事本地 reflog | 暂停 push，找最后好提交，恢复分支引用 |
| 发布标签打错 | `git show <tag>`、发布系统状态 | 新建修正标签或按平台规则撤销，避免静默移动公开标签 |

## 实验

- Lab id：`LAB-GOV-DISASTER-01`
- 场景文件：[../labs/scenarios/LAB-GOV-DISASTER-01.md](../labs/scenarios/LAB-GOV-DISASTER-01.md)
- 生成脚本：

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-DISASTER-01 --force
```

实验重点：为一个模拟团队填写分支策略表，并对“主干坏了”“远程被强推”“发布标签打错”三种事故写恢复卡片。

## 常见错误

1. **只规定分支名，不规定合并条件**：没有 CI、review 和同步要求，分支策略只是命名规范。
2. **事故时第一反应是 reset + force push**：共享历史优先考虑 `revert` 或新修复提交。
3. **发布标签可以随便移动**：公开标签是团队和发布系统的锚点，移动前必须有公告和审计。
4. **把权限给所有人**：权限越大越需要流程；主干保护是协作系统的一部分。

## 风险提示与恢复路径

> 危险动作：`git push --force`、删除远程分支、移动公开标签、直接修改主干保护规则都会影响团队共享状态。

更安全的替代：

```bash
git push --force-with-lease origin your-branch   # 仅限自己独占分支，且先 fetch
git revert <bad-sha>                            # 共享主干优先反向提交
git switch -c rescue/<incident> <last-good-sha> # 先保留证据和恢复入口
```

事故恢复顺序：

1. 冻结：暂停相关分支 push，避免继续扩大损害。
2. 观察：记录当前 HEAD、远程引用、CI 状态、相关人员本地 reflog。
3. 备份：创建 `rescue/*` 分支或镜像备份。
4. 恢复：选择 revert、修复提交、恢复引用或重新发布。
5. 复盘：把触发条件写回分支策略和 CI 规则。

## 验收

为你的团队写出一页 Git 协作最低规则，至少包含：

- 主干保护规则。
- 功能分支命名和最长生命周期。
- 合并方式和同步主干方式。
- 允许历史重写的边界。
- 发布标签责任人。
- 三条事故恢复卡片：本地丢提交、主干坏、远程误强推。

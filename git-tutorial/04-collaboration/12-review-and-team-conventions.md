# 12 Review 与团队约定

## 本章导航

- 前置章节： [11 同步主干并打开 PR](11-sync-with-main-and-open-pr.md)
- 后续章节： [13 Undo Local Changes](../05-recovery/13-undo-local-changes.md)
- 本章 Lab ID：`LAB-COLLAB-PR-01`

## 场景

你已经把任务分支推到远程，并打开了 PR。接下来 Git 不再只是个人命令工具，而是团队协作协议：reviewer 如何阅读历史、CI 如何阻止坏提交、主干如何保持可发布、哪些分支允许改写，都需要在团队内明确约定。

本章训练你把 PR 看成“历史治理接口”，而不只是一个合并按钮。一个好的 PR 应该让 reviewer 快速回答三件事：这次改动从哪里开始、包含哪些提交、如果出问题如何撤回。

## 学习目标

完成本章后，你应该能够：

1. 把 PR 理解为团队共享历史进入主干前的治理入口。
2. 准备 reviewer 能高效理解的提交历史、diff 范围和 PR 描述。
3. 比较 merge commit、squash merge、rebase merge 对主干历史的长期影响。
4. 判断 review 期间何时可以追加提交，何时不应改写远程分支历史。
5. 为小团队制定最低限度的 Git 协作约定。

## 观察点

review 前后继续使用命令行确认真实状态，不只依赖网页 UI：

```bash
git status -sb
git fetch origin
git branch -vv
git diff --stat origin/main...HEAD
git log --oneline --graph --decorate --all -n 16
```

你要能判断：

- 当前工作区是否干净，是否还有未提交或未暂存的 review 修复。
- `origin/main...HEAD` 是否正是本次 PR 的范围。
- review 讨论的是哪一组提交，PR 更新后 reviewer 是否只需看增量。
- CI 检查对应的是哪个提交 SHA，而不是旧的已过期提交。
- 最终合并方式会在主干上留下怎样的历史形态。

## 命令与判断

整理 PR 范围：

```bash
git fetch origin
git diff origin/main...HEAD
git log --oneline origin/main..HEAD
```

根据 review 意见追加一个清晰修复提交：

```bash
git add <files>
git commit -m "fix: address review feedback"
git push
```

这种方式保留 reviewer 已看过的提交上下文，适合 review 正在进行、评论已经绑定到旧提交的情况。

如果团队允许在个人 PR 分支上整理历史，可以在通知 reviewer 后执行：

```bash
git fetch origin
git rebase -i origin/main
git push --force-with-lease
```

`--force-with-lease` 仍然是改写远程分支历史，只是会先检查远程分支是否仍处在你上次观察到的位置。它不能替代团队约定，也不能用于受保护主干或多人共用分支。

## PR 描述最小模板

一个可 review 的 PR 描述至少包含：

```text
背景：为什么需要这个改动
方案：主要改了什么，边界在哪里
验证：运行了哪些自动或手工检查
风险：可能影响哪些路径，哪些点需要 reviewer 重点看
回滚：如果合并后出问题，如何撤回或缓解
```

如果 PR 中既有功能改动又有重构，应优先拆分。拆不开时，在描述中明确哪些提交或文件是行为变化，哪些只是整理。

## 团队约定最小集

一个小团队至少需要写清楚：

1. 主干是否受保护，是否允许直接 push。
2. PR 必须通过哪些检查才能合并，例如测试、lint、构建、人工 review 数量。
3. 默认合并策略：merge commit、squash merge 还是 rebase merge。
4. PR 打开后是否允许 force push；允许时要如何通知 reviewer。
5. 分支命名、提交信息和 PR 描述的最低格式。
6. 紧急修复、回滚和发布分支由谁批准。
7. CI 失败时谁负责判断是代码问题、测试问题还是基础设施问题。

这些约定不是形式主义。它们决定了未来排障时能否看懂历史，也决定了新人是否敢安全协作。

## 三种合并策略

| 策略 | 主干历史形态 | 优点 | 代价 |
|---|---|---|---|
| merge commit | 保留分支汇入主干的节点 | 真实记录协作轨迹，容易看出一次 PR 的边界 | 主干历史可能不够线性 |
| squash merge | 每个 PR 压成一个主干提交 | 主干按主题单元前进，回滚单个 PR 较直接 | 丢失分支内部细粒度提交和提交级讨论上下文 |
| rebase merge | 把 PR 提交线性放到主干后 | 主干直线清晰，适合按提交阅读 | 提交身份可能变化，需要团队接受线性治理方式 |

没有一种策略永远正确；错误的是团队没有边界，却让每个人用个人习惯塑造共享历史。

## 实验

Lab ID：`LAB-COLLAB-PR-01`

实验步骤：

1. 为一个小功能创建任务分支，准备两到三个有意图的提交。
2. 写一段 PR 描述，包含背景、方案、验证、风险、回滚。
3. 模拟 reviewer 提出两个意见：一个需要代码修复，一个只需要补充说明。
4. 先用追加提交的方式处理 review，记录 `git log --oneline origin/main..HEAD` 的变化。
5. 在另一条临时分支上模拟交互式 rebase 整理提交，再对比 reviewer 需要重新阅读的内容。
6. 为一个三人小团队写出最低 Git 协作规则，并标明是否允许 PR 分支 force push。

预期结果：学习者能把 PR 从“合并按钮”提升为“团队历史治理接口”，并能解释不同 review 更新方式对团队成本的影响。

## 常见错误

- **CI 通过就认为 PR 一定准备好了。** CI 只能覆盖部分自动检查，不能保证历史边界、风险说明和 review 语义清晰。
- **把无关重构混进功能 PR。** 它会扩大风险并降低 review 质量；reviewer 也更难判断回滚边界。
- **PR 打开后随意 force push。** 这可能让既有评论、比较基线和 reviewer 上下文失效。
- **团队合并策略靠个人偏好。** 主干历史会长期变得难以预测，新人也不知道该如何安全更新分支。
- **只在网页上看状态，不回到本地观察。** 本地分支、远程跟踪分支和 CI 对应 SHA 可能并不一致。

## 危险命令

以下动作会影响共享历史或绕过团队治理，执行前必须确认分支边界和团队规则：

```bash
git push --force
git push --force-with-lease
git rebase -i origin/main
git push origin HEAD:main
git merge --no-verify
```

危险点：

- `git push --force` 可能覆盖远程已有提交，不检查别人是否已经更新。
- `git push --force-with-lease` 比 `--force` 安全，但仍会改写远程分支历史。
- `git rebase -i origin/main` 会重写当前分支上的提交身份；PR 已打开时可能破坏 reviewer 上下文。
- `git push origin HEAD:main` 可能绕过正常 PR 流程直接更新主干，若主干未受保护会造成协作事故。
- `--no-verify` 会跳过本地 hook；如果团队依赖 hook 做提交检查，必须说明原因并补跑等价验证。

## 恢复路径

出现协作事故时，先停止继续推送，再恢复可审计状态：

1. 如果误把未审查提交直接 push 到主干，立即通知团队，保留现场输出，优先创建 revert PR 或修复 PR，不继续堆叠新的未审查提交。
2. 如果 review 过程中改写了 PR 分支历史，立即在 PR 中说明重写原因、新的提交范围和重新运行的验证结果。
3. 如果强推覆盖了别人提交，先从远程平台事件记录、同事本地 clone、CI checkout 日志或 `git reflog` 找到被覆盖 SHA，再由团队决定恢复分支还是创建补救提交。
4. 如果 CI 对旧提交已经通过但新提交未跑完，不要合并；等待当前提交 SHA 的检查完成。
5. 如果无法判断是否安全，创建备份分支保存当前状态：

```bash
git branch backup/review-incident-$(date +%Y%m%d-%H%M%S)
git status -sb
git log --oneline --graph --decorate --all -n 24
```

## 验收

你应该能回答：

1. PR 为什么是团队共享历史进入主干前的治理接口？
2. 三种合并策略分别会怎样影响主干历史和回滚边界？
3. PR 打开后改写历史需要满足哪些团队前提？
4. 一个小团队最少需要约定哪些 Git 规则，才能避免协作事故？
5. 为什么 CI 通过后仍要确认当前提交 SHA、PR 范围和回滚路径？

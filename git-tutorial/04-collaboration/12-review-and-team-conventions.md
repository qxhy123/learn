# 12 Review 与团队约定

## 场景

你的代码已经推送并打开 PR。接下来 Git 不再只是个人命令工具，而是团队协作协议：reviewer 如何阅读历史、CI 如何阻止坏提交、主干如何保持可发布、哪些分支允许改写，都需要明确约定。

## 学习目标

完成本章后，你应该能够：

1. 把 PR 理解为历史治理接口，而不只是代码审查页面。
2. 准备一个 reviewer 能高效理解的分支和 PR 描述。
3. 比较 merge commit、squash merge、rebase merge 的长期影响。
4. 为小团队制定最低限度的 Git 协作规则。

## 状态观察

review 前后继续使用命令行确认真实状态：

```bash
git status -sb
git fetch origin
git branch -vv
git diff --stat origin/main...HEAD
git log --oneline --graph --decorate --all -n 16
```

你要能判断：

- review 讨论的是哪一组提交。
- PR 更新后，reviewer 是否需要重新阅读整个 diff，还是只看增量。
- CI 检查对应的是哪个提交 SHA。
- 合并方式会在主干上留下怎样的历史形态。

## 命令

整理 PR 范围：

```bash
git diff origin/main...HEAD
git log --oneline origin/main..HEAD
```

根据 review 修改后追加提交：

```bash
git add <files>
git commit -m "fix: address review feedback"
git push
```

如果团队允许在 PR 分支上整理历史：

```bash
git rebase -i origin/main
git push --force-with-lease
```

是否这样做取决于团队约定。若 reviewer 按提交逐条评论，频繁改写历史会增加审查成本。

## 团队约定最小集

一个小团队至少需要写清楚：

1. 主干是否受保护，是否允许直接 push。
2. PR 必须通过哪些检查才能合并。
3. 默认合并策略：merge commit、squash merge 还是 rebase merge。
4. PR 打开后是否允许 force push，允许时要如何通知 reviewer。
5. 分支命名、提交信息和 PR 描述的最低格式。
6. 紧急修复、回滚和发布分支由谁批准。

这些约定不是形式主义。它们决定了未来排障时能否看懂历史，也决定了新人是否敢安全协作。

## 三种合并策略

### merge commit

保留主题分支汇入主干的事实。适合重视分支上下文、需要还原真实协作轨迹的团队。

### squash merge

把 PR 压成一个主干提交。适合希望主干每次合并都是一个完整主题单元的团队，但会丢失分支内部细粒度提交。

### rebase merge

让主干保持线性历史。适合偏好直线历史的团队，但需要明确 review 期间的提交身份变化边界。

没有一种策略永远正确；错误的是团队没有边界，却让每个人用个人习惯塑造共享历史。

## 实验

Lab ID：`LAB-COLLAB-PR-01`

实验步骤：

1. 为一个小功能创建任务分支，准备两到三个有意图的提交。
2. 写一段 PR 描述，包含背景、方案、验证、风险、回滚。
3. 分别模拟追加 review 修复提交与交互式 rebase 整理提交。
4. 对比 reviewer 在两种方式下需要重新阅读的内容。
5. 为一个三人小团队写出最低 Git 协作规则。

预期结果：学习者能把 PR 从“合并按钮”提升为“团队历史治理接口”。

## 常见错误

- **CI 通过就认为 PR 一定准备好了。** CI 只能覆盖部分自动检查，不能保证历史边界和 review 语义清晰。
- **把无关重构混进功能 PR。** 它会扩大风险并降低 review 质量。
- **PR 打开后随意 force push。** 这可能让既有评论、比较基线和 reviewer 上下文失效。
- **团队合并策略靠个人偏好。** 主干历史会长期变得难以预测。

## 危险提示与恢复路径

危险动作：在受保护分支绕过 PR 直接 push，或在 review 进行中不通知就改写 PR 分支历史。

恢复路径：

1. 直接 push 到主干后，优先创建修复 PR 或 revert PR，不继续堆叠未审查提交。
2. 改写 PR 分支后，立即在 PR 中说明重写原因和新的验证结果。
3. 如果 CI 对旧提交已经通过但新提交未跑完，不要合并；等待当前提交 SHA 的检查完成。

## 验收

你应该能回答：

1. PR 为什么是历史治理接口？
2. 三种合并策略分别会怎样影响主干历史？
3. PR 打开后改写历史需要满足哪些团队前提？
4. 一个小团队最少需要约定哪些 Git 规则，才能避免协作事故？

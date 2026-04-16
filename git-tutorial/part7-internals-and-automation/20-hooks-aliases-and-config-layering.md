# 第20章：hooks、aliases 与配置层级

## 学习目标

完成本章后，你应能：

1. 理解 Git hooks 的触发时机和典型用途
2. 配置常用 alias 提升命令效率
3. 理解 system、global、local 三层配置覆盖关系
4. 判断哪些规则适合放本地 hook，哪些必须交给 CI

---

## 20.1 hooks：在 Git 生命周期里插入自动化动作

Git hooks 是一组在特定时机触发的脚本，例如：

- `pre-commit`
- `commit-msg`
- `pre-push`

它们常用于：

- 阻止明显错误提交
- 运行格式化和测试
- 检查提交说明格式
- 拦截敏感文件或密钥

hook 的本质，不是“让 Git 更复杂”，而是让团队把一部分规则变成机器可执行的前置检查。

---

## 20.2 一个最小 `pre-commit` 示例

例如你想阻止提交包含 `console.log` 的代码：

```bash
#!/usr/bin/env bash
if git diff --cached | grep -q "console\\.log"; then
  echo "Detected console.log in staged changes"
  exit 1
fi
```

把它放进 `.git/hooks/pre-commit` 并赋予可执行权限后，每次提交前都会运行。

这个例子最重要的不是脚本本身，而是让你意识到：

- hook 主要看 staged 结果
- 它拦的是“进入历史前”的问题

---

## 20.3 哪些规则适合 hook，哪些更适合 CI

### 更适合本地 hook 的

- 快速、无副作用、能给即时反馈的检查
- 明显低级错误
- 提交说明格式

### 更适合 CI 的

- 团队必须统一执行、不能依赖个人环境的检查
- 耗时较长的测试
- 需要标准化运行环境的构建流程

一个很重要的判断标准是：

> 本地 hook 适合“尽快提醒你”，CI 适合“统一约束全团队”。

---

## 20.4 alias：让高频动作更低成本

配置别名：

```bash
git config --global alias.lg "log --oneline --graph --decorate --all"
git config --global alias.st status
git config --global alias.co switch
```

之后可以直接：

```bash
git lg
git st
git co main
```

alias 的价值不是炫技，而是让“高频观察动作”更容易变成习惯。

如果某个 alias 会让你忘记它原本干了什么，通常就写得太花了。

---

## 20.5 配置层级再看一次：system / global / local

这一章要把第 2 章的配置知识再推进一步：

- `system`：机器级约束
- `global`：用户习惯
- `local`：仓库专属策略

团队项目里，很多“为什么我这里行为不一样”的问题，最终都能回到配置层级。

最推荐的观察命令仍然是：

```bash
git config --show-origin --list
```

它比背优先级更有用，因为它直接告诉你“值来自哪里”。

---

## 20.6 自动化的边界：不要把所有规则都塞进 hook

hook 有价值，但别把它神化。

几个常见边界：

- 本地 hook 可以被绕过
- 团队成员环境不一致
- 过重的 hook 会明显拖慢开发体验
- 某些检查必须在标准容器或 CI 环境里跑

所以更成熟的做法通常是：

- hook 负责快反馈
- CI 负责强一致

---

## 20.7 一个高阶视角：Git 不只是工具，也是工作流接口

当你开始使用：

- hook
- alias
- 分层配置
- CI 触发

Git 就不再只是“命令集合”，而开始变成：

> 团队工作流的入口和边界控制器。

理解这一点，会帮助你更自然地看待第 22-24 章里的治理问题。

---

## 常见误区

- **误区 1：有了 hook 就不用 code review。**
  hook 只能拦机械问题，拦不了设计问题。

- **误区 2：alias 越短越酷越好。**
  可读性和可记忆性比炫技重要。

- **误区 3：本地 hook 能替代 CI。**
  它们解决的是不同层级的问题。

- **误区 4：配置覆盖关系不用管。**
  很多“为什么配置没生效”都和它有关。

---

## 本章练习

1. 为自己添加一个 `git lg` 别名。
2. 写一个最简单的 `pre-commit` hook，阻止提交某个关键词。
3. 在 global 和 local 中分别设置同名配置，观察优先级。
4. 思考题：为什么团队一致性规则通常不能只依赖本地 hook？

---

## 本章小结

这一章把 Git 从“命令行工具”推进到了“工作流平台”。下一章我们继续讨论仓库组合方式：多工作目录、子模块和更复杂的仓库边界。

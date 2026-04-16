# 第1章：Shell 与命令基础

## 本章解决什么问题

如果你以前主要用过 CMD、Bash，或者几乎没系统学过 Shell，那么 PowerShell 最容易让人困惑的地方是：

- 命令为什么长得像 `Verb-Noun`
- 参数为什么有时写成 `-Name value`
- 别名、函数、脚本、cmdlet 到底有什么区别
- 执行脚本时为什么还会碰到执行策略

本章先不急着讲对象和脚本，而是先把“如何稳定打开 PowerShell、找到命令、运行命令、理解命令形态”这件事做扎实。

---

## 学习目标

完成本章后，你应能：

1. 解释 PowerShell 与 CMD / Bash 的基本定位差异
2. 看懂常见命令、参数和别名的基本形态
3. 用帮助系统和命令发现机制探索新命令
4. 对执行策略形成正确的第一印象

---

## 1.1 PowerShell 到底是什么

PowerShell 同时是两样东西：

- 一个交互式 Shell
- 一门脚本语言

这意味着你可以：

- 直接输入命令查询系统状态
- 把几条命令连起来做批量处理
- 把处理逻辑保存成脚本
- 再把脚本整理成函数、模块和工具

它和 CMD 的区别不只是“命令更现代”，而是它从一开始就把系统管理、对象处理和脚本化放在一起设计。

它和 Bash 的区别也不只是平台，而是默认管道里传递的核心单位不同。这个差异会在下一章真正展开。

---

## 1.2 PowerShell 命令为什么经常长成 `Verb-Noun`

PowerShell 的很多内建命令遵循 `Verb-Noun` 约定，例如：

```powershell
Get-Process
Set-Location
New-Item
Remove-Item
```

这种写法的好处是：

- 动词先告诉你“它要做什么”
- 名词再告诉你“它作用于什么对象”
- 命令发现变得更有规律

比如你忘了怎么列服务，不一定要靠记忆，可以先想“我要获取服务”，再试：

```powershell
Get-Command -Verb Get -Noun *Service*
```

这比纯靠背命令稳得多。

---

## 1.3 一条命令通常由哪几部分组成

最常见的 PowerShell 命令一般包含：

1. 命令名
2. 参数名
3. 参数值

例如：

```powershell
Get-ChildItem -Path . -Force
```

这里：

- `Get-ChildItem` 是命令名
- `-Path` 和 `-Force` 是参数
- `.` 是 `-Path` 的参数值

你会发现：

- 有些参数需要值，如 `-Path`
- 有些参数像开关，只要出现就生效，如 `-Force`

如果你看到命令很长，不要慌，先拆结构，再看帮助。

---

## 1.4 帮助系统是第一生产力

以后遇到陌生命令，默认先这样看：

```powershell
Get-Help Get-ChildItem
Get-Help Get-ChildItem -Examples
Get-Help Get-ChildItem -Detailed
```

建议你把 `-Examples` 当成第一入口，因为它最接近“我要怎么用”。

如果你联网并且环境允许，还可以更新帮助：

```powershell
Update-Help
```

但这不是学 PowerShell 的前提。更重要的是养成“先看帮助，再动手”的习惯。

---

## 1.5 如何发现你并不知道的命令

PowerShell 不要求你先知道完整命令名。

常见做法有三类：

```powershell
Get-Command *service*
Get-Command -Verb Get
Get-Command -Noun *Item*
```

如果你知道自己想做的事，也可以从动词反推：

- 想查看：`Get-*`
- 想新建：`New-*`
- 想删除：`Remove-*`
- 想导入：`Import-*`
- 想导出：`Export-*`

这就是 `Verb-Noun` 约定带来的真正价值：它降低了“我必须先记住全名”的压力。

---

## 1.6 别名很方便，但不要把它当长期依赖

PowerShell 中常见别名包括：

```powershell
ls
cd
cat
pwd
```

它们让新手更容易上手，但你要尽快知道这些别名对应的真实命令是什么：

```powershell
Get-Alias ls
Get-Alias cat
```

原因很现实：

- 别名不一定跨平台完全一致
- 别名会让脚本可读性下降
- 团队协作时，显式命令更清楚

结论很简单：

- 交互式输入可以适度用别名
- 正式脚本、文档、教程示例尽量写完整命令名

---

## 1.7 执行策略是什么，不是什么

很多人第一次运行脚本时会遇到执行策略相关报错，于是把执行策略误认为某种“绝对安全防线”。

这是不准确的。

你现在只需要建立第一层判断：

- 执行策略影响的是脚本执行体验，尤其是 Windows 上
- 它是安全提示和管理边界的一部分
- 它不是一个真正意义上的沙箱

先查看当前状态：

```powershell
Get-ExecutionPolicy -List
```

如果以后需要调整，也应优先理解作用域：

- `Process`
- `CurrentUser`
- `LocalMachine`

初学阶段最重要的不是立刻改它，而是不要形成“报错了就全局放开”的习惯。

---

## 1.8 最小观察实验

建议你现在就做一遍：

```powershell
$PSVersionTable
Get-Command Get-Process
Get-Help Get-Process -Examples
Get-Alias ls
```

观察时重点看三件事：

1. 你的环境里到底是 `pwsh` 还是 `powershell`
2. 帮助系统能给出什么信息
3. 别名和真实命令之间是什么关系

---

## 常见误区

- **误区 1：PowerShell 就是 Windows 版 Bash。**
  它有 Shell 属性，但它的对象模型完全不同。

- **误区 2：别名越多，说明越熟。**
  真正成熟的习惯是能在需要时切回完整命令名。

- **误区 3：执行策略报错时，第一反应就是全局关闭限制。**
  更稳妥的做法是先理解作用域和当前场景。

- **误区 4：学命令靠背全名。**
  更有效的方式是借助 `Get-Command` 和命名规律。

---

## 本章练习

1. 用 `Get-Command -Verb Get` 找出 5 个你没见过的命令。
2. 对 `Get-Process` 和 `Get-Service` 分别查看 `-Examples` 帮助。
3. 用 `Get-Alias` 找出 `ls`、`pwd`、`cat` 对应的真实命令。
4. 运行一次 `Get-ExecutionPolicy -List`，说清每个作用域代表什么。
5. 思考题：为什么“先会查命令”比“先背命令表”更重要？

---

## 本章小结

本章最重要的收获只有一句话：

> 学 PowerShell 的第一步不是记住很多命令，而是建立一套“查帮助、找命令、拆结构”的稳定动作。

下一章开始，我们进入真正的核心：为什么 PowerShell 里流动的是对象，而不是纯文本。

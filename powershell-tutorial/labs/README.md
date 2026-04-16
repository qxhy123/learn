# PowerShell Tutorial Labs

这套教程刻意不依赖复杂的自动生成脚本，而是提供 4 组轻量、可重复的手工练习场景。

原因很简单：

- PowerShell 经常运行在不同机器、不同策略、不同权限环境里
- 与其强行统一生成器，不如先训练你自己搭环境、看对象、评估风险的能力

---

## 使用方式

每次练习时，建议固定准备一个临时目录，例如：

```powershell
New-Item -ItemType Directory -Path .\ps-lab -Force | Out-Null
Set-Location .\ps-lab
```

如果你在 Windows 上，也建议再开一个只读观察窗口，专门跑：

```powershell
Get-Help <CommandName> -Examples
<CommandName> | Get-Member
<CommandName> | Select-Object -First 5
```

---

## 实验 A：命令探索

适用章节：

- 第 1-2 章

目标：

- 熟悉帮助系统
- 理解 Verb-Noun 命名
- 第一次建立“命令输出是对象”的直觉

建议任务：

1. 用 `Get-Command -Verb Get` 列出部分命令
2. 对 `Get-Process`、`Get-Service`、`Get-Date` 分别跑一次 `Get-Member`
3. 用 `Select-Object` 和 `Format-Table` 比较“数据处理”和“显示格式”的差别

---

## 实验 B：数据与 Provider

适用章节：

- 第 3-4 章

目标：

- 操作文件、目录、环境变量
- 在文本、CSV、JSON 和对象之间来回转换
- 用数组、哈希表和控制流做简单数据处理

建议任务：

1. 手工创建 3 个文本文件，再用 `Get-ChildItem` 和 `Select-Object` 查看元数据
2. 新建一个包含几行用户数据的 CSV，再用 `Import-Csv` 做筛选和排序
3. 构造一个哈希表，转成 JSON，再读回对象并验证属性

---

## 实验 C：脚本与模块

适用章节：

- 第 5 章

目标：

- 把交互式命令整理成脚本
- 为脚本加入参数
- 把一个函数放进模块并导入测试

建议任务：

1. 写一个接收路径参数的脚本，输出目录下最大的 5 个文件
2. 把核心逻辑提成函数，增加 `-Top` 参数
3. 把函数放进 `.psm1` 并用 `Import-Module` 进行调用

---

## 实验 D：自动化与安全

适用章节：

- 第 6-8 章

目标：

- 在自动化场景里先观察风险，再做执行
- 练习错误处理、日志和 `-WhatIf`
- 形成“先只读、再预演、最后执行”的习惯

建议任务：

1. 写一个批量清理临时文件的函数，先只输出将删除的对象
2. 给函数加上 `SupportsShouldProcess`，测试 `-WhatIf`
3. 为失败场景加入 `try/catch` 和日志输出

---

## 练习原则

- 先在测试目录做，不要直接对真实系统目录动手
- 批量修改前，先用 `-WhatIf` 或只读命令演练
- 对不认识的输出先跑 `Get-Member`
- 如果你在教程里第一次见到某个命令，先看帮助，再继续跟练

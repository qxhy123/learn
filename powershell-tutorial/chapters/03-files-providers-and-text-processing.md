# 第3章：文件、Provider 与文本处理

## 本章解决什么问题

真正开始用 PowerShell 做事时，你很快会碰到三类对象：

- 文件和目录
- 路径背后的 Provider
- 文本、CSV、JSON 这类外部数据

如果这一章没打稳，后面脚本写出来通常会有两个问题：

- 只会处理屏幕上看见的内容，不会处理结构化对象
- 以为所有路径都只是文件系统路径，不知道 PowerShell 还能把环境变量、注册表等暴露成类似目录树

---

## 学习目标

完成本章后，你应能：

1. 稳定操作文件和目录对象
2. 解释 Provider 的基本作用
3. 在文本、CSV、JSON 和对象之间做转换
4. 区分“路径空间”和“文件内容处理”是两类问题

---

## 3.1 文件系统对象不是只有文件名

运行：

```powershell
Get-ChildItem | Select-Object -First 5
Get-ChildItem | Get-Member
```

你会看到目录项除了名字之外，还有很多属性：

- `Name`
- `FullName`
- `Length`
- `CreationTime`
- `LastWriteTime`

这意味着你完全可以按对象属性做筛选，而不是靠字符串拼接：

```powershell
Get-ChildItem -File |
    Where-Object Length -gt 1MB |
    Sort-Object Length -Descending |
    Select-Object Name, Length, LastWriteTime
```

---

## 3.2 路径不只属于文件系统

PowerShell 的 Provider 机制会把不同数据源暴露成类似路径空间。

先看有哪些 Provider：

```powershell
Get-PSProvider
```

常见的包括：

- `FileSystem`
- `Environment`
- `Alias`
- `Variable`

在 Windows 上你还经常会看到：

- `Registry`

这意味着你可以像切目录一样切到某些非文件系统空间，例如：

```powershell
Set-Location Env:
Get-ChildItem
```

这会让你更容易理解 PowerShell 的统一视角：

> 它关心的不只是文件，而是“对象 + 路径空间 + 命令操作”。

---

## 3.3 常见文件操作命令

最常用的几类命令包括：

```powershell
Get-ChildItem
New-Item
Copy-Item
Move-Item
Remove-Item
Get-Content
Set-Content
Add-Content
```

但真正的关键不是命令名，而是习惯：

- 删除前先确认对象集合
- 覆盖前先确认目标路径
- 递归操作前先确认作用域

例如：

```powershell
Get-ChildItem .\logs -File
Get-ChildItem .\logs -File | Remove-Item -WhatIf
```

先只读观察，再预演，再执行。

---

## 3.4 `Get-Content` 和 `Import-Csv` 不是一回事

这也是新手高频混淆点。

### `Get-Content`

更适合读取文本内容：

```powershell
Get-Content .\app.log
Get-Content .\config.json -Raw
```

### `Import-Csv`

更适合把结构化表格直接变成对象：

```powershell
Import-Csv .\users.csv
```

导入后再查看结构：

```powershell
Import-Csv .\users.csv | Get-Member
```

你会发现 CSV 的每一行都变成了带属性的对象。

---

## 3.5 JSON 处理的正确姿势

如果你有 JSON 文件，不要只把它当文本操作。更稳妥的方式是：

```powershell
Get-Content .\data.json -Raw | ConvertFrom-Json
```

然后再：

```powershell
$data = Get-Content .\data.json -Raw | ConvertFrom-Json
$data | Get-Member
```

写回 JSON 时：

```powershell
$data | ConvertTo-Json -Depth 5
```

这里的 `-Depth` 很重要，因为嵌套对象比较深时，默认深度可能不够。

---

## 3.6 文本处理当然也重要，但要知道边界

虽然 PowerShell 强调对象，但文本处理仍然是日常工作的一部分。

常见命令包括：

```powershell
Select-String "ERROR" .\app.log
Get-Content .\app.log | Select-Object -First 20
```

但你要区分两类场景：

- 数据本来就是结构化的：优先转对象
- 数据本来就是纯文本日志：再用文本处理

不要把“所有东西都当纯文本”或“所有东西都硬转对象”走到极端。

---

## 3.7 一个真实的小练习骨架

假设你有一个 `users.csv`，内容包括：

- `Name`
- `Department`
- `Age`

你可以先这样做：

```powershell
$users = Import-Csv .\users.csv
$users | Get-Member
$users | Where-Object Department -eq "Ops"
$users | Sort-Object Age -Descending | Select-Object -First 3
```

这就是典型的 PowerShell 工作流：

1. 读入数据
2. 看对象结构
3. 按属性筛选
4. 排序并取结果

---

## 3.8 关于注册表和环境变量的边界

Provider 很强大，但也意味着风险不同。

例如：

- 文件系统操作往往会影响文件和目录
- `Env:` 会影响环境变量
- `Registry:` 可能影响系统配置，仅 Windows 常见

所以你不能因为路径看起来都像 `X:\...`，就以为风险相同。

这也是后面要单独讲安全操作的原因。

---

## 常见误区

- **误区 1：所有路径都只是文件路径。**
  Provider 让路径空间变得更广。

- **误区 2：CSV 和 JSON 都应该先当纯文本处理。**
  更稳定的做法通常是尽快转成对象。

- **误区 3：看到 `Remove-Item` 就直接加 `-Recurse`。**
  先看对象集合，再预演，再执行。

- **误区 4：文本处理和对象处理谁更高级。**
  关键不是高级，而是场景是否匹配。

---

## 本章练习

1. 用 `Get-PSProvider` 列出当前环境中的 Provider。
2. 用 `Set-Location Env:` 和 `Get-ChildItem` 查看环境变量。
3. 创建一个简单 CSV，用 `Import-Csv` 导入后筛选其中一列。
4. 读取一个 JSON 文件，转成对象后只显示你关心的属性。
5. 思考题：为什么说“路径空间”和“文件内容”是两层不同问题？

---

## 本章小结

本章真正想让你建立的是这条链路：

> 路径让你找到对象，Provider 决定对象来自哪里，导入和转换决定对象怎样进入管道。

下一章开始，我们把这些对象放进脚本里真正组织起来：变量、表达式和控制流。

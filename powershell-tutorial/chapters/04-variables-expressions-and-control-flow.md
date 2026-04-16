# 第4章：变量、表达式与控制流

## 本章解决什么问题

前几章你主要是在“观察对象”和“串接命令”。但只靠一条管道，你很快会碰到更复杂的需求：

- 想把中间结果保存下来
- 想根据条件决定是否继续执行
- 想循环处理一批对象
- 想组织稍微复杂一点的逻辑

这时就必须进入变量、表达式和控制流。

---

## 学习目标

完成本章后，你应能：

1. 使用变量保存和复用对象
2. 区分标量、数组和哈希表的常见使用方式
3. 写出 `if`、`foreach`、`for`、`while` 等基本控制流
4. 用脚本块组织一小段可传递逻辑

---

## 4.1 变量保存的不是“文本片段”，而是对象

PowerShell 变量最重要的直觉是：

```powershell
$p = Get-Process | Select-Object -First 1
$p | Get-Member
```

这里 `$p` 保存的不是一段打印出来的文本，而是一个对象。

这意味着你可以继续访问属性：

```powershell
$p.Name
$p.Id
```

所以变量的真正作用不是“缩短输入”，而是：

- 保存中间结果
- 让后续逻辑更清晰
- 避免重复执行高成本命令

---

## 4.2 数组和哈希表是两种高频基础结构

### 数组

```powershell
$names = @("alice", "bob", "carol")
$names[0]
$names.Count
```

数组适合：

- 保持顺序
- 成批遍历
- 暂存一组同类对象

### 哈希表

```powershell
$config = @{
    Path = ".\\logs"
    Top  = 10
    Mode = "Preview"
}
```

哈希表适合：

- 用键值对组织配置
- 构造参数集合
- 作为转换前的临时结构

---

## 4.3 常见比较和逻辑判断

PowerShell 常见比较运算包括：

```powershell
-eq  # 等于
-ne  # 不等于
-gt  # 大于
-lt  # 小于
-like # 通配匹配
-match # 正则匹配
```

例如：

```powershell
if ($p.CPU -gt 100) {
    "High CPU"
}
```

这里你真正判断的是对象属性，而不是屏幕展示字符串。

---

## 4.4 `if`：最基本的分支控制

```powershell
$service = Get-Service -Name Spooler

if ($service.Status -eq "Running") {
    "Service is running"
}
else {
    "Service is not running"
}
```

写条件时，优先让判断基于明确属性，而不是模糊的字符串包含。

---

## 4.5 `foreach` 与 `ForEach-Object` 的区别

这是一个必须尽早区分的点。

### `foreach`

适合遍历已经拿到内存里的集合：

```powershell
$files = Get-ChildItem -File
foreach ($file in $files) {
    $file.Name
}
```

### `ForEach-Object`

适合在管道中逐个处理对象：

```powershell
Get-ChildItem -File | ForEach-Object {
    $_.Name
}
```

可以先记住最简判断：

- 已经有集合变量了，用 `foreach`
- 还在管道里流动，用 `ForEach-Object`

---

## 4.6 其它常见循环

### `for`

适合你确实需要索引时：

```powershell
for ($i = 0; $i -lt $names.Count; $i++) {
    $names[$i]
}
```

### `while`

适合你有明确退出条件时：

```powershell
$i = 0
while ($i -lt 3) {
    $i
    $i++
}
```

不要为了“语法更高级”而滥用复杂循环。最清晰的控制流通常就是最好的。

---

## 4.7 脚本块是很多高级能力的基础

脚本块写作：

```powershell
{ $_.CPU -gt 100 }
```

它经常出现在：

- `Where-Object`
- `ForEach-Object`
- `Invoke-Command`
- `Start-Job`

你可以把脚本块先理解成“一段可传递的 PowerShell 代码”。

例如：

```powershell
Get-Process | Where-Object { $_.CPU -gt 100 }
```

这里 `$_` 代表当前正在处理的对象。

---

## 4.8 一个小型脚本思路

例如你想找出某目录下体积最大的 5 个文件：

```powershell
$files = Get-ChildItem .\logs -File
$result = $files |
    Sort-Object Length -Descending |
    Select-Object -First 5 Name, Length, LastWriteTime

$result
```

这个例子里你已经同时用了：

- 变量
- 管道
- 排序
- 选择属性

这就是 PowerShell 进入脚本化的起点。

---

## 常见误区

- **误区 1：变量只是给字符串换个名字。**
  变量真正保存的是对象。

- **误区 2：`foreach` 和 `ForEach-Object` 完全一样。**
  它们都能遍历，但所处上下文不同。

- **误区 3：条件判断时靠显示文本猜状态。**
  应优先基于明确属性判断。

- **误区 4：循环越复杂越厉害。**
  清晰胜过花哨。

---

## 本章练习

1. 用变量保存一个进程对象，并读取它的 3 个属性。
2. 写一个数组，循环输出每个元素。
3. 写一个哈希表表示配置，并读取其中两个键的值。
4. 用 `foreach` 或 `ForEach-Object` 列出一个目录下所有文件名。
5. 思考题：为什么说“先理解对象，再学习控制流”会让脚本更稳？

---

## 本章小结

本章最重要的收获是：

> PowerShell 的控制流不是脱离对象存在的，变量、条件和循环的真正价值，是帮助你更稳定地组织对象处理逻辑。

下一章，我们把这些基础继续往前推进：写函数、脚本和模块，让逻辑真正可复用。

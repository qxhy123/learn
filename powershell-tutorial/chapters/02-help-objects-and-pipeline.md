# 第2章：帮助、对象与管道

## 本章解决什么问题

如果你学 PowerShell 只记住一句话，那应该是：

> PowerShell 管道默认传递的是对象。

很多新手不稳，根源都在这里：

- 不知道命令输出长什么样
- 不知道属性该怎么拿
- 不知道 `Select-Object`、`Where-Object`、`Format-Table` 到底分别干什么
- 把“展示给人看”和“继续给命令处理”混成一件事

本章就是整套教程的地基。

---

## 学习目标

完成本章后，你应能：

1. 解释 PowerShell 对象和文本处理的差别
2. 用 `Get-Member` 查看对象结构
3. 用 `Select-Object`、`Where-Object`、`Sort-Object` 处理对象
4. 区分“处理数据”和“只改显示格式”

---

## 2.1 先看一个最经典的例子

运行：

```powershell
Get-Process | Select-Object -First 3
```

你看到的可能像表格，但这不代表它真的只是文本。继续运行：

```powershell
Get-Process | Get-Member
```

你会看到输出对象的：

- 类型
- 属性
- 方法

例如：

- `Name`
- `Id`
- `CPU`
- `Kill()`

这说明 `Get-Process` 返回的不是“几行字符”，而是一组进程对象。

---

## 2.2 为什么这件事重要

如果管道里传的是对象，你后面就可以做非常稳定的事：

```powershell
Get-Process |
    Where-Object CPU -gt 100 |
    Sort-Object CPU -Descending |
    Select-Object Name, Id, CPU
```

这里每一步处理的都不是“凭位置猜字段的文本”，而是明确属性。

这带来两个巨大好处：

- 脚本更可读
- 处理逻辑更稳定

也正因为如此，PowerShell 的思维方式和传统“文本管道”有明显差异。

---

## 2.3 `Get-Member` 是对象世界的入口

以后只要你不确定某个命令输出了什么，先跑：

```powershell
<Command> | Get-Member
```

例如：

```powershell
Get-Service | Get-Member
Get-Date | Get-Member
Get-ChildItem | Get-Member
```

你最应该先看三类信息：

1. 对象类型
2. 常用属性
3. 是否有能直接调用的方法

不要一上来就猜属性名。先观察，再写筛选逻辑。

---

## 2.4 管道的核心不是“串命令”，而是“传对象”

你可以把管道理解成一条对象输送带。

例如：

```powershell
Get-Service | Where-Object Status -eq Running
```

这条命令不是把屏幕上的文本送给下一条命令，而是把每个服务对象依次交给 `Where-Object` 的条件判断。

所以你写筛选时，本质上是在说：

- 对每个对象
- 看它的哪个属性
- 满足什么条件才保留

---

## 2.5 最常用的 4 个对象处理命令

### `Select-Object`

选取你真正需要的属性：

```powershell
Get-Process | Select-Object Name, Id, CPU
```

### `Where-Object`

筛选满足条件的对象：

```powershell
Get-Process | Where-Object CPU -gt 100
```

### `Sort-Object`

按属性排序：

```powershell
Get-Process | Sort-Object CPU -Descending
```

### `ForEach-Object`

对管道中的每个对象执行一段逻辑：

```powershell
Get-Service | ForEach-Object {
    "$($_.Name): $($_.Status)"
}
```

这 4 个命令几乎构成了你最常见的数据处理骨架。

---

## 2.6 `Format-*` 是显示命令，不是数据处理命令

这是 PowerShell 最常见的大坑之一。

例如：

```powershell
Get-Process | Format-Table Name, Id, CPU
```

这条命令很适合“给人看”，但不适合“继续给后面的命令处理”。

为什么？

因为 `Format-Table` 和 `Format-List` 的职责是：

- 改变显示方式
- 让屏幕输出更好读

它们不是为了继续保留原始对象结构。

所以一条稳定原则是：

> `Format-*` 尽量只放在管道最后一段。

如果你还要继续导出、筛选、排序、转换，先用 `Select-Object`，不要先 `Format-Table`。

---

## 2.7 最小观察流程

以后遇到陌生命令，推荐用固定四步：

```powershell
Get-Help Get-Process -Examples
Get-Process | Get-Member
Get-Process | Select-Object -First 3
Get-Process | Select-Object Name, Id, CPU
```

这四步会让你非常快地回答：

- 它能干什么
- 它输出什么
- 这些对象大概长什么样
- 我真正关心哪些属性

---

## 2.8 参数绑定的直觉

PowerShell 的很多命令之所以能无缝接收前一个命令的输出，是因为它支持按类型或按属性做参数绑定。

你现在不必死记术语，只需要先形成直觉：

- 有些命令吃的是“整个对象”
- 有些命令吃的是“对象里的某个属性”

这也是为什么观察对象结构总是第一步。

---

## 常见误区

- **误区 1：屏幕上看起来像表格，所以它就是文本。**
  屏幕展示和对象本体不是一回事。

- **误区 2：`Format-Table` 可以放在管道中间继续处理。**
  它主要是给人看，不是给数据流继续处理。

- **误区 3：看不懂输出时先猜属性名。**
  更稳妥的方式是先跑 `Get-Member`。

- **误区 4：管道只是为了省敲几个变量。**
  它真正的价值是把对象处理步骤组织成一条稳定流水线。

---

## 本章练习

1. 对 `Get-Process`、`Get-Service`、`Get-ChildItem` 各跑一次 `Get-Member`。
2. 找出 CPU 占用最高的 5 个进程，并只显示 `Name`、`Id`、`CPU`。
3. 比较 `Select-Object Name, Id` 和 `Format-Table Name, Id` 的使用场景差别。
4. 用 `Where-Object` 找出状态为 Running 的服务。
5. 思考题：为什么“显示格式”和“数据结构”必须分开理解？

---

## 本章小结

本章最重要的判断标准是：

> 以后你看到任何管道，都先问自己：这里流动的是什么对象，我到底是在处理数据，还是只是在排版显示？

下一章，我们把这种对象思维带到更真实的场景里：路径、Provider、文件和结构化数据处理。

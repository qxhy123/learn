# 第7章：错误处理、调试与安全操作

## 本章解决什么问题

很多 PowerShell 脚本“看起来能跑”，但真正一上生产或批量环境就开始出问题，原因通常不是语法，而是：

- 错误没有被正确中止
- 出问题时看不到足够的上下文
- 有副作用的操作没有先预演
- 脚本重复执行后结果失控

所以这一章不是补充内容，而是自动化是否可靠的分水岭。

---

## 学习目标

完成本章后，你应能：

1. 区分终止性错误和非终止性错误
2. 使用 `try/catch/finally` 和 `-ErrorAction`
3. 为脚本加入日志、调试和预演能力
4. 理解 `ShouldProcess`、`-WhatIf`、幂等和安全边界的意义

---

## 7.1 PowerShell 的错误模型为什么容易让人误判

很多人第一次写 `try/catch` 时会困惑：

- 为什么有些错误能被 `catch` 到
- 为什么有些错误明明报出来了，却没有中止脚本

原因在于 PowerShell 里常见两类错误：

- **非终止性错误**：报错了，但默认继续
- **终止性错误**：直接中断当前执行流

这意味着“屏幕上出现红字”不自动等于“脚本已经停下来了”。

---

## 7.2 `-ErrorAction Stop` 是非常关键的稳定器

例如：

```powershell
Get-Item .\missing.txt -ErrorAction Stop
```

这会把某些默认非终止性的错误提升为可被 `try/catch` 处理的终止性错误。

一个更完整的例子：

```powershell
try {
    Get-Item .\missing.txt -ErrorAction Stop
}
catch {
    Write-Warning $_
}
```

如果你不理解这一点，很多脚本会在失败后继续往下跑，结果越修越乱。

---

## 7.3 `try/catch/finally` 的职责分工

### `try`

放你希望被保护的核心逻辑。

### `catch`

放错误处理逻辑，例如：

- 记录日志
- 输出上下文
- 做必要清理
- 重新抛出

### `finally`

放无论成功或失败都需要执行的收尾逻辑。

这不是 PowerShell 特有概念，但在 PowerShell 里尤其重要，因为很多自动化脚本天然带副作用。

---

## 7.4 只报错还不够，你还需要可观察性

最低限度的日志和可观察性，通常包括：

- `Write-Verbose`
- `Write-Warning`
- `Write-Error`
- `Start-Transcript`

例如：

```powershell
[CmdletBinding()]
param()

Write-Verbose "Starting cleanup"
Write-Warning "Target directory is large"
```

日志不是为了“看起来专业”，而是为了让你在未来复盘时知道：

- 当时处理了哪些对象
- 哪一步失败
- 失败前脚本处于什么状态

---

## 7.5 调试时，不要只盯着屏幕输出

调试时你最应该核对的是：

1. 当前变量里到底是什么对象
2. 这些对象属性是否符合你假设
3. 管道中间哪一步把结构改坏了

实用手段包括：

```powershell
Get-Member
Select-Object -First 3
Set-PSBreakpoint -Script .\demo.ps1 -Line 12
```

很多所谓“逻辑问题”，本质上只是对象结构和你想象的不一样。

---

## 7.6 `ShouldProcess` 和 `-WhatIf`

PowerShell 里有副作用的成熟命令，很多都会支持：

- `-WhatIf`
- `-Confirm`

你自己写高级函数时，也应尽量提供这种能力。

例如：

```powershell
function Remove-DemoFile {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [string]$Path
    )

    if ($PSCmdlet.ShouldProcess($Path, "Remove file")) {
        Remove-Item $Path
    }
}
```

这样你就能先运行：

```powershell
Remove-DemoFile -Path .\old.log -WhatIf
```

这会极大降低批量脚本出事概率。

---

## 7.7 幂等：重复执行最好不要把事情越做越坏

一个成熟自动化脚本往往需要考虑：

- 重复执行会怎样
- 对象已经存在会怎样
- 对象已经删除会怎样

例如：

- 目录已存在时先检查再创建
- 服务已停止时不要重复报错
- 文件不存在时要区分“可忽略”还是“必须失败”

幂等思维的核心不是“绝对不报错”，而是让重复执行结果可预测。

---

## 7.8 不要把执行策略和高权限当成万能解法

PowerShell 新手常见坏习惯包括：

- 一出问题就提高权限
- 一遇执行策略报错就全局放开
- 一写不出来就用 `Invoke-Expression`

这些做法会快速放大风险。

更稳妥的思路应该是：

1. 先看错误类型
2. 先缩小作用域
3. 先用只读命令和 `-WhatIf`
4. 再考虑权限、策略和执行边界

---

## 常见误区

- **误区 1：报错了就一定中断了。**
  非终止性错误默认可能继续执行。

- **误区 2：`try/catch` 没生效说明 PowerShell 不可靠。**
  很多时候是你没有把错误提升为终止性错误。

- **误区 3：日志是锦上添花。**
  真正的自动化里，日志是排障入口。

- **误区 4：`-WhatIf` 是可有可无的高级功能。**
  对有副作用的脚本来说，它是非常重要的安全边界。

---

## 本章练习

1. 写一个会访问不存在路径的脚本，观察默认错误行为。
2. 再加上 `-ErrorAction Stop` 和 `try/catch`，比较行为差异。
3. 为一个删除测试文件的函数加上 `SupportsShouldProcess`。
4. 先跑一次 `-WhatIf`，确认输出后再真正执行。
5. 思考题：为什么“脚本能跑通”和“脚本适合批量执行”是两回事？

---

## 本章小结

本章最重要的判断标准是：

> PowerShell 自动化的成熟度，不取决于它能否成功执行一次，而取决于它在失败、重复执行和副作用场景下是否仍然可控。

下一章，我们把前面的所有能力收束起来，进入真正的高阶实践：高级函数、模块边界、性能、兼容性和团队工具化。

# PowerShell 速查表

## 命令发现

| 任务 | 命令 |
|------|------|
| 看命令帮助 | `Get-Help Get-Process -Examples` |
| 搜索命令 | `Get-Command *process*` |
| 按动词找命令 | `Get-Command -Verb Get` |
| 看对象结构 | `Get-Process | Get-Member` |

## 基础观察

| 任务 | 命令 |
|------|------|
| 只看前几条 | `Get-Process | Select-Object -First 5` |
| 只看几个属性 | `Get-Process | Select-Object Name, Id, CPU` |
| 筛选对象 | `Get-Process | Where-Object CPU -gt 100` |
| 排序 | `Get-Process | Sort-Object CPU -Descending` |

## 文件与路径

| 任务 | 命令 |
|------|------|
| 查看当前目录 | `Get-Location` |
| 列出文件 | `Get-ChildItem` |
| 新建目录 | `New-Item -ItemType Directory -Path .\demo` |
| 复制文件 | `Copy-Item .\a.txt .\backup\a.txt` |
| 删除文件 | `Remove-Item .\old.log -WhatIf` |

## 数据格式

| 任务 | 命令 |
|------|------|
| 导入 CSV | `Import-Csv .\users.csv` |
| 导出 CSV | `$data | Export-Csv .\users.csv -NoTypeInformation` |
| 转 JSON | `$data | ConvertTo-Json -Depth 5` |
| 读 JSON | `Get-Content .\data.json -Raw | ConvertFrom-Json` |

## 脚本与函数

| 任务 | 命令 |
|------|------|
| 运行脚本 | `.\demo.ps1` |
| 传参数 | `.\demo.ps1 -Path .\logs -Top 10` |
| 导入模块 | `Import-Module .\MyTools\MyTools.psm1` |
| 查看 Profile 路径 | `$PROFILE` |

## 安全与调试

| 任务 | 命令 |
|------|------|
| 看执行策略 | `Get-ExecutionPolicy -List` |
| 强制错误终止 | `Get-Item missing.txt -ErrorAction Stop` |
| 记录日志 | `Start-Transcript` |
| 只预演不执行 | `Remove-Item .\tmp\* -WhatIf` |

## 远程与作业

| 任务 | 命令 |
|------|------|
| 远程执行 | `Invoke-Command -ComputerName Server01 -ScriptBlock { Get-Service }` |
| 进入远程会话 | `Enter-PSSession -ComputerName Server01` |
| 启动作业 | `Start-Job -ScriptBlock { Get-Process }` |
| 收取作业结果 | `Receive-Job -Id 1 -Keep` |

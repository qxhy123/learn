# 术语表

## Object

PowerShell 管道里的核心单位。它带有属性和方法，不只是文本行。

## Pipeline

把一个命令的输出传给下一个命令的机制。PowerShell 默认传对象。

## Cmdlet

PowerShell 内建命令的典型形态，通常遵循 `Verb-Noun` 命名，比如 `Get-Process`。

## Provider

把不同数据源暴露成类似文件系统路径的机制，例如文件系统、环境变量、注册表。

## ScriptBlock

用 `{ ... }` 表示的一段可执行代码，经常出现在筛选、循环、远程执行和作业里。

## Module

一组可复用的函数、别名、变量或脚本资源的组织单元。

## Profile

PowerShell 启动时可自动加载的配置脚本位置，通常用于个人环境定制。

## Session

一次 PowerShell 运行上下文。远程会话和本地会话都可以承载命令执行状态。

## Non-terminating Error

默认不会立即中断整个管道的错误。很多命令错误默认属于这一类。

## Terminating Error

会直接中断当前执行流的错误，通常可被 `try/catch` 捕获。

## ShouldProcess

PowerShell 用于支持 `-WhatIf` 和 `-Confirm` 的机制，常用于有副作用的高级函数。

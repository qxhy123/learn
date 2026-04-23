# 环境准备

## 开始前至少确认这四件事

1. 你在 Mac 上，且能运行较新的 Xcode
2. 终端里能执行 `swift --version`
3. 终端里能执行 `xcodebuild -version`
4. 你知道如何进入仓库根目录再运行脚本

这套教程不是纯 Markdown 阅读材料，starter、测试和验证脚本都会实际运行，所以环境坏了时不要硬着头皮继续。

## 最小验证命令

先在仓库根目录跑：

```bash
swift --version
xcodebuild -version
bash swift-tutorial/scripts/verify_layout.sh
```

然后进入 starter 再跑：

```bash
cd swift-tutorial/projects/focuslist/starter
swift test
swift build --product FocusListApp
```

如果这里有任何一步失败，先修环境，不要继续做教程正文。

## 你会频繁进入的目录

- `swift-tutorial/`：教程正文
- `swift-tutorial/projects/focuslist/starter/`：主要动手位置
- `swift-tutorial/labs/`：综合实验
- `swift-tutorial/appendix/`：速查与答疑

把这些路径先熟悉，后面会节省很多时间。

## 常见问题排查

### `swift: command not found`

先检查 Xcode Command Line Tools 是否安装，必要时运行：

```bash
xcode-select --install
```

### `swift build` 报错但脚本路径正常

先确认你是不是在 `starter/` 目录里运行的构建命令。

### 验证脚本失败

优先检查两类问题：

- 文件标题或目录被改坏
- 当前目录不是仓库根目录

## 一条建议

每次大改教程内容或 starter 代码后，都重新跑一遍验证脚本。环境稳定时，很多问题会在这一步被提前拦住。

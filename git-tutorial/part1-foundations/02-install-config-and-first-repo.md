# 第2章：安装、配置与第一个仓库

## 学习目标

完成本章后，你应能：

1. 在本机安装 Git 并确认版本
2. 配置自己的 Git 身份、默认分支和编辑器
3. 理解 `system / global / local` 三层配置覆盖关系
4. 初始化一个本地仓库并完成第一次提交

---

## 2.1 安装 Git

大多数系统都可以通过包管理器安装 Git。

### macOS

```bash
brew install git
git --version
```

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install git
git --version
```

### Windows

你可以使用官方安装包、`winget`，或者在 WSL 中直接安装：

```powershell
winget install --id Git.Git -e
git --version
```

安装完成后，先执行：

```bash
git --version
```

只要终端能输出版本号，就说明 Git 已经可用。

---

## 2.2 配置身份信息：先保证提交“知道你是谁”

Git 提交会记录作者信息，所以第一步应配置名字和邮箱：

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

查看配置：

```bash
git config --global --list
```

这里的 `--global` 表示写入当前用户级配置，通常是 `~/.gitconfig`。

如果你会同时处理公司项目和个人项目，就要尽早意识到：**全局配置是“默认值”，不是所有仓库的唯一真相。**

---

## 2.3 三层配置：system、global、local

Git 配置有三个常见层级：

1. `system`：系统级，通常影响整台机器
2. `global`：当前用户级，影响你这个账号的默认行为
3. `local`：当前仓库级，只影响这个仓库

同名配置冲突时，越局部的优先级越高。

建议你一开始就用这个命令把“配置来源”看出来：

```bash
git config --show-origin --list
```

它能告诉你：

- 每个配置值来自哪个文件
- 为什么你以为设置过的选项没有生效
- 为什么同一台机器上的不同仓库可以有不同习惯

### 一个最小观察实验

先设置全局默认主分支：

```bash
git config --global init.defaultBranch main
```

再在某个仓库里设置本地值：

```bash
git config init.defaultBranch master
git config --show-origin --list | grep init.defaultBranch
```

这时你会看到：

- 全局配置仍然存在
- 本地仓库配置覆盖了它

这个实验很重要，因为它让“配置优先级”从抽象概念变成了可见事实。

---

## 2.4 建议先建立的几个默认习惯

```bash
git config --global init.defaultBranch main
git config --global core.editor "code --wait"
git config --global pull.rebase false
```

含义分别是：

- 默认主分支名为 `main`
- 提交说明编辑器使用 VS Code
- `git pull` 默认采用 merge 策略

这里要特别注意：`pull.rebase` 不是“开了就更高级”的开关，它会影响整个团队的历史形态。初学阶段先显式理解 `fetch + merge`，通常更稳。

---

## 2.5 一个常被忽略的问题：换行符和跨平台差异

如果你在 Windows、macOS、Linux 混合环境里工作，很容易遇到换行符问题。

一个比较常见的默认建议是：

```bash
git config --global core.autocrlf input
```

它并不是万能法术，但至少提醒你一件事：

> 很多“Git 好像坏了”的文本差异，其实不是 Git 逻辑问题，而是换行符策略没统一。

这一点在团队项目里比个人实验更重要。

---

## 2.6 初始化你的第一个仓库

创建一个练习目录：

```bash
mkdir git-lab
cd git-lab
git init
```

接着查看状态：

```bash
git status -sb
```

此时一般会看到：

- 当前位于 `main` 分支
- 还没有任何提交
- 工作区是干净的

---

## 2.7 创建第一个文件并提交

先创建一个文件：

```bash
echo "# Git Lab" > README.md
git status -sb
```

此时 `README.md` 会显示为未跟踪文件（untracked）。

把它加入暂存区：

```bash
git add README.md
git status -sb
```

现在状态会变成“changes to be committed”，说明它已经进入下一次提交的候选快照。

然后提交：

```bash
git commit -m "chore: initialize git lab"
git log --oneline
```

到这里，你已经完成了 Git 的第一个完整闭环。

---

## 2.8 哪些“进阶配置”现在先知道名字就够了

例如：

```bash
git config --global fetch.prune true
git config --global rerere.enabled true
```

它们分别大致表示：

- `fetch.prune`：同步远程时顺便清理已删除远程分支的本地跟踪引用
- `rerere.enabled`：记录你解决过的冲突，以便重复场景时复用

这些配置很有用，但在你还没真正经历过冲突和远程协作之前，不必强行背下。

---

## 常见误区

- **误区 1：`git init` 会自动帮你提交。**
  不会，初始化只是在目录中创建 Git 元数据。

- **误区 2：全局配置永远就是最终配置。**
  仓库本地配置可以覆盖它。

- **误区 3：随便改 `pull.rebase` 没什么影响。**
  它会直接影响你后面每次同步历史的方式。

- **误区 4：换行符问题说明 Git 坏了。**
  更多时候是跨平台文本策略没有统一。

---

## 本章练习

1. 用 `git config --show-origin --list` 观察当前 Git 配置来自哪些文件。
2. 在一个新目录里执行 `git init`，确认 `.git/` 目录出现。
3. 创建一个文件并完成第一次提交。
4. 思考题：为什么配置来源和覆盖顺序会直接影响团队行为？

---

## 本章小结

本章你已经有了一个真正可用的 Git 仓库，也知道了配置并不只是“顺手写两行名字邮箱”。下一章开始，我们要把最核心的模型真正拆开：工作区、暂存区、HEAD 与提交之间到底是什么关系。

# 忽略文件与仓库卫生

## 场景

你运行项目后出现了日志、缓存、构建产物和本地配置文件。它们不应该进入提交，但 `git status` 一直显示很多 `??`，让真正的变更被噪音淹没。你需要用 `.gitignore` 维护仓库卫生。

## 学习目标

- 判断哪些文件应该被忽略，哪些必须提交。
- 理解 `.gitignore` 只影响未跟踪文件，不会自动移除已跟踪文件。
- 使用 `git check-ignore -v` 排查忽略规则。
- 避免提交秘密、生成物和机器本地配置。

## 观察点

```bash
git status --short
git check-ignore -v path/to/file || true
git ls-files path/to/file
```

分别回答：文件是否未跟踪、被哪条规则忽略、是否已经被 Git 跟踪。

## 命令与解释

```bash
printf "*.log\nbuild/\n.env.local\n" >> .gitignore
```

规则通常提交到仓库中，表示团队共同忽略的文件。

```bash
git check-ignore -v debug.log
```

显示是哪一个 `.gitignore` 文件的哪一行匹配了该路径。排查“为什么没有显示在 status 里”时很有用。

```bash
git rm --cached path/to/tracked-generated-file
```

把已经被跟踪的文件从暂存区/下一次提交中移除，但保留工作区文件。只有确认它确实不该被版本控制时才使用。

## 实验

**Lab id：`LAB-DAILY-IGNORE-01`**

目标：添加忽略规则，并验证 tracked 文件不受自动影响。

```bash
mkdir ignore-lab
cd ignore-lab
git init
printf "keep\n" > app.txt
git add app.txt
git commit -m "seed app"

printf "debug\n" > debug.log
mkdir build
printf "artifact\n" > build/output.txt
git status --short

printf "*.log\nbuild/\n" > .gitignore
git status --short
git check-ignore -v debug.log build/output.txt
```

预期观察：

- 添加规则前，`debug.log` 和 `build/output.txt` 作为未跟踪文件出现。
- 添加规则后，它们不再作为普通未跟踪文件干扰状态。
- `.gitignore` 本身应该被提交，因为这是团队规则。

## 风险与恢复路径

> **风险提示**：不要用 `git clean -fd` 作为“让 status 变干净”的默认手段。它会删除未跟踪文件，包括尚未提交的新文件。
>
> **恢复路径**：清理前先运行 `git clean -fdn` 预演；如果误删，Git 通常无法恢复未跟踪文件，只能依赖编辑器本地历史、系统回收站、备份或重新生成。

## 常见错误

- **以为 `.gitignore` 会删除文件**：它只影响未跟踪文件是否显示和是否容易被添加。
- **把秘密写进历史后再 ignore**：一旦提交到历史，就需要按安全流程轮换密钥，不能只靠 ignore。
- **规则过宽**：例如忽略 `*.json` 可能误伤必须提交的配置模板。

## 验收

请解释：

1. 为什么 `.gitignore` 无法自动停止跟踪已经提交过的文件？
2. `git check-ignore -v` 能帮助你回答什么问题？
3. 运行 `git clean -fd` 前为什么应该先运行 `git clean -fdn`？

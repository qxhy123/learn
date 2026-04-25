# 20 Monorepo、LFS 与大仓库：规模变大后先治理边界

## 场景

一个仓库从几千行代码增长到多个服务、前端资源、模型文件和自动生成产物。新人 clone 很慢，CI 拉取很慢，PR diff 经常混入构建产物。有人建议拆成多仓库，有人建议上 monorepo，有人想用 Git LFS。你需要先判断问题属于仓库形态、文件类型，还是团队提交习惯。

## 学习目标

完成本章后，你应该能够：

1. 用观察命令定位“大仓库慢”的主要信号。
2. 解释 monorepo、多仓库、submodule/subtree 的取舍。
3. 判断哪些文件应进入 Git LFS，哪些应进入制品库或被忽略。
4. 为大仓库制定最小卫生规则。
5. 在不重写共享历史的前提下提出清理路线。

## 观察点

在真实仓库做任何清理前，先只读观察：

```bash
git status -sb
git count-objects -vH
git branch -vv
git remote -v
git ls-files | sed -n '1,40p'
git rev-list --objects --all | wc -l
```

如果安装了 LFS：

```bash
git lfs env
git lfs ls-files
```

如果怀疑构建产物进入历史，先找当前跟踪文件：

```bash
git ls-files | grep -E '(^dist/|^build/|\.log$|\.zip$|\.mp4$|\.psd$|\.pt$|\.onnx$)' || true
```

## 命令与决策

### 先分清三类问题

| 症状 | 常见原因 | 优先动作 |
|---|---|---|
| `git status` 慢 | 未忽略的大量生成文件、文件系统扫描压力 | `.gitignore`、拆工作区、稀疏检出 |
| clone/fetch 慢 | 历史中大对象多、远程引用过多 | LFS、浅克隆、清理旧引用、制品外置 |
| PR review 慢 | 巨型 diff、跨模块耦合 | 提交拆分、CODEOWNERS、目录边界 |

不要一上来就重写历史。重写共享历史会改变大量提交 ID，必须有停机窗口、备份和全员迁移方案。

### Monorepo 不是“先进”，而是边界选择

Monorepo 优势：

- 跨模块改动一次提交完成。
- 统一工具链和代码搜索。
- 原子升级依赖更容易。

Monorepo 成本：

- 权限、CI、代码所有权更复杂。
- 无关改动噪音更多。
- 对构建缓存、稀疏检出、路径触发要求更高。

多仓库优势：

- 权限和发布边界清晰。
- 单仓库体积小。
- 团队自治更强。

多仓库成本：

- 跨仓改动需要协调版本。
- 本地环境和 CI 编排更复杂。
- 公共库升级容易出现碎片化。

### LFS 的判断线

适合 LFS：

- 需要版本化但体积较大的二进制资源。
- 设计资源、媒体、模型权重、数据样本。
- 文件内容不适合文本 diff。

不适合 LFS：

- 可重新生成的构建产物。
- 临时日志、缓存、依赖下载目录。
- 需要频繁小改且每次都产生完整二进制版本的文件，除非团队接受存储成本。

最小规则：

```bash
git lfs track '*.psd'
git lfs track '*.mp4'
git lfs track '*.onnx'
git add .gitattributes
```

如果团队尚未启用 LFS，先在新文件上应用，不要直接迁移旧历史。

## 实验

- Lab id：`LAB-GOV-LARGE-REPO-01`
- 场景文件：[../labs/scenarios/LAB-GOV-LARGE-REPO-01.md](../labs/scenarios/LAB-GOV-LARGE-REPO-01.md)
- 生成脚本：

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-LARGE-REPO-01 --force
```

实验重点：识别应忽略文件、应外置制品、可考虑 LFS 的文件，并写出不重写历史的整改计划。

## 常见错误

1. **把构建产物放进 LFS**：LFS 不是垃圾桶；能重建的产物通常应忽略或上传制品库。
2. **用重写历史解决所有体积问题**：共享仓库重写历史风险高，通常最后才考虑。
3. **只看文件大小不看变更频率**：频繁改动的大二进制会持续增加存储和带宽成本。
4. **迁移仓库形态前不定义 ownership**：monorepo 没有 CODEOWNERS、路径 CI 和评审规则，会把协作成本集中放大。

## 风险提示与恢复路径

> 危险动作：历史清理、迁移 LFS、删除远程分支或批量改 `.gitattributes` 会影响所有协作者。

恢复路线：

1. 先建立只读报告：大文件列表、当前跟踪产物、分支/标签数量。
2. 创建备份引用或镜像仓库：`git clone --mirror <url> backup.git`。
3. 新规则先对未来提交生效：`.gitignore`、`.gitattributes`、CI 阻断。
4. 若必须重写历史，明确冻结窗口、迁移命令和回滚负责人。

不要在未通知团队时运行历史重写工具。教程的 lab 只做设计和局部临时仓库演练。

## 验收

给定一个大仓库，你能输出一张三列表：

| 文件/问题 | 分类 | 处理建议 |
|---|---|---|
| `dist/app.js` | 构建产物 | 加入 `.gitignore`，CI 生成 |
| `assets/intro.mp4` | 大二进制资源 | 新文件用 LFS 或制品库 |
| `services/a + services/b` | 跨模块改动 | 用路径 ownership 和拆分提交降低 review 成本 |

并能说明：哪些动作可以今天做，哪些需要团队窗口和备份。

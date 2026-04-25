# 20 Monorepo、LFS 与大仓库：规模变大后先治理边界

- 前置章节：[19 Hooks、配置与别名](19-hooks-config-and-aliases.md)：你已经能把低风险规则放进本地 hook / CI，并能观察配置来源。
- 后续章节：[21 分支策略与灾难手册](21-branching-policy-and-disaster-playbook.md)：下一章会把大仓库里的 ownership、发布和事故处理固化成团队策略。
- 本章 Lab id：`LAB-GOV-LARGE-REPO-01`
- 核心危险命令：`git lfs migrate import`、`git filter-repo` / `git filter-branch` / BFG、`git push --force`、`git push --mirror`、批量删除远程引用。
- 恢复路径：先冻结写入并做镜像备份，再用只读报告确认问题范围；任何历史重写都必须有停机窗口、迁移说明和回滚负责人。

## 场景

一个仓库从几千行代码增长到多个服务、前端资源、模型文件和自动生成产物。新人 clone 很慢，CI 拉取很慢，PR diff 经常混入构建产物。有人建议拆成多仓库，有人建议上 monorepo，有人想用 Git LFS，还有人想直接把旧历史里的大文件删掉。

本章的任务不是替某一种仓库形态背书，而是建立一套判断顺序：先观察瓶颈，再区分“代码边界”“文件类型”“历史包袱”“团队规则”四类问题，最后选择能渐进落地、可恢复的治理动作。

## 学习目标

完成本章后，你应该能够：

1. 用只读命令定位“大仓库慢”的主要信号：对象数量、对象体积、跟踪文件、分支/标签规模和远程状态。
2. 解释 monorepo、多仓库、submodule/subtree 的适用边界，而不是把它们当作先进/落后的标签。
3. 判断哪些文件应该进入 Git LFS，哪些应该进入制品库，哪些应该被 `.gitignore` 或 CI 阻断。
4. 为大仓库制定最小卫生规则：目录 ownership、路径触发 CI、稀疏检出、提交边界和大文件门禁。
5. 在不重写共享历史的前提下提出第一阶段清理路线，并知道何时必须升级为团队级迁移项目。

## 观察点

在真实仓库做任何清理前，先只读观察。你要回答四个问题：现在慢在哪里、哪些文件被跟踪、历史里是否有大对象、团队协作边界是否清楚。

```bash
git status -sb
git branch -vv
git remote -v
git count-objects -vH
git ls-files | sed -n '1,60p'
git rev-list --objects --all | wc -l
```

观察当前跟踪文件是否包含常见生成物或大二进制：

```bash
git ls-files | grep -E '(^dist/|^build/|^coverage/|\.log$|\.zip$|\.mp4$|\.psd$|\.pt$|\.onnx$)' || true
```

如果仓库启用了 LFS，再观察 LFS 配置与指针文件：

```bash
git lfs env
git lfs ls-files
cat .gitattributes 2>/dev/null || true
```

如果你要估算工作区和 Git 对象占用，可以在本机做辅助观察：

```bash
du -sh .git . 2>/dev/null || true
git for-each-ref --format='%(refname:short) %(objectname:short) %(committerdate:short)' refs/heads refs/remotes refs/tags | sed -n '1,80p'
```

> 注意：`du`、`grep` 等命令是辅助信号，不是治理方案。不要因为看到一个大文件就立刻重写历史；先判断它是否仍被当前版本跟踪、是否影响 clone/fetch、是否有团队迁移窗口。

## 命令与决策

### 先分清四类问题

| 症状 | 常见原因 | 优先动作 | 不要先做 |
|---|---|---|---|
| `git status` 慢 | 未忽略的大量生成文件、工作区文件数过多、文件系统扫描压力 | `.gitignore`、拆工作区、稀疏检出、工具缓存 | 重写历史 |
| clone/fetch 慢 | 历史中大对象多、远程引用过多、二进制频繁变更 | LFS、浅克隆/部分克隆、清理旧引用、制品外置 | 直接 `push --mirror` 覆盖远端 |
| PR review 慢 | 巨型 diff、跨模块耦合、生成物混入提交 | 拆分提交、CODEOWNERS、路径 CI、目录边界 | 把所有问题归咎于仓库形态 |
| CI 慢 | 每次都构建全量模块、缓存无效、测试选择不精确 | 路径触发、构建缓存、模块 ownership | 只让开发者本地少提交 |

一个可靠的大仓库治理流程通常是：**先阻止新问题进入，再为当前问题分级，最后再考虑历史迁移**。历史重写会改变大量提交 ID，必须有停机窗口、镜像备份和全员迁移方案。

### Monorepo 不是“先进”，而是边界选择

Monorepo 优势：

- 跨模块改动可以一次提交、一次 review、一次 CI 证明。
- 统一工具链、统一依赖升级和全局代码搜索更容易。
- 公共接口变更可以用原子提交降低版本漂移。

Monorepo 成本：

- 权限、CI、代码所有权更复杂。
- 无关改动噪音更多，review 容易被大 diff 淹没。
- 对构建缓存、稀疏检出、路径触发和目录契约要求更高。

多仓库优势：

- 权限、发布和故障边界更清晰。
- 单仓库体积小，clone 与本地工具压力低。
- 团队自治更强，服务之间可以独立演进。

多仓库成本：

- 跨仓改动需要协调版本、发布顺序和回滚策略。
- 本地环境和 CI 编排更复杂。
- 公共库升级容易碎片化，兼容层会变厚。

Submodule/subtree 适合少量清晰边界的共享代码或第三方依赖，但它们会增加学习和协作成本。选择它们之前，先确认团队能解释：更新流程、回滚流程、CI 如何取依赖、谁负责版本同步。

### LFS 的判断线

适合 LFS：

- 需要版本化但体积较大的二进制资源。
- 设计资源、媒体、模型权重、数据样本。
- 文件内容不适合文本 diff，但需要跟随代码版本演进。

不适合 LFS：

- 可重新生成的构建产物，例如 `dist/`、`build/`、`coverage/`。
- 临时日志、缓存、依赖下载目录。
- 需要频繁小改且每次都产生完整二进制版本的文件，除非团队接受存储和带宽成本。
- 应由制品库、对象存储或模型仓库管理的发布产物。

新增 LFS 规则只影响之后加入的文件：

```bash
git lfs track '*.psd'
git lfs track '*.mp4'
git lfs track '*.onnx'
git add .gitattributes
```

这不是旧历史迁移。若某个大文件已经在普通 Git 历史里提交过，简单添加 `.gitattributes` 不会让旧提交变小。旧历史迁移属于团队级项目，见“危险命令与恢复路径”。

### 大仓库的低风险治理清单

第一阶段优先选择不会改写历史、可以小步 review 的动作：

1. **阻止新垃圾进入**：完善 `.gitignore`，用 pre-commit / CI 拦截 `dist/`、日志、压缩包和超限文件。
2. **让 review 可分层**：按目录拆提交，给核心目录设置 CODEOWNERS 或评审人约定。
3. **降低本地检出成本**：对新 clone 使用浅克隆、部分克隆或稀疏检出。
4. **降低 CI 成本**：按路径触发测试，缓存依赖和构建产物。
5. **为二进制建立去处**：明确 LFS、制品库、对象存储和忽略文件的边界。

稀疏检出适合“我只需要一个子目录工作”的场景，建议先在新 clone 或临时目录验证：

```bash
git clone --filter=blob:none --sparse <url> repo-sparse
cd repo-sparse
git sparse-checkout set services/payments libs/shared
```

这不会改变远端历史，但会改变当前工作区看到的路径。团队文档必须说明如何退出或调整稀疏范围：

```bash
git sparse-checkout list
git sparse-checkout disable
```

## 实验

**Lab id：`LAB-GOV-LARGE-REPO-01`**

- 场景文件：[../labs/scenarios/LAB-GOV-LARGE-REPO-01.md](../labs/scenarios/LAB-GOV-LARGE-REPO-01.md)
- 生成脚本：

```bash
cd git-tutorial/labs
./bin/git-lab.sh LAB-GOV-LARGE-REPO-01 --force
```

实验重点：你会拿到一个模拟“大仓库”：包含源码目录、生成物目录、大二进制样例和混乱提交。不要直接清理历史；先完成一份治理报告。

建议步骤：

1. 运行观察命令，记录 `git status -sb`、`git count-objects -vH`、跟踪文件中的可疑路径。
2. 把文件分成四类：源码、应忽略生成物、应放 LFS 的二进制、应外置到制品库的发布产物。
3. 写出第一阶段低风险改动：`.gitignore` 规则、LFS 新增规则、CI 阻断项、目录 ownership。
4. 对历史里的大对象只写“迁移提案”，不要在 lab 主路径中重写共享历史。
5. 用验收表说明哪些动作可以今天合并，哪些需要团队冻结窗口。

## 常见错误

1. **把构建产物放进 LFS**：LFS 不是垃圾桶；能重建的产物通常应忽略或上传制品库。
2. **用重写历史解决所有体积问题**：共享仓库重写历史风险高，通常最后才考虑。
3. **只看文件大小不看变更频率**：频繁改动的大二进制会持续增加存储和带宽成本。
4. **迁移仓库形态前不定义 ownership**：monorepo 没有 CODEOWNERS、路径 CI 和评审规则，会把协作成本集中放大。
5. **把 submodule 当成透明目录**：submodule 有自己的提交指针和更新流程，不说明清楚会让新人频繁卡在 detached HEAD 或版本不同步。
6. **只优化 clone，不治理新增问题**：如果 CI 和 review 仍允许大文件、生成物和跨模块巨型提交进入，仓库会很快再次变慢。

## 危险命令与恢复路径

> **危险命令：`git lfs migrate import`、`git filter-repo` / `git filter-branch` / BFG**
>
> **风险**：这些工具会改写历史提交 ID。已经 clone 的同事、打开的 PR、发布标签、CI 缓存和部署系统都可能需要迁移。
>
> **恢复路径**：迁移前必须 `git clone --mirror <url> backup.git`，记录所有分支/标签，冻结写入；迁移后先推到新远端或临时命名空间验证，不要直接覆盖生产远端。

> **危险命令：`git push --force`、`git push --mirror`、批量删除远程分支/标签**
>
> **风险**：会改变团队共享引用，可能让其他人的本地分支、PR 和发布锚点失效。
>
> **恢复路径**：先 `git fetch --all --prune` 并保存 `git for-each-ref` 输出；需要强推个人分支时优先 `--force-with-lease`；误改共享引用后立即冻结 push，从镜像备份、平台审计或同事本地 reflog 找回最后好引用。

> **危险命令：`rm -rf .git/lfs`、手动编辑 `.git/objects`、直接删除 `.git` 内部目录**
>
> **风险**：会破坏本地仓库对象数据库，轻则需要重新 clone，重则丢失未推送对象。
>
> **恢复路径**：不要手动清理 `.git` 内部目录。先确认工作区是否有未提交内容；必要时把工作区文件复制到安全目录，再重新 clone 或从备份恢复。

团队级历史迁移的最低流程：

1. **冻结**：公告窗口，暂停相关分支 push 和发布。
2. **备份**：镜像 clone，记录分支/标签、默认分支、保护规则和当前 CI 状态。
3. **演练**：在临时远端验证 clone、fetch、LFS 拉取、CI、关键标签和 PR 迁移。
4. **切换**：按计划更新远端，给全员提供重新 clone 或 hard reset 指令。
5. **回滚**：保留旧镜像和只读远端，明确谁可以恢复引用。

## 验收

给定一个大仓库，你能输出一张治理表：

| 文件/问题 | 分类 | 处理建议 | 风险等级 |
|---|---|---|---|
| `dist/app.js` | 构建产物 | 加入 `.gitignore`，CI 生成，不进入 LFS | 低 |
| `assets/intro.mp4` | 大二进制资源 | 新文件用 LFS；旧历史迁移另立项目 | 中 |
| `release/app.zip` | 发布制品 | 放制品库或对象存储，提交中只保留版本说明 | 低 |
| `services/a + services/b` 同一 PR 巨型改动 | 跨模块耦合 | 用路径 ownership、拆分提交和路径 CI 降低 review 成本 | 中 |
| 历史中 2GB 模型权重 | 历史包袱 | 先阻断新增；迁移需镜像备份、冻结窗口和全员说明 | 高 |

你还应该能回答：

1. 本章 lab id 是什么？它验证的是清理历史，还是制定可恢复治理计划？
2. 当前仓库中哪些文件应该忽略、哪些适合 LFS、哪些应该外置到制品库？
3. 为什么 `.gitattributes` 的 LFS 规则不能自动缩小旧历史？
4. 如果团队坚持重写历史，你会要求哪三项前置证据？
5. 哪些动作可以今天做，哪些必须等团队窗口和备份？

## 术语需求

请 Appendix 集成时确认或补充以下术语：monorepo、多仓库、Git LFS、制品库、稀疏检出、部分克隆、对象数据库、历史重写、镜像仓库、CODEOWNERS。

## 交付给集成阶段

- 本章引用的 lab id：`LAB-GOV-LARGE-REPO-01`。
- 前置章节：`07-scale-and-governance/19-hooks-config-and-aliases.md`。
- 后续章节：`07-scale-and-governance/21-branching-policy-and-disaster-playbook.md`。
- 危险命令：`git lfs migrate import`、`git filter-repo` / `git filter-branch` / BFG、`git push --force`、`git push --mirror`、批量删除远程引用、手动删除 `.git` 内部目录。
- 恢复路径：先冻结写入，镜像备份，临时远端演练；误改共享引用时从镜像备份、平台审计或同事 reflog 找回最后好引用。
- Labs 需求：场景应包含生成物、大二进制、LFS 候选、应外置制品和跨模块 PR 噪音；主路径只要求写治理报告，不要求真实重写共享历史。

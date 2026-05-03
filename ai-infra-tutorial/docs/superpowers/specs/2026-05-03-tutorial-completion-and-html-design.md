# AI Infra 教程：缺口补完 + 多文件 HTML 化设计文档

> 日期：2026-05-03
> 对象仓库：`/Users/yangyang/ai_projs/math/ai-infra-tutorial/`
> 上游约束：本设计取代并合并 `SPEC.md`（2026-04-24，17 个 Work Unit）。

## 0. 目标

把现有 25 章中文 markdown 教程做两件事：

1. **缺口补完**。合并两路缺口：
   - 用户清单：GPU 体系架构 / OS·CPU 微架构 / 网络协议栈 / 文件系统内部 / 分布式训练 / LLM 训练 / 推理服务 / 集群编排 / 可观测性
   - 仓库已有的 `SPEC.md` 17 个 Work Unit
2. **多文件 HTML 化**。按 `nm.html` 的浅色 paper 风格做静态站点，每章独立 HTML 文件 + iframe sidebar 导航 + mermaid 图表 + 手工 SVG。

最终交付物：补完后的 markdown 树 + `html/` 目录下完整的可独立分发静态站点。

## 1. 整体架构与目录结构

```
ai-infra-tutorial/
├── README.md                          # 更新章节列表 + 学习路径
├── 00-preface.md
├── SPEC.md                            # 完工后归档（重命名为 SPEC-archive-2026-04-24.md）
├── docs/superpowers/specs/            # 本设计文档所在
├── part0-foundations-of-systems/      # 【新增】体系结构基础
│   ├── 0a-cpu-microarchitecture.md
│   ├── 0b-memory-virtual-memory-and-io.md
│   ├── 0c-filesystems-and-storage-internals.md
│   └── 0d-network-stack-fundamentals.md
├── part1-foundations/ ... part8-advanced-and-capstone/   # 现有，扩写
├── appendix/                          # 现有，更新
└── html/                              # 【新增】HTML 输出目录
    ├── index.html                     # 教程门面 + 全章目录 + 学习路径
    ├── sidebar.html                   # sidebar 内容（被 iframe 加载）
    ├── assets/
    │   ├── style.css                  # 共用样式（提取自 nm.html）
    │   ├── nav.js                     # prev/next 注入、当前章高亮
    │   ├── mermaid.min.js             # offline mermaid v11 bundle
    │   └── tutorial-data.js           # 章节顺序/标题/Part 分组单一数据源
    ├── part0/ ... part8/              # 每 Part 一个子目录，每章一个 HTML
    └── appendix/
        ├── glossary.html
        ├── tooling-map.html
        ├── checklists.html
        └── answers.html
```

**关键决定**：

- HTML 输出与 markdown 源分离到 `html/`，原 markdown 保持纯净，便于后续维护。
- sidebar 通过 `iframe src="../sidebar.html?current=<id>"` 引入，单点维护。
- `assets/style.css` 抽出 `nm.html` 内联 CSS，每章不再内联约 600 行 CSS。
- prev/next 由 `nav.js` 根据当前 URL 在 `tutorial-data.js` 里查相邻章自动注入。
- 章节顺序变更只需改 `tutorial-data.js` 单文件，不需要批量改 30+ HTML。
- mermaid 走离线 bundle（默认）；如需 CDN 版本可后续改 `<script src>`。

## 2. Part 0「体系结构基础」四章

新增 4 章，每章 ~3000-4000 字，12-14 道练习。开篇"为什么 AI 平台工程师必须懂"，结尾 worked example。

### 2.1 Ch 0a CPU 微架构

- §0a.1 为什么 AI 工程师要懂 CPU 微架构（DataLoader 抖动、host-side bottleneck、tokenizer 上限）
- §0a.2 流水线（pipeline）：5 段经典流水、停顿、冒险、CPI 推算
- §0a.3 乱序执行（OoO）+ register renaming + ROB
- §0a.4 分支预测：BTB、预测器类型、误预测代价、AI 场景的 cold path
- §0a.5 SIMD：SSE/AVX/AVX-512、向量化判断、tokenizer / decode preprocessing 实例
- §0a.6 Cache 层级：L1/L2/L3 容量带宽延迟表、Cache line 64B、关联度
- §0a.7 MESI 协议：4 状态机、coherence traffic、多 socket 影响
- §0a.8 伪共享（false sharing）：DataLoader worker counter 实例、padding/aligned 解决
- §0a.9 Worked example：DataLoader 8 worker → 16 worker 反而变慢（perf/cache-misses/false-sharing 链路）

### 2.2 Ch 0b 内存、虚拟内存与 IO

- §0b.1 为什么 AI 工程师要懂虚拟内存（mmap 大权重、shared mem OOM、Page Cache 影响 dataset 读取）
- §0b.2 物理 / 虚拟内存 / 页表 / TLB
- §0b.3 Page / Page Cache / 脏页回写 / `vm.dirty_ratio`
- §0b.4 Huge Pages（THP vs explicit）：DataLoader 大数组、PyTorch arena、HugeTLB
- §0b.5 NUMA：node 亲和、`numactl`、GPU pinning 与 NUMA 关系
- §0b.6 用户态 / 内核态 / syscall：context switch 代价、io_uring / epoll
- §0b.7 PCIe：lane / 代际、bandwidth/latency 表、PCIe topology、GPU↔NIC↔CPU 数据路径
- §0b.8 DMA、page-locked memory（pinned）、cudaMemcpyAsync
- §0b.9 Worked example：H2D 带宽上不去 → 排查到 dataloader 没 pin_memory + NUMA 错位

### 2.3 Ch 0c 文件系统与存储内核

- §0c.1 为什么 AI 工程师要懂文件系统（checkpoint 写入慢、dataset 读取抖、对象存储 vs POSIX 选择）
- §0c.2 VFS、inode、dentry、page cache 与文件系统关系
- §0c.3 ext4：journal 模式、extent、checkpoint 大文件写放大
- §0c.4 XFS：B+tree、并发写优势、AI 场景常用原因
- §0c.5 ZFS：copy-on-write、ARC、snapshot、压缩、dataset 仓库适配
- §0c.6 文件系统对比表（吞吐 / 延迟 / 一致性 / 快照 / AI 场景适配）
- §0c.7 fsync / O_DIRECT / O_SYNC / writev：checkpoint 工程语义陷阱
- §0c.8 IOPS / 带宽 / 延迟；顺序 vs 随机；AI workload 模式
- §0c.9 对象存储（S3 / OSS）：HTTP REST 语义、最终一致性、列表/分片上传、与 POSIX 差异
- §0c.10 并行文件系统（Lustre / GPFS / BeeGFS / WekaFS）：MDS/OSS 架构、stripe、训练 dataset 契合度
- §0c.11 Worked example：800GB checkpoint 在不同 FS 上的写入时长 + 一致性影响

### 2.4 Ch 0d 网络协议栈基础

- §0d.1 为什么 AI 工程师要懂协议栈（control plane 走 TCP，data plane 才走 RDMA）
- §0d.2 OSI 模型 → 实际 Linux 协议栈分层
- §0d.3 TCP：三次握手、拥塞控制（CUBIC/BBR）、窗口、AI 长流 vs 短连接
- §0d.4 IP 路由、subnet、MTU、jumbo frame 对训练吞吐影响
- §0d.5 socket / epoll / io_uring（control plane 用）
- §0d.6 网卡 offload（GSO/TSO/LRO/RSS/RPS）
- §0d.7 RDMA verbs（QP / CQ / WR / WC）；RoCE v2 vs InfiniBand 对比；零拷贝原理
- §0d.8 GPUDirect RDMA：路径图（GPU → NIC，bypass CPU mem）
- §0d.9 集合通信库与协议栈关系：NCCL → libfabric / verbs → driver
- §0d.10 Worked example：8 节点 64-GPU AllReduce 慢一半 → 排查到 ECN 关闭 + MTU 1500

### 2.5 共同要求

每章必须含：

- 开篇"为什么 AI 平台工程师要懂"段落（不少于 300 字）
- 结尾 Worked example（不少于 600 字，含数字、命令、推理链）
- 4-8 个 mermaid 图（流程 / 状态机 / 序列 / 架构）
- 可酌情加 1-2 个手工 SVG（参考 `nm.html` 风格）
- 12-14 道练习（基础 6 + 进阶 4 + 设计 2-4）
- "深度参考阅读"列表

## 3. 现有章节扩写清单

合并 `SPEC.md` 17 WUs 与用户新增清单后的最终扩写表。每条标 **【SPEC】** / **【新】** / **【合】**。

### Part 1
- **Ch 2** 算力存储网络：补 Page Cache / NUMA 浅引用并指向 §0b（~300）【新】

### Part 2
- **Ch 4** GPU：NVSwitch 工作原理 + 拓扑图、HGX H100/H200 baseboard 物理布局、GB200/NVL72 架构（~1500）【新】
- **Ch 5** 内存互联：集群网络拓扑 Fat-tree / Rail-optimized / DragonFly+ 对比表 + Job Placement（~1200）【SPEC WU-11】+ 文件系统对比浅引用指向 §0c（~300）【新】
- **Ch 6** CUDA：SM 调度 / warp / register spill 段（~600）【新】

### Part 3
- **Ch 7** 单机训练：LLaMA-7B Worked Example 完整推演（~1500）+ MFU/HFU 定义对比（~600）+ Mixed Precision/AMP（~400）【SPEC WU-06】
- **Ch 8** 数据并行：梯度压缩（quantization / sparsification / PowerSGD）一节（~600）【新】
- **Ch 9** 模型/流水并行：并行策略选型决策树（mermaid flowchart）+ 典型配置实例表（~1500）【SPEC WU-04】+ Sequence Parallelism + Context Parallelism + 三者对比表（~700）【SPEC WU-09】+ Interleaved/Zero Bubble 详解（~500）【SPEC WU-04】
- **Ch 10** 内存/Checkpoint/恢复：NCCL Hang 完整排查流程（~600）+ Straggler Detection（~400）+ Elastic Training（~400）+ Pre-flight Validation（~400）【SPEC WU-10】+ FP8 训练管线 + HFU 整合（~400）【新】
- **Ch 10b** 对齐/后训练：PPO Worked Example（LLaMA-7B 8×H100 显存表）+ DPO/GRPO 扩写 + RM 部署选型 + Checkpoint 多模型一致性（~2500）+ 7 道新练习题【SPEC WU-01】
- **Ch 10c** Fine-tuning/Multi-Adapter：Multi-LoRA 显存预算扩写 + Adapter/Base 版本兼容章 + FTaaS pipeline 完整化 + 5 道新练习题（~2200）【SPEC WU-02】

### Part 4
- **Ch 11** 数据管道：与 §0c 联动的 dataset shard 读取章（~400）【新】
- **Ch 12** 制品/Checkpoint：与 §0c 联动的 checkpoint FS 选型小节（~300）【新】
- **Ch 13** 特征/向量：向量数据库选型决策框架表 + ANN 算法 + RAG Chunking 三模式对比 + 增量 vs 全量重建 + Prefix Caching（~2500）【SPEC WU-08】

### Part 5
- **Ch 14** 在线推理架构：（无大改）
- **Ch 15** Batching/Scheduling/KV Cache：LLaMA-70B 容量规划 Worked Example + Prefill-Decode Disaggregated（DistServe/Mooncake 架构）+ Speculative Decoding + ITL 指标（~2200）【SPEC WU-07 + 新】
- **Ch 16** 量化/编译/引擎：量化方案选型决策树 + 推理引擎选型决策树 + 校准过程（~1500）【SPEC WU-05】+ vLLM/TRT-LLM/SGLang 引擎内部机制对比（~800）【新】
- **Ch 17** 多租户/成本：Cloud vs On-Prem TCO + Spot 策略 + MFU vs Utilization 真实含义 + Chargeback（~2000）【SPEC WU-12】

### Part 6
- **Ch 18** 容器：（无大改）
- **Ch 19** K8s for AI：Volcano/Kueue 内部调度算法 + 拓扑感知调度 K8s 实现 + 亲和/反亲和（~1200）【新】
- **Ch 20** 队列/配额/弹性：MIG/MPS/Time-Slicing 对比表 + GPU 资源碎片化 + DRF 公平调度（~1500）【SPEC WU-13】

### Part 7
- **Ch 21** 可观测性：Trace 采样策略 head/tail-based + cardinality 治理 + 错误预算 burn-down + 成本归因（~1200）【新】
- **Ch 22** 评测/发布/故障：A/B vs 灰度 + 灰度质量采样 + Prompt/配置变更管理（~1500）【SPEC WU-14】
- **Ch 23** 安全/隔离/治理：Secrets 管理 + 模型安全（pickle/SafeTensors）+ 供应链（cosign/Trivy/SLSA）（~1500）【SPEC WU-15】

### Part 8
- **Ch 24** 构建 AI 平台：（无大改）
- **Ch 25** Agent：去重 + thinking tokens 泛化为 4 模式 + 推理预算工程实现 + Agent/推理服务集成（~2500）+ 7 道新练习题【SPEC WU-03】

### 附录
- **glossary.md**：新增 ~30 个术语（含 Part 0 全部）【SPEC WU-16 + 新】
- **tooling-map.md**：新增 4+ 类别（CPU profiling、FS tools、network tools、mermaid 渲染）【SPEC WU-16 + 新】
- **checklists.md**：新增 3+ 清单（CPU/FS/网络排查）【SPEC WU-16 + 新】
- **answers.md**：补全所有新增练习题答案【SPEC WU-16】

**总计**：Part 0 全新 ~14K + 现有章节扩写 ~32K = ~46K 行新增。

## 4. HTML 转换流水线

### 4.1 共用资源

`html/assets/`：

- `style.css`：从 `nm.html` 内联 CSS 抽出，加多文件适配补丁（外链字体不变）。
- `nav.js`：读 `tutorial-data.js`，根据 `window.location.pathname` 注入顶/底 prev/next 链接，并向 sidebar iframe `postMessage` 同步当前章。
- `mermaid.min.js`：mermaid v11 离线 bundle。
- `tutorial-data.js`：单一数据源，定义全部 30+ 章顺序、标题、Part 分组、路径。新增/换序只改这一个文件。

### 4.2 章节 HTML 模板（标准结构）

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>第 0a 章 · CPU 微架构 — AI Infra 教程</title>
  <link rel="stylesheet" href="../assets/style.css" />
</head>
<body class="has-sidebar">
  <iframe id="sidebar" src="../sidebar.html?current=0a"
          class="sidebar-frame" loading="eager"></iframe>
  <main class="page">
    <nav class="topnav"></nav>
    <section class="hero">
      <h1>第 0a 章 · CPU 微架构</h1>
      <p class="sub">为什么 AI 平台工程师必须懂流水线、Cache 层级与 MESI 协议</p>
      <div class="chips"><span class="chip">CPU</span></div>
      <div class="note"><strong>一句话理解：</strong>...</div>
      <div class="success"><strong>最重要原则：</strong>...</div>
    </section>
    <section class="toc">本章目录</section>
    <section class="section" id="s1">...</section>
    <pre class="mermaid">flowchart LR ...</pre>
    <section class="section" id="s2">...</section>
    <section class="refbox">参考资料</section>
    <nav class="bottomnav"></nav>
    <footer class="footer">...</footer>
  </main>
  <script src="../assets/tutorial-data.js"></script>
  <script src="../assets/mermaid.min.js"></script>
  <script src="../assets/nav.js"></script>
  <script>mermaid.initialize({ startOnLoad: true, theme: 'neutral' });</script>
</body>
</html>
```

### 4.3 Sidebar 设计

`sidebar.html` 是独立 iframe 页：

- 顶部 logo + 教程标题 + 搜索框（纯客户端 fuzzy filter）
- 8 个 Part 折叠组（含 Part 0），当前所在 Part 自动展开
- 当前章高亮（蓝色左边框 + 加粗）
- 章节列表数据从父页面共享的 `tutorial-data.js` 读取（sidebar.html 自己也 `<script src="assets/tutorial-data.js">`）
- iframe `?current=<id>` query param 告诉它高亮哪一章
- 点击章节链接 `target="_top"` 让父窗口跳转

视觉拼接细节：

- iframe 宽 280px、高 100vh、`position:fixed; left:0; top:0`、无边框
- 主内容 `margin-left:280px`
- 移动端（< 768px）：sidebar 隐藏，hamburger 按钮触发覆盖层

### 4.4 由 subagent 直接编写 HTML（不使用脚本）

每个章节由独立 subagent 拿着统一的 Conversion Spec 直接写 HTML，不通过转换脚本。

**Conversion Spec**（完整版本独立写入 `docs/conversion-spec.md`，由 Batch 3 产出）包含：

A. **必须出现的结构元素**：head（charset / viewport / title / link assets）、iframe sidebar、main.page、hero（h1+sub+chips+至少一个 callout）、本章 toc、N 个 section[id=sN]、refbox、footer、末尾三个 script。

B. **风格 token**：`.card`、`.grid-2/.grid-3/.grid-4`、`.note/.warn/.success/.danger` 四色 callout、原生表格、`<pre><code>`、`.kbd .chip .mini .caption`。

C. **mermaid 用法**：包成 `<pre class="mermaid">`，每章 4-8 个，覆盖流程 / 状态机 / 序列 / 架构。

D. **手工 SVG 用法**：mermaid 表达力不够时（菱形决策、四象限、视觉隐喻图），按 `nm.html` 风格嵌入 `<section class="figure">`。

E. **内容质量底线**：hero 必须有"一句话理解"+"最重要原则"两个 callout、≥1 个 mermaid 或手工 SVG、≥3 个表格、≥5 个 callout、refbox 含"学习路线"+"延伸阅读"。

F. **跨章引用**：markdown 跨章链接翻译成 `<a href="../partN/<n>-<slug>.html">`，章内 §N.X 翻译成 `<a href="#sN">`。

### 4.5 index.html

入口页结构：

- Hero 区：教程总标题 + 副标题 + 8 个 chips
- "如何使用本教程"简短卡片
- 8 个学习路径卡片
- 完整章节卡片墙：每 Part 一个大卡，里面网格列出全部章节

## 5. 并行执行模型与质量门

### 5.1 五个批次

**Batch 1：内容补完（24 个 subagent 并行）**

- Part 0 4 章新写：`agent-0a` `agent-0b` `agent-0c` `agent-0d`
- 现有章节扩写按 §3 清单合并打包，每章一个 agent：`agent-ch4` `agent-ch5` `agent-ch6` `agent-ch7` `agent-ch8` `agent-ch9` `agent-ch10` `agent-ch10b` `agent-ch10c` `agent-ch11-12`（Ch 11 和 Ch 12 都是小补丁，合并到一个 agent）`agent-ch13` `agent-ch15` `agent-ch16` `agent-ch17` `agent-ch19` `agent-ch20` `agent-ch21` `agent-ch22` `agent-ch23` `agent-ch25`
- Ch 1, 2, 3, 14, 18, 24 在 §3 标记"无大改"或仅 ~300 字浅引用调整，由 Ch 2 浅引用 agent 顺便处理或不动；不单独派 Batch 1 agent

**Batch 2：附录 + README 同步**（依赖 Batch 1）

- `agent-glossary-tooling-checklist`：扫一遍 Batch 1 全部新增内容，更新 4 份附录
- `agent-readme`：更新 README 章节清单 + 学习路径

**Batch 3：HTML 框架**（与 Batch 2 并行，不依赖 Batch 1 内容）

- `agent-html-skeleton`：写出 `style.css` + `tutorial-data.js` + `sidebar.html` + `nav.js` + `index.html` + 一个章节 reference HTML（Ch 1 全章作为模板示例）+ `docs/conversion-spec.md` 完整版

**Batch 4：HTML 章节大并行（35 个 subagent）**（依赖 Batch 1 + Batch 3）

- 每章一个 subagent：`agent-html-0a` `agent-html-0b` `agent-html-0c` `agent-html-0d` + `agent-html-01` ~ `agent-html-25`（含 10b、10c，共 27 章）+ 4 个附录 agent（glossary、tooling-map、checklists、answers）= 4 + 27 + 4 = 35
- 每个 subagent 拿到：自己负责章节的 markdown 源、Conversion Spec（`docs/conversion-spec.md`）、Reference HTML（Ch 1 标杆，由 Batch 3 产出）、`nm.html` 全文（风格参照）、`tutorial-data.js`（章节顺序数据，仅作上下文，prev/next 由 nav.js 注入不需手写）
- 输出该章 HTML

**Batch 5：集成 review + 修补**（主线 agent 执行）

- 抽样打开 5-8 章 HTML 检查风格一致性、链接完整性、mermaid 渲染、sidebar 同步
- Edit 直接修偏差
- 链接扫描：grep `href=` 核对所有相对路径文件存在
- `SPEC.md` 归档为 `SPEC-archive-2026-04-24.md`

### 5.2 质量门

**Markdown 阶段（Batch 1 完成后）**

- 自动检查：每个 .md 文件 ≥ 验收标准（字数 / 练习题数）
- 主线 agent 抽样 review 4-5 个文件

**HTML 阶段（Batch 4 完成后）**

- 链接完整性：所有相对路径文件存在
- 必备元素检查：每个 HTML 含 sidebar iframe / mermaid script / hero / refbox
- 抽样浏览器打开 review

### 5.3 风险与缓解

| 风险 | 缓解 |
|---|---|
| 22 个 subagent 风格漂移 | Batch 3 先产出 Ch 1 标杆 + 详尽 Conversion Spec；Batch 5 集成 review 修补 |
| 跨章引用断链 | tutorial-data.js 单一数据源 + Batch 5 链接扫描 |
| mermaid 渲染失败 | 经过验证的 mermaid v11 离线 bundle；Conversion Spec 给出可工作语法示例 |
| 内容深度不均 | Spec 明确字数下限和"必须包含"列表 |
| 总文件数大，git 噪声 | `html/` 目录加 `.gitattributes` 标记 generated；commit 分批 |
| iframe 跨域（file:// 打开） | index.html 顶部加提示，建议 `python -m http.server` 或 VS Code Live Server 打开 |

## 6. 不做的事（Scope Exclusion）

| 排除项 | 原因 |
|---|---|
| 重写现有核心章节结构 | 1-9, 14-23 主体质量已经很好，只补缺口 |
| 把 markdown 与 HTML 视为单一来源 | 二者并存：markdown 是教程主源，HTML 是分发交付物，二者独立维护 |
| 添加英文版本 | 超出当前范围 |
| 添加 hands-on 代码项目 | 教程定位是认知框架，不是实操 lab |
| 引入构建工具链（webpack / vite / pandoc） | subagent 直接写 HTML，不需要构建 |
| 章节 HTML 的服务端动态化 | 静态站点，离线可读优先 |

## 7. 工作量估算

| 项 | 数量 | 估算 |
|---|---|---|
| Part 0 全新章节 | 4 章 | ~14K 行 markdown |
| 现有章节扩写 | 22 章 | ~32K 行 markdown |
| HTML 章节文件 | ~32 个 | 平均 800-1500 行 HTML/章 |
| HTML 框架文件 | 6 个 | ~1500 行总计 |
| **总并行 subagent** | **62 个** | 分 5 批次（Batch 1: 24, Batch 2: 2, Batch 3: 1, Batch 4: 35, Batch 5: 主线 agent） |

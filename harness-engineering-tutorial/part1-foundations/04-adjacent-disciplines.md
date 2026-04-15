# 第4章：Harness Engineering 与相邻学科

> 任何新学科都不是凭空产生的。Harness Engineering 站在多个成熟学科的交汇处，理解它与邻居们的边界和联系，会让你更清楚自己需要学什么、不需要学什么。

---

## 学习目标

学完本章，你将能够：

1. 精确区分 Harness Engineering 与 Prompt Engineering、Context Engineering 的边界
2. 解释 Harness Engineering 与 MLOps/ML Engineering 的本质差异
3. 描述 Harness Engineering 如何继承和扩展 DevOps/Platform Engineering 的理念
4. 理解非确定性系统测试与传统软件测试的根本区别
5. 在团队沟通中准确定位 Harness Engineering 的职责范围

---

## 4.1 Harness Engineering vs Prompt Engineering

### Prompt Engineering 的位置

Prompt Engineering 是你最熟悉的起点，也是最容易与 Harness Engineering 混淆的学科。

```
Prompt Engineering 的范围
─────────────────────────

只关注这个框：
                    ┌─────────────────┐
   你写的 prompt →  │     Model       │  → 输出
                    └─────────────────┘

Harness Engineering 的范围
──────────────────────────

关注整个系统：
   ┌────────────────────────────────────────────┐
   │  上下文组装 → prompt 构建 → 模型调用       │
   │  → 输出解析 → 验证 → 重试 → 日志 → 监控   │
   │  → 多步编排 → 工具集成 → 状态管理 → 部署   │
   └────────────────────────────────────────────┘
```

### 核心区别

| 维度 | Prompt Engineering | Harness Engineering |
|------|-------------------|-------------------|
| 关注范围 | 模型的输入 | 模型周围的完整系统 |
| 核心产出 | 一段精心撰写的文本 | 一套可运行的软件系统 |
| 所需技能 | 语言直觉、领域知识 | 系统工程、软件开发 |
| 可测试性 | 主观评估为主 | 可构建自动化测试套件 |
| 版本管理 | prompt 版本化 | 完整的 CI/CD 流程 |
| 失败时 | 改写 prompt | 调整系统（可能改 prompt，也可能改验证逻辑、重试策略等） |

### 关键洞察

**Prompt 只是 Guide 的一种。**

回顾第 3 章的控制矩阵，prompt（包括 system prompt 和 few-shot examples）属于"推理型前馈控制"象限——只是四个象限中的一个。一个成熟的 harness 中，prompt 工程可能只占 20% 的工作量。

```
Harness 工作量分布（典型项目）

  Prompt 设计          ████████░░░░░░░░░░░░  20%
  验证/测试逻辑        ██████████████░░░░░░░  30%
  编排/状态管理        ██████████░░░░░░░░░░░  25%
  可观测性/监控        ██████░░░░░░░░░░░░░░░  15%
  部署/运维            ████░░░░░░░░░░░░░░░░░  10%
```

---

## 4.2 Harness Engineering vs Context Engineering

### Context Engineering 的兴起

Context Engineering 在 2025 年成为热门话题，它的核心主张是：**模型的表现上限由上下文质量决定**。

这个洞察是正确的。但 Context Engineering 和 Harness Engineering 的关系是什么？

### 手段 vs 系统

```
Context Engineering                 Harness Engineering
──────────────────                 ────────────────────

"怎么给模型提供                    "怎么构建一个可靠的
 最好的上下文？"                    AI agent 系统？"

  ┌──────────────┐                 ┌──────────────────────┐
  │ RAG 检索     │                 │ 上下文组装           │ ← 包含 Context Engineering
  │ 向量数据库   │                 │ ┌──────────────┐     │
  │ chunk 策略   │                 │ │ RAG / 向量库  │     │
  │ reranking    │                 │ └──────────────┘     │
  │ prompt 模板  │                 │ 模型调用 + 重试      │
  └──────────────┘                 │ 输出验证             │
                                   │ 多步编排             │
  关注：输入端                      │ 可观测性             │
  产出：高质量上下文                │ 部署运维             │
                                   └──────────────────────┘

                                   关注：整个生命周期
                                   产出：可运行的系统
```

### 对比表

| 维度 | Context Engineering | Harness Engineering |
|------|-------------------|-------------------|
| 核心问题 | 如何让模型"看到"最相关的信息 | 如何让 agent 可靠地完成任务 |
| 范围 | 模型调用前的数据准备 | 整个 agent 生命周期 |
| 关键技术 | RAG、向量检索、chunk 策略 | 控制论、编排、验证、可观测性 |
| 与 Harness 的关系 | 是 Harness 的重要子系统 | 包含 Context Engineering |
| 比喻 | 给 CPU 准备好的数据 | 整个操作系统 |

### 一句话总结

> **Context 是手段，Harness 是系统。** Context Engineering 负责"给模型看什么"，Harness Engineering 还要负责"模型看了之后怎么办"。

---

## 4.3 Harness Engineering vs MLOps / ML Engineering

### 一个关键区别

这可能是最容易产生困惑的边界。让我们先明确一点：

> **Harness Engineering 不训练模型。**

```
ML Engineering / MLOps              Harness Engineering
──────────────────────             ────────────────────

  数据收集                          ×
  数据清洗                          ×
  特征工程                          ×
  模型训练                          ×
  模型评估                          △ (评估 agent，非模型)
  模型部署                          △ (部署 agent，非模型)
  模型监控                          △ (监控 agent 行为)
  ──────────────────               ────────────────────
  模型版本管理                      Harness 版本管理
  训练流水线                        Agent 编排流水线
  A/B 测试（模型级别）               A/B 测试（harness 级别）
  GPU 集群管理                      API 调用管理
```

### 对比表

| 维度 | MLOps / ML Engineering | Harness Engineering |
|------|----------------------|-------------------|
| 对模型的操作 | 训练、微调、部署 | 调用（作为黑盒） |
| 核心技能 | 统计、线性代数、PyTorch | 系统工程、API 设计、DevOps |
| 数据关注点 | 训练数据质量 | 运行时上下文质量 |
| 基础设施 | GPU 集群、训练流水线 | API 网关、编排引擎 |
| 迭代对象 | 模型权重、超参数 | Harness 配置、验证规则 |
| 成本构成 | 训练计算成本 | 推理 API 调用成本 |

### 协作关系

在一个完整的 AI 产品团队中，ML Engineer 和 Harness Engineer 的协作通常如下：

```
ML Engineer 的产出           Harness Engineer 的输入
──────────────────           ─────────────────────

  微调后的模型 ──────────→   model_id: "ft:company-v3"
  模型能力报告 ──────────→   据此设计验证策略
  API endpoint ──────────→   配置到 model interface 层
  token 限制   ──────────→   据此设计 context 策略
```

Harness Engineer 把模型当作**服务**来使用，就像后端工程师使用数据库一样——你不需要了解 B+ 树的实现细节，但你需要知道如何设计高效的查询和处理连接错误。

---

## 4.4 Harness Engineering vs DevOps / Platform Engineering

### 最近的邻居

如果要说 Harness Engineering 与哪个学科最亲近，答案是 DevOps 和 Platform Engineering。

```
DevOps / Platform Engineering 的核心理念
────────────────────────────────────────

  基础设施即代码 (IaC)     →    Harness 即代码
  CI/CD 流水线              →    Agent 评估流水线
  监控与告警                →    Agent 可观测性
  蓝绿部署 / 金丝雀发布    →    Agent A/B 测试
  故障恢复                  →    Agent 降级策略
  安全合规                  →    Prompt Injection 防御
```

### Harness Engineering 是 DevOps 在 AI 时代的延伸

```
                    传统 DevOps
                    ────────────
                    │
                    │  确定性系统
                    │  代码 → 构建 → 测试 → 部署 → 监控
                    │
                    ▼
              ┌─────────────┐
              │   延伸到     │
              │   AI 时代    │
              └─────────────┘
                    │
                    │  非确定性系统
                    │  Harness → 评估 → 验证 → 部署 → 观测
                    │
                    ▼
              Harness Engineering
```

### 继承与扩展

| DevOps 概念 | Harness Engineering 对应 | 新增的挑战 |
|------------|------------------------|-----------|
| 单元测试 | Agent 评估 (evals) | 非确定性：同输入不同输出 |
| 集成测试 | 端到端 agent 测试 | 测试用例的"通过"标准模糊 |
| 持续集成 | 评估流水线 | 评估成本高（每次调用 API） |
| 监控指标 | Agent 质量指标 | 需要语义级别的异常检测 |
| 日志 | 对话轨迹追踪 | 数据量大、结构复杂 |
| 回滚 | Harness 版本回滚 | 可能需要同时回滚 prompt + 配置 + 代码 |
| 安全扫描 | Prompt Injection 检测 | 攻击面是自然语言 |

### Harness Engineer 需要的 DevOps 技能

如果你是一个有 DevOps 背景的工程师转向 Harness Engineering，你会发现 70% 的技能可以直接迁移：

- CI/CD 设计能力 → Agent 评估流水线
- 容器化和编排 → Agent 部署
- 监控和告警 → Agent 可观测性
- 自动化思维 → Harness 自动化

需要额外学习的 30%：

- LLM 的工作原理和限制
- Prompt 和 Context 设计
- 非确定性系统的测试方法
- AI 安全（prompt injection、数据泄露等）

---

## 4.5 Harness Engineering vs Software Testing

### 确定性 vs 非确定性

这是一个经常被低估的区别。传统软件测试建立在一个基本假设上：**相同的输入总是产生相同的输出。**

```
传统软件测试                    AI Agent 测试
─────────────                  ──────────────

  f(x) = y                     f(x) = y₁ 或 y₂ 或 ... yₙ
  始终成立                      每次可能不同

  assert add(2, 3) == 5        assert is_reasonable(
  # 100% 确定性                 #   summarize("long text...")
                                # )
                                # 什么叫"合理"？
```

### 测试策略的差异

| 维度 | 传统测试 | AI Agent 测试 |
|------|---------|--------------|
| 通过标准 | 精确匹配 | 范围匹配 / 语义评估 |
| 测试结果 | 二元（pass/fail） | 概率（80% 时间通过） |
| 测试运行时间 | 毫秒级 | 秒到分钟（API 调用） |
| 测试成本 | 几乎为零 | 每次调用收费 |
| 可重复性 | 完全可重复 | 设置 seed 也不能完全控制 |
| 回归测试 | 明确的通过/失败 | 需要统计显著性 |

### Harness Engineering 的测试方法

面对非确定性，Harness Engineering 发展出了新的测试方法论：

```python
# 传统测试
def test_add():
    assert add(2, 3) == 5  # 精确匹配

# Harness 测试：统计验证
def test_agent_summarize():
    """对 agent 的摘要功能进行统计评估"""
    text = load_test_document()
    results = []

    for _ in range(20):  # 多次运行
        summary = agent.summarize(text)
        score = evaluate_summary(summary, text)
        results.append(score)

    avg_score = sum(results) / len(results)
    pass_rate = sum(1 for r in results if r >= 0.7) / len(results)

    assert avg_score >= 0.75, f"平均质量分 {avg_score} 低于阈值 0.75"
    assert pass_rate >= 0.90, f"通过率 {pass_rate} 低于阈值 90%"
```

```python
# Harness 测试：属性验证（而非精确匹配）
def test_agent_code_generation():
    """验证生成代码的属性，而非精确内容"""
    code = agent.generate_code("排序算法")

    # 属性 1：语法正确
    compile(code, "<test>", "exec")

    # 属性 2：包含函数定义
    assert "def " in code

    # 属性 3：功能正确（通过测试用例）
    exec(code)  # 执行生成的代码
    assert sort_function([3, 1, 2]) == [1, 2, 3]

    # 不检查：具体用了什么算法、变量名是什么、注释内容等
```

---

## 4.6 全景定位：维恩图与总结

### 学科关系维恩图

```
                        ┌──────────────────────┐
                        │   ML Engineering     │
                        │  (训练和微调模型)      │
                        │         ┌────────────┼──────────┐
                        │         │            │          │
                        └─────────┼────────────┘          │
                                  │                       │
            ┌─────────────────────┼───────────┐           │
            │  Prompt Engineering │           │           │
            │  (设计输入)          │           │           │
            │        ┌────────────┼───────┐   │           │
            │        │            │       │   │           │
            │        │   Context  │       │   │           │
            │        │   Eng.     │       │   │           │
            │        │   (组装    │       │   │           │
            │        │   上下文)  │ ┌─────┼───┼───────┐   │
            │        │            │ │     │   │       │   │
            │        └────────────┼─┤     │   │       │   │
            │                     │ │ HARNESS │       │   │
            └─────────────────────┤ │  ENG.   │       │   │
                                  │ │         │       │   │
          ┌───────────────────────┤ │         │       ├───┘
          │   DevOps / Platform   │ │         │       │
          │   Engineering         │ │         │       │
          │   (部署和运维)        │ │         │       │
          │                       │ │         │       │
          └───────────────────────┘ └─────────┘       │
                                    │  Software       │
                                    │  Testing        │
                                    │  (质量保障)      │
                                    └─────────────────┘
```

### 一张总结对比表

| 学科 | 核心问题 | 与 Harness Eng. 的关系 | 技能迁移度 |
|------|---------|----------------------|-----------|
| Prompt Engineering | 如何写好 prompt | prompt 是 Guide 的一种，被 Harness 包含 | 中等：prompt 设计是必要但不充分技能 |
| Context Engineering | 如何组装最优上下文 | context 是 Harness 的子系统 | 中等：RAG 和检索技能有用 |
| ML Engineering | 如何训练和部署模型 | Harness 使用模型但不训练模型 | 低：技能集差异大 |
| MLOps | 如何管理 ML 生命周期 | 方法论可借鉴，但对象不同 | 中等：流水线和监控思维有用 |
| DevOps | 如何自动化部署和运维 | Harness 是 DevOps 在 AI 时代的延伸 | 高：70% 技能可直接迁移 |
| Platform Eng. | 如何构建内部开发平台 | Harness 平台化是高级形态 | 高：平台思维高度适用 |
| Software Testing | 如何保证软件质量 | 测试非确定性系统需要新方法 | 中等：测试思维有用，但方法需调整 |

### 你需要知道什么、不需要知道什么

```
Harness Engineer 的知识需求

  必须精通：                    了解即可：              不需要：
  ────────                     ─────────              ────────
  □ Python / TypeScript        □ ML 基础概念           □ 模型训练
  □ API 设计                   □ 向量数据库原理         □ GPU 编程
  □ CI/CD                      □ Transformer 架构      □ 梯度下降
  □ 容器化 / K8s               □ 分布式系统            □ 特征工程
  □ 可观测性                   □ 安全攻防              □ 统计建模
  □ 评估方法论                 □ 产品设计              □ 数据标注
  □ Prompt / Context 设计                             □ 论文复现
```

---

## 本章小结

| 概念 | 要点 |
|------|------|
| vs Prompt Eng. | Prompt 只是 Guide 的一种，Harness 包含但远超 prompt |
| vs Context Eng. | Context 是手段（子系统），Harness 是系统（全生命周期） |
| vs MLOps | Harness 不训练模型，把模型当服务调用 |
| vs DevOps | 最亲近的邻居，70% 技能可迁移，加 30% AI 特有知识 |
| vs Testing | 非确定性测试需要统计验证和属性验证，而非精确匹配 |
| 定位 | 站在 DevOps、Prompt Eng.、Context Eng.、Testing 的交汇处 |

---

## 动手实验

### 实验 1：学科归类练习

对以下 12 项工作任务，标注它们属于哪个学科（可多选）：

```python
"""
实验 1：学科归类练习
在每行末尾标注学科缩写：
  PE = Prompt Engineering
  CE = Context Engineering
  HE = Harness Engineering
  ML = ML Engineering / MLOps
  DO = DevOps / Platform Engineering
  ST = Software Testing
"""

tasks = [
    "1.  编写 system prompt 让模型按特定格式输出",          # ?
    "2.  设计 RAG 流水线从知识库检索相关文档",              # ?
    "3.  实现 agent 的重试和降级策略",                      # ?
    "4.  微调一个领域特定的语言模型",                       # ?
    "5.  为 agent 搭建 CI/CD 评估流水线",                  # ?
    "6.  设计 few-shot examples 集合",                     # ?
    "7.  构建 agent 行为的可观测性仪表盘",                  # ?
    "8.  编写代码生成 agent 的测试用例",                    # ?
    "9.  优化 token 使用以降低 API 成本",                  # ?
    "10. 设计多 agent 的编排协议",                         # ?
    "11. 部署 agent 到 Kubernetes 集群",                   # ?
    "12. 用 LLM-as-Judge 评估 agent 输出质量",            # ?
]

# 参考答案（翻到下方查看）
answers = {
    1: "PE, HE (Guide 设计)",
    2: "CE, HE (上下文子系统)",
    3: "HE (运行时环境层)",
    4: "ML (Harness Engineer 不做)",
    5: "HE, DO (评估即 CI/CD)",
    6: "PE, CE, HE (Guide 设计)",
    7: "HE, DO (可观测性)",
    8: "HE, ST (非确定性测试)",
    9: "CE, HE (上下文和运行时优化)",
    10: "HE (编排层)",
    11: "DO, HE (部署)",
    12: "HE, ST (推理型 Sensor)",
}

for task in tasks:
    num = int(task.strip().split(".")[0])
    print(f"{task}")
    print(f"   → {answers[num]}")
    print()
```

### 实验 2：从 DevOps 到 Harness 的映射表

```python
"""
实验 2：构建你自己的"DevOps → Harness"映射表
"""

mapping = {
    "DevOps 概念": [
        "Dockerfile",
        "GitHub Actions",
        "Prometheus + Grafana",
        "Feature Flags",
        "Blue-Green Deploy",
        "Terraform",
        "PagerDuty alerts",
        "Load Testing (k6)",
    ],
    "Harness 对应": [
        "Harness 配置文件（定义 agent 的运行环境）",
        "Eval 流水线（每次提交运行评估）",
        "Agent 质量仪表盘（追踪成功率、延迟、成本）",
        "A/B 测试不同 prompt/harness 版本",
        "Agent 版本切换（新旧 harness 平行运行）",
        "Harness 即代码（声明式配置 agent 系统）",
        "Agent 质量告警（成功率下降、成本飙升）",
        "Agent 负载测试（并发调用、token 消耗）",
    ]
}

print(f"{'DevOps 概念':<30} {'Harness 对应'}")
print("=" * 80)
for devops, harness in zip(mapping["DevOps 概念"], mapping["Harness 对应"]):
    print(f"{devops:<30} {harness}")
```

---

## 练习题

### 基础题

1. 一位同事说"我会 Prompt Engineering，所以我也会 Harness Engineering"。你如何礼貌但准确地纠正这个认识？
2. 为什么说"Harness Engineering 不训练模型"是一个重要的边界声明？它对团队分工意味着什么？
3. 列出 3 个从 DevOps 可以直接迁移到 Harness Engineering 的技能，以及 3 个需要重新学习的技能。

### 实践题

4. 你的公司有一个 DevOps 团队和一个 ML 团队。现在要组建 Harness Engineering 能力。画一个组织架构图，说明 Harness Engineer 应该归属哪个团队（或独立成团队），以及与其他团队的协作接口。
5. 为一个文本分类 agent 设计完整的测试策略。明确标注哪些测试方法来自传统软件测试（可直接使用），哪些需要适配非确定性特点。

### 思考题

6. 有人认为"Harness Engineering 只是 DevOps 换了个名字"。你同意吗？举出至少 2 个 Harness Engineering 特有的、在传统 DevOps 中不存在的挑战。
7. 随着模型能力增强，Prompt Engineering 的重要性是在上升还是下降？Harness Engineering 呢？论证你的观点。

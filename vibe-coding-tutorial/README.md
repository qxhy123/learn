# Vibe Coding 完全指南：TDD + DDD + DSL 从零到高阶

> 用测试驱动开发、领域驱动设计和领域特定语言，进入心流状态编程

---

## 什么是 Vibe Coding？

**Vibe Coding** 是一种与 AI 协作的编程方式——你用自然语言描述意图，AI 帮你生成代码。但真正的 Vibe Coding 不是"让 AI 帮你写代码"，而是**建立一套让你和 AI 都能精确表达意图的系统**。

这套系统由三个支柱构成：

| 支柱 | 作用 | 解决的问题 |
|------|------|------------|
| **TDD** (测试驱动开发) | 确定性基础 | 每次迭代都有可验证的终点 |
| **DDD** (领域驱动设计) | 语义准确性 | 代码和业务对齐，减少沟通损耗 |
| **DSL** (领域特定语言) | 表达力放大 | 用领域语言直接编写逻辑 |

---

## 教程目标

通过本教程，你将能够：

- 理解 Vibe Coding 的核心哲学：**意图先于实现**
- 掌握 TDD 的 Red-Green-Refactor 循环作为 AI 协作的节拍器
- 运用 DDD 的战略/战术工具建立清晰的领域模型
- 设计内部/外部 DSL，让代码像业务语言一样可读
- 将三者融合，在复杂项目中保持心流状态

---

## 适合读者

- 已有基本编程经验（Python 为主），想提升代码设计能力
- 正在使用 AI 编程工具（Cursor、Claude、GitHub Copilot）但感觉"失控"
- 想在复杂业务系统中保持代码可维护性

---

## 目录结构

```
vibe-coding-tutorial/
├── part1-foundations/                    # 第一部分：基础认知
│   ├── 01-what-is-vibe-coding.md        # Vibe Coding 概念与 AI 协作哲学
│   ├── 02-tdd-ddd-dsl-trinity.md        # TDD+DDD+DSL 三位一体
│   ├── 03-mindset-and-flow-state.md     # 心流状态与编程节奏
│   └── 04-environment-setup.md          # 工具链配置
│
├── part2-tdd-backbone/                   # 第二部分：TDD 作为主干
│   ├── 05-red-green-refactor-deep.md    # Red-Green-Refactor 深度实践
│   ├── 06-test-first-design-thinking.md # 测试优先的设计思维
│   ├── 07-emergent-architecture.md      # TDD 涌现出架构
│   └── 08-tdd-ai-pair-programming.md    # TDD 与 AI 结对编程
│
├── part3-ddd-modeling/                   # 第三部分：DDD 领域建模
│   ├── 09-ubiquitous-language.md        # 统一语言实战
│   ├── 10-bounded-context-design.md     # 限界上下文设计
│   ├── 11-aggregates-entities-vos.md    # 聚合根、实体与值对象
│   └── 12-domain-events-sagas.md        # 领域事件与 Saga
│
├── part4-dsl-design/                     # 第四部分：DSL 设计
│   ├── 13-internal-dsl-python.md        # Python 内部 DSL
│   ├── 14-fluent-interface-patterns.md  # 流式接口模式
│   ├── 15-external-dsl-with-lark.md     # 用 Lark 构建外部 DSL
│   └── 16-dsl-testing-with-tdd.md       # 用 TDD 测试 DSL
│
└── part5-integration-mastery/            # 第五部分：综合精通
    ├── 17-tdd-ddd-synthesis.md           # TDD 与 DDD 的融合
    ├── 18-dsl-domain-expression.md       # DSL 表达领域语言
    ├── 19-full-vibe-project.md           # 完整 Vibe 项目实战
    └── 20-advanced-patterns-future.md    # 高阶模式与未来展望
```

---

## 学习路径

### 快速入门（2小时）
> Part 1 全部 → Chapter 5 → Chapter 9

### 系统掌握（2周）
> 按顺序完整学习，每章都动手实践代码

### 深度专项
- **TDD 专线**：Chapters 1, 5, 6, 7, 8, 17
- **DDD 专线**：Chapters 1, 9, 10, 11, 12, 17
- **DSL 专线**：Chapters 1, 13, 14, 15, 16, 18

---

## 技术栈

- **语言**：Python 3.10+
- **测试框架**：pytest + unittest
- **DSL 解析**：Lark-parser
- **AI 工具**：Claude / ChatGPT（任意大模型均可）

---

## 配套代码

每章均含可运行的代码示例，遵循以下原则：
- 所有示例先有测试，后有实现（TDD）
- 代码使用领域语言命名（DDD）
- 关键逻辑用 DSL 表达（DSL）

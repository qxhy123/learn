# 从零到高阶的DSL（领域特定语言）教程

## 项目简介

本教程旨在为学习者提供一套系统、完整的 DSL（Domain Specific Language，领域特定语言）设计与实现学习资源。从"什么是DSL"出发，循序渐进地覆盖词法分析、语法解析、AST构建、内部DSL设计模式、外部DSL完整管道，最终深入到类型系统、错误处理、语言工具链（LSP）以及工业级DSL案例分析。

**本教程的独特之处**：每章都包含「核心思维模型」部分，帮助你建立设计直觉，而不仅仅是记忆代码模板。所有代码示例均使用 Python 实现（部分章节提供多语言对比），附带完整的可运行代码和深度原理分析。

---

## 目标受众

- 希望理解编程语言底层原理的开发者
- 需要为团队/产品设计配置语言、查询语言、规则引擎的工程师
- 编译原理初学者，希望通过实战项目掌握语言实现技术
- 对Makefile、SQL、正则表达式、Dockerfile等DSL感到好奇的工程师
- 希望构建语言工具（IDE插件、代码生成器、格式化器）的高级开发者

---

## 章节导航目录

### 第一部分：基础篇 —— 理解DSL的本质

| 章节 | 标题 | 核心概念 |
|------|------|----------|
| 第1章 | [什么是DSL](./part1-foundations/01-what-is-dsl.md) | DSL定义、存在意义、历史沿革 |
| 第2章 | [DSL vs 通用语言](./part1-foundations/02-dsl-vs-gpl.md) | 表达力与通用性的权衡、何时选择DSL |
| 第3章 | [DSL的分类学](./part1-foundations/03-dsl-taxonomy.md) | 内部DSL、外部DSL、语言工作台 |
| 第4章 | [DSL设计原则](./part1-foundations/04-dsl-design-principles.md) | 最小惊讶原则、可读性、可组合性 |

### 第二部分：解析技术 —— 让计算机读懂你的语言

| 章节 | 标题 | 核心概念 |
|------|------|----------|
| 第5章 | [词法分析：把字符串变成Token](./part2-parsing/05-lexer-tokenizer.md) | 正则表达式、有限自动机、手写Lexer |
| 第6章 | [语法分析与AST](./part2-parsing/06-parser-ast.md) | 上下文无关文法、BNF/EBNF、AST节点设计 |
| 第7章 | [递归下降解析器](./part2-parsing/07-recursive-descent.md) | LL(1)文法、预测分析、错误恢复 |
| 第8章 | [解析器组合子](./part2-parsing/08-parser-combinators.md) | 函数式解析、PEG文法、组合子库 |

### 第三部分：内部DSL —— 在宿主语言中雕刻语言

| 章节 | 标题 | 核心概念 |
|------|------|----------|
| 第9章 | [内部DSL模式总览](./part3-internal-dsl/09-internal-dsl-patterns.md) | 宿主语言复用、语法噪声最小化 |
| 第10章 | [构建者模式DSL](./part3-internal-dsl/10-builder-pattern.md) | Builder模式、不可变构造、类型安全 |
| 第11章 | [流畅接口设计](./part3-internal-dsl/11-fluent-interface.md) | 方法链、上下文对象、可读性设计 |
| 第12章 | [Python魔法方法DSL](./part3-internal-dsl/12-python-magic-dsl.md) | __or__重载、装饰器DSL、上下文管理器 |

### 第四部分：外部DSL —— 从零构建一门语言

| 章节 | 标题 | 核心概念 |
|------|------|----------|
| 第13章 | [外部DSL全流程](./part4-external-dsl/13-external-dsl-pipeline.md) | 完整编译管道：词法→语法→语义→执行 |
| 第14章 | [语义分析与符号表](./part4-external-dsl/14-semantic-analysis.md) | 作用域、符号解析、类型推断基础 |
| 第15章 | [代码生成与目标IR](./part4-external-dsl/15-code-generation.md) | 访问者模式生成代码、模板生成、字节码 |
| 第16章 | [解释器模式执行DSL](./part4-external-dsl/16-interpreter-pattern.md) | 树形解释器、环境传递、尾调用优化 |

### 第五部分：高阶篇 —— 工业级DSL工程

| 章节 | 标题 | 核心概念 |
|------|------|----------|
| 第17章 | [DSL类型系统设计](./part5-advanced/17-type-systems.md) | 静态类型、类型推断（Hindley-Milner）、渐进类型 |
| 第18章 | [错误报告与诊断](./part5-advanced/18-error-handling.md) | 源码位置追踪、错误恢复、诊断消息设计 |
| 第19章 | [DSL工具链：LSP与IDE支持](./part5-advanced/19-dsl-tooling.md) | Language Server Protocol、语法高亮、自动补全 |
| 第20章 | [真实世界DSL案例解析](./part5-advanced/20-real-world-dsls.md) | SQL引擎、规则引擎、配置语言、查询DSL |

---

## 学习路径建议

```
零基础  →  第1-4章（理解DSL是什么，建立直觉）
           ↓
入门    →  第5-8章（掌握解析技术，能读懂任何语言）
           ↓
实践    →  第9-12章（快速构建内部DSL，实战价值高）
           ↓
进阶    →  第13-16章（构建完整外部DSL语言）
           ↓
高阶    →  第17-20章（工业级语言工程，类型系统与工具链）
```

### 按目标快速入口

| 目标 | 推荐章节 |
|------|----------|
| 我想快速构建一个配置语言 | 第1章 → 第5-7章 → 第13章 |
| 我想为Python库设计优雅API | 第1章 → 第9-12章 |
| 我想理解SQL/正则是如何工作的 | 第1-3章 → 第5-8章 → 第20章 |
| 我想构建有IDE支持的DSL | 全部章节 → 重点第17-19章 |

---

## 贯穿全书的实战项目

本教程通过一个贯穿始终的项目演示所有概念——**QueryLang**：一个用于数据过滤和转换的迷你查询语言。

```
# QueryLang 示例语法
FROM users
WHERE age > 18 AND status = "active"
SELECT name, email
ORDER BY created_at DESC
LIMIT 10
```

随着教程推进，我们将：
- 第5章：为QueryLang写词法分析器
- 第6-7章：构建语法分析器和AST
- 第13-16章：实现完整解释器
- 第17章：添加类型检查
- 第19章：为QueryLang编写VSCode插件

---

## 技术栈

- **主要语言**：Python 3.10+（利用match语句、dataclasses、typing）
- **解析工具对比**：手写解析器 vs lark-parser vs pyparsing
- **代码生成**：Python字节码、JavaScript输出、SQL输出
- **工具链**：pygls（LSP框架）、tree-sitter（语法高亮）

---

## 关于代码风格

所有代码示例遵循以下原则：
1. **可运行性优先**：每个代码块均可直接复制运行
2. **渐进式复杂度**：先给出简单版本，再引入优化
3. **真实错误处理**：不使用 `pass` 或 `# TODO` 掩盖错误路径
4. **类型注解**：使用Python类型注解提升可读性

---

*"A language that doesn't affect the way you think about programming is not worth knowing."*
— Alan Perlis

# Python TDD 完整教程：从零到高阶

> 以 `unittest` 为基准，兼顾深度与广度的测试驱动开发学习路径

---

## 教程目标

通过本教程，你将：
- 理解 TDD 的核心哲学与工作循环
- 掌握 Python `unittest` 框架的全部能力
- 学会用测试驱动复杂系统的设计
- 了解 Mock、Stub、Spy 等测试替身的精确用法
- 能够在异步、数据库、IO 等复杂场景下编写高质量测试

---

## 目录结构

```
tdd-tutorial/
├── part1-foundations/          # 第一部分：基础
│   ├── 01-what-is-tdd.md      # TDD 是什么？哲学与收益
│   ├── 02-unittest-basics.md  # unittest 框架基础
│   └── 03-first-tdd-cycle.md  # 第一个 TDD 循环实战
│
├── part2-core-concepts/        # 第二部分：核心概念
│   ├── 04-test-organization.md      # 测试组织与命名
│   ├── 05-assertions-deep-dive.md   # 断言深度解析
│   ├── 06-fixtures-and-setup.md     # 测试夹具与生命周期
│   └── 07-test-suites-discovery.md  # 测试套件与自动发现
│
├── part3-intermediate/         # 第三部分：中级
│   ├── 08-mocking-and-patching.md   # Mock 与打桩
│   ├── 09-testing-exceptions.md     # 异常与错误测试
│   ├── 10-parameterized-tests.md    # 参数化测试
│   └── 11-testing-io-database.md    # IO 与数据库测试
│
├── part4-advanced/             # 第四部分：高阶
│   ├── 12-test-doubles-patterns.md  # 测试替身模式
│   ├── 13-async-testing.md          # 异步代码测试
│   ├── 14-coverage-and-metrics.md   # 覆盖率与质量度量
│   └── 15-best-practices.md         # TDD 最佳实践与反模式
│
├── part5-deep-dives/           # 第五部分：深度专题
│   ├── 16-property-based-testing.md    # 基于属性的测试（Hypothesis）
│   ├── 17-hexagonal-architecture-tdd.md # 六边形架构与 TDD
│   ├── 18-legacy-code-tdd.md           # 遗留代码的 TDD 策略
│   ├── 19-mock-internals.md            # Mock 内部机制深度解析
│   └── 20-tdd-design-patterns.md       # TDD 驱动出设计模式
│
└── examples/                   # 可运行示例代码
    ├── part1_examples.py
    ├── part2_examples.py
    ├── part3_examples.py
    ├── part4_examples.py
    └── part5_examples.py       # 六边形架构/Mock机制/Strategy/Observer/Command/Builder
```

---

## 学习路径建议

| 阶段 | 内容 | 目标读者 |
|------|------|----------|
| **Part 1** | TDD 哲学 + unittest 入门 | 完全零基础 |
| **Part 2** | 断言/夹具/套件 | 会写代码但不会测试 |
| **Part 3** | Mock/异常/参数化 | 已有测试基础 |
| **Part 4** | 高阶模式/异步/覆盖率 | 追求工程卓越 |
| **Part 5** | 属性测试/架构/遗留代码/设计模式 | 追求大师境界 |

---

## 快速开始

```bash
# 确认 Python 版本（需要 3.8+）
python --version

# 运行任意示例
python -m unittest examples/part1_examples.py -v

# 运行全部测试
python -m unittest discover examples/ -v
```

---

## 关键原则（贯穿全教程）

1. **Red → Green → Refactor**：先写失败的测试，再写最少代码让其通过，最后重构
2. **测试即文档**：好的测试名称比注释更清晰
3. **一个测试，一个断言**：每个测试只验证一件事
4. **测试应该快**：慢测试会被跳过，快测试才会被执行
5. **FIRST 原则**：Fast、Independent、Repeatable、Self-validating、Timely

# 第1章：什么是DSL（领域特定语言）

## 核心思维模型

> DSL是一种**为特定问题域而设计的语言**，它用该领域的概念和术语来表达程序逻辑，而不是用通用计算机科学概念。

---

## 1.1 从一个问题开始

假设你需要表达"从用户列表中找出年龄大于18岁的活跃用户，返回他们的名字和邮件"。

**用Python（通用语言）表达：**
```python
result = [
    {"name": u["name"], "email": u["email"]}
    for u in users
    if u["age"] > 18 and u["status"] == "active"
]
```

**用SQL（DSL）表达：**
```sql
SELECT name, email FROM users
WHERE age > 18 AND status = 'active'
```

**用Pandas（内部DSL）表达：**
```python
result = (users
    .query("age > 18 and status == 'active'")
    [["name", "email"]])
```

三种表达都在"做同一件事"，但它们使用的**抽象层次**和**目标受众**截然不同。SQL最接近数据库管理员的思维；Python最接近程序员的思维；Pandas介于两者之间。

**这就是DSL的本质：匹配领域专家的思维模型，而不是计算机的执行模型。**

---

## 1.2 DSL的正式定义

**DSL（Domain Specific Language，领域特定语言）** 是一种针对特定应用领域设计的编程或规格说明语言。

与**GPL（General Purpose Language，通用编程语言）** 相对，DSL的特征：

| 维度 | DSL | GPL |
|------|-----|-----|
| 适用范围 | 窄（特定领域） | 宽（任意计算） |
| 表达力 | 在领域内极强 | 通用但啰嗦 |
| 学习曲线 | 对领域专家低 | 对非程序员高 |
| 编译/运行 | 常嵌入宿主语言 | 独立运行 |
| 可图灵完备 | 通常否 | 是 |

---

## 1.3 你每天都在使用DSL

你可能没意识到，但你每天用的工具里充满了DSL：

### 查询与数据
- **SQL**：关系数据库查询语言，自1974年诞生
- **GraphQL**：图状数据查询语言
- **XPath/XQuery**：XML文档查询
- **JMESPath**：JSON查询

### 文本处理
- **正则表达式（Regex）**：模式匹配的极简DSL
  ```regex
  ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
  ```
- **sed/awk**：流编辑器DSL
- **Markdown**：文档格式化DSL（本教程就用它写的）

### 构建与配置
- **Makefile**：构建依赖描述语言
- **Dockerfile**：容器构建描述语言
- **Nginx配置**：Web服务器配置DSL
- **CSS**：样式描述语言（针对"视觉样式"领域）
- **YAML/TOML**：配置数据描述

### 测试与规格
- **Cucumber/Gherkin**：行为驱动开发DSL
  ```gherkin
  Given a user with age 20
  When they try to register
  Then registration should succeed
  ```
- **Hamcrest匹配器**：断言DSL

### 领域专用
- **LaTeX**：科学排版DSL
- **Verilog/VHDL**：硬件描述语言
- **MATLAB**：数值计算DSL
- **R语言**：统计计算DSL

---

## 1.4 DSL的历史脉络

```
1957 - FORTRAN（科学计算，算是早期领域特化语言）
1959 - COBOL（商业数据处理DSL先驱）
1974 - SQL（关系数据库查询，最成功的DSL之一）
1986 - Make（构建系统DSL）
1987 - Perl（文本处理，内嵌强大DSL能力）
1989 - TeX/LaTeX（排版DSL）
1994 - CSS（样式DSL）
1997 - XSL/XSLT（XML转换DSL）
2000s - Ruby DSL爆发（Rake, RSpec, Rails路由）
2010s - JSON/YAML配置DSL兴起
2015 - GraphQL诞生
2020s - 基础设施即代码DSL（Terraform HCL, Pulumi）
```

**关键观察**：每次技术领域成熟，就会催生对应的DSL。DSL是领域知识结晶的语言形态。

---

## 1.5 为什么要创造DSL？

### 动机一：降低认知门槛

非程序员（数据分析师、运营、金融师）能用SQL查询数据，但他们无法用C++写代码。DSL把领域知识从编程知识中解耦。

### 动机二：提高表达密度

用SQL表达一个复杂查询，往往比等价的Python代码少10倍字符数，且语义更清晰。

### 动机三：可验证性

DSL的受限语法使得静态分析、类型检查更容易。SQL注入之所以能被检测，是因为SQL有明确的语法边界。

### 动机四：可优化性

SQL查询优化器能将你写的`SELECT`语句优化为最高效的执行计划，因为SQL描述的是"想要什么"而非"如何得到"。这是DSL的声明式能力。

### 动机五：沟通媒介

架构师和产品经理能读懂Gherkin测试，参与讨论。DSL成为技术与业务之间的桥梁。

---

## 1.6 DSL的代价

DSL不是银弹，它有真实代价：

### 实现成本
构建一个DSL需要实现词法分析、语法解析、语义分析、解释/编译。这是一个完整的语言工程项目。

### 生态碎片化
每个自定义DSL都需要新的工具：解析器、调试器、文档、语法高亮。

### 学习负担
用户需要学习新语法。对于小团队或短生命周期项目，这个投入可能得不偿失。

### 错误陷阱
```
# 哪个错误更好调试？
users.filter(age__gt=18)          # Django ORM（内部DSL）→ Python异常
SELECT * FROM users WHERE ag > 18 # SQL（外部DSL）→ "column ag does not exist"
```

外部DSL的错误消息设计是一项专门技艺（第18章专题讲解）。

---

## 1.7 核心判断：何时该用DSL？

```
是否存在清晰的问题域？
├── 否 → 不需要DSL，用GPL
└── 是 → 该领域是否有大量重复的结构化表达？
         ├── 否 → 可能只需要库/API
         └── 是 → 领域专家能否理解通用语言代码？
                  ├── 是 → 内部DSL（第9-12章）
                  └── 否 → 外部DSL（第13-16章）
```

**经验法则**：如果你发现自己在写大量"配置"代码，或者在注释里解释"这段代码的意思是...（用领域术语）"，那就是DSL的信号。

---

## 1.8 本章实战：识别DSL特征

分析下面这段代码，判断哪些是DSL特征：

```python
# Flask路由（内部DSL）
@app.route('/users/<int:user_id>', methods=['GET', 'POST'])
def user_handler(user_id):
    pass

# SQLAlchemy查询（内部DSL）
users = session.query(User)\
    .filter(User.age > 18)\
    .order_by(User.name)\
    .limit(10)\
    .all()

# pytest参数化（内部DSL）
@pytest.mark.parametrize("input,expected", [
    ("hello", 5),
    ("world", 5),
    ("", 0),
])
def test_length(input, expected):
    assert len(input) == expected
```

**分析**：
- Flask路由：用`@app.route`装饰器表达HTTP路由映射，是内部DSL（利用Python装饰器语法）
- SQLAlchemy：方法链模拟SQL语法，是内部DSL（流畅接口模式）
- pytest参数化：用数据驱动测试，是内部DSL（装饰器+数据表格模式）

---

## 小结

| 概念 | 要点 |
|------|------|
| DSL定义 | 针对特定问题域的语言，匹配领域专家思维模型 |
| 核心价值 | 降低门槛、提高表达密度、支持声明式优化 |
| 真实代价 | 实现成本、工具链建设、学习负担 |
| 使用时机 | 领域清晰 + 大量结构化表达 + 非程序员用户 |

---

## 思考题

1. Makefile 的语法（目标:依赖\n\t命令）体现了哪些DSL设计决策？它的设计有哪些被人诟病的缺陷？
2. CSS是DSL还是数据格式？它是图灵完备的吗？（提示：CSS动画+计数器...）
3. 如果你要为公司的运营团队设计一个"活动规则配置语言"，你会优先考虑内部DSL还是外部DSL？为什么？

---

**下一章**：[DSL vs 通用语言：权衡的艺术](./02-dsl-vs-gpl.md)

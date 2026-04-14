# 第3章：DSL的分类学

## 核心思维模型

> DSL不是单一事物，而是一个**设计空间**。理解分类学，就是理解这个设计空间的坐标系——不同坐标对应不同的实现策略和工程成本。

---

## 3.1 最重要的分类：内部 vs 外部

### 内部DSL（Internal / Embedded DSL）

**定义**：嵌入在宿主编程语言（host language）中，利用宿主语言的语法特性来构造领域语言。

```ruby
# Ruby内部DSL：Rails路由
Rails.application.routes.draw do
  get '/users', to: 'users#index'
  post '/users', to: 'users#create'
  resources :posts do
    resources :comments
  end
end
```

这是合法的Ruby代码，但它"看起来"像一门路由配置语言。

```python
# Python内部DSL：pytest
import pytest

@pytest.fixture
def db_session():
    session = create_session()
    yield session
    session.rollback()

@pytest.mark.parametrize("email", [
    "user@example.com",
    "another.user@domain.org",
])
def test_valid_email(email):
    assert validate_email(email)
```

**内部DSL的本质**：利用宿主语言的以下特性来"伪造"新语法：
- 方法链（method chaining）
- 操作符重载（operator overloading）
- 装饰器/注解（decorators/annotations）
- 块/闭包（blocks/closures/lambdas）
- 元类/宏（metaclasses/macros）

**优点**：
- 零额外解析器开发成本
- 自动获得宿主语言的所有工具（IDE、调试器、类型检查器）
- 可以无缝混用DSL和通用代码

**缺点**：
- 受限于宿主语言的语法约束（不能完全自定义语法）
- 宿主语言语法噪声（括号、引号等）难以完全消除
- 非程序员仍需理解宿主语言基础

---

### 外部DSL（External DSL）

**定义**：完全独立于任何宿主语言，有自己的语法和解析器。

```
# 假想的工作流DSL（外部DSL）
workflow deploy_pipeline {
    stage build {
        run: "docker build -t app:${VERSION} ."
        on_failure: abort
    }
    
    stage test {
        parallel {
            run: "pytest tests/unit"
            run: "pytest tests/integration"
        }
        timeout: 10m
    }
    
    stage deploy {
        requires: [build, test]
        run: "kubectl apply -f k8s/"
        notify: slack("#deployments")
    }
}
```

这段代码不是任何通用语言，需要专门的解析器才能执行。

**真实的外部DSL例子**：
```sql
-- SQL：最成功的外部DSL
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2024-01-01'
GROUP BY u.name
HAVING COUNT(o.id) > 5
ORDER BY order_count DESC;
```

```css
/* CSS：样式外部DSL */
.container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

@media (min-width: 768px) {
  .container { flex-direction: row; }
}
```

**优点**：
- 语法完全自由，可以精确匹配领域概念
- 可以被非程序员使用（无需了解任何通用语言）
- 更强的静态分析和错误检测能力

**缺点**：
- 需要实现完整的解析器
- 需要构建配套工具链（IDE支持、调试器、文档生成）
- 与外部系统集成需要额外绑定代码

---

## 3.2 按执行方式分类

### 解释型DSL（Interpreted DSL）

DSL解析后，直接由解释器执行AST：

```
源码字符串
    ↓ 词法分析
Token流
    ↓ 语法分析
AST（抽象语法树）
    ↓ 直接解释执行
结果
```

大多数简单DSL采用这种方式。优点是实现简单，缺点是性能较低。

### 编译型DSL（Compiled DSL）

DSL被编译为目标代码（可以是其他高级语言、字节码、机器码）：

```
源码
    ↓ 前端（词法+语法+语义分析）
中间表示（IR）
    ↓ 优化
优化后IR
    ↓ 后端（代码生成）
目标代码（Python/JS/字节码/机器码）
```

SQL就是这种方式：SQL文本被数据库编译成查询执行计划。

### 转译型DSL（Transpiled DSL）

DSL被转换为另一种高级语言：

```
TypeScript → JavaScript
SCSS → CSS
CoffeeScript → JavaScript
GraphQL schema → TypeScript types
```

这是构建DSL的常见偷懒策略：不用实现运行时，只需生成目标语言代码。

---

## 3.3 按语义风格分类

### 声明式DSL（Declarative DSL）

描述"是什么"，而非"怎么做"：

```sql
-- 声明"想要什么"，不指定执行顺序
SELECT name FROM users WHERE age > 18 ORDER BY name;
```

```css
/* 声明元素样式，不指定渲染算法 */
button { background: blue; color: white; }
```

**特征**：
- 无副作用（纯粹描述性）
- 顺序无关（语句顺序不影响语义）
- 可被优化器重写

### 命令式DSL（Imperative DSL）

描述"怎么做"，有明确的执行步骤：

```makefile
# Makefile：命令式，有明确执行顺序
build: compile link
compile:
    gcc -c main.c -o main.o
link:
    gcc main.o -o main
```

### 反应式DSL（Reactive DSL）

描述数据流和事件响应：

```javascript
// RxJS（响应式DSL）
fromEvent(button, 'click')
  .pipe(
    debounceTime(300),
    switchMap(event => fetch('/api/search')),
    map(response => response.json()),
    catchError(err => of([]))
  )
  .subscribe(results => render(results));
```

---

## 3.4 按抽象层次分类

### 数据DSL（Data DSL）

主要用于描述数据结构和配置：

```toml
# TOML配置DSL
[server]
host = "localhost"
port = 8080

[database]
url = "postgres://localhost/mydb"
pool_size = 10

[[plugins]]
name = "auth"
enabled = true
```

### 工作流DSL（Workflow DSL）

描述流程、步骤和依赖：

```yaml
# GitHub Actions工作流DSL
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest
```

### 规则DSL（Rules DSL）

描述业务规则和条件逻辑：

```
# 保险规则引擎DSL（假想）
rule "高风险客户":
    when:
        客户.年龄 < 25
        AND 客户.驾龄 < 2
        AND 车辆.价值 > 500000
    then:
        保费系数 = 1.8
        要求.加收附加险 = true
```

### 查询DSL（Query DSL）

用于数据检索和过滤：

```json
// Elasticsearch Query DSL
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "python" } }
      ],
      "filter": [
        { "term": { "status": "published" } },
        { "range": { "date": { "gte": "2024-01-01" } } }
      ]
    }
  }
}
```

---

## 3.5 语言工作台（Language Workbench）

语言工作台是专为构建DSL而设计的工具平台，代表了DSL工程的最高抽象层次：

### JetBrains MPS

```
// MPS中的DSL定义（概念化表示）
concept UserStory
  properties:
    title: string
    description: string
  children:
    acceptance_criteria: AcceptanceCriteria[1..*]
  
  editor:
    As a <role>
    I want <goal>
    So that <benefit>
```

MPS允许你用图形化方式定义DSL的语法、语义、编辑器和生成器，无需手写解析器代码。

### Xtext（Eclipse生态）

```
// Xtext文法定义（类EBNF）
grammar org.example.Greeting with org.eclipse.xtext.common.Terminals

generate greeting "http://www.example.org/Greeting"

Model:
    greetings+=Greeting*;

Greeting:
    'Hello' name=ID '!';
```

Xtext从文法定义自动生成解析器、AST类和Eclipse IDE插件。

**语言工作台的核心价值**：把DSL的"工具链问题"（IDE支持、调试、重构）也一并解决，而不只是解决解析问题。

---

## 3.6 完整分类图

```
DSL
├── 内部DSL（Internal/Embedded）
│   ├── 流畅接口（Fluent Interface）
│   ├── 方法链（Method Chaining）
│   ├── 装饰器DSL（Decorator-based）
│   ├── 操作符重载DSL（Operator Overloading）
│   └── 宏DSL（Macro-based，Lisp/Rust/C++）
│
├── 外部DSL（External）
│   ├── 解释型
│   │   ├── 树遍历解释器
│   │   └── 基于栈的解释器
│   ├── 编译型
│   │   ├── 本地代码生成
│   │   └── 字节码生成
│   └── 转译型
│       ├── 同层转译（DSL→高级语言）
│       └── 模板生成
│
└── 语言工作台（Language Workbench）
    ├── JetBrains MPS
    ├── Xtext
    └── Spoofax
```

---

## 3.7 选择指南

```python
def choose_dsl_type(context: dict) -> str:
    """根据上下文选择DSL类型的决策逻辑"""
    
    # 如果受众是非程序员，必须外部DSL
    if not context["audience_can_code"]:
        return "外部DSL"
    
    # 如果宿主语言支持足够的元编程
    if context["host_language"] in ["Ruby", "Python", "Kotlin", "Scala"]:
        if context["syntax_flexibility_needed"] == "low":
            return "内部DSL（流畅接口）"
        elif context["syntax_flexibility_needed"] == "medium":
            return "内部DSL（装饰器/操作符重载）"
    
    # 如果需要专业IDE工具支持
    if context["needs_ide_support"] and context["team_size"] > 20:
        return "语言工作台（MPS/Xtext）"
    
    # 默认：外部DSL（手写解析器）
    return "外部DSL（手写递归下降）"
```

---

## 3.8 本章实战：分类现有DSL

对以下DSL进行分类（内部/外部、声明式/命令式、执行方式）：

1. **正则表达式** `^[a-z]+$`
2. **Gradle构建脚本**（Kotlin DSL）`plugins { kotlin("jvm") version "1.9.0" }`
3. **ANTLR文法定义** `expr: expr '*' expr | INT ;`
4. **Prometheus PromQL** `rate(http_requests_total{job="api"}[5m])`
5. **AWS CloudFormation** YAML模板

**参考答案**：
1. 正则：外部DSL，声明式，解释型（由正则引擎执行）
2. Gradle Kotlin DSL：内部DSL（Kotlin），命令式，编译型（JVM字节码）
3. ANTLR：外部DSL，声明式，编译型（生成Java/Python解析器代码）
4. PromQL：外部DSL，声明式（查询DSL），解释型
5. CloudFormation：外部DSL，声明式（数据DSL），解释型（由AWS运行时执行）

---

## 小结

| 分类维度 | 类型 | 适用场景 |
|----------|------|----------|
| 宿主关系 | 内部DSL | 程序员受众，快速实现 |
| 宿主关系 | 外部DSL | 非程序员，语法完全自定义 |
| 执行方式 | 解释型 | 简单DSL，开发快 |
| 执行方式 | 编译/转译型 | 性能要求高，或复用现有运行时 |
| 语义风格 | 声明式 | 配置、查询、规则 |
| 语义风格 | 命令式 | 工作流、脚本 |

---

**上一章**：[DSL vs 通用语言](./02-dsl-vs-gpl.md)
**下一章**：[DSL设计原则](./04-dsl-design-principles.md)

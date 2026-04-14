# 第2章：DSL vs 通用语言——权衡的艺术

## 核心思维模型

> DSL和GPL不是对立关系，而是**表达力-通用性权衡谱系**上的不同点位。理解这个谱系，才能做出正确的语言选择。

---

## 2.1 表达力谱系

把编程语言按"领域特化程度"排列：

```
通用性 ←────────────────────────────────────→ 领域特化

汇编  C  Python  Ruby  SQL  Regex  Makefile  Gherkin
 │                              │                │
 │                              │                └── 纯粹描述，几乎无计算
 │                              └── 声明式查询
 └── 接近硬件，最通用
```

关键洞察：**越往右，在目标领域的表达力越强，但通用计算能力越弱**。

---

## 2.2 Greenspun第十定律

> "Any sufficiently complicated C or Fortran program contains an ad hoc, informally-specified, bug-ridden, slow implementation of half of Common Lisp."
> —— Philip Greenspun

改写成DSL版：
> "任何足够复杂的配置文件，最终都会演变成一门非正式的、缺乏规范的、充满bug的DSL。"

典型案例：
- **Nginx配置**：起初是简单的`key value`配置，后来加入`if`、`map`、`lua`模块，事实上成了编程语言
- **Kubernetes YAML**：加入`helm`模板后，YAML里出现了`{{- if .Values.enabled }}`这样的代码
- **CMake**：从构建配置演变为功能完整（但极其难用）的脚本语言

**教训**：与其让配置文件悄悄长成怪物，不如有意识地设计一门DSL。

---

## 2.3 四个核心权衡维度

### 维度一：表达密度（Expressiveness）

```python
# 目标：验证一个字符串是有效的邮件地址

# 方案A：Python（通用语言）
import re
def is_valid_email(s: str) -> bool:
    pattern = re.compile(
        r'^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+'
        r'@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?'
        r'(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    )
    return bool(pattern.match(s))

# 方案B：正则DSL（直接使用）
pattern = r'^[\w.!#$%&\'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+$'

# 方案C：假想的邮件验证DSL
validate email:
    local-part: alphanum | special-chars
    "@"
    domain: alphanum separated by "."
    tld: alpha{2,}
```

方案B（正则）的密度最高，但可读性依赖于读者的正则功底。方案C可读性最好，但需要实现一个解析器。**没有免费的午餐。**

### 维度二：受众范围（Audience）

| 语言 | 主要受众 | 次要受众 |
|------|----------|----------|
| Python | 程序员 | 数据科学家 |
| SQL | 数据分析师 | DBA、程序员 |
| Gherkin | 业务分析师 | 测试工程师 |
| CSS | 前端开发者 | 设计师（有时） |
| Terraform HCL | DevOps工程师 | 开发者 |

**设计原则**：DSL的语法应该优先服务**主要受众**的思维模型。SQL用`WHERE`而不是`filter`，因为数据分析师理解"条件过滤"的概念更自然地映射到"where"这个英文词。

### 维度三：静态分析能力（Analyzability）

DSL受限的语法带来更强的静态分析能力：

```sql
-- SQL解析器可以在执行前检测：
SELECT nme FROM users  -- 列名拼写错误（静态检测）
WHERE age > "18"       -- 类型不匹配（某些DBMS可检测）
```

```python
# Python：运行时才报错
users[0]["nme"]  # KeyError at runtime
```

**这是DSL的隐藏优势**：因为语言受限，编译器/解析器能做更多的静态保证。这正是类型系统（第17章）和错误诊断（第18章）的基础。

### 维度四：可组合性（Composability）

通用语言的可组合性通常优于DSL：

```python
# Python：可以组合任意逻辑
result = process(
    transform(
        filter_data(users, lambda u: u.age > 18),
        add_field("full_name", lambda u: f"{u.first} {u.last}")
    ),
    sort_by="full_name"
)
```

```sql
-- SQL：组合能力有限（子查询、CTE是主要手段）
WITH active_users AS (
    SELECT *, first_name || ' ' || last_name AS full_name
    FROM users WHERE age > 18
)
SELECT * FROM active_users ORDER BY full_name;
```

SQL的CTE是一种补偿机制——因为SQL不支持函数抽象，所以引入了CTE来解决组合问题。

---

## 2.4 "语言谱系"框架

Martin Fowler在《Domain-Specific Languages》中提出了一个有用的分类：

```
               ┌─────────────────────────────────┐
               │         计算完备性               │
               │  低                          高  │
     ┌─────────┼────────────────────────────────┤
     │ 配置性  │  INI/TOML    │  YAML    │  Lua  │
     │         │  JSON        │  HCL     │  配置 │
     ├─────────┼─────────────-┼──────────┼───────┤
     │ 声明式  │  CSS         │  SQL     │       │
     │         │  正则表达式  │  GraphQL │       │
     ├─────────┼──────────────┼──────────┼───────┤
     │ 过程式  │              │  Make    │  Ant  │
     │         │              │  CMake   │  DSL  │
     └─────────┴──────────────┴──────────┴───────┘
```

**设计建议**：明确你的DSL在这个矩阵中的位置，然后**抵制功能蔓延（feature creep）**，不要让DSL爬格子。

---

## 2.5 案例研究：GraphQL vs REST

REST API是否算DSL？严格说不算（它是架构风格），但GraphQL是一个很好的DSL设计案例：

**REST方式（通用HTTP约定）：**
```
GET /users/123
GET /users/123/posts
GET /users/123/posts/456/comments
```
需要3次请求，且通常过度获取（over-fetching）或欠获取（under-fetching）数据。

**GraphQL DSL：**
```graphql
query {
  user(id: 123) {
    name
    email
    posts(first: 5) {
      title
      comments(first: 3) {
        text
        author { name }
      }
    }
  }
}
```

1次请求，精确描述需要的数据。GraphQL的成功恰好体现了DSL的价值：
1. **领域明确**：图状数据查询
2. **受众清晰**：前端开发者
3. **表达密度高**：嵌套结构直接反映数据图
4. **可静态分析**：GraphQL schema提供完整类型信息

---

## 2.6 反模式：错误使用DSL

### 反模式一：用DSL替代函数抽象

```yaml
# 这是一个真实的CI配置（类Gitlab CI）
steps:
  - name: build
    if: ${{ github.event_name == 'push' && (startsWith(github.ref, 'refs/heads/main') || startsWith(github.ref, 'refs/tags/v')) }}
    run: |
      if [[ "$GITHUB_REF" == refs/tags/* ]]; then
        VERSION=${GITHUB_REF#refs/tags/}
      else
        VERSION="latest"
      fi
      docker build -t myapp:$VERSION .
```

这段YAML已经在试图用字符串模拟编程语言。正确做法：在YAML中调用脚本文件，把逻辑放回真正的语言里。

### 反模式二：把DSL设计成通用语言

```
# 假想的"简单配置语言"，最终长成了这样：
define user_validator(user):
    if user.age > 18:
        if user.email matches /^.+@.+\..+$/:
            return valid
        else:
            return invalid("email format error")
    else:
        return invalid("age must > 18")
```

这已经是一门编程语言了。为什么不直接用Python？**DSL应该有明确的"不支持清单"**。

### 反模式三：为DSL添加过多逃生舱口（escape hatch）

```
{% raw %}{{ python_expression }}{% endraw %}  # Jinja2的Python逃逸
{%- exec -%}import os; os.system('rm -rf /'){% endexec %}  # 危险！
```

太多逃生舱口意味着DSL的抽象不够好，用户不断突破边界。

---

## 2.7 实用决策框架

```
问题：我该为X领域创建DSL吗？

Step 1: 量化表达重复度
  → 写出10个典型用例
  → 计数通用语言代码的"噪声比"（注释/脚手架代码占比）
  → 如果噪声比 > 50%，DSL值得考虑

Step 2: 评估受众
  → 谁会写这些代码？他们懂Python/JavaScript吗？
  → 如果受众是非程序员，几乎必须用外部DSL

Step 3: 评估生命周期
  → 这个DSL会用多久？有多少用户？
  → 短期项目 + 少量用户 → 用库/API
  → 长期产品 + 大量非程序员用户 → 外部DSL值得投入

Step 4: 考虑"内部DSL"的可行性
  → 宿主语言支持足够的元编程吗？（Python: yes，Java: partially，Go: no）
  → 内部DSL实现成本低10倍，优先考虑
```

---

## 2.8 本章实战：分析设计决策

分析 **Terraform HCL** 的设计决策：

```hcl
# Terraform HCL
resource "aws_instance" "web" {
  ami           = "ami-0c94855ba95b798c7"
  instance_type = "t2.micro"

  tags = {
    Name = "HelloWorld"
  }

  lifecycle {
    create_before_destroy = true
  }
}
```

**问题**：
1. HCL为什么选择比YAML更复杂的语法？（提示：YAML不支持函数调用和表达式）
2. `lifecycle`块体现了什么DSL设计模式？
3. 如果你是HashiCorp工程师，你会在HCL中加入`for`循环吗？（事实：HCL确实加了`for_each`）

**分析参考**：
- HCL选择自定义语法，而非JSON/YAML，是因为需要支持`${interpolation}`和函数调用——这是YAML无法干净地支持的
- `lifecycle`块是**嵌套DSL**模式，不同的块类型有不同的语义，用块名而非键名区分语义
- `for_each`的加入是功能蔓延的例子，但Terraform认为它是必要的权衡

---

## 小结

| 维度 | DSL优势 | DSL劣势 |
|------|---------|---------|
| 表达密度 | 领域内极高 | 领域外为零 |
| 受众范围 | 对非程序员友好 | 需要新学习成本 |
| 静态分析 | 受限语法易于分析 | 工具链需自建 |
| 可组合性 | 声明式易优化 | 过程式组合差 |

---

## 思考题

1. Python的列表推导式 `[x for x in list if condition]` 是DSL吗？它从哪门语言借鉴了这个语法？
2. 为什么 Makefile 的目标/依赖语法如此长寿（40年+），即使它有很多公认的缺陷？
3. 比较 Docker Compose YAML 和 Kubernetes YAML：它们都是配置DSL，但设计哲学有何不同？

---

**上一章**：[什么是DSL](./01-what-is-dsl.md)
**下一章**：[DSL的分类学](./03-dsl-taxonomy.md)

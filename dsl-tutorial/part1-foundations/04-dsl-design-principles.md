# 第4章：DSL设计原则

## 核心思维模型

> 好的DSL设计不是"把所有需求都放进语言里"，而是**找到最小完备集**——用最少的概念，覆盖最多的用例，同时保持语法的可预测性和优雅性。

---

## 4.1 第一原则：与领域专家的词汇对齐

DSL的语法必须使用领域内约定的术语，而非技术术语。

**反面教材**：
```
# 技术化的配置语言（差）
execute_validation(
    target_entity="USER",
    validation_function="age_check",
    param_key="age",
    param_comparator="GREATER_THAN",
    param_value=18
)
```

**正面教材**：
```
# 领域化的规则DSL（好）
rule "成年验证":
    用户的年龄 必须大于 18
```

**实践方法**：
1. 与领域专家做"词汇卡片"练习——让他们用自然语言描述5-10个典型场景
2. 记录他们反复使用的名词、动词和修饰语
3. 这些词就是DSL关键字的候选

---

## 4.2 第二原则：最小惊讶原则（POLA）

**Principle of Least Astonishment**：DSL的行为应该符合用户的直觉预期。

```python
# 反例：违反最小惊讶原则
# 假设有一个日期DSL：
duration = 2 months + 30 days  # 用户预期：约90天
# 但实现中 "2 months" 取平均30.44天 → 结果是91.88天
# 在月份边界上行为不一致 → 让用户困惑
```

```
# 正例：显式优于隐式
duration = 60 days              # 明确，无歧义
duration = 2 calendar_months   # 明确声明日历月语义
```

**POLA的几个子原则**：

### 一致性（Consistency）
同类事物用同类语法：
```
# 好：所有类型声明形式一致
let x: int = 5
let name: string = "Alice"
let items: list[int] = [1, 2, 3]

# 差：混用不同风格
int x = 5           # C风格
name = "Alice"      # 动态推断
items := [1, 2, 3]  # Go风格
```

### 幂等性（Idempotency）
重复应用不改变结果：
```
# 好的配置DSL：多次设置同一属性，最后一次生效
config:
    timeout: 30s
    timeout: 60s  # 最终生效，覆盖上面的

# 差的配置DSL：重复设置导致意外累加
config:
    retry: 3      # 意外变成 retry: 3+3=6
    retry: 3
```

### 对称性（Symmetry）
如果A能做某事，B也能以类似语法做类似事：
```
# 好：操作对称
FROM source SELECT fields  # 查询
INTO target INSERT fields  # 插入（结构镜像）
```

---

## 4.3 第三原则：渐进式复杂度（Progressive Disclosure）

简单用例应该语法简单，复杂用例可以使用更复杂语法，但不强迫简单用例了解复杂语法。

```
# 层次一：最简单用法（80%的用例）
validate email

# 层次二：带参数（15%的用例）
validate email:
    max_length: 254
    allow_subdomains: true

# 层次三：完整控制（5%的用例）
validate email:
    max_length: 254
    allow_subdomains: true
    custom_rule: /^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/i
    on_error: "请输入有效的邮件地址"
```

**对应设计模式**：默认值（Sensible Defaults）。复杂参数都应该有合理默认值，用户只在需要覆盖时才写。

---

## 4.4 第四原则：清晰优于简洁

```
# "简洁"的正则风格（难读）
@>18&active:name,email|10

# "清晰"的DSL风格（可读）
FROM users
WHERE age > 18 AND status = "active"
SELECT name, email
LIMIT 10
```

当简洁和清晰冲突时，**优先选择清晰**。DSL的用户通常比设计者少，清晰的代码降低了他人理解你意图的成本。

**反例警告**：APL语言
```
⍝ APL：求质数（极度简洁但完全不可读）
p←{⍵/⍨{∧/⍵≠0}¨⍵|⍨¨⊂⍳⍵-1}⍳100
```

APL是学术上有趣的反面案例：它的操作符密度如此之高，以至于成为了"只写语言"（write-only language）。

---

## 4.5 第五原则：有边界的错误处理

DSL必须提供**有意义的错误消息**，并明确指出错误位置。

```python
# 差的错误报告
ParseError: unexpected token

# 好的错误报告
line 5, column 12: unexpected token '='
  WHERE age = > 18
            ^
  Did you mean 'age > 18' or 'age >= 18'?
```

**错误消息的五个要素**：
1. **位置**：行号、列号
2. **上下文**：显示错误发生的代码片段
3. **指示**：用箭头或下划线标出问题所在
4. **解释**：用领域语言（而非技术术语）说明错误
5. **建议**：如果可能，提供修复建议

这是第18章的核心主题，这里先建立意识。

---

## 4.6 第六原则：正交性（Orthogonality）

**正交**：语言的概念之间相互独立，每个特性只做一件事，特性之间的组合方式可预测。

```
# 非正交设计（差）：
# "过滤"和"排序"是一个不可分割的操作
filter_and_sort(data, age > 18, by="name")

# 正交设计（好）：
# "过滤"和"排序"是独立的，可以任意组合
data | filter(age > 18) | sort(by="name")
data | sort(by="name") | filter(age > 18)  # 顺序可以变化
data | filter(age > 18)                     # 可以只过滤不排序
data | sort(by="name")                      # 可以只排序不过滤
```

正交性是组合爆炸的解决方案：n个正交特性能提供 2^n 种组合，而不是需要为每种组合单独设计语法。

---

## 4.7 第七原则：明确的逃生舱口（Escape Hatches）

DSL必然无法覆盖所有用例。为高级用户提供逃生舱口（能直接执行宿主语言代码），但要明确标注这是"超出DSL设计范围"的行为：

```
# 规则引擎DSL
rule "标准折扣":
    when 订单金额 > 1000:
        折扣 = 0.1

# 逃生舱口：嵌入Python代码（明确标注为高级用法）
rule "复杂折扣计算":
    when 订单金额 > 1000:
        [python]
        discount = complex_pricing_model(
            order.amount,
            user.loyalty_tier,
            season_factor
        )
        [/python]
```

**设计要点**：
- 逃生舱口的语法应该**视觉上与DSL代码明显区分**
- 文档中明确说明逃生舱口的副作用和限制
- 考虑安全问题（第17章类型系统会涵盖沙箱化）

---

## 4.8 第八原则：版本演化策略

DSL一旦发布，就需要考虑如何演化而不破坏现有用户：

```
# 版本化的DSL声明
#!querylang v2.0

FROM users
WHERE age > 18
-- v2.0新增语法：窗口函数
WINDOW OVER (PARTITION BY department ORDER BY salary)
SELECT name, salary, RANK() as salary_rank
```

**演化策略**：

| 策略 | 适用场景 | 代价 |
|------|----------|------|
| 语义版本（SemVer）| 公开DSL | 需维护多版本解析器 |
| 弃用警告 | 渐进迁移 | 需要lint工具支持 |
| 向后兼容添加 | 最低风险 | 语言可能变臃肿 |
| 大版本重写 | 根本缺陷 | 迁移成本高 |

---

## 4.9 实战：设计 QueryLang v1.0

结合本章原则，设计我们贯穿全书的 **QueryLang**：

### 需求收集（与"领域专家"对话）

```
用户A（数据分析师）："我想从用户表里拿出18岁以上的活跃用户，
                     看他们的名字和邮件，按名字排序，先看前10个"

用户B（运营）："给我找上周注册的用户，按注册时间倒排"

用户C："我要统计每个城市有多少用户，城市按用户数排"
```

### 词汇提取

从用户描述中提取：
- 动词：**从**（FROM）、**找**（WHERE）、**看**（SELECT）、**排**（ORDER）、**统计**（GROUP BY）
- 名词：表名、字段名
- 修饰：**前N个**（LIMIT）、**倒排**（DESC）、**上周**（时间范围）

### QueryLang 语法设计

```
# QueryLang v1.0 语法（EBNF表示）
query      := from_clause where_clause? select_clause order_clause? limit_clause?
from_clause   := 'FROM' identifier
where_clause  := 'WHERE' condition ('AND' | 'OR' condition)*
condition     := field operator value
select_clause := 'SELECT' ('*' | field (',' field)*)
order_clause  := 'ORDER BY' field ('ASC' | 'DESC')?
limit_clause  := 'LIMIT' number

# 使用示例（原则验证）
FROM users                                # 最简用法（渐进式复杂度）
WHERE age > 18 AND status = "active"      # 自然语言风格（词汇对齐）
SELECT name, email                        # SELECT前有WHERE（一致性）
ORDER BY name                             # 默认ASC（最小惊讶）
LIMIT 10
```

### 原则验证清单

- [x] **词汇对齐**：FROM/WHERE/SELECT 是数据分析师熟悉的词
- [x] **最小惊讶**：`ORDER BY name` 默认升序，与SQL保持一致
- [x] **渐进复杂度**：最简查询只需 `FROM table SELECT *`
- [x] **清晰优于简洁**：关键字大写，语义清晰
- [x] **正交性**：WHERE/ORDER BY/LIMIT 可以任意组合或省略
- [ ] **错误处理**：待第18章实现
- [ ] **逃生舱口**：v1.0暂不支持，待需求驱动

---

## 4.10 反模式汇总

### 反模式一：关键字污染

```
# 差：过多关键字，用户需要记大量语法
FETCHDATA FROM TABLE users WHERECLAUSE age ISMORETHAN 18 SORTBY name ASCENDING
```

每个关键字都需要记忆成本，应最小化关键字数量。

### 反模式二：语法歧义

```
# 差：歧义语法
select name, email from users where age > 18 order name
# "order name"是 "ORDER BY name"吗？还是"order"是字段名？
```

### 反模式三：静默失败

```
# 差：错误被静默忽略
FROM users
WHERE ags > 18    # 字段名拼写错误，但系统不报错，返回空结果
SELECT name
```

用户不知道是没有数据，还是查询本身有问题。**DSL应该快速失败（fail fast）**。

### 反模式四：隐式全局状态

```
# 差：操作有隐式顺序依赖
SET DEFAULT TABLE users    # 设置全局默认表
WHERE age > 18             # 隐式依赖上面的SET
SELECT name
```

---

## 小结

| 原则 | 核心问题 |
|------|----------|
| 词汇对齐 | "领域专家能认出这些关键字吗？" |
| 最小惊讶 | "这个行为符合直觉吗？" |
| 渐进复杂度 | "简单用例有简单语法吗？" |
| 清晰优于简洁 | "陌生人能读懂吗？" |
| 有意义的错误 | "用户能从错误消息中自救吗？" |
| 正交性 | "特性能独立组合吗？" |
| 逃生舱口 | "无法表达的用例有出路吗？" |
| 版本演化 | "未来如何改变而不破坏现有代码？" |

---

## 思考题

1. Python的`f-string`（`f"Hello {name}"`）是一种DSL吗？它满足哪些设计原则，违反了哪些？
2. YAML的缩进敏感语法（vs JSON的括号语法）是最小惊讶原则的好例子还是坏例子？为什么？
3. 设计一个"工作日历DSL"：允许用户描述"每周一上午9点到10点"的重复事件。尝试应用本章的所有设计原则。

---

**上一章**：[DSL的分类学](./03-dsl-taxonomy.md)
**下一章**：[词法分析：把字符串变成Token](../part2-parsing/05-lexer-tokenizer.md)

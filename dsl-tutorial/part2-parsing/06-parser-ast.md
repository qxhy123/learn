# 第6章：语法分析与AST

## 核心思维模型

> 词法分析把字符流变成Token流，语法分析把Token流变成**树**。这棵树（AST）是程序语义的结构化表示——它不是线性字符序列，而是反映程序逻辑嵌套关系的层次结构。

---

## 6.1 从Token到结构

词法分析后，我们有一个扁平的Token流：

```
[FROM, "users", WHERE, "age", GT, 18, AND, "status", EQ, "active", SELECT, "name", "email"]
```

但这个流没有表达"AND连接了两个条件"，也没有表达"WHERE限定了FROM的结果"。我们需要把这个序列**解析为有结构的树**。

**目标结构（AST）**：

```
Query
├── from: TableRef("users")
├── where: BinaryOp(AND)
│   ├── left: Comparison(GT)
│   │   ├── field: Field("age")
│   │   └── value: Integer(18)
│   └── right: Comparison(EQ)
│       ├── field: Field("status")
│       └── value: String("active")
└── select: SelectList
    ├── Field("name")
    └── Field("email")
```

---

## 6.2 上下文无关文法（CFG）

语法分析器依据**文法（Grammar）**工作。文法定义了语言的合法结构。

**BNF（Backus-Naur Form）标记**：

```
<query>     ::= <from_clause> <where_clause>? <select_clause> <order_clause>? <limit_clause>?
<from_clause>   ::= "FROM" <identifier>
<where_clause>  ::= "WHERE" <condition>
<condition>     ::= <condition> "AND" <condition>
               |   <condition> "OR"  <condition>
               |   <comparison>
               |   "(" <condition> ")"
<comparison>    ::= <field> <operator> <value>
<operator>      ::= ">" | "<" | ">=" | "<=" | "=" | "!="
<field>         ::= <identifier>
<value>         ::= <integer> | <float> | <string> | <boolean>
<select_clause> ::= "SELECT" <select_list>
<select_list>   ::= "*" | <field> ("," <field>)*
<order_clause>  ::= "ORDER" "BY" <field> ("ASC" | "DESC")?
<limit_clause>  ::= "LIMIT" <integer>
```

**EBNF（Extended BNF）—— 更简洁的写法**：

```ebnf
query       = from_clause where_clause? select_clause order_clause? limit_clause?
from_clause = 'FROM' identifier
where_clause = 'WHERE' condition
condition   = and_expr ('OR' and_expr)*
and_expr    = comparison ('AND' comparison)*
comparison  = field operator value
              | '(' condition ')'
operator    = '>' | '<' | '>=' | '<=' | '=' | '!='
select_clause = 'SELECT' ('*' | field (',' field)*)
order_clause = 'ORDER' 'BY' field ('ASC' | 'DESC')?
limit_clause = 'LIMIT' INTEGER
```

注意 `condition` 的分层设计（`condition → and_expr → comparison`）：这是处理运算符**优先级**的标准方式——OR的优先级低于AND。

---

## 6.3 AST节点设计

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# ─── 基类 ───────────────────────────────────────────────

@dataclass
class ASTNode:
    """所有AST节点的基类"""
    line: int = field(default=0, compare=False, repr=False)
    column: int = field(default=0, compare=False, repr=False)
    
    def accept(self, visitor):
        """访问者模式接口（第15章代码生成会用到）"""
        method_name = f"visit_{type(self).__name__}"
        visitor_method = getattr(visitor, method_name, visitor.generic_visit)
        return visitor_method(self)

# ─── 查询节点 ────────────────────────────────────────────

@dataclass
class Query(ASTNode):
    from_clause: FromClause
    where_clause: Optional[WhereClause] = None
    select_clause: Optional[SelectClause] = None
    order_clause: Optional[OrderClause] = None
    limit_clause: Optional[LimitClause] = None

@dataclass
class FromClause(ASTNode):
    table: str  # 表名

@dataclass
class WhereClause(ASTNode):
    condition: Condition  # 条件表达式

@dataclass
class SelectClause(ASTNode):
    star: bool = False              # SELECT *
    fields: list[str] = field(default_factory=list)  # SELECT f1, f2

@dataclass
class OrderClause(ASTNode):
    field: str
    direction: str = "ASC"  # "ASC" or "DESC"

@dataclass
class LimitClause(ASTNode):
    count: int

# ─── 条件/表达式节点 ──────────────────────────────────────

Condition = "BinaryOp | Comparison"  # 类型别名

@dataclass
class BinaryOp(ASTNode):
    """AND / OR 二元逻辑运算"""
    operator: str   # "AND" or "OR"
    left: ASTNode
    right: ASTNode

@dataclass
class Comparison(ASTNode):
    """字段比较：field op value"""
    field: str
    operator: str   # ">", "<", ">=", "<=", "=", "!="
    value: ASTNode

# ─── 值节点 ──────────────────────────────────────────────

@dataclass
class IntegerLiteral(ASTNode):
    value: int

@dataclass
class FloatLiteral(ASTNode):
    value: float

@dataclass
class StringLiteral(ASTNode):
    value: str

@dataclass
class BooleanLiteral(ASTNode):
    value: bool

@dataclass
class FieldRef(ASTNode):
    name: str
```

---

## 6.4 AST的可视化

为了调试，实现一个AST打印器：

```python
class ASTPrinter:
    """将AST打印为缩进树形格式"""
    
    def print(self, node: ASTNode, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = []
        
        match node:
            case Query():
                lines.append(f"{prefix}Query")
                lines.append(self.print(node.from_clause, indent + 1))
                if node.where_clause:
                    lines.append(self.print(node.where_clause, indent + 1))
                if node.select_clause:
                    lines.append(self.print(node.select_clause, indent + 1))
                if node.order_clause:
                    lines.append(self.print(node.order_clause, indent + 1))
                if node.limit_clause:
                    lines.append(self.print(node.limit_clause, indent + 1))
            
            case FromClause(table=t):
                lines.append(f"{prefix}FROM {t}")
            
            case WhereClause(condition=c):
                lines.append(f"{prefix}WHERE")
                lines.append(self.print(c, indent + 1))
            
            case BinaryOp(operator=op, left=l, right=r):
                lines.append(f"{prefix}BinaryOp({op})")
                lines.append(self.print(l, indent + 1))
                lines.append(self.print(r, indent + 1))
            
            case Comparison(field=f, operator=op, value=v):
                lines.append(f"{prefix}Compare({f} {op})")
                lines.append(self.print(v, indent + 2))
            
            case IntegerLiteral(value=v):
                lines.append(f"{prefix}Int({v})")
            
            case StringLiteral(value=v):
                lines.append(f"{prefix}Str({v!r})")
            
            case SelectClause(star=True):
                lines.append(f"{prefix}SELECT *")
            
            case SelectClause(fields=fs):
                lines.append(f"{prefix}SELECT {', '.join(fs)}")
            
            case _:
                lines.append(f"{prefix}{type(node).__name__}")
        
        return '\n'.join(lines)


# 测试
printer = ASTPrinter()
query = Query(
    from_clause=FromClause("users"),
    where_clause=WhereClause(
        BinaryOp("AND",
            Comparison("age", ">", IntegerLiteral(18)),
            Comparison("status", "=", StringLiteral("active"))
        )
    ),
    select_clause=SelectClause(fields=["name", "email"])
)

print(printer.print(query))
# Query
#   FROM users
#   WHERE
#     BinaryOp(AND)
#       Compare(age >)
#           Int(18)
#       Compare(status =)
#           Str('active')
#   SELECT name, email
```

---

## 6.5 解析树 vs 抽象语法树

初学者常混淆这两个概念：

**解析树（Parse Tree / Concrete Syntax Tree）**：
- 忠实反映文法的每一个产生式
- 包含所有语法标记（括号、分隔符、关键字）
- 通常很"胖"，节点多

**抽象语法树（Abstract Syntax Tree）**：
- 省略纯语法标记（如括号、逗号）
- 只保留**语义相关**的结构
- 更紧凑，是后续处理的标准输入

```
输入：age > 18 AND status = "active"

解析树：
condition
├── and_expr
│   ├── comparison
│   │   ├── field: "age"
│   │   ├── operator: ">"
│   │   └── value: integer("18")
│   ├── token: "AND"       ← 纯语法标记，AST中省略
│   └── comparison
│       ├── field: "status"
│       ├── operator: "="
│       └── value: string("active")

AST：
BinaryOp(AND)
├── Comparison(age, >, 18)
└── Comparison(status, =, "active")
```

---

## 6.6 运算符优先级与结合性

以下这个查询有歧义：

```
WHERE age > 18 OR age < 10 AND status = "active"
```

有两种解析方式：
```
# 方式A：AND优先级高于OR
(age > 18) OR ((age < 10) AND (status = "active"))

# 方式B：OR优先级高于AND
((age > 18) OR (age < 10)) AND (status = "active")
```

在SQL（和我们的QueryLang）中，AND优先级高于OR。这通过**文法分层**来实现：

```ebnf
condition  = or_expr
or_expr    = and_expr ('OR' and_expr)*    # OR优先级最低
and_expr   = not_expr ('AND' not_expr)*   # AND次之
not_expr   = 'NOT' not_expr | comparison  # NOT再次
comparison = field operator value
           | '(' condition ')'            # 括号允许显式覆盖优先级
```

文法中越靠近叶子节点，优先级越高。这是**运算符优先级解析**的标准技巧。

---

## 6.7 本章实战：手绘AST

对以下查询，画出完整AST结构：

```sql
FROM orders
WHERE amount > 100 AND (status = "paid" OR status = "pending")
SELECT id, amount, status
ORDER BY amount DESC
LIMIT 5
```

**答案参考**：
```
Query
├── FROM: "orders"
├── WHERE:
│   BinaryOp(AND)
│   ├── Comparison(amount, >, Int(100))
│   └── BinaryOp(OR)
│       ├── Comparison(status, =, Str("paid"))
│       └── Comparison(status, =, Str("pending"))
├── SELECT: [id, amount, status]
├── ORDER BY: amount DESC
└── LIMIT: 5
```

---

## 小结

| 概念 | 要点 |
|------|------|
| 文法（Grammar） | 用BNF/EBNF定义语言的合法结构 |
| 解析树 | 忠实记录所有语法符号，通常过于详细 |
| AST | 省略纯语法符号，保留语义结构，是后续处理的标准形式 |
| 运算符优先级 | 通过文法分层实现，越靠近叶子优先级越高 |
| 访问者模式 | AST遍历的标准模式，第15章代码生成会深入使用 |

---

**上一章**：[词法分析](./05-lexer-tokenizer.md)
**下一章**：[递归下降解析器](./07-recursive-descent.md)

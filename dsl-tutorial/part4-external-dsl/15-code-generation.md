# 第15章：代码生成与目标IR

## 核心思维模型

> 代码生成是"把树变成文本"的过程——访问AST的每个节点，根据目标语言的语法规则生成对应代码。掌握**访问者模式（Visitor Pattern）**是代码生成的核心技术。

---

## 15.1 为什么需要代码生成？

解释器（第16章）是最简单的执行方式，但代码生成提供了更多可能：

1. **SQL生成**：把QueryLang查询转换为真实SQL，在数据库上执行（利用数据库的索引和优化）
2. **Python代码生成**：把DSL转换为Python代码，可以保存/复用
3. **字节码生成**：直接生成Python字节码，获得VM级别的性能
4. **多目标输出**：同一个DSL可以生成PostgreSQL、MySQL、SQLite等不同方言

---

## 15.2 访问者模式（Visitor Pattern）

**问题**：如何在不修改AST节点类的情况下，添加新的操作（如SQL生成、类型检查、优化）？

**解决方案**：访问者模式——把操作从数据结构中分离出来。

```python
from abc import ABC, abstractmethod
from querylang.ast_nodes import *

class ASTVisitor(ABC):
    """AST访问者基类"""
    
    def visit(self, node: ASTNode):
        """分发到具体的visit方法"""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node: ASTNode):
        """默认访问行为（可以遍历子节点）"""
        raise NotImplementedError(
            f"{type(self).__name__} 未实现 visit_{type(node).__name__}"
        )
```

---

## 15.3 SQL代码生成器

### 15.3.1 基础SQL生成

```python
class SQLGenerator(ASTVisitor):
    """
    将QueryLang AST生成标准SQL
    
    目标：生成ANSI SQL，可以在PostgreSQL/MySQL/SQLite上运行
    """
    
    def __init__(self, dialect: str = "ansi"):
        self.dialect = dialect
        self._indent = 0
    
    def generate(self, query: Query) -> str:
        return self.visit(query)
    
    # ─── 查询节点 ──────────────────────────────────────────
    
    def visit_Query(self, node: Query) -> str:
        parts = []
        
        # SELECT子句（SQL中SELECT在前，FROM在后）
        if node.select_clause:
            parts.append(self.visit(node.select_clause))
        else:
            parts.append("SELECT *")
        
        # FROM子句
        parts.append(self.visit(node.from_clause))
        
        # WHERE子句
        if node.where_clause:
            parts.append(self.visit(node.where_clause))
        
        # ORDER BY子句
        if node.order_clause:
            parts.append(self.visit(node.order_clause))
        
        # LIMIT子句
        if node.limit_clause:
            parts.append(self.visit(node.limit_clause))
        
        return "\n".join(parts)
    
    def visit_FromClause(self, node: FromClause) -> str:
        return f"FROM {self._quote_identifier(node.table)}"
    
    def visit_WhereClause(self, node: WhereClause) -> str:
        condition_sql = self.visit(node.condition)
        return f"WHERE {condition_sql}"
    
    def visit_SelectClause(self, node: SelectClause) -> str:
        if node.star:
            return "SELECT *"
        fields = ", ".join(self._quote_identifier(f) for f in node.fields)
        return f"SELECT {fields}"
    
    def visit_OrderClause(self, node: OrderClause) -> str:
        return f"ORDER BY {self._quote_identifier(node.field)} {node.direction}"
    
    def visit_LimitClause(self, node: LimitClause) -> str:
        return f"LIMIT {node.count}"
    
    # ─── 条件节点 ──────────────────────────────────────────
    
    def visit_BinaryOp(self, node: BinaryOp) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        # AND/OR需要括号处理优先级
        if node.operator == "OR":
            if isinstance(node.left, BinaryOp) and node.left.operator == "AND":
                left = f"({left})"
            if isinstance(node.right, BinaryOp) and node.right.operator == "AND":
                right = f"({right})"
        
        return f"{left} {node.operator} {right}"
    
    def visit_Comparison(self, node: Comparison) -> str:
        field = self._quote_identifier(node.field)
        value = self.visit(node.value)
        return f"{field} {node.operator} {value}"
    
    # ─── 值节点 ────────────────────────────────────────────
    
    def visit_IntegerLiteral(self, node: IntegerLiteral) -> str:
        return str(node.value)
    
    def visit_FloatLiteral(self, node: FloatLiteral) -> str:
        return str(node.value)
    
    def visit_StringLiteral(self, node: StringLiteral) -> str:
        # 转义单引号
        escaped = node.value.replace("'", "''")
        return f"'{escaped}'"
    
    def visit_BooleanLiteral(self, node: BooleanLiteral) -> str:
        if self.dialect == "mysql":
            return "1" if node.value else "0"
        return "TRUE" if node.value else "FALSE"
    
    # ─── 辅助方法 ──────────────────────────────────────────
    
    def _quote_identifier(self, name: str) -> str:
        """根据方言选择标识符引号"""
        if self.dialect in ("postgresql", "ansi"):
            return f'"{name}"'
        elif self.dialect == "mysql":
            return f"`{name}`"
        elif self.dialect == "sqlite":
            return f'"{name}"'
        return name
```

### 15.3.2 方言特化

```python
class PostgreSQLGenerator(SQLGenerator):
    """PostgreSQL特定SQL生成"""
    
    def __init__(self):
        super().__init__(dialect="postgresql")
    
    def visit_LimitClause(self, node: LimitClause) -> str:
        # PostgreSQL支持LIMIT/OFFSET
        return f"LIMIT {node.count}"
    
    def visit_BooleanLiteral(self, node: BooleanLiteral) -> str:
        return "TRUE" if node.value else "FALSE"


class MySQLGenerator(SQLGenerator):
    """MySQL特定SQL生成"""
    
    def __init__(self):
        super().__init__(dialect="mysql")
    
    def visit_StringLiteral(self, node: StringLiteral) -> str:
        # MySQL用双反斜杠转义
        escaped = node.value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
```

---

## 15.4 Python代码生成器

把QueryLang查询转换为等价的Python代码（用于导出/保存可复用的查询）：

```python
class PythonGenerator(ASTVisitor):
    """
    将QueryLang AST生成等价的Python代码
    生成使用标准Python列表推导式的代码
    """
    
    def generate(self, query: Query) -> str:
        lines = [
            "# 由 QueryLang 自动生成",
            "def run_query(tables: dict) -> list:",
            f"    rows = list(tables[{self.visit(query.from_clause)}])",
        ]
        
        if query.where_clause:
            condition = self.visit_condition(query.where_clause.condition)
            lines.append(f"    rows = [row for row in rows if {condition}]")
        
        if query.order_clause:
            order = query.order_clause
            reverse = order.direction == "DESC"
            lines.append(
                f"    rows = sorted(rows, "
                f"key=lambda r: r.get({order.field!r}), "
                f"reverse={reverse})"
            )
        
        if query.limit_clause:
            lines.append(f"    rows = rows[:{query.limit_clause.count}]")
        
        if query.select_clause and not query.select_clause.star:
            fields_str = str(query.select_clause.fields)
            lines.append(
                f"    rows = [{{{', '.join(repr(f) + ': row.get(' + repr(f) + ')' for f in query.select_clause.fields)}}} for row in rows]"
            )
        
        lines.append("    return rows")
        return "\n".join(lines)
    
    def visit_FromClause(self, node: FromClause) -> str:
        return repr(node.table)
    
    def visit_condition(self, condition) -> str:
        match condition:
            case BinaryOp(operator="AND", left=l, right=r):
                return f"({self.visit_condition(l)}) and ({self.visit_condition(r)})"
            case BinaryOp(operator="OR", left=l, right=r):
                return f"({self.visit_condition(l)}) or ({self.visit_condition(r)})"
            case Comparison(field=f, operator=op, value=v):
                field_expr = f"row.get({f!r})"
                value_expr = self.visit_value(v)
                py_op = {"=": "==", "!=": "!="}.get(op, op)
                return f"{field_expr} {py_op} {value_expr}"
    
    def visit_value(self, node) -> str:
        match node:
            case IntegerLiteral(value=v): return str(v)
            case FloatLiteral(value=v): return str(v)
            case StringLiteral(value=v): return repr(v)
            case BooleanLiteral(value=v): return str(v)
```

---

## 15.5 中间表示（IR）

复杂的编译器在AST和目标代码之间引入一层**中间表示（IR, Intermediate Representation）**：

```
源码 → AST → 高层IR → 优化 → 低层IR → 目标代码
```

对于QueryLang，可以定义一个关系代数IR：

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RelationalOp:
    """关系代数操作基类"""
    pass

@dataclass
class Scan(RelationalOp):
    """全表扫描"""
    table: str
    
@dataclass
class Filter(RelationalOp):
    """过滤（σ）"""
    source: RelationalOp
    predicate: object  # 条件AST节点
    
@dataclass
class Project(RelationalOp):
    """投影（π）"""
    source: RelationalOp
    fields: list[str] | None  # None表示SELECT *
    
@dataclass
class Sort(RelationalOp):
    """排序（τ）"""
    source: RelationalOp
    key: str
    ascending: bool = True

@dataclass
class Limit(RelationalOp):
    """限制行数"""
    source: RelationalOp
    count: int

class ASTToIR:
    """将QueryLang AST转换为关系代数IR"""
    
    def convert(self, query: Query) -> RelationalOp:
        # 从底层往上构建
        plan: RelationalOp = Scan(query.from_clause.table)
        
        if query.where_clause:
            plan = Filter(plan, query.where_clause.condition)
        
        if query.order_clause:
            plan = Sort(
                plan,
                query.order_clause.field,
                ascending=(query.order_clause.direction == "ASC")
            )
        
        if query.limit_clause:
            plan = Limit(plan, query.limit_clause.count)
        
        if query.select_clause and not query.select_clause.star:
            plan = Project(plan, query.select_clause.fields)
        
        return plan

class IROptimizer:
    """关系代数优化（谓词下推等）"""
    
    def optimize(self, plan: RelationalOp) -> RelationalOp:
        # 谓词下推：把Filter尽量往Scan方向移
        return self._push_down_filter(plan)
    
    def _push_down_filter(self, plan: RelationalOp) -> RelationalOp:
        match plan:
            case Project(source=Sort(source=Filter(source=s, predicate=p), key=k, ascending=a), fields=fs):
                # Project → Sort → Filter → Scan
                # 优化为 Project → Sort → Scan（Filter已经在Scan层）
                # 实际上Filter已经在正确位置，不需要移动
                return plan
            
            case Sort(source=Limit(source=inner), key=k, ascending=a):
                # 不能交换Sort和Limit的顺序！这是一个约束
                return plan
            
            case _:
                return plan
```

---

## 15.6 代码生成测试

```python
def test_sql_generation():
    source = '''FROM users
WHERE age > 18 AND status = "active"
SELECT name, email
ORDER BY name
LIMIT 10'''
    
    from querylang.lexer import Lexer
    from querylang.parser import Parser
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    
    # ANSI SQL
    sql = SQLGenerator().generate(ast)
    assert 'SELECT "name", "email"' in sql
    assert 'FROM "users"' in sql
    assert 'WHERE "age" > 18 AND "status" = \'active\'' in sql
    assert 'ORDER BY "name" ASC' in sql
    assert 'LIMIT 10' in sql

def test_mysql_sql_generation():
    source = 'FROM users WHERE active = true SELECT name'
    
    from querylang.lexer import Lexer
    from querylang.parser import Parser
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    
    sql = MySQLGenerator().generate(ast)
    assert '`users`' in sql   # MySQL反引号
    assert '1' in sql          # MySQL用1表示TRUE

def test_python_code_generation():
    source = 'FROM users WHERE age > 18 SELECT name'
    
    from querylang.lexer import Lexer
    from querylang.parser import Parser
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    
    code = PythonGenerator().generate(ast)
    assert "def run_query" in code
    assert "tables['users']" in code
    assert "row.get('age')" in code
```

---

## 小结

| 概念 | 要点 |
|------|------|
| 访问者模式 | 把操作从AST分离，支持多种代码生成目标 |
| SQL生成 | 注意标识符引号、字符串转义、方言差异 |
| 多目标生成 | 同一AST，不同Visitor生成不同目标代码 |
| IR | 解耦前端（解析）和后端（生成），支持优化pass |
| 谓词下推 | 关系代数优化：把过滤尽早执行 |

---

**上一章**：[语义分析与符号表](./14-semantic-analysis.md)
**下一章**：[解释器模式执行DSL](./16-interpreter-pattern.md)

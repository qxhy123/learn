# 第14章：语义分析与符号表

## 核心思维模型

> 语法分析器确保代码"结构合法"，语义分析器确保代码"含义合理"。语法分析是**形式层面**的检查，语义分析是**内容层面**的检查。这两者的区别，就像"句子语法正确但意思荒谬"。

---

## 14.1 语法正确但语义错误的例子

```
# 语法完全合法，但语义错误：
FROM nonexistent_table       # 表不存在
WHERE age > "not_a_number"   # 类型不匹配
SELECT xyz, abc              # 字段不存在
ORDER BY unknown_field       # 排序字段不存在
```

语法分析器对这些代码"毫无异议"，因为它们的结构完全符合文法。但执行时会出错。

**语义分析的任务**：在执行前，静态地检查这些错误。

---

## 14.2 符号表（Symbol Table）

符号表是语义分析的核心数据结构，记录**所有已知名字（标识符）的信息**：

```python
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum, auto

class SymbolKind(Enum):
    TABLE   = auto()
    COLUMN  = auto()
    ALIAS   = auto()
    FUNCTION = auto()

@dataclass
class Symbol:
    name: str
    kind: SymbolKind
    type: str = "unknown"   # 数据类型：int, float, string, bool
    nullable: bool = True   # 是否可为NULL
    metadata: dict = field(default_factory=dict)
    
    def __repr__(self):
        return f"Symbol({self.name!r}, {self.kind.name}, type={self.type})"

class SymbolTable:
    """
    符号表：支持嵌套作用域
    
    作用域链：内层可以看到外层，但外层看不到内层
    """
    
    def __init__(self, parent: Optional['SymbolTable'] = None, scope_name: str = "global"):
        self._symbols: dict[str, Symbol] = {}
        self._parent = parent
        self._scope_name = scope_name
        self._children: list['SymbolTable'] = []
    
    def define(self, symbol: Symbol) -> None:
        """在当前作用域定义符号"""
        if symbol.name in self._symbols:
            raise SemanticError(
                f"符号 '{symbol.name}' 在作用域 '{self._scope_name}' 中已定义"
            )
        self._symbols[symbol.name] = symbol
    
    def define_or_update(self, symbol: Symbol) -> None:
        """定义或更新符号（用于允许覆盖的场景）"""
        self._symbols[symbol.name] = symbol
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """查找符号，沿作用域链向上查找"""
        if name in self._symbols:
            return self._symbols[name]
        if self._parent:
            return self._parent.lookup(name)
        return None
    
    def lookup_local(self, name: str) -> Optional[Symbol]:
        """只在当前作用域查找，不向上"""
        return self._symbols.get(name)
    
    def create_child(self, scope_name: str) -> 'SymbolTable':
        """创建子作用域"""
        child = SymbolTable(parent=self, scope_name=scope_name)
        self._children.append(child)
        return child
    
    def all_symbols(self) -> list[Symbol]:
        """列出当前作用域所有符号"""
        return list(self._symbols.values())
    
    def __repr__(self):
        return f"SymbolTable({self._scope_name}, {list(self._symbols.keys())})"


class SemanticError(Exception):
    def __init__(self, message: str, line: int = None, column: int = None):
        super().__init__(message)
        self.line = line
        self.column = column
```

---

## 14.3 QueryLang 语义分析器

```python
from querylang.ast_nodes import *

class SemanticAnalyzer:
    """
    QueryLang语义分析器
    
    职责：
    1. 验证表名存在
    2. 验证字段名存在于对应表
    3. 验证条件中的类型兼容性
    4. 推断表达式类型
    5. 建立符号表供后续分析使用
    """
    
    def __init__(self, schema: dict[str, dict[str, str]] = None,
                 available_tables: list[str] = None):
        """
        Args:
            schema: 表结构 {"table_name": {"field": "type", ...}}
            available_tables: 仅提供表名（无类型信息时使用）
        """
        self.schema = schema or {}
        self._available_tables = set(available_tables or []) | set(schema.keys())
        self.errors: list[SemanticError] = []
        self.symbol_table = SymbolTable(scope_name="global")
        self._current_table: str | None = None
    
    def analyze(self, ast: Query) -> SymbolTable:
        """
        分析整个查询，返回符号表
        如果发现语义错误，收集所有错误后统一抛出
        """
        self._analyze_query(ast)
        
        if self.errors:
            messages = "\n".join(f"  - {e}" for e in self.errors)
            raise SemanticError(f"语义分析发现 {len(self.errors)} 个错误:\n{messages}")
        
        return self.symbol_table
    
    # ─── 查询分析 ──────────────────────────────────────────
    
    def _analyze_query(self, query: Query):
        # 必须先分析FROM，因为后续分析需要知道当前表
        self._analyze_from(query.from_clause)
        
        if query.where_clause:
            self._analyze_where(query.where_clause)
        
        if query.select_clause:
            self._analyze_select(query.select_clause)
        
        if query.order_clause:
            self._analyze_order(query.order_clause)
        
        if query.limit_clause:
            self._analyze_limit(query.limit_clause)
    
    def _analyze_from(self, from_clause: FromClause):
        table_name = from_clause.table
        
        if table_name not in self._available_tables:
            # 找相似表名（fuzzy match）
            similar = self._find_similar(table_name, self._available_tables)
            hint = f"，是否要用: {similar!r}" if similar else ""
            self._error(
                f"表 '{table_name}' 不存在{hint}",
                from_clause.line, from_clause.column
            )
            return
        
        self._current_table = table_name
        
        # 如果有完整schema，把字段注册到符号表
        if table_name in self.schema:
            table_scope = self.symbol_table.create_child(f"table:{table_name}")
            for field_name, field_type in self.schema[table_name].items():
                table_scope.define(Symbol(
                    name=field_name,
                    kind=SymbolKind.COLUMN,
                    type=field_type,
                    nullable=True,
                ))
            self.symbol_table.define(Symbol(
                name=table_name,
                kind=SymbolKind.TABLE,
                metadata={"scope": table_scope},
            ))
    
    def _analyze_where(self, where_clause: WhereClause):
        self._analyze_condition(where_clause.condition)
    
    def _analyze_condition(self, condition):
        match condition:
            case BinaryOp(operator="AND" | "OR", left=l, right=r):
                self._analyze_condition(l)
                self._analyze_condition(r)
            
            case Comparison(field=field, operator=op, value=value_node, line=line, column=col):
                # 检查字段是否存在（如果有schema）
                if self._current_table and self._current_table in self.schema:
                    table_fields = self.schema[self._current_table]
                    if field not in table_fields:
                        similar = self._find_similar(field, table_fields.keys())
                        hint = f"，是否要用: {similar!r}" if similar else ""
                        self._error(
                            f"字段 '{field}' 在表 '{self._current_table}' 中不存在{hint}",
                            line, col
                        )
                        return
                    
                    # 类型检查
                    field_type = table_fields[field]
                    self._check_type_compatibility(field, field_type, op, value_node, line)
            
            case _:
                pass  # 未知类型，忽略（允许扩展）
    
    def _check_type_compatibility(self, field, field_type, op, value_node, line):
        """检查比较运算的类型兼容性"""
        value_type = self._infer_value_type(value_node)
        
        # 类型兼容性规则
        compatible = {
            ("int", "int"): True,
            ("int", "float"): True,
            ("float", "int"): True,
            ("float", "float"): True,
            ("string", "string"): True,
            ("bool", "bool"): True,
        }
        
        # 跨类型比较（例如数字和字符串比较）
        if not compatible.get((field_type, value_type), False):
            # 警告级别，不阻止执行
            self._warning(
                f"字段 '{field}'（类型: {field_type}）与值（类型: {value_type}）类型不匹配，"
                f"可能导致意外结果",
                line
            )
        
        # 特定运算符不适用于某些类型
        numeric_only_ops = {"<", ">", "<=", ">="}
        if op in numeric_only_ops and field_type == "string":
            self._error(
                f"运算符 '{op}' 不适用于字符串字段 '{field}'，"
                f"请使用 '=' 或 '!='",
                line
            )
    
    def _analyze_select(self, select_clause: SelectClause):
        if select_clause.star:
            return  # SELECT * 总是合法的
        
        # 检查选择的字段是否存在
        if self._current_table and self._current_table in self.schema:
            table_fields = self.schema[self._current_table]
            for field in select_clause.fields:
                if field not in table_fields:
                    similar = self._find_similar(field, table_fields.keys())
                    hint = f"，是否要用: {similar!r}" if similar else ""
                    self._error(
                        f"SELECT中的字段 '{field}' 在表 '{self._current_table}' 中不存在{hint}",
                        select_clause.line
                    )
    
    def _analyze_order(self, order_clause: OrderClause):
        if self._current_table and self._current_table in self.schema:
            table_fields = self.schema[self._current_table]
            if order_clause.field not in table_fields:
                similar = self._find_similar(order_clause.field, table_fields.keys())
                hint = f"，是否要用: {similar!r}" if similar else ""
                self._error(
                    f"ORDER BY字段 '{order_clause.field}' 不存在{hint}",
                    order_clause.line
                )
    
    def _analyze_limit(self, limit_clause: LimitClause):
        if limit_clause.count <= 0:
            self._error(
                f"LIMIT必须大于0，得到: {limit_clause.count}",
                limit_clause.line
            )
    
    # ─── 辅助方法 ──────────────────────────────────────────
    
    def _infer_value_type(self, node) -> str:
        match node:
            case IntegerLiteral(): return "int"
            case FloatLiteral(): return "float"
            case StringLiteral(): return "string"
            case BooleanLiteral(): return "bool"
            case _: return "unknown"
    
    def _find_similar(self, name: str, candidates) -> Optional[str]:
        """简单的字符串相似性搜索（Levenshtein距离）"""
        def levenshtein(a, b):
            if len(a) < len(b): a, b = b, a
            if not b: return len(a)
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a):
                curr = [i + 1]
                for j, cb in enumerate(b):
                    curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca!=cb)))
                prev = curr
            return prev[-1]
        
        best = min(candidates, key=lambda c: levenshtein(name.lower(), c.lower()), default=None)
        if best and levenshtein(name.lower(), best.lower()) <= 3:
            return best
        return None
    
    def _error(self, message: str, line=None, column=None):
        self.errors.append(SemanticError(message, line, column))
    
    def _warning(self, message: str, line=None):
        # 可以记录警告而不终止分析
        print(f"[警告] 第{line}行: {message}" if line else f"[警告] {message}")
```

---

## 14.4 Schema推断（无显式Schema时）

当没有预定义Schema时，可以从数据中推断：

```python
class SchemaInferrer:
    """从数据推断表结构"""
    
    def infer(self, tables: dict[str, list[dict]]) -> dict[str, dict[str, str]]:
        schema = {}
        for table_name, rows in tables.items():
            if not rows:
                schema[table_name] = {}
                continue
            
            # 合并所有行的字段类型
            field_types = {}
            for row in rows:
                for field, value in row.items():
                    inferred_type = self._infer_type(value)
                    if field not in field_types:
                        field_types[field] = inferred_type
                    elif field_types[field] != inferred_type:
                        # 类型冲突 → 降级为string
                        field_types[field] = "string"
            
            schema[table_name] = field_types
        
        return schema
    
    def _infer_type(self, value) -> str:
        if isinstance(value, bool): return "bool"
        if isinstance(value, int): return "int"
        if isinstance(value, float): return "float"
        if isinstance(value, str): return "string"
        return "unknown"


# 使用
tables = {"users": [{"name": "Alice", "age": 25}]}
schema = SchemaInferrer().infer(tables)
# {"users": {"name": "string", "age": "int"}}
```

---

## 14.5 语义分析测试

```python
import pytest

SCHEMA = {
    "users": {
        "id": "int",
        "name": "string",
        "age": "int",
        "status": "string",
        "email": "string",
    }
}

class TestSemanticAnalyzer:
    
    def _analyze(self, source):
        from querylang.lexer import Lexer
        from querylang.parser import Parser
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        analyzer = SemanticAnalyzer(schema=SCHEMA)
        analyzer.analyze(ast)
        return analyzer
    
    def test_valid_query(self):
        """完全合法的查询不应报错"""
        self._analyze('FROM users WHERE age > 18 SELECT name, email')
    
    def test_unknown_table(self):
        with pytest.raises(SemanticError, match="不存在"):
            self._analyze('FROM nonexistent SELECT *')
    
    def test_unknown_field_in_where(self):
        with pytest.raises(SemanticError, match="不存在"):
            self._analyze('FROM users WHERE nme > 18 SELECT name')
    
    def test_typo_suggestion(self):
        """字段名拼写错误时应给出建议"""
        try:
            self._analyze('FROM users WHERE nme = "Alice" SELECT *')
            assert False, "应该报错"
        except SemanticError as e:
            assert "name" in str(e)  # 建议正确字段名
    
    def test_unknown_field_in_select(self):
        with pytest.raises(SemanticError, match="不存在"):
            self._analyze('FROM users SELECT nonexistent_field')
    
    def test_limit_must_be_positive(self):
        with pytest.raises(SemanticError, match="大于0"):
            self._analyze('FROM users SELECT * LIMIT 0')
    
    def test_type_incompatibility_warning(self):
        """类型不匹配应给出警告（不报错）"""
        # 数字字段与字符串比较：警告但不阻止
        import warnings
        # 这里仅验证不抛出错误
        self._analyze('FROM users WHERE age > "18" SELECT name')
```

---

## 小结

| 概念 | 要点 |
|------|------|
| 语义分析 vs 语法分析 | 语法检查结构，语义检查含义 |
| 符号表 | 记录标识符的类型、作用域、元数据 |
| 错误收集 | 不要在第一个错误处停止，收集所有错误后统一报告 |
| Schema推断 | 从数据中自动推断类型，降低使用门槛 |
| 相似名建议 | Levenshtein距离找相似名，显著改善用户体验 |

---

**上一章**：[外部DSL全流程](./13-external-dsl-pipeline.md)
**下一章**：[代码生成与目标IR](./15-code-generation.md)

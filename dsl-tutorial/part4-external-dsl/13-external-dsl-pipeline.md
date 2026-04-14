# 第13章：外部DSL全流程

## 核心思维模型

> 构建外部DSL是一个**编译器工程**问题。整个流程是一条流水线：源码字符串流经每个阶段，逐步从"人类可读的文本"转变为"机器可执行的指令"。理解这条流水线，你就掌握了所有编程语言的骨架。

---

## 13.1 完整编译管道

```
源码文本（字符串）
    │
    ▼ 词法分析（Lexing）         ← 第5章
Token流
    │
    ▼ 语法分析（Parsing）        ← 第6-7章
抽象语法树（AST）
    │
    ▼ 语义分析（Semantic Analysis）  ← 第14章
带注解的AST（类型、作用域）
    │
    ▼ 执行/代码生成               ← 第15-16章
  ┌─┴─┐
  │   │
  ▼   ▼
解释 代码生成
执行   │
      Python/SQL/字节码
```

本章把前面所有章节学到的技术整合成一个完整的、可运行的 **QueryLang 解释器**。

---

## 13.2 完整项目结构

```
querylang/
├── __init__.py
├── lexer.py          # 词法分析器（第5章）
├── ast_nodes.py      # AST节点定义（第6章）
├── parser.py         # 递归下降解析器（第7章）
├── semantic.py       # 语义分析（第14章）
├── interpreter.py    # 树遍历解释器（第16章）
├── codegen.py        # SQL代码生成（第15章）
└── errors.py         # 统一错误处理（第18章）
```

---

## 13.3 整合：端到端 QueryLang 引擎

把前面各章的代码整合为一个完整的引擎：

```python
# querylang/engine.py
from dataclasses import dataclass
from typing import Any

@dataclass
class QueryResult:
    """查询执行结果"""
    rows: list[dict]
    total_rows: int
    query_time_ms: float

class QueryLangEngine:
    """
    完整的QueryLang执行引擎
    
    流程：源码 → 词法 → 语法 → 语义 → 执行
    """
    
    def __init__(self, data_source=None):
        self.data_source = data_source or {}
        self._lexer = None  # 延迟初始化
    
    def execute(self, source: str, tables: dict[str, list[dict]] = None) -> QueryResult:
        """
        执行QueryLang查询
        
        Args:
            source: QueryLang源码字符串
            tables: 数据字典 {"table_name": [{"field": value, ...}, ...]}
        
        Returns:
            QueryResult 包含查询结果
        """
        import time
        start_time = time.perf_counter()
        
        data = tables or self.data_source
        
        # ─── 阶段1：词法分析 ───────────────────────────────
        try:
            from querylang.lexer import Lexer
            lexer = Lexer(source)
            tokens = lexer.tokenize()
        except LexError as e:
            raise QueryLangError(
                f"词法错误：{e}",
                phase="lexical",
                source=source,
                line=getattr(e, 'line', None),
                column=getattr(e, 'column', None),
            )
        
        # ─── 阶段2：语法分析 ───────────────────────────────
        try:
            from querylang.parser import Parser
            parser = Parser(tokens)
            ast = parser.parse()
        except ParseError as e:
            raise QueryLangError(
                f"语法错误：{e}",
                phase="syntactic",
                source=source,
                line=getattr(e, 'line', None),
                column=getattr(e, 'column', None),
            )
        
        # ─── 阶段3：语义分析 ───────────────────────────────
        try:
            from querylang.semantic import SemanticAnalyzer
            analyzer = SemanticAnalyzer(available_tables=list(data.keys()))
            analyzer.analyze(ast)
        except SemanticError as e:
            raise QueryLangError(
                f"语义错误：{e}",
                phase="semantic",
                source=source,
                line=getattr(e, 'line', None),
            )
        
        # ─── 阶段4：执行 ─────────────────────────────────
        from querylang.interpreter import Interpreter
        interpreter = Interpreter(data)
        rows = interpreter.execute(ast)
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        return QueryResult(
            rows=rows,
            total_rows=len(rows),
            query_time_ms=elapsed_ms,
        )
    
    def explain(self, source: str) -> str:
        """解析并显示AST（调试用）"""
        from querylang.lexer import Lexer
        from querylang.parser import Parser
        from querylang.ast_nodes import ASTPrinter
        
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        return ASTPrinter().print(ast)


class QueryLangError(Exception):
    """统一的错误类型，包含阶段信息"""
    
    def __init__(self, message, phase=None, source=None, line=None, column=None):
        super().__init__(message)
        self.phase = phase
        self.source = source
        self.line = line
        self.column = column
    
    def format_with_context(self) -> str:
        """生成带上下文的错误消息"""
        lines = [str(self)]
        
        if self.source and self.line:
            source_lines = self.source.split('\n')
            if self.line <= len(source_lines):
                lines.append(f"\n  第 {self.line} 行: {source_lines[self.line - 1]}")
                if self.column:
                    lines.append(f"  {'─' * (self.column + 7)}^")
        
        return '\n'.join(lines)
```

---

## 13.4 完整解释器实现

```python
# querylang/interpreter.py
from querylang.ast_nodes import *

class Interpreter:
    """
    树遍历解释器：直接遍历AST执行查询
    """
    
    def __init__(self, tables: dict[str, list[dict]]):
        self.tables = tables
    
    def execute(self, query: Query) -> list[dict]:
        """执行完整查询"""
        # Step 1: FROM - 获取数据源
        table_name = query.from_clause.table
        if table_name not in self.tables:
            raise RuntimeError(f"表 '{table_name}' 不存在")
        
        rows = list(self.tables[table_name])  # 复制，不修改原数据
        
        # Step 2: WHERE - 过滤
        if query.where_clause:
            rows = [row for row in rows 
                    if self._eval_condition(query.where_clause.condition, row)]
        
        # Step 3: ORDER BY - 排序
        if query.order_clause:
            field = query.order_clause.field
            reverse = query.order_clause.direction == "DESC"
            rows = sorted(rows, key=lambda r: r.get(field), reverse=reverse)
        
        # Step 4: LIMIT/OFFSET - 分页
        if query.limit_clause:
            rows = rows[:query.limit_clause.count]
        
        # Step 5: SELECT - 投影
        if query.select_clause:
            rows = self._project(rows, query.select_clause)
        
        return rows
    
    def _eval_condition(self, condition, row: dict) -> bool:
        """递归求值条件表达式"""
        match condition:
            case BinaryOp(operator="AND", left=l, right=r):
                return self._eval_condition(l, row) and self._eval_condition(r, row)
            
            case BinaryOp(operator="OR", left=l, right=r):
                return self._eval_condition(l, row) or self._eval_condition(r, row)
            
            case Comparison(field=field, operator=op, value=value_node):
                left_val = row.get(field)
                right_val = self._eval_value(value_node)
                return self._compare(left_val, op, right_val)
            
            case _:
                raise RuntimeError(f"未知条件类型: {type(condition)}")
    
    def _compare(self, left, op: str, right) -> bool:
        """执行比较运算"""
        # 类型强制转换
        if isinstance(left, str) and isinstance(right, (int, float)):
            try:
                left = type(right)(left)
            except (ValueError, TypeError):
                pass
        
        match op:
            case ">":  return left is not None and left > right
            case "<":  return left is not None and left < right
            case ">=": return left is not None and left >= right
            case "<=": return left is not None and left <= right
            case "=":  return left == right
            case "!=": return left != right
            case _: raise RuntimeError(f"未知运算符: {op}")
    
    def _eval_value(self, node) -> any:
        """求值字面量节点"""
        match node:
            case IntegerLiteral(value=v): return v
            case FloatLiteral(value=v): return v
            case StringLiteral(value=v): return v
            case BooleanLiteral(value=v): return v
            case _: raise RuntimeError(f"未知值类型: {type(node)}")
    
    def _project(self, rows: list[dict], select: SelectClause) -> list[dict]:
        """SELECT投影：选择需要的字段"""
        if select.star:
            return rows
        
        fields = select.fields
        return [{field: row.get(field) for field in fields} for row in rows]
```

---

## 13.5 综合测试

```python
import pytest

# 测试数据
USERS = [
    {"id": 1, "name": "Alice",   "age": 25, "status": "active",   "email": "alice@example.com"},
    {"id": 2, "name": "Bob",     "age": 17, "status": "active",   "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "age": 30, "status": "inactive", "email": "charlie@example.com"},
    {"id": 4, "name": "Diana",   "age": 22, "status": "active",   "email": "diana@example.com"},
    {"id": 5, "name": "Eve",     "age": 15, "status": "inactive", "email": "eve@example.com"},
]

ORDERS = [
    {"id": 101, "user_id": 1, "amount": 150.0, "status": "paid"},
    {"id": 102, "user_id": 1, "amount": 75.50, "status": "pending"},
    {"id": 103, "user_id": 3, "amount": 200.0, "status": "paid"},
    {"id": 104, "user_id": 4, "amount": 50.0,  "status": "cancelled"},
]

engine = QueryLangEngine()
tables = {"users": USERS, "orders": ORDERS}

class TestQueryLangEngine:
    
    def test_select_all(self):
        result = engine.execute("FROM users SELECT *", tables)
        assert result.total_rows == 5
    
    def test_where_condition(self):
        result = engine.execute(
            'FROM users WHERE age > 18 SELECT name',
            tables
        )
        names = [r["name"] for r in result.rows]
        assert "Alice" in names
        assert "Diana" in names
        assert "Bob" not in names
    
    def test_and_condition(self):
        result = engine.execute(
            'FROM users WHERE age > 18 AND status = "active" SELECT name',
            tables
        )
        assert result.total_rows == 2
        names = {r["name"] for r in result.rows}
        assert names == {"Alice", "Diana"}
    
    def test_or_condition(self):
        result = engine.execute(
            'FROM users WHERE status = "inactive" OR age < 18 SELECT name',
            tables
        )
        assert result.total_rows == 3
    
    def test_order_by_asc(self):
        result = engine.execute(
            'FROM users WHERE age > 18 SELECT name ORDER BY name ASC',
            tables
        )
        names = [r["name"] for r in result.rows]
        assert names == sorted(names)
    
    def test_order_by_desc(self):
        result = engine.execute(
            'FROM users SELECT name, age ORDER BY age DESC',
            tables
        )
        ages = [r["age"] for r in result.rows]
        assert ages == sorted(ages, reverse=True)
    
    def test_limit(self):
        result = engine.execute(
            'FROM users SELECT name ORDER BY name LIMIT 2',
            tables
        )
        assert result.total_rows == 2
    
    def test_multiline_query(self):
        result = engine.execute("""
FROM users
WHERE age > 18 AND status = "active"
SELECT name, email
ORDER BY name
LIMIT 10
        """, tables)
        assert result.total_rows == 2
    
    def test_explain(self):
        plan = engine.explain('FROM users WHERE age > 18 SELECT name')
        assert "FROM" in plan
        assert "WHERE" in plan
```

---

## 13.6 添加REPL（交互式解释器）

```python
# querylang/repl.py
import readline  # 支持历史记录和方向键

def start_repl(tables: dict):
    """启动QueryLang交互式解释器"""
    engine = QueryLangEngine()
    print("QueryLang REPL v1.0")
    print("输入 '.help' 查看帮助，'.quit' 退出\n")
    
    history = []
    
    while True:
        try:
            # 多行输入支持（以空行结束）
            lines = []
            while True:
                prompt = ">>> " if not lines else "... "
                line = input(prompt)
                
                if line.strip() == '':
                    if lines:
                        break
                    continue
                
                # 特殊命令
                if line.startswith('.'):
                    handle_special_command(line, tables, engine)
                    break
                
                lines.append(line)
                # 单行查询（包含SELECT）直接执行
                if 'SELECT' in line.upper() or 'select' in line:
                    break
            
            if not lines:
                continue
            
            source = '\n'.join(lines)
            history.append(source)
            
            # 执行查询
            result = engine.execute(source, tables)
            
            # 格式化输出
            if result.rows:
                headers = list(result.rows[0].keys())
                print_table(result.rows, headers)
            else:
                print("(0 rows)")
            
            print(f"\n[{result.total_rows} rows, {result.query_time_ms:.2f}ms]")
        
        except QueryLangError as e:
            print(f"\n错误: {e.format_with_context()}")
        except KeyboardInterrupt:
            print("\n(使用 .quit 退出)")
        except EOFError:
            print("\n再见！")
            break

def print_table(rows: list[dict], headers: list[str]):
    """格式化打印表格"""
    if not rows:
        return
    
    # 计算列宽
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ''))))
    
    # 打印表头
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    separator = "-+-".join('-' * col_widths[h] for h in headers)
    print(header_line)
    print(separator)
    
    # 打印数据
    for row in rows:
        row_line = " | ".join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers)
        print(row_line)

def handle_special_command(cmd: str, tables: dict, engine: QueryLangEngine):
    match cmd.strip():
        case '.help':
            print("可用命令：")
            print("  .tables     - 列出所有可用表")
            print("  .schema     - 显示表结构")
            print("  .explain    - 解析查询并显示AST")
            print("  .quit       - 退出")
        case '.tables':
            for name, rows in tables.items():
                print(f"  {name} ({len(rows)} rows)")
        case '.quit' | '.exit':
            import sys; sys.exit(0)
        case _:
            print(f"未知命令: {cmd}")


# 启动
if __name__ == "__main__":
    demo_tables = {
        "users": [
            {"id": 1, "name": "Alice", "age": 25, "status": "active"},
            {"id": 2, "name": "Bob",   "age": 17, "status": "active"},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "amount": 150.0, "status": "paid"},
        ]
    }
    start_repl(demo_tables)
```

---

## 13.7 管道各阶段的调试输出

```python
class DebugEngine(QueryLangEngine):
    """带调试输出的引擎"""
    
    def execute(self, source: str, tables=None, *, verbose=False):
        if not verbose:
            return super().execute(source, tables)
        
        print(f"{'='*60}")
        print(f"源码:\n{source}")
        print(f"{'─'*60}")
        
        # 词法分析
        from querylang.lexer import Lexer
        tokens = Lexer(source).tokenize()
        print("Token流:")
        for t in tokens:
            if t.type.name != 'EOF':
                print(f"  {t.type.name:12} {t.value!r:20} ({t.line}:{t.column})")
        
        # 语法分析
        from querylang.parser import Parser
        from querylang.ast_nodes import ASTPrinter
        ast = Parser(tokens).parse()
        print(f"\nAST:\n{ASTPrinter().print(ast)}")
        
        # 执行
        print(f"\n执行结果:")
        result = super().execute(source, tables)
        for row in result.rows[:5]:
            print(f"  {row}")
        if result.total_rows > 5:
            print(f"  ... 还有 {result.total_rows - 5} 行")
        
        return result
```

---

## 小结

| 阶段 | 输入 | 输出 | 可能的错误 |
|------|------|------|-----------|
| 词法分析 | 字符串 | Token流 | 非法字符、未闭合字符串 |
| 语法分析 | Token流 | AST | 语法错误、缺少关键字 |
| 语义分析 | AST | 带注解AST | 未知表/字段、类型错误 |
| 执行 | AST + 数据 | 结果集 | 运行时类型错误 |

---

**上一章**：[Python魔法方法DSL](../part3-internal-dsl/12-python-magic-dsl.md)
**下一章**：[语义分析与符号表](./14-semantic-analysis.md)

# 第18章：错误报告与诊断

## 核心思维模型

> 错误消息是DSL与用户的**最后一道沟通**：当用户写错了，好的错误消息让他们立刻知道哪里错了、为什么错了、怎么改对。坏的错误消息让用户绝望地放弃。**错误消息是DSL的用户界面。**

---

## 18.1 好错误 vs 坏错误

```
坏错误：
  ParseError: unexpected token at position 47

好错误：
  第3行，第15列：期望字段名，但遇到了数字
  
  FROM users
  WHERE 18 > age
        ^^
  提示：比较的左侧应为字段名，右侧应为值
  建议写法：WHERE age > 18
```

**好错误的五要素**：
1. **位置**：精确到行号和列号
2. **上下文**：显示错误附近的源码
3. **指示**：用符号标出问题所在
4. **说明**：用领域语言（非技术语言）解释错误
5. **建议**：如果可能，提供修复方向

---

## 18.2 源码位置追踪系统

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SourceSpan:
    """源码中的一段范围"""
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    source: str  # 完整源码（用于上下文显示）
    
    @classmethod
    def from_token(cls, token, source: str) -> 'SourceSpan':
        end_col = token.column + len(token.value)
        return cls(token.line, token.column, token.line, end_col, source)
    
    @classmethod
    def merge(cls, start: 'SourceSpan', end: 'SourceSpan') -> 'SourceSpan':
        """合并两个span（从start的开始到end的结束）"""
        return cls(
            start.start_line, start.start_col,
            end.end_line, end.end_col,
            start.source
        )
    
    def get_source_lines(self) -> list[str]:
        """获取span覆盖的源码行"""
        lines = self.source.split('\n')
        return lines[self.start_line - 1: self.end_line]
    
    def render(self, message: str, color: bool = True) -> str:
        """渲染带上下文的错误消息"""
        lines = []
        source_lines = self.source.split('\n')
        
        # 显示标题
        lines.append(f"第 {self.start_line} 行，第 {self.start_col} 列：{message}")
        
        # 显示上下文（前1行）
        if self.start_line > 1:
            prev_line = source_lines[self.start_line - 2]
            lines.append(f"  {self.start_line - 1:4d} │ {prev_line}")
        
        # 显示错误行
        error_line = source_lines[self.start_line - 1]
        if color:
            lines.append(f"  {self.start_line:4d} │ {error_line}")
        else:
            lines.append(f"  {self.start_line:4d} │ {error_line}")
        
        # 显示指示符
        indicator_start = self.start_col - 1
        indicator_len = max(self.end_col - self.start_col, 1)
        
        prefix = " " * 7  # "  NNNN │ " 的长度
        indicator = " " * indicator_start + "^" * indicator_len
        lines.append(f"       │ {indicator}")
        
        # 显示上下文（后1行）
        if self.end_line < len(source_lines):
            next_line = source_lines[self.end_line]
            lines.append(f"  {self.end_line + 1:4d} │ {next_line}")
        
        return '\n'.join(lines)
```

---

## 18.3 结构化诊断系统

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

class DiagnosticLevel(Enum):
    ERROR   = "错误"
    WARNING = "警告"
    INFO    = "信息"
    HINT    = "提示"

@dataclass
class Diagnostic:
    """一条诊断消息（错误/警告/提示）"""
    level: DiagnosticLevel
    code: str               # 错误码：E001, W002等
    message: str            # 主消息
    span: Optional[SourceSpan] = None
    notes: list[str] = field(default_factory=list)    # 附加说明
    suggestions: list[str] = field(default_factory=list)  # 修复建议
    
    def render(self) -> str:
        lines = []
        
        # 标题行
        level_str = f"[{self.level.value}]"
        lines.append(f"{level_str} {self.code}: {self.message}")
        
        # 源码上下文
        if self.span:
            lines.append(self.span.render(self.message))
        
        # 附加说明
        for note in self.notes:
            lines.append(f"  注：{note}")
        
        # 修复建议
        for i, suggestion in enumerate(self.suggestions, 1):
            lines.append(f"  建议{i if len(self.suggestions) > 1 else ''}：{suggestion}")
        
        return '\n'.join(lines)
    
    def __str__(self) -> str:
        return self.render()


class DiagnosticEngine:
    """
    诊断消息收集和报告引擎
    
    设计参考Rust编译器的诊断系统
    """
    
    def __init__(self, source: str):
        self.source = source
        self._diagnostics: list[Diagnostic] = []
    
    def error(self, code: str, message: str, span=None, 
              notes=None, suggestions=None) -> 'DiagnosticEngine':
        self._diagnostics.append(Diagnostic(
            DiagnosticLevel.ERROR, code, message,
            span, notes or [], suggestions or []
        ))
        return self
    
    def warning(self, code: str, message: str, span=None,
                notes=None, suggestions=None) -> 'DiagnosticEngine':
        self._diagnostics.append(Diagnostic(
            DiagnosticLevel.WARNING, code, message,
            span, notes or [], suggestions or []
        ))
        return self
    
    def hint(self, code: str, message: str, span=None) -> 'DiagnosticEngine':
        self._diagnostics.append(Diagnostic(
            DiagnosticLevel.HINT, code, message, span
        ))
        return self
    
    @property
    def has_errors(self) -> bool:
        return any(d.level == DiagnosticLevel.ERROR for d in self._diagnostics)
    
    def all_errors(self) -> list[Diagnostic]:
        return [d for d in self._diagnostics if d.level == DiagnosticLevel.ERROR]
    
    def report(self) -> str:
        """生成完整报告"""
        if not self._diagnostics:
            return "✓ 无错误"
        
        lines = []
        errors = sum(1 for d in self._diagnostics if d.level == DiagnosticLevel.ERROR)
        warnings = sum(1 for d in self._diagnostics if d.level == DiagnosticLevel.WARNING)
        
        for d in self._diagnostics:
            lines.append(d.render())
            lines.append("")
        
        summary = []
        if errors: summary.append(f"{errors} 个错误")
        if warnings: summary.append(f"{warnings} 个警告")
        lines.append(f"{'─'*50}")
        lines.append("分析完成：" + "，".join(summary))
        
        return '\n'.join(lines)
    
    def raise_if_errors(self):
        """如果有错误则抛出异常"""
        if self.has_errors:
            raise DSLDiagnosticError(self._diagnostics)


class DSLDiagnosticError(Exception):
    """包含结构化诊断信息的异常"""
    
    def __init__(self, diagnostics: list[Diagnostic]):
        self.diagnostics = diagnostics
        errors = [d for d in diagnostics if d.level == DiagnosticLevel.ERROR]
        super().__init__(f"{len(errors)} 个错误")
    
    def __str__(self):
        return '\n'.join(d.render() for d in self.diagnostics)
```

---

## 18.4 QueryLang 诊断集成

```python
class DiagnosticParser:
    """
    集成了诊断引擎的解析器
    产生更好的错误消息
    """
    
    # 错误码定义
    ERROR_CODES = {
        "E001": "缺少FROM子句",
        "E002": "未知表名",
        "E003": "未知字段",
        "E004": "类型不匹配",
        "E005": "非法运算符",
        "E006": "缺少必须的关键字",
        "W001": "字段类型可能不兼容",
        "W002": "LIMIT值很大，可能影响性能",
    }
    
    def __init__(self, source: str, schema: dict = None):
        self.source = source
        self.schema = schema or {}
        self.diagnostics = DiagnosticEngine(source)
    
    def parse_and_analyze(self):
        """解析并分析，收集所有诊断信息"""
        from querylang.lexer import Lexer, LexError
        from querylang.parser import Parser, ParseError
        
        # ─── 词法分析 ──────────────────────────────────────
        try:
            tokens = Lexer(self.source).tokenize()
        except LexError as e:
            span = self._make_span(getattr(e, 'line', 1), getattr(e, 'column', 1))
            self.diagnostics.error(
                "E099",
                f"无法识别的字符：{str(e)}",
                span=span,
                suggestions=["检查是否有特殊字符或未闭合的引号"]
            )
            self.diagnostics.raise_if_errors()
            return
        
        # ─── 语法分析 ──────────────────────────────────────
        try:
            ast = Parser(tokens).parse()
        except ParseError as e:
            token = getattr(e, 'token', None)
            span = self._span_from_token(token) if token else None
            
            # 根据错误上下文提供不同的建议
            suggestions = self._get_parse_suggestions(e, token)
            
            self.diagnostics.error(
                "E006",
                str(e),
                span=span,
                suggestions=suggestions
            )
            self.diagnostics.raise_if_errors()
            return
        
        # ─── 语义分析 ──────────────────────────────────────
        if self.schema:
            self._check_table(ast)
            self._check_fields(ast)
            self._check_types(ast)
        
        return ast
    
    def _get_parse_suggestions(self, error, token) -> list[str]:
        """根据具体错误提供针对性建议"""
        error_str = str(error).lower()
        
        if "from" in error_str:
            return ["查询必须以 FROM <表名> 开头", "示例：FROM users SELECT *"]
        
        if "select" in error_str or "select" in (token.value.lower() if token else ""):
            return ["在FROM之后添加 SELECT 子句", "SELECT * 选择所有字段"]
        
        if "运算符" in error_str:
            return [
                "支持的比较运算符：>, <, >=, <=, =, !=",
                "示例：WHERE age > 18"
            ]
        
        return ["检查语法是否符合QueryLang规范"]
    
    def _check_table(self, ast):
        """检查表名是否存在"""
        table_name = ast.from_clause.table
        if table_name not in self.schema:
            similar = self._find_similar(table_name, self.schema.keys())
            suggestions = []
            if similar:
                suggestions.append(f"是否要用表名 '{similar}'？")
            suggestions.append(f"可用的表：{', '.join(self.schema.keys())}")
            
            span = self._make_span(ast.from_clause.line, ast.from_clause.column,
                                   len(table_name))
            self.diagnostics.error(
                "E002",
                f"表 '{table_name}' 不存在",
                span=span,
                suggestions=suggestions
            )
    
    def _check_fields(self, ast):
        """检查所有引用的字段"""
        if ast.from_clause.table not in self.schema:
            return
        
        table_fields = set(self.schema[ast.from_clause.table].keys())
        
        if ast.where_clause:
            self._check_condition_fields(ast.where_clause.condition, table_fields, ast)
        
        if ast.select_clause and not ast.select_clause.star:
            for field in ast.select_clause.fields:
                if field not in table_fields:
                    similar = self._find_similar(field, table_fields)
                    suggestions = []
                    if similar:
                        suggestions.append(f"是否要用字段名 '{similar}'？")
                    suggestions.append(f"可用字段：{', '.join(sorted(table_fields))}")
                    
                    self.diagnostics.error(
                        "E003",
                        f"字段 '{field}' 在表 '{ast.from_clause.table}' 中不存在",
                        suggestions=suggestions
                    )
    
    def _check_condition_fields(self, condition, valid_fields, ast):
        match condition:
            case BinaryOp(left=l, right=r):
                self._check_condition_fields(l, valid_fields, ast)
                self._check_condition_fields(r, valid_fields, ast)
            case Comparison(field=f):
                if f not in valid_fields:
                    similar = self._find_similar(f, valid_fields)
                    self.diagnostics.error(
                        "E003",
                        f"WHERE条件中的字段 '{f}' 不存在",
                        suggestions=[f"是否要用: '{similar}'" if similar else "检查字段名拼写"]
                    )
    
    def _check_types(self, ast):
        """类型检查"""
        if ast.limit_clause and ast.limit_clause.count > 10000:
            self.diagnostics.warning(
                "W002",
                f"LIMIT {ast.limit_clause.count} 值很大",
                notes=["大量数据传输可能影响性能"],
                suggestions=["考虑添加更多WHERE条件减少结果集"]
            )
    
    def _make_span(self, line, col, length=1) -> SourceSpan:
        return SourceSpan(line, col, line, col + length, self.source)
    
    def _span_from_token(self, token) -> SourceSpan:
        return SourceSpan(
            token.line, token.column,
            token.line, token.column + len(token.value),
            self.source
        )
    
    def _find_similar(self, name, candidates):
        def dist(a, b):
            if len(a) < len(b): a, b = b, a
            if not b: return len(a)
            prev = list(range(len(b) + 1))
            for ca in a:
                curr = [prev[0] + 1]
                for j, cb in enumerate(b):
                    curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca!=cb)))
                prev = curr
            return prev[-1]
        best = min(candidates, key=lambda c: dist(name.lower(), c.lower()), default=None)
        return best if best and dist(name.lower(), best.lower()) <= 3 else None
```

---

## 18.5 错误消息效果对比

```python
# 演示对比：同一个错误，不同质量的报告

SOURCE = '''FROM usres
WHERE nme = "Alice"
SELECT email
LIMIT 50000'''

# 基础错误（第13章的简单实现）：
# ParseError: 表 'usres' 不存在

# 诊断错误（本章实现）：
"""
[错误] E002: 表 'usres' 不存在

     1 │ FROM usres
       │      ^^^^^
  
  建议1：是否要用表名 'users'？
  建议2：可用的表：users, orders, products

[错误] E003: WHERE条件中的字段 'nme' 不存在

     2 │ WHERE nme = "Alice"
       │       ^^^
  
  建议：是否要用: 'name'

[警告] W002: LIMIT 50000 值很大

     4 │ LIMIT 50000
       │       ^^^^^
  
  注：大量数据传输可能影响性能
  建议：考虑添加更多WHERE条件减少结果集

──────────────────────────────────────────────────
分析完成：2 个错误，1 个警告
"""
```

---

## 18.6 测试错误消息质量

```python
import pytest

class TestDiagnosticMessages:
    
    def _run(self, source: str, schema=None) -> DiagnosticEngine:
        schema = schema or {"users": {"name": "string", "age": "int"}}
        dp = DiagnosticParser(source, schema)
        try:
            dp.parse_and_analyze()
        except DSLDiagnosticError:
            pass
        return dp.diagnostics
    
    def test_unknown_table_shows_suggestion(self):
        diag = self._run("FROM usres SELECT *")
        errors = diag.all_errors()
        assert len(errors) == 1
        assert "users" in str(errors[0])  # 建议正确拼写
    
    def test_unknown_field_shows_available_fields(self):
        diag = self._run("FROM users WHERE nme = 'Alice' SELECT *")
        errors = diag.all_errors()
        assert any("name" in str(e) for e in errors)
    
    def test_multiple_errors_all_reported(self):
        diag = self._run("FROM usres WHERE nme > 18 SELECT xyz")
        errors = diag.all_errors()
        assert len(errors) >= 2  # 表名错误 + 字段错误
    
    def test_large_limit_warns(self):
        diag = self._run("FROM users SELECT * LIMIT 100000")
        assert any(d.level == DiagnosticLevel.WARNING for d in diag._diagnostics)
```

---

## 小结

| 概念 | 要点 |
|------|------|
| 五要素 | 位置 + 上下文 + 指示 + 说明 + 建议 |
| SourceSpan | 追踪每个Token和节点的源码位置 |
| 错误码 | 结构化的错误分类，支持工具化处理 |
| 多错误收集 | 不在第一个错误处停止，收集所有错误 |
| 拼写建议 | Levenshtein距离，让用户自救 |

---

**上一章**：[DSL类型系统设计](./17-type-systems.md)
**下一章**：[DSL工具链：LSP与IDE支持](./19-dsl-tooling.md)

# 第8章：解析器组合子

## 核心思维模型

> 解析器组合子是函数式编程的解析器方法：**每个解析器是一个函数，通过高阶函数组合产生更复杂的解析器**。这种方式让解析器与文法保持高度同构，是函数式DSL设计的精华。

---

## 8.1 什么是解析器组合子？

递归下降解析器的问题在于：解析逻辑散落在各个方法中，不够组合化。

**解析器组合子的思路**：

```python
# 一个解析器是一个函数：
# (tokens, position) → (result, new_position) | None

# 基础解析器：匹配一个特定类型的Token
def token(token_type):
    def parser(tokens, pos):
        if pos < len(tokens) and tokens[pos].type == token_type:
            return tokens[pos], pos + 1
        return None
    return parser

# 组合子：顺序执行两个解析器
def seq(p1, p2):
    def parser(tokens, pos):
        r1 = p1(tokens, pos)
        if r1 is None: return None
        v1, pos1 = r1
        r2 = p2(tokens, pos1)
        if r2 is None: return None
        v2, pos2 = r2
        return (v1, v2), pos2
    return parser

# 组合子：尝试第一个，失败则尝试第二个
def alt(p1, p2):
    def parser(tokens, pos):
        r1 = p1(tokens, pos)
        if r1 is not None: return r1
        return p2(tokens, pos)
    return parser
```

现在可以用这些基础组件组合出任意复杂的解析器。

---

## 8.2 构建完整的组合子库

```python
from typing import TypeVar, Callable, Generic, Optional, Any
from dataclasses import dataclass

T = TypeVar('T')
ParseResult = tuple[T, int] | None
ParserFn = Callable[[list[Token], int], ParseResult]

class Parser(Generic[T]):
    """解析器包装类，提供面向对象的组合接口"""
    
    def __init__(self, fn: ParserFn):
        self._fn = fn
    
    def __call__(self, tokens: list[Token], pos: int) -> ParseResult:
        return self._fn(tokens, pos)
    
    def map(self, transform: Callable) -> 'Parser':
        """转换解析结果"""
        def parser(tokens, pos):
            result = self(tokens, pos)
            if result is None: return None
            value, new_pos = result
            return transform(value), new_pos
        return Parser(parser)
    
    def then(self, other: 'Parser') -> 'Parser':
        """顺序组合，返回两个结果的元组"""
        def parser(tokens, pos):
            r1 = self(tokens, pos)
            if r1 is None: return None
            v1, pos1 = r1
            r2 = other(tokens, pos1)
            if r2 is None: return None
            v2, pos2 = r2
            return (v1, v2), pos2
        return Parser(parser)
    
    def skip_then(self, other: 'Parser') -> 'Parser':
        """顺序组合，丢弃左侧结果"""
        return self.then(other).map(lambda pair: pair[1])
    
    def then_skip(self, other: 'Parser') -> 'Parser':
        """顺序组合，丢弃右侧结果"""
        return self.then(other).map(lambda pair: pair[0])
    
    def or_(self, other: 'Parser') -> 'Parser':
        """选择组合：先试左，失败则试右"""
        def parser(tokens, pos):
            result = self(tokens, pos)
            if result is not None: return result
            return other(tokens, pos)
        return Parser(parser)
    
    def optional(self, default=None) -> 'Parser':
        """零或一次"""
        def parser(tokens, pos):
            result = self(tokens, pos)
            if result is not None: return result
            return default, pos
        return Parser(parser)
    
    def many(self) -> 'Parser':
        """零或多次，返回列表"""
        def parser(tokens, pos):
            results = []
            while True:
                result = self(tokens, pos)
                if result is None: break
                value, pos = result
                results.append(value)
            return results, pos
        return Parser(parser)
    
    def many1(self) -> 'Parser':
        """一或多次，至少一个"""
        def parser(tokens, pos):
            first = self(tokens, pos)
            if first is None: return None
            value, pos = first
            results = [value]
            while True:
                result = self(tokens, pos)
                if result is None: break
                v, pos = result
                results.append(v)
            return results, pos
        return Parser(parser)
    
    def sep_by(self, sep: 'Parser') -> 'Parser':
        """由sep分隔的零或多次"""
        def parser(tokens, pos):
            first = self(tokens, pos)
            if first is None: return [], pos
            value, pos = first
            results = [value]
            while True:
                sep_result = sep(tokens, pos)
                if sep_result is None: break
                _, pos = sep_result
                item = self(tokens, pos)
                if item is None: break
                v, pos = item
                results.append(v)
            return results, pos
        return Parser(parser)


# ─── 基础解析器工厂函数 ─────────────────────────────────────

def tok(token_type: TokenType) -> Parser:
    """匹配特定类型的Token"""
    def fn(tokens, pos):
        if pos < len(tokens) and tokens[pos].type == token_type:
            return tokens[pos], pos + 1
        return None
    return Parser(fn)

def tok_value(token_type: TokenType, value: str) -> Parser:
    """匹配特定类型和值的Token"""
    def fn(tokens, pos):
        if (pos < len(tokens) and 
            tokens[pos].type == token_type and 
            tokens[pos].value.upper() == value.upper()):
            return tokens[pos], pos + 1
        return None
    return Parser(fn)

def any_of(*types: TokenType) -> Parser:
    """匹配多个类型之一"""
    def fn(tokens, pos):
        if pos < len(tokens) and tokens[pos].type in types:
            return tokens[pos], pos + 1
        return None
    return Parser(fn)

def lazy(fn: Callable[[], Parser]) -> Parser:
    """惰性求值，解决递归文法中的循环引用"""
    def parser(tokens, pos):
        return fn()(tokens, pos)
    return Parser(parser)

def skip_newlines() -> Parser:
    """跳过所有换行"""
    return tok(TokenType.NEWLINE).many().map(lambda _: None)
```

---

## 8.3 用组合子构建QueryLang解析器

```python
# ─── 原子解析器 ──────────────────────────────────────────────

# 各类Token的基础解析器
from_kw     = tok(TokenType.FROM)
where_kw    = tok(TokenType.WHERE)
select_kw   = tok(TokenType.SELECT)
order_kw    = tok(TokenType.ORDER)
by_kw       = tok(TokenType.BY)
limit_kw    = tok(TokenType.LIMIT)
and_kw      = tok(TokenType.AND)
or_kw       = tok(TokenType.OR)
comma       = tok(TokenType.COMMA)
star        = tok(TokenType.STAR)
ident       = tok(TokenType.IDENT)
integer     = tok(TokenType.INTEGER).map(lambda t: IntegerLiteral(int(t.value)))
float_num   = tok(TokenType.FLOAT).map(lambda t: FloatLiteral(float(t.value)))
string      = tok(TokenType.STRING).map(lambda t: StringLiteral(t.value))
boolean     = tok(TokenType.BOOLEAN).map(lambda t: BooleanLiteral(t.value.lower() == 'true'))

# 运算符
operator = any_of(
    TokenType.GT, TokenType.LT, TokenType.GTE,
    TokenType.LTE, TokenType.EQ, TokenType.NEQ
)

# ─── 组合解析器 ──────────────────────────────────────────────

# 值：整数 | 浮点 | 字符串 | 布尔
value_parser = integer.or_(float_num).or_(string).or_(boolean)

# 比较：IDENT op value
def make_comparison(parts):
    field_tok, op_tok, val = parts
    return Comparison(field_tok.value, op_tok.value, val)

comparison_parser = (
    ident.then(operator).then(value_parser)
    .map(lambda p: make_comparison((p[0][0], p[0][1], p[1])))
)

# 条件（递归！用lazy处理）
def condition_parser_fn():
    return or_expr_parser

condition_parser = lazy(condition_parser_fn)

# AND表达式
def fold_and(parts):
    first, rest = parts
    result = first
    for item in rest:
        result = BinaryOp("AND", result, item)
    return result

and_expr_parser = (
    comparison_parser.then(
        and_kw.skip_then(comparison_parser).many()
    ).map(fold_and)
)

# OR表达式（优先级低于AND）
def fold_or(parts):
    first, rest = parts
    result = first
    for item in rest:
        result = BinaryOp("OR", result, item)
    return result

or_expr_parser = (
    and_expr_parser.then(
        or_kw.skip_then(and_expr_parser).many()
    ).map(fold_or)
)

# ─── 子句解析器 ──────────────────────────────────────────────

# FROM users
from_parser = (
    from_kw.skip_then(ident)
    .map(lambda t: FromClause(t.value))
)

# WHERE condition
where_parser = (
    where_kw.skip_then(or_expr_parser)
    .map(lambda c: WhereClause(c))
)

# SELECT * | SELECT f1, f2, ...
select_star = (
    select_kw.skip_then(star)
    .map(lambda _: SelectClause(star=True))
)

select_fields = (
    select_kw.skip_then(
        ident.sep_by(comma)
    ).map(lambda fields: SelectClause(fields=[f.value for f in fields]))
)

select_parser = select_star.or_(select_fields)

# ORDER BY field [ASC|DESC]
direction_parser = (
    tok(TokenType.ASC).map(lambda _: "ASC")
    .or_(tok(TokenType.DESC).map(lambda _: "DESC"))
    .optional("ASC")
)

order_parser = (
    order_kw.skip_then(by_kw).skip_then(
        ident.then(direction_parser)
    ).map(lambda p: OrderClause(p[0].value, p[1]))
)

# LIMIT n
limit_parser = (
    limit_kw.skip_then(tok(TokenType.INTEGER))
    .map(lambda t: LimitClause(int(t.value)))
)

# ─── 顶层查询解析器 ───────────────────────────────────────────

nl = skip_newlines()  # 换行跳过

def make_query(parts):
    from_c, _, where_c, _, select_c, _, order_c, _, limit_c = (
        parts[0], parts[1], parts[2], parts[3], parts[4],
        parts[5], parts[6], parts[7], parts[8]
    )
    return Query(from_c, where_c, select_c, order_c, limit_c)

query_parser = (
    from_parser
    .then(nl).map(lambda p: p[0])  # 跳过换行
    # 注：实际实现中需要更精细地处理可选子句
)

# 更实用的方式：手动解析可选子句
def parse_with_combinators(tokens: list[Token]) -> Query:
    pos = 0
    
    # 跳过换行
    while pos < len(tokens) and tokens[pos].type == TokenType.NEWLINE:
        pos += 1
    
    # FROM（必须）
    result = from_parser(tokens, pos)
    if result is None:
        raise ParseError("查询必须以FROM开头", tokens[pos])
    from_clause, pos = result
    
    # 跳过换行
    while pos < len(tokens) and tokens[pos].type == TokenType.NEWLINE:
        pos += 1
    
    where_clause = None
    select_clause = None
    order_clause = None
    limit_clause = None
    
    # 可选子句
    for parser_fn, setter in [
        (where_parser, lambda v: locals().__setitem__('where_clause', v)),
        (select_parser, lambda v: locals().__setitem__('select_clause', v)),
        (order_parser, lambda v: locals().__setitem__('order_clause', v)),
        (limit_parser, lambda v: locals().__setitem__('limit_clause', v)),
    ]:
        while pos < len(tokens) and tokens[pos].type == TokenType.NEWLINE:
            pos += 1
        result = parser_fn(tokens, pos)
        if result is not None:
            value, pos = result
            # 根据类型分配
            if isinstance(value, WhereClause): where_clause = value
            elif isinstance(value, SelectClause): select_clause = value
            elif isinstance(value, OrderClause): order_clause = value
            elif isinstance(value, LimitClause): limit_clause = value
    
    return Query(from_clause, where_clause, select_clause, order_clause, limit_clause)
```

---

## 8.4 PEG文法（Parsing Expression Grammar）

解析器组合子的理论基础是 **PEG**，它与CFG（上下文无关文法）的关键区别：

| 维度 | CFG | PEG |
|------|-----|-----|
| 选择语义 | 非确定性（`A|B`都有可能匹配） | 有序选择（先试A，失败才试B）|
| 二义性 | 可能有（需要消解） | 没有（有序选择消除二义性）|
| 回溯 | 不回溯（LL/LR解析器） | 可能回溯（memoization优化）|
| 工具 | ANTLR, yacc, bison | PEG.js, Lark, pyparsing |

PEG的有序选择 `/`（"有序选择"）：

```
# PEG文法：先试INTEGER，失败才试IDENT
value ← INTEGER / FLOAT / STRING / BOOLEAN / IDENT
```

这意味着PEG文法中规则顺序**很重要**，`INTEGER / IDENT` 和 `IDENT / INTEGER` 语义不同。

---

## 8.5 使用lark-parser库

`lark` 是Python最流行的解析器库，支持EBNF风格文法：

```python
from lark import Lark, Transformer, v_args

QUERYLANG_GRAMMAR = r"""
    query       : from_clause newlines? where_clause? newlines? select_clause? newlines? order_clause? newlines? limit_clause?
    
    from_clause : "FROM"i IDENT
    
    where_clause : "WHERE"i condition
    condition   : or_expr
    or_expr     : and_expr ("OR"i and_expr)*
    and_expr    : comparison ("AND"i comparison)*
    comparison  : IDENT OPERATOR value
                | "(" condition ")"
    
    OPERATOR    : ">=" | "<=" | "!=" | ">" | "<" | "="
    
    select_clause : "SELECT"i ("*" | IDENT ("," IDENT)*)
    order_clause  : "ORDER"i "BY"i IDENT ("ASC"i | "DESC"i)?
    limit_clause  : "LIMIT"i INT
    
    value       : INT       -> integer
                | FLOAT     -> float_val
                | ESCAPED_STRING -> string
                | BOOLEAN   -> boolean
    
    BOOLEAN     : "true"i | "false"i
    newlines    : NEWLINE+
    
    %import common.CNAME -> IDENT
    %import common.INT
    %import common.FLOAT
    %import common.ESCAPED_STRING
    %import common.NEWLINE
    %import common.WS
    %ignore WS
"""

@v_args(inline=True)
class QueryTransformer(Transformer):
    """把lark解析树转换为我们的AST节点"""
    
    def query(self, *args):
        # 过滤None（可选子句）
        clauses = [a for a in args if a is not None]
        from_c = next((c for c in clauses if isinstance(c, FromClause)), None)
        where_c = next((c for c in clauses if isinstance(c, WhereClause)), None)
        select_c = next((c for c in clauses if isinstance(c, SelectClause)), None)
        order_c = next((c for c in clauses if isinstance(c, OrderClause)), None)
        limit_c = next((c for c in clauses if isinstance(c, LimitClause)), None)
        return Query(from_c, where_c, select_c, order_c, limit_c)
    
    def from_clause(self, ident):
        return FromClause(str(ident))
    
    def where_clause(self, condition):
        return WhereClause(condition)
    
    def or_expr(self, first, *rest):
        result = first
        for item in rest:
            result = BinaryOp("OR", result, item)
        return result
    
    def and_expr(self, first, *rest):
        result = first
        for item in rest:
            result = BinaryOp("AND", result, item)
        return result
    
    def comparison(self, field, op, value):
        return Comparison(str(field), str(op), value)
    
    def integer(self, n):
        return IntegerLiteral(int(n))
    
    def string(self, s):
        return StringLiteral(str(s)[1:-1])  # 去引号
    
    def float_val(self, f):
        return FloatLiteral(float(f))

# 使用
lark_parser = Lark(QUERYLANG_GRAMMAR, start='query', parser='earley')

def parse_query_lark(source: str) -> Query:
    tree = lark_parser.parse(source)
    return QueryTransformer().transform(tree)

# 测试
ast = parse_query_lark("""
FROM users
WHERE age > 18 AND status = "active"
SELECT name, email
""")
print(ast)
```

---

## 8.6 组合子 vs 递归下降 vs 解析器生成器

| 维度 | 手写递归下降 | 解析器组合子 | lark/ANTLR |
|------|------------|------------|------------|
| 代码量 | 中 | 少（声明式）| 最少（文法即代码）|
| 错误消息 | 最好（手动定制）| 中（可定制）| 差（通用消息）|
| 调试难度 | 容易 | 中 | 难 |
| 性能 | 最好 | 中（函数调用开销）| 取决于算法 |
| 文法维护 | 隐式在代码中 | 半隐式 | 显式文法文件 |
| 适合场景 | 性能敏感、复杂错误恢复 | 快速原型 | 复杂文法、工具链生成 |

---

## 小结

| 概念 | 要点 |
|------|------|
| 解析器组合子 | 解析器是函数，组合子是高阶函数 |
| 核心原语 | `then`（顺序）、`or_`（选择）、`many`（重复）|
| PEG vs CFG | PEG有序选择无歧义，CFG非确定性可能有歧义 |
| lazy | 解决递归文法的循环引用 |
| lark库 | 文法即代码，自动生成解析器 |

---

**上一章**：[递归下降解析器](./07-recursive-descent.md)
**下一章**：[内部DSL模式总览](../part3-internal-dsl/09-internal-dsl-patterns.md)

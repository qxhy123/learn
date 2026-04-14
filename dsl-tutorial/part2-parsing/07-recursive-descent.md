# 第7章：递归下降解析器

## 核心思维模型

> 递归下降是最直观的解析方法：**文法的每条规则对应一个函数，规则的递归对应函数的调用**。读懂文法，你就知道如何写解析器。

---

## 7.1 递归下降的核心思想

对于文法规则：
```ebnf
query = from_clause where_clause? select_clause order_clause? limit_clause?
```

直接翻译为函数：

```python
def parse_query(self):
    from_clause = self.parse_from_clause()       # 必须
    where_clause = self.parse_where_clause()     # 可选
    select_clause = self.parse_select_clause()   # 可选
    order_clause = self.parse_order_clause()     # 可选
    limit_clause = self.parse_limit_clause()     # 可选
    return Query(from_clause, where_clause, select_clause, order_clause, limit_clause)
```

这就是**递归下降（Recursive Descent）**：从顶层规则（query）开始，递归调用子规则函数，直到叶子节点（Token）。

---

## 7.2 解析器基础设施

```python
from typing import Optional

class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        super().__init__(message)
        self.token = token
        self.line = token.line
        self.column = token.column

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
    
    # ─── 基础操作 ──────────────────────────────────────────
    
    @property
    def current(self) -> Token:
        """当前Token"""
        return self.tokens[self.pos]
    
    def peek(self, offset: int = 1) -> Token:
        """向前窥视"""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]  # 返回EOF
    
    def advance(self) -> Token:
        """消费当前Token，返回它"""
        token = self.current
        if token.type != TokenType.EOF:
            self.pos += 1
        return token
    
    def check(self, *types: TokenType) -> bool:
        """检查当前Token是否匹配任一类型，不消费"""
        return self.current.type in types
    
    def match(self, *types: TokenType) -> Optional[Token]:
        """如果当前Token匹配，消费并返回；否则返回None"""
        if self.check(*types):
            return self.advance()
        return None
    
    def expect(self, type: TokenType, message: str = None) -> Token:
        """断言当前Token是指定类型，消费并返回；否则抛出错误"""
        if self.current.type == type:
            return self.advance()
        msg = message or f"期望 {type.name}，得到 {self.current.type.name} ({self.current.value!r})"
        raise ParseError(msg, self.current)
    
    def skip_newlines(self):
        """跳过换行符"""
        while self.check(TokenType.NEWLINE):
            self.advance()
```

---

## 7.3 实现QueryLang解析器

### 7.3.1 顶层：解析查询

```python
    def parse(self) -> Query:
        """入口：解析完整查询"""
        self.skip_newlines()
        query = self.parse_query()
        self.expect(TokenType.EOF, "查询结束后有多余内容")
        return query
    
    def parse_query(self) -> Query:
        from_clause = self.parse_from_clause()
        self.skip_newlines()
        
        where_clause = None
        select_clause = None
        order_clause = None
        limit_clause = None
        
        # 解析可选子句（顺序灵活）
        while not self.check(TokenType.EOF):
            if self.check(TokenType.WHERE):
                where_clause = self.parse_where_clause()
            elif self.check(TokenType.SELECT):
                select_clause = self.parse_select_clause()
            elif self.check(TokenType.ORDER):
                order_clause = self.parse_order_clause()
            elif self.check(TokenType.LIMIT):
                limit_clause = self.parse_limit_clause()
            elif self.check(TokenType.NEWLINE):
                self.skip_newlines()
            else:
                break
        
        return Query(from_clause, where_clause, select_clause, order_clause, limit_clause)
```

### 7.3.2 FROM子句

```python
    def parse_from_clause(self) -> FromClause:
        token = self.expect(TokenType.FROM, "查询必须以FROM开头")
        table_token = self.expect(TokenType.IDENT, "FROM后需要表名")
        return FromClause(table=table_token.value, line=token.line, column=token.column)
```

### 7.3.3 WHERE子句（含运算符优先级处理）

```python
    def parse_where_clause(self) -> WhereClause:
        token = self.expect(TokenType.WHERE)
        condition = self.parse_condition()
        return WhereClause(condition=condition, line=token.line, column=token.column)
    
    def parse_condition(self) -> ASTNode:
        """
        condition = and_expr ('OR' and_expr)*
        
        这个层处理 OR，优先级最低
        """
        left = self.parse_and_expr()
        
        while self.check(TokenType.OR):
            op_token = self.advance()
            right = self.parse_and_expr()
            left = BinaryOp(
                operator="OR",
                left=left,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_and_expr(self) -> ASTNode:
        """
        and_expr = comparison ('AND' comparison)*
        
        这个层处理 AND，优先级高于 OR
        """
        left = self.parse_comparison()
        
        while self.check(TokenType.AND):
            op_token = self.advance()
            right = self.parse_comparison()
            left = BinaryOp(
                operator="AND",
                left=left,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_comparison(self) -> ASTNode:
        """
        comparison = field operator value
                   | '(' condition ')'
        """
        # 括号分组
        if self.match(TokenType.LPAREN):  # 假设我们添加了LPAREN/RPAREN
            condition = self.parse_condition()
            self.expect(TokenType.RPAREN, "括号未闭合")
            return condition
        
        # 字段 运算符 值
        field_token = self.expect(TokenType.IDENT, "条件左侧应为字段名")
        
        # 解析运算符
        op_token = self.current
        if not self.match(TokenType.GT, TokenType.LT, TokenType.GTE,
                          TokenType.LTE, TokenType.EQ, TokenType.NEQ):
            raise ParseError(
                f"期望比较运算符(>, <, >=, <=, =, !=)，得到 {op_token.value!r}",
                op_token
            )
        
        value = self.parse_value()
        
        return Comparison(
            field=field_token.value,
            operator=op_token.value,
            value=value,
            line=field_token.line,
            column=field_token.column
        )
    
    def parse_value(self) -> ASTNode:
        """解析字面量值"""
        token = self.current
        
        if self.match(TokenType.INTEGER):
            return IntegerLiteral(value=int(token.value), line=token.line, column=token.column)
        elif self.match(TokenType.FLOAT):
            return FloatLiteral(value=float(token.value), line=token.line, column=token.column)
        elif self.match(TokenType.STRING):
            return StringLiteral(value=token.value, line=token.line, column=token.column)
        elif self.match(TokenType.BOOLEAN):
            return BooleanLiteral(value=token.value.lower() == 'true', line=token.line, column=token.column)
        else:
            raise ParseError(
                f"期望值（整数、浮点数、字符串或布尔值），得到 {token.type.name} ({token.value!r})",
                token
            )
```

### 7.3.4 SELECT子句

```python
    def parse_select_clause(self) -> SelectClause:
        token = self.expect(TokenType.SELECT)
        
        if self.match(TokenType.STAR):
            return SelectClause(star=True, line=token.line, column=token.column)
        
        fields = []
        fields.append(self.expect(TokenType.IDENT, "SELECT后需要字段名").value)
        
        while self.match(TokenType.COMMA):
            fields.append(self.expect(TokenType.IDENT, "逗号后需要字段名").value)
        
        return SelectClause(fields=fields, line=token.line, column=token.column)
```

### 7.3.5 ORDER BY 和 LIMIT

```python
    def parse_order_clause(self) -> OrderClause:
        token = self.expect(TokenType.ORDER)
        self.expect(TokenType.BY, "ORDER后需要BY")
        field_token = self.expect(TokenType.IDENT, "ORDER BY后需要字段名")
        
        direction = "ASC"
        if self.match(TokenType.DESC):
            direction = "DESC"
        elif self.match(TokenType.ASC):
            direction = "ASC"
        
        return OrderClause(
            field=field_token.value,
            direction=direction,
            line=token.line,
            column=token.column
        )
    
    def parse_limit_clause(self) -> LimitClause:
        token = self.expect(TokenType.LIMIT)
        count_token = self.expect(TokenType.INTEGER, "LIMIT后需要整数")
        count = int(count_token.value)
        
        if count <= 0:
            raise ParseError(f"LIMIT必须大于0，得到 {count}", count_token)
        
        return LimitClause(count=count, line=token.line, column=token.column)
```

---

## 7.4 错误恢复（Error Recovery）

专业的解析器不会在第一个错误处停止，而是**尽量继续解析**，收集更多错误：

```python
class RobustParser(Parser):
    def __init__(self, tokens: list[Token]):
        super().__init__(tokens)
        self.errors: list[ParseError] = []
    
    def synchronize(self, *sync_tokens: TokenType):
        """
        错误恢复：跳过Token直到找到同步点
        同步点通常是子句开始的关键字
        """
        while not self.check(TokenType.EOF):
            if self.check(*sync_tokens):
                return
            self.advance()
    
    def parse_query_robust(self) -> tuple[Query | None, list[ParseError]]:
        """容错版本：尽量解析，收集所有错误"""
        errors = []
        from_clause = None
        where_clause = None
        select_clause = None
        
        try:
            from_clause = self.parse_from_clause()
        except ParseError as e:
            errors.append(e)
            self.synchronize(TokenType.WHERE, TokenType.SELECT, TokenType.EOF)
        
        if self.check(TokenType.WHERE):
            try:
                where_clause = self.parse_where_clause()
            except ParseError as e:
                errors.append(e)
                self.synchronize(TokenType.SELECT, TokenType.ORDER, TokenType.LIMIT, TokenType.EOF)
        
        if self.check(TokenType.SELECT):
            try:
                select_clause = self.parse_select_clause()
            except ParseError as e:
                errors.append(e)
        
        if errors:
            return None, errors
        
        return Query(from_clause, where_clause, select_clause), []
```

---

## 7.5 LL(k)文法与展望冲突

递归下降解析器是 **LL(1)** 解析器（从左到右扫描，最左推导，向前看1个Token）。

**LL(1)问题**：有些文法需要看更多Token才能决定走哪条路径：

```ebnf
# 这个文法有展望冲突（lookahead conflict）
expression = identifier          # 可能是字段引用
           | identifier '(' args ')' # 也可能是函数调用
```

看到 `identifier` 时，不确定下一步走哪条规则，需要再看一个Token（`(`）。

**解决方法一：LL(2)展望**

```python
def parse_expression(self):
    if self.check(TokenType.IDENT) and self.peek(1).type == TokenType.LPAREN:
        return self.parse_function_call()
    else:
        return self.parse_field_ref()
```

**解决方法二：提取左公因子（Left Factoring）**

```ebnf
# 重写文法，消除展望冲突
expression = identifier ('(' args ')' )?
```

```python
def parse_expression(self):
    name_token = self.expect(TokenType.IDENT)
    if self.match(TokenType.LPAREN):
        args = self.parse_args()
        self.expect(TokenType.RPAREN)
        return FunctionCall(name_token.value, args)
    return FieldRef(name_token.value)
```

---

## 7.6 完整解析器集成测试

```python
def run_parser(source: str) -> Query:
    """端到端：源码 → Token → AST"""
    lexer = RegexLexer()
    tokens = lexer.tokenize(source)
    parser = Parser(tokens)
    return parser.parse()

# 测试用例
test_queries = [
    "FROM users SELECT *",
    
    """FROM users
WHERE age > 18
SELECT name, email""",
    
    """FROM orders
WHERE amount >= 100 AND status = "paid"
SELECT id, amount
ORDER BY amount DESC
LIMIT 20""",
    
    # OR条件
    """FROM products
WHERE category = "electronics" OR category = "books"
SELECT name, price""",
]

printer = ASTPrinter()
for query_str in test_queries:
    print(f"\n{'='*50}")
    print(f"输入:\n{query_str}")
    print(f"\nAST:")
    try:
        ast = run_parser(query_str)
        print(printer.print(ast))
    except (LexError, ParseError) as e:
        print(f"错误: {e}")
```

---

## 7.7 性能：解析器的时间复杂度

良好设计的递归下降解析器是 **O(n)** 的（n为Token数量），因为：
- 每个Token最多被消费一次
- 不回溯（第8章介绍的PEG解析器可能回溯）

**退化为O(n²)的情况**：
- 错误恢复中大量的 `synchronize` 调用
- 无限展望（在某些位置扫描任意多个Token来决定分支）

---

## 小结

| 概念 | 要点 |
|------|------|
| 递归下降 | 每条文法规则 = 一个函数，最直观的解析方式 |
| check/match/expect | 三个基础原语，组合出所有解析逻辑 |
| 运算符优先级 | 文法分层（condition→and_expr→comparison）|
| 错误恢复 | 同步点跳过，收集多个错误 |
| LL(1)限制 | 需要展望时，用LL(k)或提取左公因子 |

---

**上一章**：[语法分析与AST](./06-parser-ast.md)
**下一章**：[解析器组合子](./08-parser-combinators.md)

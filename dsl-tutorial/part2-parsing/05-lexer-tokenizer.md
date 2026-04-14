# 第5章：词法分析——把字符串变成Token

## 核心思维模型

> 词法分析是语言处理的**第一道门**：把无结构的字符流切割成有意义的原子单元（Token）。它解决的核心问题是：**这几个字符在一起是什么意思？**

---

## 5.1 为什么需要词法分析？

计算机看到的 DSL 代码只是一个字符串：

```
"FROM users WHERE age > 18 SELECT name"
```

在语法分析器处理之前，需要把这个字符串切分成有意义的单元：

```
["FROM", "users", "WHERE", "age", ">", "18", "SELECT", "name"]
  ↑关键字   ↑标识符  ↑关键字  ↑标识符 ↑运算符 ↑数字   ↑关键字  ↑标识符
```

这个切分过程叫**词法分析（Lexical Analysis）**，执行这个过程的程序叫**词法分析器（Lexer）**，也叫**分词器（Tokenizer）**或**扫描器（Scanner）**。

---

## 5.2 Token的结构

每个Token包含至少三个信息：

```python
from dataclasses import dataclass
from enum import Enum, auto

class TokenType(Enum):
    # QueryLang关键字
    FROM     = auto()
    WHERE    = auto()
    SELECT   = auto()
    ORDER    = auto()
    BY       = auto()
    LIMIT    = auto()
    AND      = auto()
    OR       = auto()
    ASC      = auto()
    DESC     = auto()
    
    # 字面量
    INTEGER  = auto()   # 整数：18, 100
    FLOAT    = auto()   # 浮点：3.14
    STRING   = auto()   # 字符串："active"
    BOOLEAN  = auto()   # 布尔：true, false
    
    # 标识符
    IDENT    = auto()   # 字段名、表名：age, users
    
    # 运算符
    GT       = auto()   # >
    LT       = auto()   # <
    GTE      = auto()   # >=
    LTE      = auto()   # <=
    EQ       = auto()   # =
    NEQ      = auto()   # !=
    
    # 标点符号
    COMMA    = auto()   # ,
    STAR     = auto()   # *
    
    # 特殊
    EOF      = auto()   # 文件结束
    NEWLINE  = auto()   # 换行

@dataclass
class Token:
    type: TokenType
    value: str      # 原始字符串值
    line: int       # 行号（用于错误报告）
    column: int     # 列号（用于错误报告）
    
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"
```

---

## 5.3 手写词法分析器

### 5.3.1 基础结构

```python
class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0          # 当前位置
        self.line = 1         # 当前行号
        self.column = 1       # 当前列号
        self.tokens: list[Token] = []
    
    def current_char(self) -> str | None:
        """当前字符，到达末尾返回None"""
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]
    
    def peek(self, offset: int = 1) -> str | None:
        """向前窥视offset个字符"""
        pos = self.pos + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> str:
        """消费当前字符，移动指针"""
        char = self.source[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def make_token(self, type: TokenType, value: str) -> Token:
        """创建Token（使用当前行列号）"""
        return Token(type, value, self.line, self.column - len(value))
```

### 5.3.2 跳过空白字符

```python
    def skip_whitespace(self):
        """跳过空格、制表符（但保留换行）"""
        while self.current_char() in (' ', '\t', '\r'):
            self.advance()
    
    def skip_comment(self):
        """跳过 -- 单行注释"""
        if self.current_char() == '-' and self.peek() == '-':
            while self.current_char() not in ('\n', None):
                self.advance()
```

### 5.3.3 识别各类Token

```python
    def read_number(self) -> Token:
        """读取整数或浮点数"""
        start_col = self.column
        digits = []
        
        while self.current_char() is not None and self.current_char().isdigit():
            digits.append(self.advance())
        
        # 检查是否是浮点数
        if self.current_char() == '.' and self.peek() and self.peek().isdigit():
            digits.append(self.advance())  # 消费 '.'
            while self.current_char() is not None and self.current_char().isdigit():
                digits.append(self.advance())
            return Token(TokenType.FLOAT, ''.join(digits), self.line, start_col)
        
        return Token(TokenType.INTEGER, ''.join(digits), self.line, start_col)
    
    def read_string(self) -> Token:
        """读取双引号字符串，支持转义"""
        start_col = self.column
        self.advance()  # 消费开引号 "
        chars = []
        
        while self.current_char() is not None and self.current_char() != '"':
            if self.current_char() == '\\':
                self.advance()  # 消费反斜杠
                escape = self.advance()
                escape_map = {'n': '\n', 't': '\t', '"': '"', '\\': '\\'}
                chars.append(escape_map.get(escape, escape))
            else:
                chars.append(self.advance())
        
        if self.current_char() is None:
            raise LexError(f"未闭合的字符串，始于 {self.line}:{start_col}")
        
        self.advance()  # 消费闭引号 "
        return Token(TokenType.STRING, ''.join(chars), self.line, start_col)
    
    # 关键字映射表
    KEYWORDS = {
        'FROM': TokenType.FROM,
        'WHERE': TokenType.WHERE,
        'SELECT': TokenType.SELECT,
        'ORDER': TokenType.ORDER,
        'BY': TokenType.BY,
        'LIMIT': TokenType.LIMIT,
        'AND': TokenType.AND,
        'OR': TokenType.OR,
        'ASC': TokenType.ASC,
        'DESC': TokenType.DESC,
        'true': TokenType.BOOLEAN,
        'false': TokenType.BOOLEAN,
    }
    
    def read_identifier_or_keyword(self) -> Token:
        """读取标识符，判断是否是关键字"""
        start_col = self.column
        chars = []
        
        while self.current_char() is not None and (
            self.current_char().isalnum() or self.current_char() == '_'
        ):
            chars.append(self.advance())
        
        word = ''.join(chars)
        # 关键字不区分大小写（FROM 和 from 等效）
        token_type = self.KEYWORDS.get(word.upper()) or self.KEYWORDS.get(word)
        
        if token_type:
            return Token(token_type, word, self.line, start_col)
        return Token(TokenType.IDENT, word, self.line, start_col)
```

### 5.3.4 主循环

```python
    def tokenize(self) -> list[Token]:
        """主词法分析循环"""
        while self.current_char() is not None:
            self.skip_whitespace()
            self.skip_comment()
            
            char = self.current_char()
            if char is None:
                break
            
            # 换行
            if char == '\n':
                self.tokens.append(self.make_token(TokenType.NEWLINE, '\n'))
                self.advance()
            
            # 数字
            elif char.isdigit():
                self.tokens.append(self.read_number())
            
            # 字符串
            elif char == '"':
                self.tokens.append(self.read_string())
            
            # 标识符或关键字
            elif char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier_or_keyword())
            
            # 运算符（注意先检查双字符运算符）
            elif char == '>' and self.peek() == '=':
                self.advance(); self.advance()
                self.tokens.append(self.make_token(TokenType.GTE, '>='))
            elif char == '<' and self.peek() == '=':
                self.advance(); self.advance()
                self.tokens.append(self.make_token(TokenType.LTE, '<='))
            elif char == '!' and self.peek() == '=':
                self.advance(); self.advance()
                self.tokens.append(self.make_token(TokenType.NEQ, '!='))
            elif char == '>':
                self.advance()
                self.tokens.append(self.make_token(TokenType.GT, '>'))
            elif char == '<':
                self.advance()
                self.tokens.append(self.make_token(TokenType.LT, '<'))
            elif char == '=':
                self.advance()
                self.tokens.append(self.make_token(TokenType.EQ, '='))
            elif char == ',':
                self.advance()
                self.tokens.append(self.make_token(TokenType.COMMA, ','))
            elif char == '*':
                self.advance()
                self.tokens.append(self.make_token(TokenType.STAR, '*'))
            
            else:
                raise LexError(
                    f"未知字符 {char!r} 在 {self.line}:{self.column}"
                )
        
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens
```

---

## 5.4 完整可运行示例

```python
class LexError(Exception):
    pass

# 整合上面所有代码后，测试：
def test_lexer():
    source = '''FROM users
WHERE age > 18 AND status = "active"
SELECT name, email
ORDER BY created_at DESC
LIMIT 10'''
    
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    for token in tokens:
        print(token)

# 输出：
# Token(FROM, 'FROM', 1:1)
# Token(IDENT, 'users', 1:6)
# Token(NEWLINE, '\n', 1:11)
# Token(WHERE, 'WHERE', 2:1)
# Token(IDENT, 'age', 2:7)
# Token(GT, '>', 2:11)
# Token(INTEGER, '18', 2:13)
# Token(AND, 'AND', 2:16)
# Token(IDENT, 'status', 2:20)
# Token(EQ, '=', 2:27)
# Token(STRING, 'active', 2:29)
# ... 以此类推
```

---

## 5.5 正则表达式驱动的词法分析器

手写词法分析器清晰但繁琐。另一种方式是用正则表达式驱动：

```python
import re
from typing import Iterator

class RegexLexer:
    """基于正则表达式规则的词法分析器"""
    
    # 规则：(模式, Token类型)，顺序重要！
    TOKEN_RULES = [
        (r'--[^\n]*',           None),              # 注释，跳过
        (r'[ \t\r]+',           None),              # 空白，跳过
        (r'\n',                 TokenType.NEWLINE),
        (r'\b(?:FROM|from)\b',  TokenType.FROM),
        (r'\b(?:WHERE|where)\b',TokenType.WHERE),
        (r'\b(?:SELECT|select)\b', TokenType.SELECT),
        (r'\b(?:ORDER|order)\b',TokenType.ORDER),
        (r'\b(?:BY|by)\b',      TokenType.BY),
        (r'\b(?:LIMIT|limit)\b',TokenType.LIMIT),
        (r'\b(?:AND|and)\b',    TokenType.AND),
        (r'\b(?:OR|or)\b',      TokenType.OR),
        (r'\b(?:ASC|asc)\b',    TokenType.ASC),
        (r'\b(?:DESC|desc)\b',  TokenType.DESC),
        (r'\btrue\b',           TokenType.BOOLEAN),
        (r'\bfalse\b',          TokenType.BOOLEAN),
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.IDENT),
        (r'\d+\.\d+',           TokenType.FLOAT),
        (r'\d+',                TokenType.INTEGER),
        (r'"(?:[^"\\]|\\.)*"',  TokenType.STRING),
        (r'>=',                 TokenType.GTE),
        (r'<=',                 TokenType.LTE),
        (r'!=',                 TokenType.NEQ),
        (r'>',                  TokenType.GT),
        (r'<',                  TokenType.LT),
        (r'=',                  TokenType.EQ),
        (r',',                  TokenType.COMMA),
        (r'\*',                 TokenType.STAR),
    ]
    
    def __init__(self):
        # 预编译：把所有规则合并成一个大正则（命名组）
        pattern_parts = []
        self.group_names = []
        for i, (pattern, token_type) in enumerate(self.TOKEN_RULES):
            group_name = f"G{i}"
            pattern_parts.append(f"(?P<{group_name}>{pattern})")
            self.group_names.append((group_name, token_type))
        self.master_pattern = re.compile('|'.join(pattern_parts))
    
    def tokenize(self, source: str) -> list[Token]:
        tokens = []
        line = 1
        line_start = 0
        
        for match in self.master_pattern.finditer(source):
            column = match.start() - line_start + 1
            
            # 找到匹配的规则
            for group_name, token_type in self.group_names:
                if match.group(group_name) is not None:
                    value = match.group(group_name)
                    
                    if token_type is None:
                        # 跳过（空白、注释）
                        break
                    
                    if token_type == TokenType.NEWLINE:
                        line += 1
                        line_start = match.end()
                    
                    if token_type == TokenType.STRING:
                        value = value[1:-1]  # 去掉引号
                    
                    tokens.append(Token(token_type, value, line, column))
                    break
            else:
                raise LexError(f"无法匹配字符：{match.group()!r} 在 {line}:{column}")
        
        # 检查是否有未处理的字符
        processed_length = sum(
            len(m.group()) for m in self.master_pattern.finditer(source)
        )
        if processed_length < len(source):
            raise LexError("输入包含无法识别的字符")
        
        tokens.append(Token(TokenType.EOF, '', line, 1))
        return tokens
```

---

## 5.6 有限自动机（DFA）原理

词法分析器的理论基础是**有限自动机（Finite Automaton）**：

```
识别整数或浮点数的DFA：

        digit          digit
  ───────────────→ S1 ──────────────→ S2 (整数接受态)
 │                  │
 │                  │ '.'
 │                  ↓
 │                 S3
 │                  │ digit
 │                  ↓
 │                 S4 ──────────────→ ... (浮点数接受态)
 │                  │ digit
 │                  ↓
 │                 S4 (循环)
```

Python的手写词法分析器实质上就是在**手动实现这个状态机**。正则表达式引擎会自动把正则转换为DFA来执行。

---

## 5.7 性能优化

对于大文件，词法分析器的性能很重要：

```python
# 优化一：生成器模式（惰性求值）
def tokenize_lazy(self) -> Iterator[Token]:
    """生成器版本，不需要一次性将所有Token加载到内存"""
    while self.current_char() is not None:
        # ... 同上，但用 yield 代替 append
        yield self.read_next_token()
    yield Token(TokenType.EOF, '', self.line, self.column)

# 优化二：预分配buffer（避免字符串拼接）
def read_identifier_fast(self) -> Token:
    start = self.pos
    while self.pos < len(self.source) and (
        self.source[self.pos].isalnum() or self.source[self.pos] == '_'
    ):
        self.pos += 1
    # 切片比join快
    word = self.source[start:self.pos]
    ...
```

---

## 5.8 本章实战：为 QueryLang 实现完整Lexer

运行以下代码，验证你的理解：

```python
# complete_lexer.py
# 把上面所有代码整合，测试这些用例

test_cases = [
    # 基础查询
    'FROM users SELECT *',
    
    # 带条件
    'FROM orders WHERE amount >= 100.50 AND status = "paid"',
    
    # 多行查询
    '''FROM users
WHERE age > 18
SELECT name, email
ORDER BY name ASC
LIMIT 10''',
    
    # 错误用例（应该抛出LexError）
    'FROM users WHERE age > @invalid',
]

lexer = RegexLexer()
for source in test_cases:
    print(f"\n--- 输入: {source[:50]}...")
    try:
        tokens = lexer.tokenize(source)
        for t in tokens:
            if t.type != TokenType.EOF:
                print(f"  {t}")
    except LexError as e:
        print(f"  词法错误: {e}")
```

---

## 小结

| 概念 | 要点 |
|------|------|
| Token结构 | 类型 + 原始值 + 位置信息（行/列） |
| 手写Lexer | 字符级状态机，清晰但冗长 |
| 正则Lexer | 规则表驱动，简洁但规则顺序敏感 |
| 关键字处理 | 先匹配为IDENT，再查关键字表 |
| 位置追踪 | 每个Token记录行号和列号，用于错误报告 |
| 性能 | 生成器模式 + 字符串切片代替join |

---

**上一章**：[DSL设计原则](../part1-foundations/04-dsl-design-principles.md)
**下一章**：[语法分析与AST](./06-parser-ast.md)

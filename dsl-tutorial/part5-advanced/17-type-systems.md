# 第17章：DSL类型系统设计

## 核心思维模型

> 类型系统是DSL的**安全网**：在程序执行前，静态地证明某类错误不可能发生。类型系统不是为了给编译器用的，而是为了给**用户**的——它把"运行时崩溃"提前到"编写代码时报错"。

---

## 17.1 为什么DSL需要类型系统？

```
# 没有类型系统的QueryLang
FROM users
WHERE age > "not_a_number"    -- 字符串和数字比较：运行时才发现
SELECT count + name           -- 数字加字符串：语义无意义

# 有类型系统的QueryLang
FROM users
WHERE age > "not_a_number"
          ~~~~~~~~~~~~~~~
          错误：字段'age'类型为int，不能与string比较
          建议：WHERE age > 18
```

类型系统提供：
1. **早期错误发现**：编写DSL时就发现类型错误，而非运行时
2. **IDE自动补全**：知道字段类型，可以提示可用操作符
3. **文档化**：类型注解是可执行的文档
4. **优化依据**：编译器可以根据类型选择最优执行计划

---

## 17.2 类型系统的基础：类型层次

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
from abc import ABC, abstractmethod

# ─── 类型层次定义 ───────────────────────────────────────────

class Type(ABC):
    """所有类型的基类"""
    
    @abstractmethod
    def is_compatible_with(self, other: 'Type') -> bool:
        """类型兼容性检查（宽松：other是否可以用在期望self的地方）"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

@dataclass(frozen=True)
class PrimitiveType(Type):
    name: str  # "int", "float", "string", "bool"
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, PrimitiveType):
            # 数值类型兼容
            if self.name == "float" and other.name == "int":
                return True  # int可以用在期望float的地方
            return self.name == other.name
        return False
    
    def __str__(self) -> str:
        return self.name

@dataclass(frozen=True)
class NullableType(Type):
    """可空类型：T?"""
    inner: Type
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, NullableType):
            return self.inner.is_compatible_with(other.inner)
        # 非空类型可以用在可空位置
        return self.inner.is_compatible_with(other)
    
    def __str__(self) -> str:
        return f"{self.inner}?"

@dataclass(frozen=True)
class ListType(Type):
    """列表类型：List[T]"""
    element_type: Type
    
    def is_compatible_with(self, other: Type) -> bool:
        if isinstance(other, ListType):
            return self.element_type.is_compatible_with(other.element_type)
        return False
    
    def __str__(self) -> str:
        return f"List[{self.element_type}]"

# ─── 预定义基本类型 ─────────────────────────────────────────

INT     = PrimitiveType("int")
FLOAT   = PrimitiveType("float")
STRING  = PrimitiveType("string")
BOOL    = PrimitiveType("bool")
ANY     = PrimitiveType("any")       # 顶类型（任何类型都兼容）
NEVER   = PrimitiveType("never")     # 底类型（不可能的类型）
```

---

## 17.3 类型推断（Type Inference）

类型推断让用户**无需显式写类型**，编译器自动推断：

```python
from querylang.ast_nodes import *

class TypeInferrer:
    """
    QueryLang类型推断器
    
    实现简化版Hindley-Milner类型推断：
    1. 为每个表达式分配类型变量
    2. 根据操作符和上下文生成约束
    3. 用联合算法求解约束
    """
    
    def __init__(self, schema: dict[str, dict[str, Type]]):
        self.schema = schema
        self._type_errors: list[str] = []
        self._inferred_types: dict[int, Type] = {}  # node_id → type
    
    def infer(self, query: Query) -> dict:
        """
        推断整个查询的类型信息
        返回：{node: type} 的映射
        """
        from_table = query.from_clause.table
        if from_table not in self.schema:
            return {}
        
        table_schema = self.schema[from_table]
        
        if query.where_clause:
            self._check_condition_type(query.where_clause.condition, table_schema)
        
        if query.select_clause and not query.select_clause.star:
            for field in query.select_clause.fields:
                if field not in table_schema:
                    self._error(f"SELECT中的字段 '{field}' 不存在")
        
        return self._inferred_types
    
    def _check_condition_type(self, condition, schema: dict) -> Type:
        """检查条件表达式的类型，应该返回bool"""
        match condition:
            case BinaryOp(operator="AND" | "OR", left=l, right=r):
                lt = self._check_condition_type(l, schema)
                rt = self._check_condition_type(r, schema)
                
                if not BOOL.is_compatible_with(lt):
                    self._error(f"AND/OR的左侧应为bool类型，得到 {lt}")
                if not BOOL.is_compatible_with(rt):
                    self._error(f"AND/OR的右侧应为bool类型，得到 {rt}")
                
                return BOOL
            
            case Comparison(field=field, operator=op, value=value_node):
                field_type = schema.get(field, ANY)
                value_type = self._infer_value_type(value_node)
                
                self._check_comparison_type(field, field_type, op, value_type)
                return BOOL
            
            case _:
                return ANY
    
    def _check_comparison_type(self, field, field_type, op, value_type):
        """检查比较运算的类型规则"""
        
        # 规则1：等值比较（=, !=）：两侧类型必须相同
        if op in ("=", "!="):
            if not field_type.is_compatible_with(value_type) and \
               not value_type.is_compatible_with(field_type):
                if field_type != ANY and value_type != ANY:
                    self._error(
                        f"字段 '{field}'（类型: {field_type}）"
                        f"与值（类型: {value_type}）类型不兼容"
                    )
        
        # 规则2：大小比较（>, <, >=, <=）：必须是可排序类型
        orderable = (INT, FLOAT, STRING)
        if op in (">", "<", ">=", "<="):
            if field_type not in orderable and field_type != ANY:
                self._error(
                    f"运算符 '{op}' 要求可排序类型，"
                    f"字段 '{field}' 类型为 {field_type}（不可排序）"
                )
        
        # 规则3：数值字段不能与字符串值比较（除非= !=）
        if field_type == INT and value_type == STRING and op not in ("=", "!="):
            self._error(
                f"不能将整数字段 '{field}' 与字符串值进行 '{op}' 比较"
            )
    
    def _infer_value_type(self, node) -> Type:
        match node:
            case IntegerLiteral(): return INT
            case FloatLiteral(): return FLOAT
            case StringLiteral(): return STRING
            case BooleanLiteral(): return BOOL
            case _: return ANY
    
    def _error(self, message: str):
        self._type_errors.append(message)
    
    def get_errors(self) -> list[str]:
        return list(self._type_errors)
    
    def has_errors(self) -> bool:
        return bool(self._type_errors)
```

---

## 17.4 渐进类型（Gradual Typing）

渐进类型允许DSL**部分有类型**——有Schema时做严格检查，没有时退化为动态类型：

```python
class GradualTypeChecker:
    """
    渐进类型检查器
    
    - 有Schema → 严格检查
    - 部分Schema → 已知字段严格，未知字段宽松
    - 无Schema → any类型，不检查
    """
    
    STRICTNESS_LEVELS = {
        "strict": "所有字段必须在schema中，类型必须严格匹配",
        "normal": "字段必须在schema中，类型宽松兼容",
        "relaxed": "字段不必须在schema中，只检查明显的类型错误",
        "off": "不做任何类型检查",
    }
    
    def __init__(self, schema=None, strictness="normal"):
        self.schema = schema or {}
        self.strictness = strictness
    
    def check(self, ast, table_name: str) -> list[str]:
        errors = []
        
        if self.strictness == "off":
            return errors
        
        if table_name not in self.schema:
            if self.strictness == "strict":
                errors.append(f"表 '{table_name}' 未在schema中定义")
            return errors
        
        table_schema = self.schema[table_name]
        inferrer = TypeInferrer(self.schema)
        inferrer.infer(ast)
        
        return inferrer.get_errors()
```

---

## 17.5 Hindley-Milner 类型推断（简化版）

HM类型推断是函数式语言（Haskell、ML）的类型系统核心。对于高级DSL，实现简化版HM让DSL支持泛型：

```python
# 类型变量（用于泛型推断）
@dataclass
class TypeVar(Type):
    id: int
    _counter = 0
    
    @classmethod
    def fresh(cls) -> 'TypeVar':
        cls._counter += 1
        return cls(cls._counter)
    
    def is_compatible_with(self, other: Type) -> bool:
        return True  # 类型变量与任何类型兼容（求解前）
    
    def __str__(self) -> str:
        return f"'a{self.id}"

# 类型约束
@dataclass
class TypeConstraint:
    left: Type
    right: Type  # left必须等于right

class UnificationSolver:
    """
    类型约束求解器（联合算法/Unification）
    
    给定约束集合，找到使所有约束都满足的类型替换
    """
    
    def __init__(self):
        self.substitution: dict[int, Type] = {}  # TypeVar.id → Type
    
    def unify(self, t1: Type, t2: Type) -> bool:
        """
        尝试统一两个类型
        返回True表示成功，False表示类型冲突
        """
        t1 = self._apply(t1)
        t2 = self._apply(t2)
        
        # 相同类型：成功
        if t1 == t2:
            return True
        
        # 类型变量：绑定到另一个类型
        if isinstance(t1, TypeVar):
            if self._occurs_in(t1, t2):
                return False  # 循环类型：'a = List['a]
            self.substitution[t1.id] = t2
            return True
        
        if isinstance(t2, TypeVar):
            return self.unify(t2, t1)
        
        # 复合类型：递归统一
        if isinstance(t1, ListType) and isinstance(t2, ListType):
            return self.unify(t1.element_type, t2.element_type)
        
        # 类型不兼容
        return False
    
    def _apply(self, t: Type) -> Type:
        """应用当前替换到类型"""
        if isinstance(t, TypeVar) and t.id in self.substitution:
            return self._apply(self.substitution[t.id])
        return t
    
    def _occurs_in(self, var: TypeVar, t: Type) -> bool:
        """检查类型变量是否出现在类型中（防止无限类型）"""
        if isinstance(t, TypeVar):
            return var.id == t.id
        if isinstance(t, ListType):
            return self._occurs_in(var, t.element_type)
        return False
    
    def solve(self, constraints: list[TypeConstraint]) -> dict[int, Type]:
        for c in constraints:
            if not self.unify(c.left, c.right):
                raise TypeError(
                    f"类型约束冲突: {c.left} ≠ {c.right}"
                )
        return self.substitution
```

---

## 17.6 实战：为QueryLang添加计算字段类型

```python
# 扩展QueryLang支持计算字段，并推断其类型
# SELECT name, age * 2 as double_age, price * 0.9 as discounted_price

@dataclass
class ArithExpr:
    """算术表达式（新增AST节点）"""
    left: object
    operator: str  # +, -, *, /
    right: object

class ArithTypeInferrer:
    """推断算术表达式的类型"""
    
    ARITH_RULES = {
        # (left_type, right_type) → result_type
        (INT, INT): INT,
        (INT, FLOAT): FLOAT,
        (FLOAT, INT): FLOAT,
        (FLOAT, FLOAT): FLOAT,
        (STRING, STRING): None,  # 字符串加法需要特殊处理
    }
    
    def infer_arith(self, expr: ArithExpr, schema: dict) -> Type:
        left_type = self._get_type(expr.left, schema)
        right_type = self._get_type(expr.right, schema)
        
        if expr.operator == "+" and left_type == STRING:
            # 字符串拼接：STRING + STRING → STRING
            if right_type == STRING:
                return STRING
            raise TypeError(f"字符串不能与 {right_type} 相加")
        
        # 数值运算
        result = self.ARITH_RULES.get((left_type, right_type))
        if result is None:
            raise TypeError(
                f"不支持的运算: {left_type} {expr.operator} {right_type}"
            )
        return result
    
    def _get_type(self, node, schema) -> Type:
        match node:
            case IntegerLiteral(): return INT
            case FloatLiteral(): return FLOAT
            case StringLiteral(): return STRING
            case str(): return schema.get(node, ANY)  # 字段名
            case ArithExpr(): return self.infer_arith(node, schema)
            case _: return ANY
```

---

## 17.7 类型系统测试

```python
import pytest

TYPED_SCHEMA = {
    "users": {
        "id": INT,
        "name": STRING,
        "age": INT,
        "score": FLOAT,
        "active": BOOL,
    }
}

class TestTypeSystem:
    
    def _check(self, source: str) -> list[str]:
        from querylang.lexer import Lexer
        from querylang.parser import Parser
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        inferrer = TypeInferrer(TYPED_SCHEMA)
        inferrer.infer(ast)
        return inferrer.get_errors()
    
    def test_valid_int_comparison(self):
        errors = self._check('FROM users WHERE age > 18 SELECT name')
        assert not errors
    
    def test_valid_string_comparison(self):
        errors = self._check('FROM users WHERE name = "Alice" SELECT *')
        assert not errors
    
    def test_invalid_type_comparison(self):
        errors = self._check('FROM users WHERE age > "not_a_number" SELECT *')
        assert any("类型不兼容" in e or "类型" in e for e in errors)
    
    def test_ordering_bool_field(self):
        """bool字段不能排序"""
        errors = self._check('FROM users SELECT * ORDER BY active ASC')
        # bool不是可排序类型，但这里取决于是否检查ORDER BY的类型
        # 这是一个设计决策：是否在ORDER BY中也做类型检查
    
    def test_type_variable_unification(self):
        """类型推断测试"""
        solver = UnificationSolver()
        a = TypeVar.fresh()
        b = TypeVar.fresh()
        
        constraints = [
            TypeConstraint(a, INT),
            TypeConstraint(b, a),
        ]
        
        result = solver.solve(constraints)
        # a=INT, b=INT（通过传递性）
        assert solver._apply(b) == INT
```

---

## 小结

| 概念 | 要点 |
|------|------|
| 类型层次 | 基本类型 + 复合类型（可空、列表） |
| 类型推断 | 自动推断无需显式注解，遇到冲突报错 |
| 渐进类型 | 部分类型信息时优雅降级，不强制全Schema |
| HM推断 | 类型变量 + 约束 + 联合算法，支持泛型 |
| 排序/比较规则 | 不同运算符对类型有不同要求 |

---

**上一章**：[解释器模式执行DSL](../part4-external-dsl/16-interpreter-pattern.md)
**下一章**：[错误报告与诊断](./18-error-handling.md)

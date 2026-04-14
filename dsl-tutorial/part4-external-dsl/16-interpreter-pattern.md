# 第16章：解释器模式执行DSL

## 核心思维模型

> 解释器是"活着的AST"：每次执行都是一次树遍历，把AST节点的结构转变为实际操作。相比代码生成，解释器牺牲了性能，但换取了更好的**可移植性**和**运行时灵活性**（动态修改、内省、热重载）。

---

## 16.1 解释器 vs 编译器

```
                    ┌──────────────────┐
                    │     源程序        │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    解析 + 语义    │
                    └────────┬─────────┘
                             │
               ┌─────────────▼──────────────┐
               │                            │
    ┌──────────▼───────┐        ┌──────────▼───────┐
    │   解释器          │        │   代码生成器       │
    │（遍历AST直接执行） │        │（生成目标语言代码） │
    └──────────┬───────┘        └──────────┬───────┘
               │                           │
    ┌──────────▼───────┐        ┌──────────▼───────┐
    │   运行时结果       │        │   目标代码 / 字节码│
    └──────────────────┘        └──────────┬───────┘
                                           │
                                ┌──────────▼───────┐
                                │    运行时结果      │
                                └──────────────────┘

性能：    解释器 < 编译器
灵活性：  解释器 > 编译器
调试：    解释器 > 编译器
可移植：  解释器 > 编译器
```

---

## 16.2 环境（Environment）与作用域

解释器需要一个**环境（Environment）**来存储变量的当前值：

```python
from typing import Any, Optional

class Environment:
    """
    执行环境：存储变量绑定，支持嵌套作用域
    
    这是动态语言解释器的核心数据结构。
    """
    
    def __init__(self, parent: Optional['Environment'] = None):
        self._bindings: dict[str, Any] = {}
        self._parent = parent
    
    def define(self, name: str, value: Any) -> None:
        """在当前环境定义变量"""
        self._bindings[name] = value
    
    def get(self, name: str) -> Any:
        """查找变量，沿作用域链向上"""
        if name in self._bindings:
            return self._bindings[name]
        if self._parent:
            return self._parent.get(name)
        raise NameError(f"未定义的变量: {name!r}")
    
    def set(self, name: str, value: Any) -> None:
        """更新变量（在定义它的作用域中）"""
        if name in self._bindings:
            self._bindings[name] = value
        elif self._parent:
            self._parent.set(name, value)
        else:
            raise NameError(f"未定义的变量: {name!r}")
    
    def child(self) -> 'Environment':
        """创建子作用域"""
        return Environment(parent=self)
    
    def __repr__(self):
        return f"Env({self._bindings})"
```

---

## 16.3 完整的规则引擎解释器

让我们为一个更复杂的DSL——**规则引擎DSL**——构建完整解释器。这个DSL比QueryLang更有代表性，因为它包含变量赋值、条件分支和函数调用。

### 规则引擎DSL语法

```
# RuleLang：业务规则描述语言
rule "premium_discount":
    when:
        user.subscription = "premium"
        AND order.amount > 1000
    then:
        discount = 0.15
        order.final_amount = order.amount * (1 - discount)
        notify(user.email, "您享受了15%的优惠")

rule "first_order_bonus":
    when:
        user.order_count = 0
    then:
        discount = 0.10
        order.final_amount = order.amount * (1 - discount)
```

### 完整解释器

```python
from querylang.ast_nodes import *
from dataclasses import dataclass, field
from typing import Any, Callable

# ─── RuleLang AST节点 ────────────────────────────────────────

@dataclass
class RuleNode:
    name: str
    when_clause: 'ConditionBlock'
    then_clause: 'ActionBlock'

@dataclass
class ConditionBlock:
    condition: Any  # 重用QueryLang的条件节点

@dataclass
class ActionBlock:
    actions: list['Action']

@dataclass
class Assignment:
    """变量赋值：discount = 0.15"""
    target: str           # 变量名（可能是 "order.final_amount" 这样的路径）
    value: 'Expr'

@dataclass
class FunctionCall:
    """函数调用：notify(email, message)"""
    name: str
    args: list['Expr']

@dataclass
class GetField:
    """字段访问：user.email"""
    object_name: str
    field_name: str

@dataclass
class BinaryArith:
    """算术运算：amount * 0.85"""
    left: 'Expr'
    operator: str   # +, -, *, /
    right: 'Expr'


# ─── 解释器 ──────────────────────────────────────────────────

class RuleInterpreter:
    """
    规则引擎解释器
    
    设计特点：
    1. 环境传递：每次规则执行有独立环境
    2. 上下文对象：user/order作为可变上下文
    3. 内置函数：通过函数注册表扩展
    4. 副作用追踪：记录所有修改，支持干运行
    """
    
    def __init__(self):
        self._global_env = Environment()
        self._functions: dict[str, Callable] = {}
        self._audit_log: list[dict] = []
        
        # 注册内置函数
        self._register_builtins()
    
    def _register_builtins(self):
        """注册内置函数"""
        self._functions["notify"] = lambda *args: self._audit_log.append({
            "type": "notification",
            "to": args[0] if args else None,
            "message": args[1] if len(args) > 1 else "",
        })
        
        self._functions["log"] = lambda *args: print("[规则日志]", *args)
        
        self._functions["max"] = lambda *args: max(args)
        self._functions["min"] = lambda *args: min(args)
        self._functions["round"] = lambda v, n=2: round(v, n)
    
    def register_function(self, name: str, func: Callable) -> None:
        """注册自定义函数（扩展机制）"""
        self._functions[name] = func
    
    def execute_rule(self, rule: RuleNode, context: dict) -> dict:
        """
        执行单条规则
        
        Args:
            rule: 规则AST节点
            context: 执行上下文 {"user": {...}, "order": {...}}
        
        Returns:
            修改后的context
        """
        env = self._global_env.child()
        
        # 把context对象注入环境
        for obj_name, obj_value in context.items():
            env.define(obj_name, dict(obj_value))  # 复制，不修改原对象
        
        # 求值when条件
        condition_met = self._eval_condition(rule.when_clause.condition, env)
        
        if condition_met:
            # 执行then动作
            self._execute_actions(rule.then_clause.actions, env)
            self._audit_log.append({
                "type": "rule_fired",
                "rule": rule.name,
                "context_snapshot": {k: env.get(k) for k in context.keys()},
            })
        
        # 返回修改后的context
        return {k: env.get(k) for k in context.keys()}
    
    def execute_rules(self, rules: list[RuleNode], context: dict) -> dict:
        """执行规则集，按顺序执行所有匹配的规则"""
        current_context = context
        for rule in rules:
            current_context = self.execute_rule(rule, current_context)
        return current_context
    
    # ─── 条件求值 ──────────────────────────────────────────
    
    def _eval_condition(self, condition, env: Environment) -> bool:
        match condition:
            case BinaryOp(operator="AND", left=l, right=r):
                # 短路求值
                return self._eval_condition(l, env) and self._eval_condition(r, env)
            
            case BinaryOp(operator="OR", left=l, right=r):
                return self._eval_condition(l, env) or self._eval_condition(r, env)
            
            case Comparison(field=field, operator=op, value=value_node):
                left_val = self._resolve_field(field, env)
                right_val = self._eval_expr(value_node, env)
                return self._compare(left_val, op, right_val)
            
            case _:
                raise RuntimeError(f"未知条件类型: {type(condition)}")
    
    def _resolve_field(self, field: str, env: Environment) -> Any:
        """解析字段引用，支持 user.age 这样的路径"""
        if '.' in field:
            parts = field.split('.', 1)
            obj = env.get(parts[0])
            if isinstance(obj, dict):
                return obj.get(parts[1])
            return getattr(obj, parts[1], None)
        return env.get(field)
    
    def _compare(self, left, op: str, right) -> bool:
        if left is None:
            return op == "=" and right is None
        match op:
            case ">":  return left > right
            case "<":  return left < right
            case ">=": return left >= right
            case "<=": return left <= right
            case "=":  return left == right
            case "!=": return left != right
            case _: raise RuntimeError(f"未知运算符: {op}")
    
    # ─── 动作执行 ──────────────────────────────────────────
    
    def _execute_actions(self, actions: list, env: Environment):
        for action in actions:
            self._execute_action(action, env)
    
    def _execute_action(self, action, env: Environment):
        match action:
            case Assignment(target=target, value=value_node):
                value = self._eval_expr(value_node, env)
                self._set_field(target, value, env)
            
            case FunctionCall(name=name, args=arg_nodes):
                args = [self._eval_expr(a, env) for a in arg_nodes]
                if name not in self._functions:
                    raise RuntimeError(f"未知函数: {name!r}")
                self._functions[name](*args)
            
            case _:
                raise RuntimeError(f"未知动作类型: {type(action)}")
    
    def _set_field(self, target: str, value: Any, env: Environment):
        """设置字段值，支持 order.final_amount 这样的路径"""
        if '.' in target:
            parts = target.split('.', 1)
            obj = env.get(parts[0])
            if isinstance(obj, dict):
                obj[parts[1]] = value
        else:
            try:
                env.set(target, value)
            except NameError:
                env.define(target, value)
    
    # ─── 表达式求值 ────────────────────────────────────────
    
    def _eval_expr(self, node, env: Environment) -> Any:
        match node:
            case IntegerLiteral(value=v): return v
            case FloatLiteral(value=v): return v
            case StringLiteral(value=v): return v
            case BooleanLiteral(value=v): return v
            
            case GetField(object_name=obj, field_name=field):
                o = env.get(obj)
                return o.get(field) if isinstance(o, dict) else getattr(o, field)
            
            case BinaryArith(left=l, operator=op, right=r):
                lv = self._eval_expr(l, env)
                rv = self._eval_expr(r, env)
                match op:
                    case "+": return lv + rv
                    case "-": return lv - rv
                    case "*": return lv * rv
                    case "/":
                        if rv == 0: raise ZeroDivisionError("除以零")
                        return lv / rv
            
            case FunctionCall(name=name, args=arg_nodes):
                args = [self._eval_expr(a, env) for a in arg_nodes]
                if name not in self._functions:
                    raise RuntimeError(f"未知函数: {name!r}")
                return self._functions[name](*args)
            
            case _:
                raise RuntimeError(f"无法求值的表达式: {type(node)}")
    
    # ─── 审计日志 ──────────────────────────────────────────
    
    def get_audit_log(self) -> list[dict]:
        return list(self._audit_log)
    
    def clear_audit_log(self):
        self._audit_log.clear()
```

---

## 16.4 尾调用优化

对于递归定义的DSL（如Lisp风格），需要尾调用优化（TCO）避免栈溢出：

```python
class TailCall(Exception):
    """用异常实现尾调用优化"""
    def __init__(self, func, *args):
        self.func = func
        self.args = args

def trampoline(func, *args):
    """
    蹦床（Trampoline）：将尾递归转换为迭代
    
    函数返回TailCall时，继续执行；返回普通值时，终止
    """
    result = func(*args)
    while isinstance(result, TailCall):
        result = result.func(*result.args)
    return result

# 示例：递归求和（尾调用版本）
def sum_tail(n: int, acc: int = 0):
    if n <= 0:
        return acc
    return TailCall(sum_tail, n - 1, acc + n)

# 使用trampoline，不会栈溢出
result = trampoline(sum_tail, 100000)  # 1 + 2 + ... + 100000
print(result)  # 5000050000
```

---

## 16.5 解释器优化：字节码

当解释器性能成为瓶颈时，可以先编译为字节码，再解释字节码：

```python
from enum import Enum, auto
from dataclasses import dataclass

class Opcode(Enum):
    LOAD_CONST  = auto()  # 压入常量
    LOAD_FIELD  = auto()  # 压入字段值
    STORE_FIELD = auto()  # 弹出值存入字段
    COMPARE     = auto()  # 比较栈顶两个值
    JUMP_IF_FALSE = auto() # 条件跳转
    CALL        = auto()  # 函数调用
    RETURN      = auto()  # 返回

@dataclass
class Instruction:
    opcode: Opcode
    arg: any = None

class BytecodeVM:
    """基于栈的字节码虚拟机（简化版）"""
    
    def __init__(self):
        self.stack: list = []
        self.ip: int = 0  # 指令指针
    
    def execute(self, instructions: list[Instruction], env: dict) -> any:
        self.ip = 0
        while self.ip < len(instructions):
            instr = instructions[self.ip]
            self.ip += 1
            
            match instr.opcode:
                case Opcode.LOAD_CONST:
                    self.stack.append(instr.arg)
                
                case Opcode.LOAD_FIELD:
                    obj_name, field = instr.arg.split('.')
                    self.stack.append(env[obj_name][field])
                
                case Opcode.COMPARE:
                    right = self.stack.pop()
                    left = self.stack.pop()
                    op = instr.arg
                    result = self._compare(left, op, right)
                    self.stack.append(result)
                
                case Opcode.JUMP_IF_FALSE:
                    condition = self.stack.pop()
                    if not condition:
                        self.ip = instr.arg
                
                case Opcode.RETURN:
                    return self.stack.pop() if self.stack else None
        
        return self.stack[-1] if self.stack else None
    
    def _compare(self, left, op, right):
        return {"=": left==right, "!=": left!=right,
                ">": left>right, "<": left<right}[op]
```

---

## 16.6 完整规则引擎测试

```python
def test_rule_interpreter():
    # 构造规则AST（通常由解析器生成）
    premium_rule = RuleNode(
        name="premium_discount",
        when_clause=ConditionBlock(
            BinaryOp(
                "AND",
                Comparison("user.subscription", "=", StringLiteral("premium")),
                Comparison("order.amount", ">", IntegerLiteral(1000))
            )
        ),
        then_clause=ActionBlock([
            Assignment("discount", FloatLiteral(0.15)),
            Assignment("order.final_amount", 
                BinaryArith(
                    GetField("order", "amount"),
                    "*",
                    BinaryArith(IntegerLiteral(1), "-", GetField(None, "discount"))
                )
            ),
        ])
    )
    
    interpreter = RuleInterpreter()
    
    # 测试规则触发
    context = {
        "user": {"id": 1, "subscription": "premium", "email": "vip@test.com"},
        "order": {"id": 101, "amount": 1500.0}
    }
    
    result = interpreter.execute_rule(premium_rule, context)
    assert "final_amount" in result["order"]
    
    # 测试规则不触发（金额不足）
    small_order_context = {
        "user": {"subscription": "premium"},
        "order": {"amount": 500.0}
    }
    result2 = interpreter.execute_rule(premium_rule, small_order_context)
    assert "final_amount" not in result2["order"]
```

---

## 小结

| 概念 | 要点 |
|------|------|
| 环境（Environment） | 变量绑定的容器，支持嵌套作用域 |
| 树遍历解释器 | 直接遍历AST执行，实现简单，性能较低 |
| 尾调用优化 | Trampoline技术，把递归转为迭代 |
| 字节码VM | 解释器性能优化：先编译为字节码，再解释 |
| 副作用追踪 | 审计日志记录所有规则触发和修改 |

---

**上一章**：[代码生成与目标IR](./15-code-generation.md)
**下一章**：[DSL类型系统设计](../part5-advanced/17-type-systems.md)

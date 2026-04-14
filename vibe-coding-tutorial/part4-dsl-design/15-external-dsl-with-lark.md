# 第15章：用 Lark 构建外部 DSL

> "当内部 DSL 的表达力不够时，设计一门自己的语言。"

---

## 15.1 外部 DSL 的适用场景

外部 DSL 适合以下情况：
- 领域专家需要**直接编写**业务规则（不是程序员）
- 规则需要**独立存储**（数据库、配置文件）和**动态加载**
- 语法需要比 Python 更**简洁或更受限制**（防止误用）
- 需要**多语言**支持（用 Python 解析，Java 也能解析）

```python
# 内部 DSL（Python 代码）
rule = (
    Rule("VIP折扣")
        .when(customer.tier == "vip")
        .then(discount(10))
)

# 外部 DSL（独立的语言文件）
# pricing_rules.dsl
"""
rule "VIP折扣"
when customer.tier = "vip"
then discount 10%
end
"""

# 外部 DSL 的优势：
# - 业务分析师可以直接编写和修改
# - 存储在数据库中，无需重新部署
# - 可以被其他系统（如规则引擎）使用
```

---

## 15.2 Lark 基础

```bash
pip install lark
```

### 核心概念

```python
from lark import Lark, Transformer, Token, Tree

# 1. 语法（Grammar）：定义语言的结构
grammar = """
    // 语法规则
    expr: term (("+"|"-") term)*
    term: NUMBER
    
    // 终结符（Terminals）
    NUMBER: /[0-9]+/
    
    // 忽略空白
    %ignore " "
"""

# 2. 解析器（Parser）：将文本转换为解析树
parser = Lark(grammar, start="expr")

# 3. 解析树（Parse Tree）：层次结构的中间表示
tree = parser.parse("1 + 2 + 3")
print(tree.pretty())
# expr
#   term
#     1
#   +
#   term
#     2
#   +
#   term
#     3

# 4. 变换器（Transformer）：将解析树转换为目标对象
class EvalTransformer(Transformer):
    def expr(self, items):
        result = items[0]
        i = 1
        while i < len(items):
            op = str(items[i])
            val = items[i+1]
            if op == "+": result += val
            elif op == "-": result -= val
            i += 2
        return result
    
    def term(self, items):
        return int(items[0])

transformer = EvalTransformer()
result = transformer.transform(tree)  # 6
```

---

## 15.3 实战：构建定价规则 DSL

### 第一步：定义语法

```python
PRICING_DSL_GRAMMAR = """
    // 顶层：规则列表
    start: rule+
    
    // 单条规则
    rule: "rule" STRING "when" condition+ "then" action+ "end"
    
    // 条件
    condition: field_path OPERATOR value
    
    // 动作
    action: "discount" DECIMAL "%"      -> percent_discount
           | "fixed_off" DECIMAL "yuan" -> fixed_discount
           | "free_shipping"            -> free_shipping
    
    // 字段路径
    field_path: IDENTIFIER ("." IDENTIFIER)*
    
    // 值
    value: STRING
         | DECIMAL
         | BOOLEAN
    
    // 操作符
    OPERATOR: "=" | "!=" | ">" | "<" | ">=" | "<="
    
    // 基本终结符
    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    STRING: /\"[^\"]*\"/
    DECIMAL: /[0-9]+(\.[0-9]+)?/
    BOOLEAN: "true" | "false"
    
    // 忽略空白和注释
    %ignore /[ \\t\\n\\r]+/
    %ignore /\\/\\/[^\\n]*/
"""
```

### 第二步：写测试（TDD！）

```python
# tests/unit/dsl/test_pricing_dsl_parser.py
import pytest
from myproject.dsl.pricing import PricingDSLParser

class TestPricingDSLParser:
    
    def setup_method(self):
        self.parser = PricingDSLParser()
    
    def test_parse_simple_percent_discount_rule(self):
        """解析简单的百分比折扣规则"""
        dsl = '''
        rule "VIP折扣"
        when customer.tier = "vip"
        then discount 10%
        end
        '''
        rules = self.parser.parse(dsl)
        
        assert len(rules) == 1
        rule = rules[0]
        assert rule.name == "VIP折扣"
        assert len(rule.conditions) == 1
        assert len(rule.actions) == 1
    
    def test_rule_condition_has_correct_field_and_value(self):
        """条件包含正确的字段路径和值"""
        dsl = '''
        rule "测试"
        when order.total > 500
        then discount 5%
        end
        '''
        rules = self.parser.parse(dsl)
        condition = rules[0].conditions[0]
        
        assert condition.field_path == "order.total"
        assert condition.operator == ">"
        assert condition.value == 500.0
    
    def test_parse_multiple_conditions(self):
        """规则可以有多个条件"""
        dsl = '''
        rule "大额VIP折扣"
        when customer.tier = "vip"
        when order.total > 1000
        then discount 15%
        end
        '''
        rules = self.parser.parse(dsl)
        assert len(rules[0].conditions) == 2
    
    def test_parse_multiple_rules(self):
        """可以解析多条规则"""
        dsl = '''
        rule "规则1"
        when customer.tier = "vip"
        then discount 10%
        end
        
        rule "规则2"
        when order.total > 500
        then fixed_off 50 yuan
        end
        '''
        rules = self.parser.parse(dsl)
        assert len(rules) == 2
    
    def test_parse_free_shipping_action(self):
        """解析免运费动作"""
        dsl = '''
        rule "免运费"
        when order.total > 200
        then free_shipping
        end
        '''
        rules = self.parser.parse(dsl)
        action = rules[0].actions[0]
        assert action.type == "free_shipping"
    
    def test_invalid_syntax_raises_parse_error(self):
        """无效语法抛出解析错误"""
        with pytest.raises(DSLParseError):
            self.parser.parse('rule "没有 end"')
```

### 第三步：实现解析器

```python
# src/myproject/dsl/pricing.py
from lark import Lark, Transformer, Token
from dataclasses import dataclass, field
from typing import List, Any

GRAMMAR = """
    start: rule+
    rule: "rule" STRING "when" condition+ "then" action+ "end"
    condition: field_path OPERATOR value
    action: "discount" DECIMAL "%"      -> percent_discount
           | "fixed_off" DECIMAL "yuan" -> fixed_discount
           | "free_shipping"            -> free_shipping
    field_path: IDENTIFIER ("." IDENTIFIER)*
    value: STRING | DECIMAL | BOOLEAN
    OPERATOR: "=" | "!=" | ">" | "<" | ">=" | "<="
    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    STRING: /"[^"]*"/
    DECIMAL: /[0-9]+(\\.[0-9]+)?/
    BOOLEAN: "true" | "false"
    %ignore /[ \\t\\n\\r]+/
    %ignore /\\/\\/[^\\n]*/
"""

@dataclass
class Condition:
    field_path: str
    operator: str
    value: Any

@dataclass
class Action:
    type: str
    amount: float = 0.0

@dataclass
class PricingRule:
    name: str
    conditions: List[Condition]
    actions: List[Action]
    
    def evaluate(self, context: dict) -> bool:
        """检查所有条件是否满足"""
        return all(self._check_condition(c, context) for c in self.conditions)
    
    def _check_condition(self, condition: Condition, context: dict) -> bool:
        # 从上下文中获取字段值
        value = self._get_field_value(condition.field_path, context)
        target = condition.value
        
        ops = {
            "=": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
        }
        return ops[condition.operator](value, target)
    
    def _get_field_value(self, path: str, context: dict) -> Any:
        parts = path.split(".")
        obj = context
        for part in parts:
            if isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        return obj


class PricingDSLTransformer(Transformer):
    
    def start(self, rules):
        return list(rules)
    
    def rule(self, items):
        # items: [name_token, *conditions, *actions]
        name = str(items[0]).strip('"')
        conditions = [i for i in items[1:] if isinstance(i, Condition)]
        actions = [i for i in items[1:] if isinstance(i, Action)]
        return PricingRule(name=name, conditions=conditions, actions=actions)
    
    def condition(self, items):
        field_path, operator, value = items
        return Condition(
            field_path=str(field_path),
            operator=str(operator),
            value=value
        )
    
    def field_path(self, items):
        return ".".join(str(i) for i in items)
    
    def value(self, items):
        v = items[0]
        if isinstance(v, Token):
            if v.type == "STRING":
                return str(v).strip('"')
            elif v.type == "DECIMAL":
                return float(v)
            elif v.type == "BOOLEAN":
                return str(v) == "true"
        return v
    
    def percent_discount(self, items):
        return Action(type="percent_discount", amount=float(items[0]))
    
    def fixed_discount(self, items):
        return Action(type="fixed_discount", amount=float(items[0]))
    
    def free_shipping(self, items):
        return Action(type="free_shipping")


class PricingDSLParser:
    
    def __init__(self):
        self._parser = Lark(GRAMMAR, start="start", parser="earley")
        self._transformer = PricingDSLTransformer()
    
    def parse(self, dsl_text: str) -> List[PricingRule]:
        try:
            tree = self._parser.parse(dsl_text)
            return self._transformer.transform(tree)
        except Exception as e:
            raise DSLParseError(f"DSL 解析失败：{e}") from e
```

---

## 15.4 DSL 应用引擎

```python
class PricingEngine:
    """将 DSL 规则应用到订单"""
    
    def __init__(self, rules: List[PricingRule]):
        self._rules = rules
    
    @classmethod
    def from_dsl(cls, dsl_text: str) -> 'PricingEngine':
        parser = PricingDSLParser()
        rules = parser.parse(dsl_text)
        return cls(rules)
    
    @classmethod
    def from_file(cls, path: str) -> 'PricingEngine':
        with open(path) as f:
            return cls.from_dsl(f.read())
    
    def apply(self, price: Money, context: dict) -> Money:
        result = price
        for rule in self._rules:
            if rule.evaluate(context):
                result = self._execute_actions(result, rule.actions, context)
        return result
    
    def _execute_actions(self, price: Money, actions: List[Action], context: dict) -> Money:
        result = price
        for action in actions:
            if action.type == "percent_discount":
                factor = Decimal(str(1 - action.amount / 100))
                result = result.multiply(factor)
            elif action.type == "fixed_discount":
                discount = Money(Decimal(str(action.amount)), result.currency)
                result = result.subtract(discount)
            elif action.type == "free_shipping":
                shipping = context.get("shipping_fee", Money(Decimal("0"), result.currency))
                result = result.subtract(shipping)
        return result


# 使用：从文件加载规则，运行时动态更新
engine = PricingEngine.from_file("rules/pricing_rules.dsl")
final_price = engine.apply(
    price=Money(Decimal("200"), "CNY"),
    context={
        "customer": {"tier": "vip"},
        "order": {"total": 200},
        "shipping_fee": Money(Decimal("15"), "CNY")
    }
)
```

---

## 15.5 测试完整解析+执行链

```python
class TestPricingDSLIntegration:
    
    def test_vip_discount_rule_reduces_price(self):
        dsl = '''
        rule "VIP折扣"
        when customer.tier = "vip"
        then discount 10%
        end
        '''
        engine = PricingEngine.from_dsl(dsl)
        context = {"customer": {"tier": "vip"}, "order": {"total": 100}}
        
        result = engine.apply(Money(Decimal("100"), "CNY"), context)
        
        assert result == Money(Decimal("90.00"), "CNY")
    
    def test_rule_not_applied_when_condition_fails(self):
        dsl = '''
        rule "VIP折扣"
        when customer.tier = "vip"
        then discount 10%
        end
        '''
        engine = PricingEngine.from_dsl(dsl)
        context = {"customer": {"tier": "normal"}, "order": {"total": 100}}
        
        result = engine.apply(Money(Decimal("100"), "CNY"), context)
        
        assert result == Money(Decimal("100"), "CNY")  # 未应用折扣
    
    def test_multiple_rules_stack(self):
        dsl = '''
        rule "VIP折扣"
        when customer.tier = "vip"
        then discount 10%
        end
        
        rule "大额优惠"
        when order.total > 500
        then fixed_off 50 yuan
        end
        '''
        engine = PricingEngine.from_dsl(dsl)
        context = {"customer": {"tier": "vip"}, "order": {"total": 600}}
        
        # 600 * 0.9 = 540, 540 - 50 = 490
        result = engine.apply(Money(Decimal("600"), "CNY"), context)
        assert result == Money(Decimal("490.00"), "CNY")
```

---

## 总结

用 Lark 构建外部 DSL 的步骤：
1. **设计语法**：考虑领域专家如何自然表达
2. **TDD 先行**：先写解析测试，再实现解析器
3. **实现 Transformer**：将解析树转换为领域对象
4. **构建引擎**：将 DSL 对象应用到业务逻辑

外部 DSL 的价值在于：**让非程序员也能直接修改业务规则**。

---

**下一章**：[用 TDD 测试 DSL](16-dsl-testing-with-tdd.md)

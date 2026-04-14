# 第16章：用 TDD 测试 DSL

> "DSL 是代码，代码需要测试。测试 DSL 有自己的技艺。"

---

## 16.1 DSL 测试的特殊性

测试 DSL 与测试普通代码的不同：

| 维度 | 普通代码测试 | DSL 测试 |
|------|------------|---------|
| 测试对象 | 函数/类行为 | 语法解析 + 语义执行 |
| 测试用例来源 | 业务规约 | 语法用例 + 语义用例 + 错误处理 |
| 错误类型 | 逻辑错误 | 语法错误 + 语义错误 + 运行时错误 |
| 可读性要求 | 高 | 更高（DSL 本身就是为了可读性）|

### DSL 测试的三层次

```
层次3：语义测试（DSL 是否做了正确的事？）
         ↑
层次2：语法测试（DSL 能否被正确解析？）
         ↑
层次1：词法测试（DSL 的基本元素是否被正确识别？）
```

---

## 16.2 词法测试：验证基本元素

```python
# tests/unit/dsl/test_pricing_dsl_lexer.py
from lark import Lark
import pytest

class TestPricingDSLLexer:
    """词法测试：验证终结符被正确识别"""
    
    def setup_method(self):
        # 用最简单的语法测试词法
        self.lexer = Lark(PRICING_DSL_GRAMMAR, start="start", parser="earley")
    
    def test_recognizes_quoted_string(self):
        """能识别带引号的字符串"""
        dsl = 'rule "VIP折扣" when x = "vip" then discount 10% end'
        tree = self.lexer.parse(dsl)
        # 没有抛出异常即为通过
        assert tree is not None
    
    def test_recognizes_decimal_number(self):
        """能识别小数"""
        dsl = 'rule "测试" when x > 99.9 then discount 5% end'
        tree = self.lexer.parse(dsl)
        assert tree is not None
    
    def test_recognizes_field_path_with_dot(self):
        """能识别带点号的字段路径"""
        dsl = 'rule "测试" when customer.tier = "vip" then discount 5% end'
        tree = self.lexer.parse(dsl)
        assert tree is not None
    
    def test_ignores_whitespace_and_newlines(self):
        """忽略空白和换行"""
        dsl = '''
        rule   "测试"
            when   x = "y"
            then   discount   10%
        end
        '''
        tree = self.lexer.parse(dsl)
        assert tree is not None
    
    def test_ignores_comments(self):
        """忽略注释"""
        dsl = '''
        // 这是注释
        rule "测试"  // 行内注释
        when x = "y"
        then discount 10%
        end
        '''
        tree = self.lexer.parse(dsl)
        assert tree is not None
```

---

## 16.3 语法测试：验证结构

```python
# tests/unit/dsl/test_pricing_dsl_syntax.py

class TestPricingDSLSyntax:
    """语法测试：验证解析结果的结构"""
    
    def setup_method(self):
        self.parser = PricingDSLParser()
    
    # === 有效语法测试 ===
    
    def test_rule_has_name(self):
        dsl = 'rule "VIP折扣" when x = "v" then discount 10% end'
        rules = self.parser.parse(dsl)
        assert rules[0].name == "VIP折扣"
    
    def test_rule_with_single_condition(self):
        dsl = 'rule "R" when customer.tier = "vip" then discount 10% end'
        rules = self.parser.parse(dsl)
        assert len(rules[0].conditions) == 1
    
    def test_rule_with_multiple_conditions(self):
        dsl = '''
        rule "R"
        when customer.tier = "vip"
        when order.total > 100
        then discount 10%
        end
        '''
        rules = self.parser.parse(dsl)
        assert len(rules[0].conditions) == 2
    
    def test_rule_with_multiple_actions(self):
        dsl = '''
        rule "R"
        when x = "y"
        then discount 10%
        then free_shipping
        end
        '''
        rules = self.parser.parse(dsl)
        assert len(rules[0].actions) == 2
    
    def test_multiple_rules_parsed(self):
        dsl = '''
        rule "R1" when x = "y" then discount 10% end
        rule "R2" when a = "b" then fixed_off 50 yuan end
        '''
        rules = self.parser.parse(dsl)
        assert len(rules) == 2
    
    # === 无效语法测试 ===
    
    def test_missing_end_raises_error(self):
        dsl = 'rule "VIP折扣" when x = "v" then discount 10%'
        with pytest.raises(DSLParseError) as exc:
            self.parser.parse(dsl)
        assert "解析失败" in str(exc.value)
    
    def test_missing_when_clause_raises_error(self):
        dsl = 'rule "VIP折扣" then discount 10% end'
        with pytest.raises(DSLParseError):
            self.parser.parse(dsl)
    
    def test_missing_then_clause_raises_error(self):
        dsl = 'rule "VIP折扣" when x = "y" end'
        with pytest.raises(DSLParseError):
            self.parser.parse(dsl)
    
    def test_invalid_operator_raises_error(self):
        dsl = 'rule "R" when x ?? "y" then discount 10% end'
        with pytest.raises(DSLParseError):
            self.parser.parse(dsl)
    
    def test_negative_discount_raises_error(self):
        """负折扣率应该在语义检验时报错"""
        dsl = 'rule "R" when x = "y" then discount -10% end'
        with pytest.raises(DSLParseError):
            self.parser.parse(dsl)
    
    def test_discount_over_100_raises_error(self):
        """折扣率超过100%应该报错"""
        dsl = 'rule "R" when x = "y" then discount 110% end'
        with pytest.raises(DSLSemanticError):
            self.parser.parse(dsl)
```

---

## 16.4 语义测试：验证执行结果

```python
# tests/unit/dsl/test_pricing_dsl_semantics.py

class TestPricingDSLSemantics:
    """语义测试：验证 DSL 执行的业务效果"""
    
    def setup_method(self):
        self.engine = None
    
    def _make_engine(self, dsl: str) -> PricingEngine:
        return PricingEngine.from_dsl(dsl)
    
    def _make_context(self, tier="normal", total=100, coupon=None):
        return {
            "customer": {"tier": tier},
            "order": {"total": total},
            "coupon_code": coupon,
            "shipping_fee": Money(Decimal("15"), "CNY")
        }
    
    # === 折扣动作测试 ===
    
    def test_percent_discount_reduces_by_percentage(self):
        engine = self._make_engine(
            'rule "R" when customer.tier = "vip" then discount 10% end'
        )
        result = engine.apply(
            Money(Decimal("100"), "CNY"),
            self._make_context(tier="vip")
        )
        assert result == Money(Decimal("90.00"), "CNY")
    
    def test_fixed_discount_reduces_by_fixed_amount(self):
        engine = self._make_engine(
            'rule "R" when customer.tier = "vip" then fixed_off 20 yuan end'
        )
        result = engine.apply(
            Money(Decimal("100"), "CNY"),
            self._make_context(tier="vip")
        )
        assert result == Money(Decimal("80.00"), "CNY")
    
    def test_free_shipping_removes_shipping_fee(self):
        engine = self._make_engine(
            'rule "R" when order.total > 200 then free_shipping end'
        )
        context = self._make_context(total=300)
        context["shipping_fee"] = Money(Decimal("15"), "CNY")
        
        base_price = Money(Decimal("315"), "CNY")  # 商品300 + 运费15
        result = engine.apply(base_price, context)
        assert result == Money(Decimal("300.00"), "CNY")
    
    # === 条件测试 ===
    
    def test_condition_equal_matches_correctly(self):
        engine = self._make_engine(
            'rule "R" when customer.tier = "vip" then discount 10% end'
        )
        vip_price = engine.apply(Money(Decimal("100"), "CNY"), self._make_context(tier="vip"))
        normal_price = engine.apply(Money(Decimal("100"), "CNY"), self._make_context(tier="normal"))
        
        assert vip_price == Money(Decimal("90.00"), "CNY")
        assert normal_price == Money(Decimal("100.00"), "CNY")
    
    def test_condition_greater_than_threshold(self):
        engine = self._make_engine(
            'rule "R" when order.total > 500 then discount 5% end'
        )
        high = engine.apply(Money(Decimal("600"), "CNY"), self._make_context(total=600))
        low = engine.apply(Money(Decimal("400"), "CNY"), self._make_context(total=400))
        
        assert high == Money(Decimal("570.00"), "CNY")  # 折扣应用
        assert low == Money(Decimal("400.00"), "CNY")   # 未达到阈值
    
    def test_boundary_condition_exactly_at_threshold(self):
        """边界：刚好等于阈值的情况"""
        engine = self._make_engine(
            'rule "R" when order.total > 500 then discount 5% end'
        )
        # 刚好500，大于500不成立
        exactly_500 = engine.apply(
            Money(Decimal("500"), "CNY"),
            self._make_context(total=500)
        )
        assert exactly_500 == Money(Decimal("500.00"), "CNY")  # 未折扣
    
    # === 多规则叠加测试 ===
    
    def test_multiple_applicable_rules_stack(self):
        """多条适用规则依次叠加"""
        dsl = '''
        rule "VIP" when customer.tier = "vip" then discount 10% end
        rule "大额" when order.total > 500 then fixed_off 50 yuan end
        '''
        engine = PricingEngine.from_dsl(dsl)
        context = self._make_context(tier="vip", total=600)
        
        # 600 * 0.9 = 540, 540 - 50 = 490
        result = engine.apply(Money(Decimal("600"), "CNY"), context)
        assert result == Money(Decimal("490.00"), "CNY")
    
    def test_rule_order_matters_for_stacking(self):
        """规则叠加顺序影响结果"""
        # 先折扣后固定金额
        dsl1 = '''
        rule "先折扣" when x = "y" then discount 10% end
        rule "后减元" when x = "y" then fixed_off 10 yuan end
        '''
        # 先固定金额后折扣
        dsl2 = '''
        rule "先减元" when x = "y" then fixed_off 10 yuan end
        rule "后折扣" when x = "y" then discount 10% end
        '''
        context = {"customer": {"tier": "normal"}, "order": {"total": 100}, "x": "y"}
        price = Money(Decimal("100"), "CNY")
        
        engine1 = PricingEngine.from_dsl(dsl1)
        engine2 = PricingEngine.from_dsl(dsl2)
        
        # 100 * 0.9 - 10 = 80
        assert engine1.apply(price, context) == Money(Decimal("80.00"), "CNY")
        # (100 - 10) * 0.9 = 81
        assert engine2.apply(price, context) == Money(Decimal("81.00"), "CNY")
```

---

## 16.5 属性测试：用 Hypothesis 模糊测试 DSL

```python
from hypothesis import given, strategies as st

class TestPricingDSLPropertyBased:
    """基于属性的测试：发现边界情况"""
    
    @given(
        tier=st.sampled_from(["normal", "vip", "premium"]),
        amount=st.decimals(min_value=0.01, max_value=10000, places=2)
    )
    def test_result_never_exceeds_input_with_discounts(self, tier, amount):
        """折扣后的价格永远不超过原价"""
        dsl = '''
        rule "VIP" when customer.tier = "vip" then discount 10% end
        rule "Premium" when customer.tier = "premium" then discount 20% end
        '''
        engine = PricingEngine.from_dsl(dsl)
        price = Money(amount, "CNY")
        context = {"customer": {"tier": tier}, "order": {"total": float(amount)}}
        
        result = engine.apply(price, context)
        assert result.amount <= price.amount
    
    @given(
        discount_rate=st.floats(min_value=0, max_value=100, allow_nan=False)
    )
    def test_discount_rate_in_valid_range(self, discount_rate):
        """有效范围内的折扣率不应引起错误"""
        dsl = f'rule "R" when x = "y" then discount {discount_rate:.1f}% end'
        try:
            engine = PricingEngine.from_dsl(dsl)
            # 只要解析成功，引擎就是有效的
            assert engine is not None
        except DSLParseError:
            # 某些浮点表示可能导致语法问题，这是可以接受的
            pass
```

---

## 16.6 内部 DSL 的测试

内部 DSL 测试更关注 API 的行为：

```python
class TestFluentOrderDSL:
    """流式订单 DSL 的测试"""
    
    def test_add_single_item(self):
        order = (
            OrderDSL
                .for_customer("c1")
                .add("书", qty=1, price=Money(Decimal("29.9"), "CNY"))
                .place()
                .preview()
        )
        assert order.item_count == 1
    
    def test_add_multiple_items(self):
        order = (
            OrderDSL
                .for_customer("c1")
                .add("书1", qty=1, price=Money(Decimal("29.9"), "CNY"))
                .add("书2", qty=2, price=Money(Decimal("19.9"), "CNY"))
                .place()
                .preview()
        )
        assert order.item_count == 2
        assert order.total == Money(Decimal("69.70"), "CNY")
    
    def test_dsl_is_chainable(self):
        """DSL 支持完整链式调用"""
        builder = OrderDSL.for_customer("c1")
        assert builder is not None
        
        with_items = builder.add("书", qty=1, price=Money(Decimal("10"), "CNY"))
        assert with_items is builder  # 返回 self
        
        placed = with_items.place()
        assert placed is not None
    
    def test_dsl_error_message_is_clear(self):
        """DSL 错误提示清晰"""
        with pytest.raises(EmptyOrderError) as exc:
            OrderDSL.for_customer("c1").place()
        
        # 错误信息应该有业务含义
        assert "空订单" in str(exc.value) or "items" in str(exc.value).lower()
```

---

## 16.7 DSL 测试的最佳实践

### 1. 测试 DSL 文件加载

```python
class TestDSLFileLoading:
    
    def test_loads_rules_from_file(self, tmp_path):
        """从文件加载规则"""
        rule_file = tmp_path / "test_rules.dsl"
        rule_file.write_text('rule "测试" when x = "y" then discount 10% end')
        
        engine = PricingEngine.from_file(str(rule_file))
        assert engine is not None
    
    def test_error_on_file_not_found(self):
        """文件不存在时报告清晰错误"""
        with pytest.raises(FileNotFoundError):
            PricingEngine.from_file("/nonexistent/rules.dsl")
```

### 2. 错误信息测试

```python
class TestDSLErrorMessages:
    
    def test_syntax_error_shows_line_number(self):
        """语法错误显示行号"""
        dsl = """
        rule "正确规则"
        when x = "y"
        then discount 10%
        end
        
        rule "有错误的规则"
        when ???  // 语法错误在第7行
        """
        with pytest.raises(DSLParseError) as exc:
            PricingDSLParser().parse(dsl)
        assert "7" in str(exc.value) or "line" in str(exc.value).lower()
    
    def test_semantic_error_names_the_rule(self):
        """语义错误指出是哪条规则"""
        dsl = 'rule "超出范围" when x = "y" then discount 110% end'
        with pytest.raises(DSLSemanticError) as exc:
            PricingDSLParser().parse(dsl)
        assert "超出范围" in str(exc.value)
```

---

## 总结

DSL 测试分三层：
1. **词法层**：基本符号被正确识别
2. **语法层**：结构被正确解析（含无效语法的错误处理）
3. **语义层**：DSL 执行了正确的业务效果

本章按照从底层到顶层的顺序讲解（词法 → 语法 → 语义），
这是为了帮助读者理解 DSL 解析的分层结构。

在实际 TDD 开发中，也可以从语义层（业务需求）开始，逐步向下推导。
两种顺序各有优势，选择取决于团队对领域的熟悉程度：
- **自底向上**（本章顺序）：适合首次构建 DSL，逐层理解解析机制
- **自顶向下**（Outside-In）：适合领域已明确，从业务效果反推语法设计

---

**下一章**：[TDD 与 DDD 的融合](../part5-integration-mastery/17-tdd-ddd-synthesis.md)

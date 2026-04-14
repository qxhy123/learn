# 第17章：TDD 与 DDD 的融合

> "TDD 给了你安全网，DDD 给了你方向，两者结合给了你前进的底气。"

---

## 17.1 融合的核心思想

TDD 和 DDD 在单独使用时各有局限：

- **TDD 没有 DDD**：测试通过了，但代码和业务脱节，重构时方向不明
- **DDD 没有 TDD**：模型设计得很好，但实现时容易漂移，重构时缺乏安全网

融合后：
- DDD 告诉你**要测试什么**（领域行为、不变量、业务规则）
- TDD 告诉你**如何验证**（先写测试，看到红灯，再实现到绿灯）

```
DDD 提供：          TDD 提供：
- 统一语言          - 明确的"完成"标准
- 聚合根边界   →    - 每个边界的测试覆盖
- 业务规则          - 每条规则的测试用例
- 领域事件          - 事件发布的验证
```

---

## 17.2 DDD 概念到 TDD 测试的映射

### 值对象 → 相等性和不可变性测试

```python
@dataclass(frozen=True)
class EmailAddress:
    value: str

# 测试体现值对象特征
class TestEmailAddress:
    def test_equality_by_value(self):
        """值对象按值相等"""
        assert EmailAddress("a@b.com") == EmailAddress("a@b.com")
    
    def test_different_values_not_equal(self):
        assert EmailAddress("a@b.com") != EmailAddress("c@d.com")
    
    def test_immutable(self):
        """值对象不可变"""
        email = EmailAddress("a@b.com")
        with pytest.raises(AttributeError):
            email.value = "other@example.com"
    
    def test_validates_on_creation(self):
        """创建时自我验证"""
        with pytest.raises(InvalidEmailError):
            EmailAddress("not-an-email")
```

### 实体 → 身份和状态变化测试

```python
class TestOrderItem:
    def test_equality_by_id_not_value(self):
        """实体按 ID 比较，不按值"""
        item1 = OrderItem(id="i1", product="书", quantity=1)
        item2 = OrderItem(id="i2", product="书", quantity=1)  # 相同内容
        item3 = OrderItem(id="i1", product="书", quantity=2)  # 相同ID，不同内容
        
        assert item1 != item2  # 不同 ID = 不同实体
        assert item1 == item3  # 相同 ID = 相同实体（即使内容不同）
    
    def test_state_can_change(self):
        """实体状态可以变化"""
        item = OrderItem(id="i1", product="书", quantity=1)
        item.update_quantity(5)
        assert item.quantity == 5
```

### 聚合根 → 不变量和事件测试

```python
class TestOrderAggregate:
    """聚合根测试的四个维度"""
    
    # 维度1：工厂方法（创建时的不变量）
    def test_cannot_create_order_without_items(self):
        with pytest.raises(EmptyOrderError):
            Order.place(customer=make_customer(), items=[])
    
    # 维度2：命令方法（状态转换的不变量）
    def test_cannot_add_item_after_confirmation(self):
        order = make_confirmed_order()
        with pytest.raises(InvalidOrderStateError):
            order.add_item(make_item())
    
    # 维度3：聚合内部一致性
    def test_total_always_reflects_current_items(self):
        order = make_draft_order(items=[make_item(price=10)])
        assert order.total == Money(Decimal("10"), "CNY")
        
        order.add_item(make_item(price=20))
        assert order.total == Money(Decimal("30"), "CNY")
    
    # 维度4：领域事件
    def test_emits_correct_event_on_state_change(self):
        order = make_draft_order()
        order.confirm()
        
        events = order.pull_events()
        assert any(isinstance(e, OrderConfirmed) for e in events)
```

### 领域服务 → 跨聚合业务规则测试

```python
class TestTransferService:
    """领域服务：跨两个聚合的业务操作"""
    
    def test_transfer_reduces_source_and_increases_target(self):
        source = Account(id="a1", balance=Money(Decimal("100"), "CNY"))
        target = Account(id="a2", balance=Money(Decimal("0"), "CNY"))
        
        TransferService().transfer(
            from_account=source,
            to_account=target,
            amount=Money(Decimal("30"), "CNY")
        )
        
        assert source.balance == Money(Decimal("70"), "CNY")
        assert target.balance == Money(Decimal("30"), "CNY")
    
    def test_transfer_is_atomic_on_failure(self):
        """转账失败时，两个账户都不变"""
        source = Account(id="a1", balance=Money(Decimal("10"), "CNY"))
        target = Account(id="a2", balance=Money(Decimal("0"), "CNY"))
        
        with pytest.raises(InsufficientFundsError):
            TransferService().transfer(source, target, Money(Decimal("100"), "CNY"))
        
        # 两者状态均未变
        assert source.balance == Money(Decimal("10"), "CNY")
        assert target.balance == Money(Decimal("0"), "CNY")
```

---

## 17.3 统一语言驱动测试命名

测试名是最好的活文档——用 DDD 统一语言写：

```python
# ❌ 技术味道太重
def test_status_field_update():
def test_list_append_and_sum():
def test_exception_thrown():

# ✅ 领域语言，读测试名就是读业务需求
def test_vip_customer_receives_double_points_on_purchase():
def test_draft_order_total_updates_when_item_added():
def test_inventory_reservation_fails_when_stock_is_zero():
def test_patron_cannot_borrow_book_when_limit_reached():
```

---

## 17.4 限界上下文边界的测试策略

```python
# 每个上下文有自己的测试根目录
tests/
├── order_context/
│   ├── unit/
│   │   ├── domain/         # 纯领域逻辑（无 Mock）
│   │   └── application/    # 用例（Mock 基础设施）
│   └── integration/        # 跨层测试（真实基础设施）
│
├── catalog_context/
│   ├── unit/
│   └── integration/
│
└── shared/                 # 跨上下文的集成测试
    └── test_order_catalog_integration.py


# 上下文边界验证测试
class TestContextBoundaries:
    
    def test_order_context_uses_catalog_only_through_acl(self):
        """订单上下文只通过防腐层使用商品目录"""
        import ast, pathlib
        
        order_files = list(pathlib.Path("src/order_context").rglob("*.py"))
        acl_path = "src/order_context/acl"
        
        for file in order_files:
            if acl_path in str(file):
                continue  # ACL 本身可以导入 catalog
            
            source = file.read_text()
            assert "catalog_context" not in source, \
                f"{file} 直接导入了 catalog_context，应该通过 ACL"
```

---

## 17.5 Double Loop TDD（双循环 TDD）

结合 DDD 的 Outside-In TDD：

```
外循环（验收测试）：
  DDD 场景 → 验收测试（Red）
  ↓  （整个用例完成后才绿）
  
内循环（单元测试）：  
  TDD 单元测试 → Red → Green → Refactor
  ↓ （每几分钟一次）
  
验收测试变绿 → 外循环完成
```

```python
# 外循环：从用户故事出发的验收测试
class TestUserStory_VIPCustomerOrderDiscount:
    """
    用户故事：
    作为 VIP 客户，
    当我下单时，
    我希望自动获得 10% 折扣，
    以便于享受会员特权。
    """
    
    def test_vip_order_gets_10_percent_discount(self):
        # 这个测试一开始会 Red，直到所有内层组件实现
        app = build_test_app()
        
        response = app.place_order(
            customer_id="vip-customer-001",
            items=[{"product_id": "p1", "qty": 1, "unit_price": 100.0}]
        )
        
        assert response.final_price == 90.0
        assert response.discount_amount == 10.0


# 内循环：TDD 实现具体组件
class TestVIPDiscountRule:
    """内循环单元测试：VIP 折扣规则"""
    
    def test_applies_10_percent_to_vip_orders(self):
        rule = VIPDiscountRule()
        order = make_order(customer_tier="vip", total=100)
        assert rule.apply(order).total == 90
    
    def test_does_not_apply_to_normal_customers(self):
        rule = VIPDiscountRule()
        order = make_order(customer_tier="normal", total=100)
        assert rule.apply(order).total == 100
```

---

## 17.6 重构时 DDD+TDD 的协同

### 重命名（DDD 词汇演化）

```python
# 场景：业务部门将 "User" 改为 "Customer"

# Step 1：更新词汇表（文档）
# docs/ubiquitous-language.md: User → Customer

# Step 2：运行现有测试，确认全绿
pytest  # 全绿 ✓

# Step 3：使用 IDE 全局重命名
# User → Customer（所有引用）

# Step 4：再次运行测试，确认全绿
pytest  # 全绿 ✓ → 重命名完成，没有遗漏
```

### 提取聚合（DDD 战术重构）

```python
# 场景：发现 Order 类太大，需要将 Payment 提取为独立聚合

# Step 1：为 Payment 的行为写测试
class TestPaymentAggregate:
    def test_payment_can_be_processed(self): ...
    def test_payment_can_be_refunded(self): ...

# Step 2：实现 Payment，使测试通过（绿灯）

# Step 3：从 Order 中移除 Payment 相关代码
# 已有 Order 的测试会保护这个重构

# Step 4：测试 Order 和 Payment 的交互
def test_order_attaches_payment_after_processing(): ...
```

---

## 17.7 综合案例：用 TDD+DDD 实现图书馆借阅系统

```python
# 完整的 TDD+DDD 实现流程

# === Step 1：建立统一语言 ===
"""
Patron（读者）、Book（图书）、Copy（副本）、
Borrowing（借阅记录）、Reservation（预约）
"""

# === Step 2：识别聚合根 ===
"""
聚合根：
- Patron（包含：borrowings, reservations）
- Copy（包含：其当前状态）
- Book（包含：copies）
"""

# === Step 3：写 DDD 概念的测试（外循环）===
class TestLibraryBorrowingScenario:
    def test_patron_borrows_available_book(self):
        patron = Patron(id=PatronId("p1"), active_borrowings=[])
        copy = Copy(id=CopyId("c1"), status=CopyStatus.AVAILABLE)
        
        borrowing = LibraryService.borrow(patron, copy)
        
        assert copy.status == CopyStatus.ON_LOAN
        assert borrowing in patron.active_borrowings

# === Step 4：TDD 实现各组件（内循环）===
class TestPatron:
    def test_borrow_increments_active_count(self): ...
    def test_cannot_borrow_more_than_limit(self): ...
    def test_return_decrements_active_count(self): ...

class TestCopy:
    def test_available_copy_can_be_borrowed(self): ...
    def test_on_loan_copy_cannot_be_borrowed(self): ...
    def test_returned_copy_becomes_available(self): ...
```

---

## 总结

TDD 和 DDD 的融合要点：
1. **DDD 驱动测试的语言**：用统一语言命名测试
2. **TDD 验证 DDD 的模型**：每个聚合行为都有测试
3. **双循环**：外循环是用户故事，内循环是单元测试
4. **重构时互补**：DDD 指引方向，TDD 提供安全网

---

**下一章**：[DSL 表达领域语言](18-dsl-domain-expression.md)

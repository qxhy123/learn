# 第23章：DDD测试策略

## 学习目标

- 理解DDD的领域测试金字塔
- 掌握各层次的测试方法和重点
- 学会设计领域规则的表达性测试
- 建立完整的DDD测试体系

---

## 23.1 DDD测试金字塔

传统测试金字塔在DDD中有了新的含义：

```
                    ╱‾‾‾‾‾‾‾‾‾‾╲
                   ╱  E2E测试    ╲         少量，覆盖关键用户旅程
                  ╱──────────────╲
                 ╱  集成测试       ╲        适量，测试上下文边界
                ╱──────────────────╲
               ╱  应用服务测试       ╲      适量，测试用例编排
              ╱──────────────────────╲
             ╱    领域模型测试          ╲   大量！测试业务规则
            ╱────────────────────────────╲

DDD的核心：领域模型测试是最重要的一层
  - 快速（纯Python，无外部依赖）
  - 稳定（不依赖具体实现）
  - 表达性强（读起来像业务规范）
```

---

## 23.2 领域模型测试（最重要）

领域模型测试应该**像业务规则文档**一样可读：

```python
import pytest
from decimal import Decimal
from datetime import datetime

class TestOrder:
    """订单聚合的领域规则测试"""
    
    # ===== 测试数据工厂 =====
    
    def make_order(self, status=None) -> Order:
        order = Order(OrderId(), "customer-001")
        order.add_item(
            ProductSnapshot("p1", "iPhone 15", Money(Decimal("7999"))),
            quantity=1
        )
        if status == "placed":
            order.place()
        elif status == "paid":
            order.place()
            order.mark_paid("payment-001", Money(Decimal("7999")))
        return order
    
    # ===== 下单规则 =====
    
    def test_order_can_be_placed_when_has_items(self):
        """有商品的订单可以下单"""
        order = self.make_order()
        order.place()
        assert order.status == OrderStatus.PLACED
    
    def test_empty_order_cannot_be_placed(self):
        """空订单不能下单"""
        order = Order(OrderId(), "customer-001")
        
        with pytest.raises(OrderException, match="订单.*商品"):
            order.place()
    
    def test_placed_order_cannot_be_placed_again(self):
        """已下单的订单不能再次下单"""
        order = self.make_order(status="placed")
        
        with pytest.raises(OrderException):
            order.place()
    
    # ===== 取消规则 =====
    
    def test_draft_order_can_be_cancelled(self):
        """草稿订单可以取消"""
        order = self.make_order()
        order.cancel("不想买了")
        assert order.status == OrderStatus.CANCELLED
    
    def test_placed_order_can_be_cancelled(self):
        """已下单的订单可以取消"""
        order = self.make_order(status="placed")
        order.cancel("更换商品")
        assert order.status == OrderStatus.CANCELLED
    
    def test_shipped_order_cannot_be_cancelled(self):
        """已发货订单不可取消"""
        order = self.make_order(status="paid")
        order.ship("SF-123456")  # 发货
        
        with pytest.raises(OrderException, match="已发货"):
            order.cancel("不想要了")
    
    # ===== 价格计算规则 =====
    
    def test_order_total_equals_sum_of_items(self):
        """订单总价等于所有商品小计之和"""
        order = Order(OrderId(), "customer-001")
        order.add_item(
            ProductSnapshot("p1", "iPhone", Money(Decimal("7999"))),
            quantity=2
        )
        order.add_item(
            ProductSnapshot("p2", "iPad", Money(Decimal("4999"))),
            quantity=1
        )
        
        expected_total = Decimal("7999") * 2 + Decimal("4999")
        assert order.total.amount == expected_total
    
    def test_order_total_updates_when_item_quantity_changes(self):
        """修改商品数量后总价自动更新"""
        order = Order(OrderId(), "customer-001")
        order.add_item(
            ProductSnapshot("p1", "iPhone", Money(Decimal("7999"))),
            quantity=1
        )
        
        order.update_item_quantity("p1", 2)
        
        assert order.total.amount == Decimal("7999") * 2
    
    # ===== 不变量测试 =====
    
    def test_order_cannot_exceed_max_item_types(self):
        """订单商品种数不能超过上限"""
        order = Order(OrderId(), "customer-001")
        
        for i in range(50):  # 添加50种商品（假设上限是50）
            order.add_item(
                ProductSnapshot(f"p{i}", f"Product {i}", Money(Decimal("10"))),
                quantity=1
            )
        
        with pytest.raises(OrderException, match="超过"):
            order.add_item(
                ProductSnapshot("p51", "One More", Money(Decimal("10"))),
                quantity=1
            )
    
    # ===== 领域事件测试 =====
    
    def test_order_placed_event_is_published_on_place(self):
        """下单时发布 OrderPlaced 事件"""
        order = self.make_order()
        order.place()
        
        events = order.collect_events()
        
        assert len(events) == 1
        assert isinstance(events[0], OrderPlaced)
        assert events[0].order_id == str(order.id)
    
    def test_order_cancelled_event_contains_reason(self):
        """取消事件包含取消原因"""
        order = self.make_order(status="placed")
        reason = "商品缺货"
        order.cancel(reason)
        
        events = order.collect_events()
        cancelled_event = next(e for e in events if isinstance(e, OrderCancelled))
        assert cancelled_event.reason == reason
```

### 值对象测试

```python
class TestMoney:
    """货币值对象测试"""
    
    def test_same_amount_and_currency_are_equal(self):
        assert Money(Decimal("100"), "CNY") == Money(Decimal("100"), "CNY")
    
    def test_different_amounts_are_not_equal(self):
        assert Money(Decimal("100"), "CNY") != Money(Decimal("200"), "CNY")
    
    def test_money_addition(self):
        result = Money(Decimal("100"), "CNY") + Money(Decimal("50"), "CNY")
        assert result == Money(Decimal("150"), "CNY")
    
    def test_cannot_add_different_currencies(self):
        with pytest.raises(ValueError, match="货币"):
            Money(Decimal("100"), "CNY") + Money(Decimal("100"), "USD")
    
    def test_negative_amount_raises_error(self):
        with pytest.raises(ValueError):
            Money(Decimal("-1"), "CNY")
    
    def test_money_is_immutable(self):
        money = Money(Decimal("100"), "CNY")
        with pytest.raises(Exception):
            money.amount = Decimal("200")  # frozen dataclass 不允许修改


class TestDateRange:
    def test_contains_date_within_range(self):
        range_ = DateRange(date(2024, 1, 1), date(2024, 1, 31))
        assert range_.contains(date(2024, 1, 15))
    
    def test_does_not_contain_date_outside_range(self):
        range_ = DateRange(date(2024, 1, 1), date(2024, 1, 31))
        assert not range_.contains(date(2024, 2, 1))
    
    def test_end_before_start_raises_error(self):
        with pytest.raises(ValueError):
            DateRange(date(2024, 1, 31), date(2024, 1, 1))
```

---

## 23.3 应用服务测试

应用服务测试验证用例的编排逻辑，使用内存仓储替代真实数据库：

```python
class TestOrderApplicationService:
    """应用服务测试：验证用例编排，不测试业务规则"""
    
    def setup_method(self):
        self.order_repo = InMemoryOrderRepository()
        self.event_bus = CollectingEventBus()  # 收集事件供断言用
        self.product_catalog = FakeProductCatalog({
            "p1": ProductSnapshot("p1", "iPhone", Money(Decimal("7999")))
        })
        
        self.service = OrderApplicationService(
            order_repo=self.order_repo,
            product_catalog=self.product_catalog,
            event_bus=self.event_bus,
        )
    
    def test_place_order_creates_and_persists_order(self):
        """下单用例：创建订单并持久化"""
        command = PlaceOrderCommand(
            customer_id="cust-001",
            items=[{"product_id": "p1", "quantity": 1}]
        )
        
        order_id = self.service.place_order(command)
        
        # 验证订单被持久化
        saved_order = self.order_repo.get(OrderId(order_id))
        assert saved_order is not None
        assert saved_order.status == OrderStatus.PLACED
    
    def test_place_order_publishes_domain_event(self):
        """下单用例：发布领域事件"""
        command = PlaceOrderCommand(
            customer_id="cust-001",
            items=[{"product_id": "p1", "quantity": 1}]
        )
        
        self.service.place_order(command)
        
        # 验证领域事件被发布
        events = self.event_bus.published_events
        assert any(isinstance(e, OrderPlaced) for e in events)
    
    def test_cancel_nonexistent_order_raises_not_found(self):
        """取消不存在的订单应该抛出 NotFound 异常"""
        with pytest.raises(OrderNotFoundException):
            self.service.cancel_order(CancelOrderCommand(
                order_id="nonexistent-id",
                reason="test"
            ))
    
    def test_cancel_order_updates_persisted_status(self):
        """取消订单后，持久化的状态应该更新"""
        # Given：一个已下单的订单
        order_id = self._create_placed_order()
        
        # When：取消
        self.service.cancel_order(CancelOrderCommand(order_id, "changed mind"))
        
        # Then：持久化状态已更新
        order = self.order_repo.get(OrderId(order_id))
        assert order.status == OrderStatus.CANCELLED
    
    def _create_placed_order(self) -> str:
        command = PlaceOrderCommand(
            customer_id="cust-001",
            items=[{"product_id": "p1", "quantity": 1}]
        )
        return self.service.place_order(command)


# 测试辅助：收集事件的事件总线
class CollectingEventBus(EventBus):
    def __init__(self):
        self.published_events: List[DomainEvent] = []
    
    def publish(self, event: DomainEvent) -> None:
        self.published_events.append(event)
```

---

## 23.4 集成测试

集成测试验证跨层次的集成（如仓储与数据库）：

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture
def db_session():
    """测试数据库会话（使用SQLite内存数据库）"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

class TestSqlAlchemyOrderRepository:
    """仓储集成测试：验证持久化逻辑"""
    
    def test_save_and_get_order(self, db_session):
        repo = SqlAlchemyOrderRepository(db_session)
        
        # Given
        order = Order(OrderId(), "customer-001")
        order.add_item(
            ProductSnapshot("p1", "iPhone", Money(Decimal("7999"))),
            quantity=1
        )
        order.place()
        
        # When
        repo.save(order)
        db_session.flush()
        
        retrieved = repo.get(order.id)
        
        # Then
        assert retrieved.id == order.id
        assert retrieved.status == OrderStatus.PLACED
        assert retrieved.total == order.total
    
    def test_order_not_found_raises_exception(self, db_session):
        repo = SqlAlchemyOrderRepository(db_session)
        
        with pytest.raises(OrderNotFoundException):
            repo.get(OrderId())  # 随机ID，不存在
```

---

## 23.5 测试命名最佳实践

```python
# DDD测试命名模式：
# test_[主语]_[动词/条件]_[结果]
# 或 Given-When-Then 描述

class TestOrderCancellation:
    
    # 推荐：描述性强，读起来像业务规范
    def test_draft_order_can_be_cancelled_by_customer(self): ...
    def test_paid_order_cannot_be_cancelled_after_shipment_started(self): ...
    def test_cancellation_records_reason_in_domain_event(self): ...
    
    # 不推荐：技术性命名
    def test_cancel_method(self): ...
    def test_cancel_raises_exception(self): ...
    def test_order_status_after_cancel(self): ...
```

---

## 23.6 测试覆盖率策略

```python
# 领域层：追求高覆盖率（90%+）
# 每个业务规则都应该有对应的测试

# 应用层：覆盖所有用例的主路径和重要异常路径（80%+）

# 基础设施层：针对关键路径做集成测试

# 不要追求100%覆盖率：
# ❌ 不测试简单的getter/setter
# ❌ 不测试框架本身
# ✅ 测试所有业务规则
# ✅ 测试所有异常路径
# ✅ 测试所有领域事件

# 使用 pytest 的 mark 分类：
@pytest.mark.unit          # 纯单元测试（无外部依赖）
@pytest.mark.integration   # 集成测试（需要数据库等）
@pytest.mark.e2e           # 端到端测试

# CI/CD 策略：
# ├── 每次提交：运行 unit 测试（快，<30秒）
# ├── 每次PR：运行 unit + integration（中速，<5分钟）
# └── 每次发布：运行全部测试（慢，<20分钟）
```

---

## 本章小结

| 测试层次 | 重点 | 依赖 | 数量 |
|---------|------|------|------|
| 领域模型测试 | 业务规则正确性 | 无（纯Python） | 最多 |
| 应用服务测试 | 用例编排逻辑 | 内存仓储/Mock | 适量 |
| 集成测试 | 持久化/外部集成 | 真实数据库 | 少量 |
| E2E测试 | 用户旅程 | 完整系统 | 最少 |

**黄金原则**：领域模型测试是最有价值的，它既快速又直接验证业务规则。

---

**上一章：** [第22章：防腐层](./22-anti-corruption-layer.md)  
**下一章：** [第24章：综合案例——电商系统完整设计](./24-case-study.md)

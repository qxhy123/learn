# 第十七章：六边形架构与 TDD

## 17.1 测试告诉你的设计问题

当你发现测试变得复杂时，通常是架构问题的信号：

```python
# 难以测试的代码：业务逻辑与 I/O 混在一起
def process_order(order_id: int):
    conn = psycopg2.connect("host=prod_db ...")   # 硬编码数据库
    order = conn.execute(f"SELECT * FROM orders WHERE id={order_id}")
    
    if order["total"] > 1000:
        requests.post("https://api.payment.com/charge", ...)  # 硬编码 HTTP
        smtplib.SMTP("smtp.gmail.com").sendmail(...)           # 硬编码邮件

# 要测试这个函数，你需要：
# - 真实数据库（或复杂的 psycopg2 mock）
# - 真实 HTTP（或复杂的 requests mock）
# - 真实 SMTP（或复杂的 smtplib mock）
```

---

## 17.2 六边形架构（Hexagonal Architecture）

由 Alistair Cockburn 提出，又称**端口与适配器（Ports & Adapters）**：

```
        ┌─────────────────────────────────────────────┐
        │                                             │
        │              应用核心（Domain）              │
        │         纯 Python，无 I/O，可直接测试       │
        │                                             │
        │   ┌──────────┐        ┌──────────┐          │
        │   │ Port（接口）│      │ Port（接口）│        │
        └───┤ 入站端口  ├────────┤ 出站端口  ├─────────┘
            └────┬─────┘        └─────┬────┘
                 │                    │
            ┌────┴─────┐        ┌─────┴────┐
            │ Adapter  │        │ Adapter  │
            │（HTTP层）│        │（DB 层） │
            └──────────┘        └──────────┘
```

**关键原则**：
- **内层（Domain）** 不依赖任何外部框架或 I/O
- **外层（Adapter）** 实现 Port 定义的接口
- 测试时用 Fake/Stub 替换 Adapter

---

## 17.3 TDD 驱动出六边形架构

让我们通过 TDD 来**发现**这个架构，而不是提前设计它。

### 需求：订单处理系统

```
- 加载订单数据
- 如果总价 > 1000，触发支付
- 发送确认通知
- 返回处理结果
```

### 第一轮：先写测试，让接口浮现

```python
class TestOrderProcessor(unittest.TestCase):

    def test_high_value_order_triggers_payment(self):
        # 测试迫使我们思考：OrderProcessor 需要什么依赖？
        # → 它需要能"加载订单"和"触发支付"的东西
        # → 这两个"东西"就是 Port（接口）
        
        order_repo = MagicMock()   # ← Port: 订单存储
        payment_port = MagicMock() # ← Port: 支付服务
        notifier = MagicMock()     # ← Port: 通知服务
        
        order_repo.find_by_id.return_value = Order(id=1, total=1500.0)
        payment_port.charge.return_value = {"status": "ok", "ref": "p_123"}
        
        processor = OrderProcessor(
            orders=order_repo,
            payment=payment_port,
            notifier=notifier,
        )
        result = processor.process(order_id=1)
        
        payment_port.charge.assert_called_once_with(order_id=1, amount=1500.0)
        self.assertEqual(result.payment_ref, "p_123")
```

测试写完，Port 接口自然浮现了。

### 第二轮：定义 Port 接口

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    id: int
    total: float
    user_email: str = ""


@dataclass
class ProcessResult:
    order_id: int
    payment_ref: Optional[str]
    notified: bool


# 出站端口（Driven Ports）
class OrderRepository(ABC):
    @abstractmethod
    def find_by_id(self, order_id: int) -> Optional[Order]: ...

class PaymentService(ABC):
    @abstractmethod
    def charge(self, order_id: int, amount: float) -> dict: ...

class NotificationService(ABC):
    @abstractmethod
    def send(self, email: str, message: str) -> bool: ...
```

### 第三轮：实现应用核心（纯逻辑，无 I/O）

```python
class OrderProcessor:
    """应用核心：纯业务逻辑，依赖注入所有外部接口"""

    PAYMENT_THRESHOLD = 1000.0

    def __init__(
        self,
        orders: OrderRepository,
        payment: PaymentService,
        notifier: NotificationService,
    ):
        self._orders = orders
        self._payment = payment
        self._notifier = notifier

    def process(self, order_id: int) -> ProcessResult:
        order = self._orders.find_by_id(order_id)
        if order is None:
            raise LookupError(f"Order {order_id} not found")

        payment_ref = None
        if order.total > self.PAYMENT_THRESHOLD:
            result = self._payment.charge(
                order_id=order.id,
                amount=order.total,
            )
            payment_ref = result.get("ref")

        notified = False
        if order.user_email:
            notified = self._notifier.send(
                email=order.user_email,
                message=f"Order {order_id} processed. Amount: ${order.total:.2f}",
            )

        return ProcessResult(
            order_id=order_id,
            payment_ref=payment_ref,
            notified=notified,
        )
```

---

## 17.4 完整的测试套件

```python
class TestOrderProcessorUnit(unittest.TestCase):
    """纯单元测试：全用 Mock/Fake，极快"""

    def setUp(self):
        self.mock_orders = MagicMock(spec=OrderRepository)
        self.mock_payment = MagicMock(spec=PaymentService)
        self.mock_notifier = MagicMock(spec=NotificationService)
        self.processor = OrderProcessor(
            orders=self.mock_orders,
            payment=self.mock_payment,
            notifier=self.mock_notifier,
        )

    def test_high_value_order_triggers_payment(self):
        self.mock_orders.find_by_id.return_value = Order(1, total=1500.0)
        self.mock_payment.charge.return_value = {"ref": "p_001", "status": "ok"}

        result = self.processor.process(1)

        self.mock_payment.charge.assert_called_once_with(order_id=1, amount=1500.0)
        self.assertEqual(result.payment_ref, "p_001")

    def test_low_value_order_skips_payment(self):
        self.mock_orders.find_by_id.return_value = Order(2, total=50.0)

        result = self.processor.process(2)

        self.mock_payment.charge.assert_not_called()
        self.assertIsNone(result.payment_ref)

    def test_order_with_email_sends_notification(self):
        self.mock_orders.find_by_id.return_value = Order(
            3, total=200.0, user_email="user@example.com"
        )
        self.mock_notifier.send.return_value = True

        result = self.processor.process(3)

        self.mock_notifier.send.assert_called_once_with(
            email="user@example.com",
            message=unittest.mock.ANY,
        )
        self.assertTrue(result.notified)

    def test_nonexistent_order_raises(self):
        self.mock_orders.find_by_id.return_value = None

        with self.assertRaises(LookupError):
            self.processor.process(999)

    def test_payment_threshold_is_exclusive(self):
        """总价恰好等于阈值时不触发支付"""
        self.mock_orders.find_by_id.return_value = Order(4, total=1000.0)

        self.processor.process(4)

        self.mock_payment.charge.assert_not_called()
```

---

## 17.5 Fake Adapter：内存实现

```python
class FakeOrderRepository(OrderRepository):
    """Fake：内存实现，用于集成测试"""

    def __init__(self):
        self._orders: dict[int, Order] = {}

    def save(self, order: Order) -> Order:
        self._orders[order.id] = order
        return order

    def find_by_id(self, order_id: int) -> Optional[Order]:
        return self._orders.get(order_id)


class FakePaymentService(PaymentService):
    def __init__(self):
        self.charges: list[dict] = []
        self._should_fail = False

    def charge(self, order_id: int, amount: float) -> dict:
        if self._should_fail:
            raise RuntimeError("Payment gateway unavailable")
        ref = f"fake_p_{len(self.charges) + 1}"
        self.charges.append({"order_id": order_id, "amount": amount, "ref": ref})
        return {"ref": ref, "status": "ok"}

    def simulate_failure(self):
        self._should_fail = True


class FakeNotificationService(NotificationService):
    def __init__(self):
        self.sent: list[dict] = []

    def send(self, email: str, message: str) -> bool:
        self.sent.append({"email": email, "message": message})
        return True


class TestOrderProcessorIntegration(unittest.TestCase):
    """集成测试：用 Fake 替代真实 Adapter"""

    def setUp(self):
        self.orders = FakeOrderRepository()
        self.payment = FakePaymentService()
        self.notifier = FakeNotificationService()
        self.processor = OrderProcessor(
            orders=self.orders,
            payment=self.payment,
            notifier=self.notifier,
        )
        # 预置测试数据
        self.orders.save(Order(1, total=1500.0, user_email="alice@ex.com"))
        self.orders.save(Order(2, total=50.0,   user_email="bob@ex.com"))

    def test_full_processing_flow(self):
        result = self.processor.process(1)

        # 验证状态（Fake 暴露了内部状态）
        self.assertEqual(len(self.payment.charges), 1)
        self.assertEqual(self.payment.charges[0]["amount"], 1500.0)
        self.assertEqual(len(self.notifier.sent), 1)
        self.assertEqual(self.notifier.sent[0]["email"], "alice@ex.com")
        self.assertIsNotNone(result.payment_ref)

    def test_payment_failure_propagates(self):
        self.payment.simulate_failure()

        with self.assertRaises(RuntimeError):
            self.processor.process(1)

        # 通知未发送（支付失败后应中断）
        self.assertEqual(len(self.notifier.sent), 0)
```

---

## 17.6 真实 Adapter 实现（生产代码）

```python
import sqlite3
import requests

class SQLiteOrderRepository(OrderRepository):
    """真实实现：SQLite 适配器"""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    def find_by_id(self, order_id: int) -> Optional[Order]:
        row = self.conn.execute(
            "SELECT id, total, user_email FROM orders WHERE id=?",
            (order_id,)
        ).fetchone()
        if row:
            return Order(id=row[0], total=row[1], user_email=row[2])
        return None


class StripePaymentService(PaymentService):
    """真实实现：Stripe 支付适配器"""

    def __init__(self, api_key: str):
        self._api_key = api_key

    def charge(self, order_id: int, amount: float) -> dict:
        response = requests.post(
            "https://api.stripe.com/v1/charges",
            auth=(self._api_key, ""),
            data={"amount": int(amount * 100), "currency": "usd"},
        )
        response.raise_for_status()
        return response.json()
```

---

## 17.7 测试策略金字塔

```
         /\
        /  \        E2E: 少量，使用真实 Adapter
       /    \       测试完整业务流程
      /──────\
     /        \     集成: 适量，使用 Fake Adapter
    /          \    验证 Adapter 符合 Port 契约
   /────────────\
  /              \  单元: 大量，使用 Mock
 /                \ 测试核心业务逻辑
/──────────────────\
```

### 契约测试（Contract Tests）

验证 Fake 与真实 Adapter 行为一致：

```python
class OrderRepositoryContractTest:
    """抽象契约：所有 OrderRepository 实现都必须通过"""

    def get_repo(self) -> OrderRepository:
        raise NotImplementedError

    def test_find_saved_order(self):
        repo = self.get_repo()
        order = Order(id=1, total=100.0)
        # 假设实现有 save 方法
        repo.save(order)
        found = repo.find_by_id(1)
        self.assertIsNotNone(found)
        self.assertEqual(found.total, 100.0)

    def test_find_nonexistent_returns_none(self):
        repo = self.get_repo()
        self.assertIsNone(repo.find_by_id(9999))


class TestFakeOrderRepository(OrderRepositoryContractTest, unittest.TestCase):
    def get_repo(self):
        return FakeOrderRepository()


# 针对真实 DB 的测试（只在 CI 环境运行）
@unittest.skipUnless(os.environ.get("RUN_DB_TESTS"), "需要 DB 环境")
class TestSQLiteOrderRepository(OrderRepositoryContractTest, unittest.TestCase):
    def get_repo(self):
        return SQLiteOrderRepository(":memory:")
```

---

## 17.8 六边形架构的 TDD 工作流

```
1. 写测试（用 Mock 驱动出 Port 接口）
      ↓
2. 实现应用核心（纯逻辑，依赖注入）
      ↓
3. 写 Fake Adapter（用于集成测试）
      ↓
4. 写契约测试（验证 Fake 行为正确）
      ↓
5. 写真实 Adapter（匹配 Port 接口）
      ↓
6. 真实 Adapter 自动通过契约测试
```

---

## 17.9 本章小结

- **测试难写**是设计问题的信号，驱使你抽象 Port
- 六边形架构：Domain 纯净 → Port 定义契约 → Adapter 实现 I/O
- TDD 工作流自然导出三类测试：单元（Mock）、集成（Fake）、契约（验证 Fake 与真实一致）
- Fake Adapter 优于 Mock：有真实逻辑，暴露内部状态，更接近真实行为
- 契约测试确保 Fake 和真实 Adapter 行为一致，防止"Fake 测试通过但真实实现有 bug"

**下一章**：遗留代码的 TDD 策略——在没有测试网的情况下安全重构。

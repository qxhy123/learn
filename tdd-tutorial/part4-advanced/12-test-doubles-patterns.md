# 第十二章：测试替身模式

## 12.1 测试替身分类

Gerard Meszaros 在《xUnit Test Patterns》中定义了五种测试替身：

```
Test Double（测试替身）
├── Dummy    # 占位符，只是为了满足接口，从不被使用
├── Stub     # 返回预设数据，不验证调用
├── Spy      # 记录调用，事后可验证
├── Mock     # 预先设定期望，验证交互
└── Fake     # 真实实现的简化版（如内存数据库）
```

理解区别能帮你选择最合适的工具，写出更清晰的测试意图。

---

## 12.2 Dummy：占位符

Dummy 仅用于填充参数，测试中从不会真正使用它：

```python
class TestOrderService(unittest.TestCase):

    def test_order_id_is_generated(self):
        # logger 是必须参数，但这个测试不关心日志
        dummy_logger = MagicMock()   # Dummy：占位，不做断言

        service = OrderService(logger=dummy_logger)
        order = service.create_order(user_id=1)

        self.assertIsNotNone(order.id)
        # 注意：我们没有对 dummy_logger 做任何断言

    def test_user_required_fields(self):
        # 这里 email 是必须参数，但测试只关心 name
        dummy_email = "dummy@example.com"   # Dummy
        user = User(name="Alice", email=dummy_email)
        self.assertEqual(user.name, "Alice")
```

---

## 12.3 Stub：返回预设数据

Stub 的目的是提供"合适的返回值"，不验证调用：

```python
class TestPricingEngine(unittest.TestCase):

    def test_total_price_includes_tax(self):
        # Stub：预设返回值，不关心是否被调用
        tax_service_stub = MagicMock()
        tax_service_stub.get_tax_rate.return_value = 0.1  # 固定返回 10%

        engine = PricingEngine(tax_service=tax_service_stub)
        total = engine.calculate_total(base_price=100.0)

        self.assertAlmostEqual(total, 110.0)
        # 注意：没有 assert_called 这类断言 → 这是 Stub 而非 Mock

    def test_price_with_no_tax_zone(self):
        no_tax_stub = MagicMock()
        no_tax_stub.get_tax_rate.return_value = 0.0

        engine = PricingEngine(tax_service=no_tax_stub)
        total = engine.calculate_total(base_price=100.0)

        self.assertAlmostEqual(total, 100.0)
```

### Stub vs Mock 的关键区别

```python
# Stub：只关心返回值（状态验证）
stub = MagicMock()
stub.get_rate.return_value = 0.1
result = engine.calculate(stub)
assert result == expected_value   # 验证状态

# Mock：关心交互（行为验证）
mock = MagicMock()
engine.notify(mock)
mock.send.assert_called_once_with("notification")  # 验证行为
```

---

## 12.4 Spy：记录调用

Spy 允许你在事后检查调用，而不预先设定期望：

```python
class TestEventEmitter(unittest.TestCase):

    def test_subscribers_notified_of_event(self):
        # Spy：记录所有调用
        received_events = []

        def spy_handler(event):
            received_events.append(event)

        emitter = EventEmitter()
        emitter.subscribe("user.created", spy_handler)
        emitter.emit("user.created", {"id": 1, "name": "Alice"})

        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0]["name"], "Alice")

    def test_spy_with_magicmock(self):
        """MagicMock 自带 Spy 能力"""
        spy = MagicMock(wraps=real_send_email)   # wraps=真实函数

        mailer = Mailer(send_func=spy)
        mailer.send_welcome("user@example.com")

        # 验证调用（Spy 行为）
        spy.assert_called_once()
        # 真实函数也被执行了（与纯 Mock 不同）
```

---

## 12.5 Mock：预设期望

Mock 在调用前设定期望，测试是否完全符合预期交互：

```python
class TestPaymentOrchestrator(unittest.TestCase):

    def test_complete_payment_flow(self):
        # Mock：预设期望的调用序列
        mock_gateway = MagicMock(spec=PaymentGateway)
        mock_gateway.charge.return_value = {"transaction_id": "tx_123", "status": "success"}

        mock_notifier = MagicMock(spec=NotificationService)

        orchestrator = PaymentOrchestrator(
            gateway=mock_gateway,
            notifier=mock_notifier
        )
        orchestrator.process(user_id=1, amount=99.99)

        # 验证交互（Mock 的核心）
        mock_gateway.charge.assert_called_once_with(user_id=1, amount=99.99)
        mock_notifier.send_receipt.assert_called_once_with(
            user_id=1,
            transaction_id="tx_123"
        )
```

---

## 12.6 Fake：真实实现的简化版

Fake 有真实的业务逻辑，只是在某些方面做了简化（如内存存储代替数据库）：

```python
# 生产代码：真实仓库（依赖数据库）
class UserRepository:
    def __init__(self, db):
        self.db = db

    def save(self, user):
        self.db.execute("INSERT INTO users ...", user)
        return user

    def find_by_id(self, user_id):
        row = self.db.execute("SELECT ... WHERE id=?", user_id)
        return User.from_row(row) if row else None


# Fake：内存实现（有真实逻辑，无 I/O）
class FakeUserRepository:
    def __init__(self):
        self._users = {}
        self._next_id = 1

    def save(self, user):
        user.id = self._next_id
        self._next_id += 1
        self._users[user.id] = user
        return user

    def find_by_id(self, user_id):
        return self._users.get(user_id)

    def find_all(self):
        return list(self._users.values())


# 使用 Fake 的测试
class TestUserService(unittest.TestCase):

    def setUp(self):
        self.repo = FakeUserRepository()   # Fake
        self.service = UserService(repo=self.repo)

    def test_register_user(self):
        user = self.service.register("Alice", "alice@example.com")
        self.assertIsNotNone(user.id)

    def test_find_registered_user(self):
        self.service.register("Bob", "bob@example.com")
        found = self.service.find_user_by_email("bob@example.com")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Bob")
```

**Fake vs Mock 的适用场景**：
- Fake：业务逻辑涉及存储、查询的复杂交互，Mock 难以模拟
- Mock：只需验证某方法被正确调用，不需要真实逻辑

---

## 12.7 测试替身选择指南

```
我需要什么？
│
├── 参数占位，不使用       → Dummy
│
├── 控制返回值（状态验证）  → Stub
│   │
│   └── 还需要记录调用？   → Spy
│
├── 验证方法被正确调用     → Mock
│   （行为验证）
│
└── 需要真实逻辑但无 I/O   → Fake
```

---

## 12.8 组合使用：复杂场景

```python
class TestCheckoutFlow(unittest.TestCase):

    def setUp(self):
        # Fake：内存购物车（有真实逻辑）
        self.cart = InMemoryCart()
        self.cart.add_item("apple", 3.0)
        self.cart.add_item("banana", 1.5)

        # Stub：税率服务（只需要返回值）
        self.tax_service = MagicMock()
        self.tax_service.get_rate.return_value = 0.08

        # Mock：支付网关（需要验证调用）
        self.payment_gateway = MagicMock(spec=PaymentGateway)
        self.payment_gateway.charge.return_value = {"status": "ok"}

        # Dummy：日志器（不关心）
        self.logger = MagicMock()

        self.checkout = CheckoutService(
            cart=self.cart,
            tax=self.tax_service,
            gateway=self.payment_gateway,
            logger=self.logger,
        )

    def test_checkout_charges_correct_amount_with_tax(self):
        self.checkout.complete(user_id=1, payment_method="card_123")

        # 验证 Mock（支付网关）被正确调用
        self.payment_gateway.charge.assert_called_once_with(
            user_id=1,
            method="card_123",
            amount=unittest.mock.ANY  # 先验证调用，精确金额另外测
        )

        actual_amount = self.payment_gateway.charge.call_args.kwargs['amount']
        self.assertAlmostEqual(actual_amount, 4.5 * 1.08, places=2)
```

---

## 12.9 本章小结

| 替身类型 | 核心特征 | 验证方式 | 适用场景 |
|----------|----------|----------|----------|
| Dummy | 不使用 | 无 | 满足接口签名 |
| Stub | 返回预设值 | 状态验证 | 控制依赖的返回值 |
| Spy | 记录调用 | 事后验证 | 需要观察副作用 |
| Mock | 预设期望 | 行为验证 | 关键交互必须发生 |
| Fake | 有真实逻辑 | 状态验证 | 复杂依赖的内存替代 |

**下一章**：异步代码的测试——`asyncio` 与 `unittest` 的协作。

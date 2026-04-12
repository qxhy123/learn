# 第二十章：TDD 驱动出设计模式

## 20.1 TDD 与设计模式的关系

设计模式不是"发明"出来的，而是从重复的 TDD 问题中**浮现**出来的。当你持续地：
- 发现"Mock 太多了"
- 发现"修改一处破坏多个测试"
- 发现"测试难以构造对象"

...你就会自然地走向经典设计模式。

本章展示几个重要模式如何从 TDD 中自然浮现。

---

## 20.2 Strategy 模式：从"难以 Mock"中浮现

### 问题：条件分支难以测试

```python
# 难以测试：支付逻辑硬编码在类内部
class Checkout:
    def pay(self, amount, method):
        if method == "stripe":
            import stripe
            stripe.Charge.create(amount=amount)
        elif method == "paypal":
            import paypal
            paypal.Payment.create(amount=amount)
        elif method == "crypto":
            import blockchain_sdk
            blockchain_sdk.transfer(amount)
```

测试时需要 Mock `stripe`、`paypal`、`blockchain_sdk` 三个库——且每加一种支付方式，测试就更复杂。

### TDD 驱动：先写测试，让 Strategy 浮现

```python
class TestCheckout(unittest.TestCase):

    def test_pay_delegates_to_payment_strategy(self):
        # 测试迫使我们思考：Checkout 需要什么接口？
        # → 一个"可以执行支付"的对象
        # → 这就是 Strategy！

        mock_strategy = MagicMock()
        checkout = Checkout(payment_strategy=mock_strategy)
        checkout.pay(100.0)

        mock_strategy.execute.assert_called_once_with(100.0)

    def test_pay_returns_strategy_result(self):
        mock_strategy = MagicMock()
        mock_strategy.execute.return_value = {"ref": "tx_001"}
        checkout = Checkout(payment_strategy=mock_strategy)

        result = checkout.pay(100.0)
        self.assertEqual(result["ref"], "tx_001")
```

### 实现 Strategy 模式

```python
from abc import ABC, abstractmethod


class PaymentStrategy(ABC):
    @abstractmethod
    def execute(self, amount: float) -> dict: ...


class StripePayment(PaymentStrategy):
    def execute(self, amount: float) -> dict:
        # 真实 Stripe 实现
        return {"ref": f"stripe_{amount}"}


class PayPalPayment(PaymentStrategy):
    def execute(self, amount: float) -> dict:
        return {"ref": f"paypal_{amount}"}


class Checkout:
    def __init__(self, payment_strategy: PaymentStrategy):
        self._strategy = payment_strategy

    def pay(self, amount: float) -> dict:
        return self._strategy.execute(amount)
```

### 每种 Strategy 独立测试

```python
class TestStripePayment(unittest.TestCase):
    """Strategy 可以独立测试，不涉及 Checkout"""

    @patch('myapp.payment.stripe')
    def test_stripe_creates_charge(self, mock_stripe):
        mock_stripe.Charge.create.return_value = {"id": "ch_001"}
        strategy = StripePayment(api_key="sk_test")
        result = strategy.execute(99.0)
        mock_stripe.Charge.create.assert_called_once()
```

---

## 20.3 Observer 模式：从"通知难以测试"中浮现

### 问题：通知逻辑硬编码

```python
class Order:
    def place(self):
        self._save_to_db()
        send_email(self.user_email, "Order placed")    # 硬编码
        update_inventory(self.items)                   # 硬编码
        log_to_analytics(self)                         # 硬编码
```

每次添加新的"副作用"，都要修改 `Order` 类——违反开闭原则，测试也越来越复杂。

### TDD 驱动：从测试需求中发现 Observer

```python
class TestOrderPlacement(unittest.TestCase):

    def test_placing_order_notifies_observers(self):
        # 我不想 Mock 三个具体的副作用函数
        # 我只想知道：Order 通知了什么？
        # → 这推动出 Observer 接口

        observer = MagicMock()  # 通用 Observer
        order = Order(items=["apple"], user_email="a@b.com")
        order.add_observer(observer)
        order.place()

        observer.on_order_placed.assert_called_once_with(order)

    def test_multiple_observers_all_notified(self):
        obs1 = MagicMock()
        obs2 = MagicMock()
        order = Order(items=["apple"], user_email="a@b.com")
        order.add_observer(obs1)
        order.add_observer(obs2)
        order.place()

        obs1.on_order_placed.assert_called_once()
        obs2.on_order_placed.assert_called_once()
```

### 实现 Observer 模式

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


class OrderObserver(ABC):
    @abstractmethod
    def on_order_placed(self, order: 'Order') -> None: ...


@dataclass
class Order:
    items: list
    user_email: str
    _observers: List[OrderObserver] = field(default_factory=list, repr=False)

    def add_observer(self, observer: OrderObserver) -> None:
        self._observers.append(observer)

    def place(self) -> None:
        self._save_to_db()
        self._notify_observers()

    def _notify_observers(self) -> None:
        for observer in self._observers:
            observer.on_order_placed(self)

    def _save_to_db(self) -> None:
        pass  # 真实实现


# 具体 Observer（各自独立测试）
class EmailNotifier(OrderObserver):
    def __init__(self, emailer):
        self._emailer = emailer

    def on_order_placed(self, order: Order) -> None:
        self._emailer.send(order.user_email, "Order placed!")


class InventoryUpdater(OrderObserver):
    def __init__(self, inventory):
        self._inventory = inventory

    def on_order_placed(self, order: Order) -> None:
        for item in order.items:
            self._inventory.decrement(item)
```

```python
class TestEmailNotifier(unittest.TestCase):
    """Observer 实现独立测试"""

    def test_sends_confirmation_to_order_email(self):
        mock_emailer = MagicMock()
        notifier = EmailNotifier(emailer=mock_emailer)
        order = Order(items=["apple"], user_email="alice@example.com")

        notifier.on_order_placed(order)

        mock_emailer.send.assert_called_once_with(
            "alice@example.com", "Order placed!"
        )
```

---

## 20.4 Command 模式：从"操作可撤销性测试"中浮现

### 问题：操作历史和撤销难以测试

```python
# 难以测试：编辑器直接执行操作，无法撤销
class TextEditor:
    def __init__(self):
        self.content = ""

    def insert(self, pos, text):
        self.content = self.content[:pos] + text + self.content[pos:]

    def delete(self, pos, length):
        self.content = self.content[:pos] + self.content[pos + length:]
```

### TDD 驱动：测试撤销功能，发现需要 Command

```python
class TestTextEditorUndo(unittest.TestCase):

    def test_undo_insert(self):
        # 要支持撤销，操作必须是"可记录的"
        # → 每个操作封装为 Command 对象
        editor = TextEditor()
        editor.execute(InsertCommand(pos=0, text="Hello"))
        editor.undo()
        self.assertEqual(editor.content, "")

    def test_undo_multiple_operations(self):
        editor = TextEditor()
        editor.execute(InsertCommand(pos=0, text="Hello"))
        editor.execute(InsertCommand(pos=5, text=" World"))
        editor.undo()
        self.assertEqual(editor.content, "Hello")
        editor.undo()
        self.assertEqual(editor.content, "")

    def test_redo_after_undo(self):
        editor = TextEditor()
        editor.execute(InsertCommand(pos=0, text="Hello"))
        editor.undo()
        editor.redo()
        self.assertEqual(editor.content, "Hello")
```

### 实现 Command 模式

```python
from abc import ABC, abstractmethod
from typing import List


class Command(ABC):
    @abstractmethod
    def execute(self, content: str) -> str: ...

    @abstractmethod
    def undo(self, content: str) -> str: ...


class InsertCommand(Command):
    def __init__(self, pos: int, text: str):
        self.pos = pos
        self.text = text

    def execute(self, content: str) -> str:
        return content[:self.pos] + self.text + content[self.pos:]

    def undo(self, content: str) -> str:
        return content[:self.pos] + content[self.pos + len(self.text):]


class DeleteCommand(Command):
    def __init__(self, pos: int, length: int):
        self.pos = pos
        self.length = length
        self._deleted_text = ""   # 保存删除的内容用于撤销

    def execute(self, content: str) -> str:
        self._deleted_text = content[self.pos:self.pos + self.length]
        return content[:self.pos] + content[self.pos + self.length:]

    def undo(self, content: str) -> str:
        return content[:self.pos] + self._deleted_text + content[self.pos:]


class TextEditor:
    def __init__(self):
        self.content = ""
        self._history: List[Command] = []
        self._redo_stack: List[Command] = []

    def execute(self, command: Command) -> None:
        self.content = command.execute(self.content)
        self._history.append(command)
        self._redo_stack.clear()

    def undo(self) -> None:
        if not self._history:
            return
        command = self._history.pop()
        self.content = command.undo(self.content)
        self._redo_stack.append(command)

    def redo(self) -> None:
        if not self._redo_stack:
            return
        command = self._redo_stack.pop()
        self.content = command.execute(self.content)
        self._history.append(command)
```

```python
class TestInsertCommand(unittest.TestCase):
    """Command 本身独立测试"""

    def test_insert_at_beginning(self):
        cmd = InsertCommand(pos=0, text="Hello")
        result = cmd.execute("")
        self.assertEqual(result, "Hello")

    def test_insert_undo_restores_original(self):
        cmd = InsertCommand(pos=0, text="Hello")
        after_insert = cmd.execute("")
        restored = cmd.undo(after_insert)
        self.assertEqual(restored, "")

    def test_insert_in_middle(self):
        cmd = InsertCommand(pos=5, text=" World")
        result = cmd.execute("Hello")
        self.assertEqual(result, "Hello World")
```

---

## 20.5 Builder 模式：从"复杂测试数据构造"中浮现

### 问题：测试中构造对象太繁琐

```python
# 每个测试都要构造完整的对象，但只关心一个字段
def test_vip_user_gets_discount(self):
    user = User(
        id=1,
        name="Alice",
        email="alice@example.com",
        created_at=datetime(2020, 1, 1),
        role="vip",           # ← 只关心这个字段
        address="123 Main St",
        phone="555-1234",
        preferences={"theme": "dark"},
    )
    discount = calculate_discount(user)
    self.assertEqual(discount, 0.2)
```

### TDD 驱动：发现需要 Builder

```python
class TestUserDiscount(unittest.TestCase):

    def _make_user(self, **overrides) -> User:
        """Builder 函数：提供合理默认值，允许覆盖关键字段"""
        defaults = {
            "id": 1,
            "name": "Test User",
            "email": "test@example.com",
            "created_at": datetime(2020, 1, 1),
            "role": "normal",
            "address": "123 Test St",
            "phone": "000-0000",
            "preferences": {},
        }
        defaults.update(overrides)
        return User(**defaults)

    def test_vip_user_gets_20_percent_discount(self):
        user = self._make_user(role="vip")   # 只关心 role！
        self.assertAlmostEqual(calculate_discount(user), 0.2)

    def test_normal_user_gets_no_discount(self):
        user = self._make_user(role="normal")
        self.assertAlmostEqual(calculate_discount(user), 0.0)

    def test_senior_user_gets_15_percent_discount(self):
        user = self._make_user(role="senior")
        self.assertAlmostEqual(calculate_discount(user), 0.15)
```

### 流式 Builder（链式调用风格）

```python
class UserBuilder:
    """流式 Builder：测试数据构造器"""

    def __init__(self):
        self._data = {
            "id": 1,
            "name": "Default User",
            "email": "default@example.com",
            "role": "normal",
            "active": True,
            "preferences": {},
        }

    def with_id(self, user_id: int) -> 'UserBuilder':
        self._data["id"] = user_id
        return self

    def named(self, name: str) -> 'UserBuilder':
        self._data["name"] = name
        return self

    def with_email(self, email: str) -> 'UserBuilder':
        self._data["email"] = email
        return self

    def as_vip(self) -> 'UserBuilder':
        self._data["role"] = "vip"
        return self

    def as_admin(self) -> 'UserBuilder':
        self._data["role"] = "admin"
        return self

    def inactive(self) -> 'UserBuilder':
        self._data["active"] = False
        return self

    def build(self) -> User:
        return User(**self._data)


class TestWithBuilder(unittest.TestCase):

    def test_vip_discount(self):
        user = UserBuilder().as_vip().build()
        self.assertAlmostEqual(calculate_discount(user), 0.2)

    def test_inactive_user_no_discount(self):
        user = UserBuilder().inactive().build()
        self.assertAlmostEqual(calculate_discount(user), 0.0)

    def test_named_vip_user(self):
        user = (UserBuilder()
                .named("Alice")
                .with_email("alice@ex.com")
                .as_vip()
                .build())
        self.assertEqual(user.name, "Alice")
        self.assertEqual(user.role, "vip")
```

---

## 20.6 TDD 到设计模式的映射

| TDD 痛苦信号 | 根本原因 | 导出的模式 |
|-------------|---------|-----------|
| 条件分支需要多个 Mock | 行为硬编码 | Strategy |
| 一个操作触发多个副作用 | 通知机制混乱 | Observer |
| 操作难以撤销/重放 | 操作未封装 | Command |
| 构造对象代码冗长 | 构造逻辑复杂 | Builder |
| 对象创建与使用耦合 | 硬编码实例化 | Factory |
| 全局状态导致测试相互污染 | 单例/全局变量 | Dependency Injection |
| 依赖难以替换 | 实现依赖而非接口 | Adapter/Port |

---

## 20.7 从 TDD 反向发现模式

```
TDD 工作流：

写测试 → 发现 Mock 太多 → 引入依赖注入
写测试 → 发现条件分支爆炸 → 引入 Strategy
写测试 → 发现通知代码膨胀 → 引入 Observer
写测试 → 发现撤销测试困难 → 引入 Command
写测试 → 发现对象构建复杂 → 引入 Builder

反方向读：
设计模式 = 反复出现的 TDD 解法
```

---

## 20.8 本章小结

- **Strategy**：把可变行为抽象为接口，从"条件分支难以 Mock"中浮现
- **Observer**：解耦通知机制，从"副作用难以隔离测试"中浮现
- **Command**：将操作封装为对象，从"撤销/重放测试"需求中浮现
- **Builder**：分离复杂构造，从"测试数据构造冗长"中浮现
- 设计模式不是提前规划的，而是从**持续 TDD** 中自然生长出来的

---

## Part 5 总结

| 章节 | 主题 | 核心收获 |
|------|------|---------|
| 16 | 属性测试 | Hypothesis 验证不变量，自动收缩 |
| 17 | 六边形架构 | TDD 驱动出端口适配器设计 |
| 18 | 遗留代码 | 特征测试、接缝、绞杀者、Sprout |
| 19 | Mock 内部 | `__getattr__`、call 录制、autospec、patch 定位 |
| 20 | 设计模式 | Strategy/Observer/Command/Builder 从测试中浮现 |

**整个教程的终极信息**：TDD 不只是"先写测试"，它是一种迫使你持续改善设计的思维工具。每一次"测试难写"都是设计向你发出的信号——倾听它，你就会写出更好的代码。

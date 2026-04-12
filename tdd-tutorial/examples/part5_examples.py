"""
Part 5 示例：深度专题 - 属性测试、六边形架构、遗留代码、Mock内部、设计模式
运行方式：python -m unittest examples/part5_examples.py -v
注意：属性测试部分需要 pip install hypothesis
"""
import unittest
from unittest.mock import MagicMock, create_autospec, call, ANY, patch, Mock
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import os


# ════════════════════════════════════════════════════════════════════════════
# 第十七章：六边形架构与 TDD
# ════════════════════════════════════════════════════════════════════════════

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


class OrderRepository(ABC):
    @abstractmethod
    def find_by_id(self, order_id: int) -> Optional[Order]: ...


class PaymentService(ABC):
    @abstractmethod
    def charge(self, order_id: int, amount: float) -> dict: ...


class NotificationService(ABC):
    @abstractmethod
    def send(self, email: str, message: str) -> bool: ...


class OrderProcessor:
    PAYMENT_THRESHOLD = 1000.0

    def __init__(self, orders: OrderRepository, payment: PaymentService,
                 notifier: NotificationService):
        self._orders = orders
        self._payment = payment
        self._notifier = notifier

    def process(self, order_id: int) -> ProcessResult:
        order = self._orders.find_by_id(order_id)
        if order is None:
            raise LookupError(f"Order {order_id} not found")

        payment_ref = None
        if order.total > self.PAYMENT_THRESHOLD:
            result = self._payment.charge(order_id=order.id, amount=order.total)
            payment_ref = result.get("ref")

        notified = False
        if order.user_email:
            notified = self._notifier.send(
                email=order.user_email,
                message=f"Order {order_id} processed."
            )

        return ProcessResult(order_id=order_id, payment_ref=payment_ref,
                             notified=notified)


class FakeOrderRepository(OrderRepository):
    def __init__(self):
        self._orders: dict = {}
        self._next_id = 1

    def save(self, order: Order) -> Order:
        if order.id is None:
            order = Order(self._next_id, order.total, order.user_email)
            self._next_id += 1
        self._orders[order.id] = order
        return order

    def find_by_id(self, order_id: int) -> Optional[Order]:
        return self._orders.get(order_id)


class FakePaymentService(PaymentService):
    def __init__(self):
        self.charges: list = []
        self._fail = False

    def charge(self, order_id: int, amount: float) -> dict:
        if self._fail:
            raise RuntimeError("Gateway down")
        ref = f"fake_{len(self.charges)+1}"
        self.charges.append({"order_id": order_id, "amount": amount, "ref": ref})
        return {"ref": ref, "status": "ok"}

    def simulate_failure(self):
        self._fail = True


class FakeNotificationService(NotificationService):
    def __init__(self):
        self.sent: list = []

    def send(self, email: str, message: str) -> bool:
        self.sent.append({"email": email, "message": message})
        return True


class TestOrderProcessorUnit(unittest.TestCase):
    def setUp(self):
        self.mock_orders = MagicMock(spec=OrderRepository)
        self.mock_payment = MagicMock(spec=PaymentService)
        self.mock_notifier = MagicMock(spec=NotificationService)
        self.processor = OrderProcessor(
            orders=self.mock_orders, payment=self.mock_payment,
            notifier=self.mock_notifier)

    def test_high_value_order_charges_payment(self):
        self.mock_orders.find_by_id.return_value = Order(1, total=1500.0)
        self.mock_payment.charge.return_value = {"ref": "p001", "status": "ok"}
        result = self.processor.process(1)
        self.mock_payment.charge.assert_called_once_with(order_id=1, amount=1500.0)
        self.assertEqual(result.payment_ref, "p001")

    def test_low_value_order_skips_payment(self):
        self.mock_orders.find_by_id.return_value = Order(2, total=50.0)
        result = self.processor.process(2)
        self.mock_payment.charge.assert_not_called()
        self.assertIsNone(result.payment_ref)

    def test_threshold_is_exclusive(self):
        self.mock_orders.find_by_id.return_value = Order(3, total=1000.0)
        self.processor.process(3)
        self.mock_payment.charge.assert_not_called()

    def test_missing_order_raises(self):
        self.mock_orders.find_by_id.return_value = None
        with self.assertRaises(LookupError):
            self.processor.process(999)

    def test_order_with_email_notified(self):
        self.mock_orders.find_by_id.return_value = Order(4, 200.0, "a@b.com")
        self.mock_notifier.send.return_value = True
        result = self.processor.process(4)
        self.mock_notifier.send.assert_called_once()
        self.assertTrue(result.notified)


class TestOrderProcessorIntegration(unittest.TestCase):
    def setUp(self):
        self.orders = FakeOrderRepository()
        self.payment = FakePaymentService()
        self.notifier = FakeNotificationService()
        self.processor = OrderProcessor(
            orders=self.orders, payment=self.payment, notifier=self.notifier)
        self.orders.save(Order(1, 1500.0, "alice@ex.com"))
        self.orders.save(Order(2, 50.0, "bob@ex.com"))

    def test_full_high_value_flow(self):
        result = self.processor.process(1)
        self.assertEqual(len(self.payment.charges), 1)
        self.assertEqual(len(self.notifier.sent), 1)
        self.assertIsNotNone(result.payment_ref)

    def test_payment_failure_propagates(self):
        self.payment.simulate_failure()
        with self.assertRaises(RuntimeError):
            self.processor.process(1)
        self.assertEqual(len(self.notifier.sent), 0)


# ════════════════════════════════════════════════════════════════════════════
# 第十九章：Mock 内部机制
# ════════════════════════════════════════════════════════════════════════════

class Calculator:
    def add(self, a: int, b: int) -> int: return a + b
    def divide(self, a: int, b: int) -> float: return a / b


class TestMockInternals(unittest.TestCase):

    def test_child_mocks_are_cached(self):
        m = MagicMock()
        child1 = m.foo
        child2 = m.foo
        self.assertIs(child1, child2)

    def test_call_args_structure(self):
        m = MagicMock()
        m(1, 2, key="val")
        self.assertEqual(m.call_args.args, (1, 2))
        self.assertEqual(m.call_args.kwargs, {"key": "val"})

    def test_call_args_list_records_all_calls(self):
        m = MagicMock()
        m(1); m(2); m(3)
        self.assertEqual(m.call_count, 3)
        self.assertEqual(m.call_args_list, [call(1), call(2), call(3)])

    def test_side_effect_list_returns_in_order(self):
        m = MagicMock()
        m.side_effect = [10, 20, 30]
        self.assertEqual(m(), 10)
        self.assertEqual(m(), 20)
        self.assertEqual(m(), 30)

    def test_side_effect_function(self):
        m = MagicMock()
        m.side_effect = lambda n: n * 2
        self.assertEqual(m(5), 10)
        self.assertEqual(m(3), 6)

    def test_side_effect_exception(self):
        m = MagicMock()
        m.side_effect = ValueError("test error")
        with self.assertRaises(ValueError):
            m()

    def test_side_effect_overrides_return_value(self):
        m = MagicMock(return_value=42)
        m.side_effect = lambda: 100
        self.assertEqual(m(), 100)
        m.side_effect = None
        self.assertEqual(m(), 42)

    def test_spec_prevents_nonexistent_attrs(self):
        m = MagicMock(spec=Calculator)
        m.add(1, 2)  # OK
        with self.assertRaises(AttributeError):
            m.nonexistent_method()

    def test_autospec_checks_signatures(self):
        auto = create_autospec(Calculator)
        auto.add(1, 2)  # OK
        with self.assertRaises(TypeError):
            auto.add(1, 2, 3)  # too many args

    def test_reset_mock_clears_call_count(self):
        m = MagicMock()
        m(); m()
        self.assertEqual(m.call_count, 2)
        m.reset_mock()
        self.assertEqual(m.call_count, 0)

    def test_magicmock_supports_len(self):
        m = MagicMock()
        m.__len__.return_value = 42
        self.assertEqual(len(m), 42)

    def test_magicmock_supports_iteration(self):
        m = MagicMock()
        m.__iter__.return_value = iter([1, 2, 3])
        self.assertEqual(list(m), [1, 2, 3])

    def test_any_matches_any_argument(self):
        m = MagicMock()
        m("hello", 42, key="world")
        m.assert_called_with(ANY, 42, key=ANY)

    def test_assert_has_calls_checks_sequence(self):
        m = MagicMock()
        m(1); m(2); m(3)
        m.assert_has_calls([call(1), call(2)])  # 包含即可（非全等）
        m.assert_has_calls([call(3), call(1)], any_order=True)


# ════════════════════════════════════════════════════════════════════════════
# 第二十章：TDD 驱动出设计模式
# ════════════════════════════════════════════════════════════════════════════

# --- Strategy 模式 ---

class PaymentStrategy(ABC):
    @abstractmethod
    def execute(self, amount: float) -> dict: ...


class FakePaymentStrategy(PaymentStrategy):
    def __init__(self, ref="fake_ref"):
        self.calls: list = []
        self._ref = ref

    def execute(self, amount: float) -> dict:
        self.calls.append(amount)
        return {"ref": self._ref, "status": "ok"}


class Checkout:
    def __init__(self, payment_strategy: PaymentStrategy):
        self._strategy = payment_strategy

    def pay(self, amount: float) -> dict:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        return self._strategy.execute(amount)


class TestCheckoutStrategy(unittest.TestCase):

    def test_pay_delegates_to_strategy(self):
        strategy = MagicMock(spec=PaymentStrategy)
        strategy.execute.return_value = {"ref": "tx_001"}
        checkout = Checkout(payment_strategy=strategy)
        checkout.pay(100.0)
        strategy.execute.assert_called_once_with(100.0)

    def test_pay_returns_strategy_result(self):
        strategy = FakePaymentStrategy(ref="p_999")
        checkout = Checkout(payment_strategy=strategy)
        result = checkout.pay(50.0)
        self.assertEqual(result["ref"], "p_999")

    def test_zero_amount_raises(self):
        checkout = Checkout(MagicMock())
        with self.assertRaises(ValueError):
            checkout.pay(0)

    def test_strategies_are_interchangeable(self):
        """同一个 Checkout 可以切换不同策略"""
        s1 = FakePaymentStrategy("stripe_ref")
        s2 = FakePaymentStrategy("paypal_ref")

        checkout1 = Checkout(s1)
        checkout2 = Checkout(s2)

        r1 = checkout1.pay(100.0)
        r2 = checkout2.pay(100.0)

        self.assertEqual(r1["ref"], "stripe_ref")
        self.assertEqual(r2["ref"], "paypal_ref")


# --- Observer 模式 ---

class OrderObserver(ABC):
    @abstractmethod
    def on_order_placed(self, order: 'ObservableOrder') -> None: ...


@dataclass
class ObservableOrder:
    items: list
    user_email: str
    id: int = 0
    _observers: List[OrderObserver] = field(default_factory=list, repr=False)

    def add_observer(self, observer: OrderObserver) -> None:
        self._observers.append(observer)

    def place(self) -> None:
        self._notify()

    def _notify(self) -> None:
        for obs in self._observers:
            obs.on_order_placed(self)


class TestOrderObserver(unittest.TestCase):

    def test_single_observer_notified(self):
        obs = MagicMock(spec=OrderObserver)
        order = ObservableOrder(items=["apple"], user_email="a@b.com")
        order.add_observer(obs)
        order.place()
        obs.on_order_placed.assert_called_once_with(order)

    def test_multiple_observers_all_notified(self):
        obs1, obs2, obs3 = MagicMock(), MagicMock(), MagicMock()
        order = ObservableOrder(items=["apple"], user_email="a@b.com")
        order.add_observer(obs1)
        order.add_observer(obs2)
        order.add_observer(obs3)
        order.place()
        obs1.on_order_placed.assert_called_once()
        obs2.on_order_placed.assert_called_once()
        obs3.on_order_placed.assert_called_once()

    def test_no_observers_no_error(self):
        order = ObservableOrder(items=[], user_email="")
        order.place()  # 不报错即通过


# --- Command 模式 ---

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
        cmd = self._history.pop()
        self.content = cmd.undo(self.content)
        self._redo_stack.append(cmd)

    def redo(self) -> None:
        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        self.content = cmd.execute(self.content)
        self._history.append(cmd)


class TestInsertCommand(unittest.TestCase):

    def test_insert_at_start(self):
        cmd = InsertCommand(0, "Hello")
        self.assertEqual(cmd.execute(""), "Hello")

    def test_insert_in_middle(self):
        cmd = InsertCommand(5, " World")
        self.assertEqual(cmd.execute("Hello"), "Hello World")

    def test_undo_restores_original(self):
        cmd = InsertCommand(0, "Hello")
        after = cmd.execute("")
        self.assertEqual(cmd.undo(after), "")


class TestTextEditorCommand(unittest.TestCase):

    def test_undo_single_operation(self):
        editor = TextEditor()
        editor.execute(InsertCommand(0, "Hello"))
        editor.undo()
        self.assertEqual(editor.content, "")

    def test_undo_multiple_operations(self):
        editor = TextEditor()
        editor.execute(InsertCommand(0, "Hello"))
        editor.execute(InsertCommand(5, " World"))
        editor.undo()
        self.assertEqual(editor.content, "Hello")
        editor.undo()
        self.assertEqual(editor.content, "")

    def test_redo_after_undo(self):
        editor = TextEditor()
        editor.execute(InsertCommand(0, "Hello"))
        editor.undo()
        editor.redo()
        self.assertEqual(editor.content, "Hello")

    def test_new_command_clears_redo_stack(self):
        editor = TextEditor()
        editor.execute(InsertCommand(0, "Hello"))
        editor.undo()
        editor.execute(InsertCommand(0, "World"))  # 新命令
        editor.redo()   # 应该什么都不做
        self.assertEqual(editor.content, "World")

    def test_undo_on_empty_editor_is_noop(self):
        editor = TextEditor()
        editor.undo()   # 不报错即通过
        self.assertEqual(editor.content, "")


# --- Builder 模式 ---

@dataclass
class User:
    id: int
    name: str
    email: str
    role: str = "normal"
    active: bool = True


class UserBuilder:
    def __init__(self):
        self._data = {"id": 1, "name": "Test User",
                      "email": "test@example.com", "role": "normal", "active": True}

    def with_id(self, v): self._data["id"] = v; return self
    def named(self, v): self._data["name"] = v; return self
    def as_vip(self): self._data["role"] = "vip"; return self
    def as_admin(self): self._data["role"] = "admin"; return self
    def inactive(self): self._data["active"] = False; return self
    def build(self): return User(**self._data)


def calculate_discount(user: User) -> float:
    if not user.active:
        return 0.0
    return {"vip": 0.20, "admin": 0.0, "normal": 0.0}.get(user.role, 0.0)


class TestUserBuilder(unittest.TestCase):

    def test_vip_gets_discount(self):
        user = UserBuilder().as_vip().build()
        self.assertAlmostEqual(calculate_discount(user), 0.20)

    def test_normal_gets_no_discount(self):
        user = UserBuilder().build()
        self.assertAlmostEqual(calculate_discount(user), 0.0)

    def test_inactive_vip_gets_no_discount(self):
        user = UserBuilder().as_vip().inactive().build()
        self.assertAlmostEqual(calculate_discount(user), 0.0)

    def test_builder_allows_focusing_on_relevant_field(self):
        """Builder 让测试只关注相关字段"""
        user = UserBuilder().as_vip().build()
        # 不需要关心 id、name、email 等与此测试无关的字段
        self.assertEqual(user.role, "vip")


if __name__ == '__main__':
    unittest.main(verbosity=2)

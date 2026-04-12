"""
Part 3 示例：中级技巧 - Mock、异常、参数化、IO
运行方式：python -m unittest examples/part3_examples.py -v
"""
import unittest
from unittest.mock import MagicMock, patch, mock_open, call, ANY
import io
import os
import tempfile


# ── 生产代码 ────────────────────────────────────────────────────────────────

class PaymentGateway:
    """真实支付网关（测试中会被 Mock）"""
    def charge(self, user_id: int, amount: float) -> dict:
        raise NotImplementedError("Use real gateway in prod")


class EmailService:
    """邮件服务（测试中会被 Mock）"""
    def send(self, to: str, subject: str, body: str) -> bool:
        raise NotImplementedError("Use real email service in prod")


class OrderService:
    def __init__(self, gateway: PaymentGateway, emailer: EmailService):
        self._gateway = gateway
        self._emailer = emailer

    def place_order(self, user_id: int, user_email: str, amount: float) -> dict:
        if amount <= 0:
            raise ValueError(f"Amount must be positive, got {amount}")

        result = self._gateway.charge(user_id=user_id, amount=amount)
        if result.get("status") != "success":
            raise RuntimeError(f"Payment failed: {result.get('error', 'unknown')}")

        self._emailer.send(
            to=user_email,
            subject="Order Confirmation",
            body=f"Your order of ${amount:.2f} was placed successfully."
        )
        return {"order_id": result["transaction_id"], "amount": amount}


def parse_quantity(value) -> int:
    """解析数量，带严格验证"""
    if value is None:
        raise TypeError("quantity cannot be None")
    if not isinstance(value, int):
        raise TypeError(f"quantity must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"quantity must be non-negative, got {value}")
    if value > 9999:
        raise ValueError(f"quantity exceeds maximum (9999), got {value}")
    return value


def read_config(path: str) -> dict:
    with open(path) as f:
        import json
        return json.load(f)


# ── Mock 测试 ────────────────────────────────────────────────────────────────

class TestOrderService(unittest.TestCase):
    """第八章：Mock 与打桩"""

    def setUp(self):
        self.mock_gateway = MagicMock(spec=PaymentGateway)
        self.mock_emailer = MagicMock(spec=EmailService)
        self.service = OrderService(
            gateway=self.mock_gateway,
            emailer=self.mock_emailer,
        )

    def test_successful_order_charges_gateway(self):
        self.mock_gateway.charge.return_value = {
            "status": "success", "transaction_id": "tx_001"
        }
        self.service.place_order(
            user_id=1, user_email="alice@example.com", amount=99.99
        )
        self.mock_gateway.charge.assert_called_once_with(
            user_id=1, amount=99.99
        )

    def test_successful_order_sends_confirmation_email(self):
        self.mock_gateway.charge.return_value = {
            "status": "success", "transaction_id": "tx_002"
        }
        self.service.place_order(
            user_id=2, user_email="bob@example.com", amount=50.0
        )
        self.mock_emailer.send.assert_called_once()
        call_kwargs = self.mock_emailer.send.call_args.kwargs
        self.assertEqual(call_kwargs["to"], "bob@example.com")
        self.assertIn("Confirmation", call_kwargs["subject"])

    def test_payment_failure_raises_runtime_error(self):
        self.mock_gateway.charge.return_value = {
            "status": "failed", "error": "card declined"
        }
        with self.assertRaisesRegex(RuntimeError, "card declined"):
            self.service.place_order(
                user_id=3, user_email="carol@example.com", amount=100.0
            )
        # 邮件不应该发出
        self.mock_emailer.send.assert_not_called()

    def test_zero_amount_raises_before_gateway_call(self):
        with self.assertRaises(ValueError):
            self.service.place_order(
                user_id=1, user_email="a@b.com", amount=0
            )
        self.mock_gateway.charge.assert_not_called()

    def test_gateway_network_error_propagates(self):
        self.mock_gateway.charge.side_effect = ConnectionError("timeout")
        with self.assertRaises(ConnectionError):
            self.service.place_order(1, "a@b.com", 50.0)

    def test_successful_order_returns_order_id(self):
        self.mock_gateway.charge.return_value = {
            "status": "success", "transaction_id": "tx_999"
        }
        result = self.service.place_order(1, "a@b.com", 25.0)
        self.assertEqual(result["order_id"], "tx_999")
        self.assertAlmostEqual(result["amount"], 25.0)


# ── 异常测试 ────────────────────────────────────────────────────────────────

class TestParseQuantity(unittest.TestCase):
    """第九章：异常与错误场景"""

    # 有效值
    def test_zero_is_valid(self):
        self.assertEqual(parse_quantity(0), 0)

    def test_positive_int_is_valid(self):
        self.assertEqual(parse_quantity(42), 42)

    def test_max_boundary_is_valid(self):
        self.assertEqual(parse_quantity(9999), 9999)

    # None
    def test_none_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "None"):
            parse_quantity(None)

    # 类型错误
    def test_string_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "int"):
            parse_quantity("5")

    def test_float_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "float"):
            parse_quantity(3.5)

    # 值错误
    def test_negative_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_quantity(-1)

    def test_over_max_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "9999"):
            parse_quantity(10000)

    # 边界
    def test_boundary_cases_with_subtest(self):
        boundaries = [
            (0,    True,  0),
            (9999, True,  9999),
            (-1,   False, ValueError),
            (10000, False, ValueError),
        ]
        for value, should_pass, expected in boundaries:
            with self.subTest(value=value):
                if should_pass:
                    self.assertEqual(parse_quantity(value), expected)
                else:
                    with self.assertRaises(expected):
                        parse_quantity(value)


# ── 参数化测试 ────────────────────────────────────────────────────────────────

class TestParameterized(unittest.TestCase):
    """第十章：参数化测试"""

    ARITHMETIC_CASES = [
        (2, 3, 5),
        (0, 0, 0),
        (-1, 1, 0),
        (100, 200, 300),
    ]

    def test_addition_table(self):
        for a, b, expected in self.ARITHMETIC_CASES:
            with self.subTest(a=a, b=b):
                self.assertEqual(a + b, expected)

    # 字符串操作的参数化
    UPPER_CASES = [
        ("hello",   "HELLO"),
        ("World",   "WORLD"),
        ("",        ""),
        ("123abc",  "123ABC"),
    ]

    def test_string_upper(self):
        for s, expected in self.UPPER_CASES:
            with self.subTest(input=s):
                self.assertEqual(s.upper(), expected)


# ── IO 测试 ────────────────────────────────────────────────────────────────

class TestConfigReader(unittest.TestCase):
    """第十一章：IO 与文件测试"""

    def test_reads_json_config_with_mock_open(self):
        config_content = '{"host": "localhost", "port": 5432}'
        with patch('builtins.open', mock_open(read_data=config_content)):
            config = read_config("any_path.json")
        self.assertEqual(config["host"], "localhost")
        self.assertEqual(config["port"], 5432)

    def test_file_not_found_raises(self):
        with patch('builtins.open', side_effect=FileNotFoundError("no file")):
            with self.assertRaises(FileNotFoundError):
                read_config("missing.json")

    def test_reads_real_temp_file(self):
        """使用真实临时文件的集成测试"""
        import json
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump({"key": "value", "number": 42}, f)
            temp_path = f.name

        try:
            config = read_config(temp_path)
            self.assertEqual(config["key"], "value")
            self.assertEqual(config["number"], 42)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)

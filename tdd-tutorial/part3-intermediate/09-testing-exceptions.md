# 第九章：异常与错误场景测试

## 9.1 为什么异常测试重要

异常是系统契约的一部分。如果 `parse_age(-1)` 应该抛出 `ValueError`，那么：
- 不抛出 → 静默地接受了无效输入（bug）
- 抛出 `TypeError` → 错误类型（契约违背）
- 抛出 `ValueError` 但消息不对 → 调用方无法区分错误原因

完整的异常测试覆盖：类型、消息、属性、上下文。

---

## 9.2 assertRaises 的完整用法

### 基础：验证异常类型

```python
class TestAgeParser(unittest.TestCase):

    def test_negative_age_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_age(-1)

    def test_string_age_raises_type_error(self):
        with self.assertRaises(TypeError):
            parse_age("twenty")

    def test_valid_age_does_not_raise(self):
        # 隐式验证：不抛出异常就是通过
        result = parse_age(25)
        self.assertEqual(result, 25)
```

### 中级：验证异常消息

```python
    def test_negative_age_error_message(self):
        with self.assertRaisesRegex(ValueError, r"age.*must be.*positive"):
            parse_age(-1)

    def test_too_old_error_mentions_max(self):
        with self.assertRaisesRegex(ValueError, r"150"):
            parse_age(200)
```

### 高级：检查异常对象属性

```python
    def test_validation_error_has_field_info(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_user_form({"age": -1, "name": ""})

        exc = ctx.exception
        # 检查异常携带的结构化信息
        self.assertIn("age", exc.fields)
        self.assertIn("name", exc.fields)
        self.assertEqual(exc.fields["age"], "must be positive")
        self.assertEqual(exc.error_count, 2)
```

---

## 9.3 异常链（Exception Chaining）

Python 的 `raise X from Y` 建立异常链，测试时需要验证：

```python
# 生产代码
def fetch_user(user_id):
    try:
        return db.query(f"SELECT * FROM users WHERE id={user_id}")
    except DatabaseError as e:
        raise ServiceError(f"Cannot fetch user {user_id}") from e


# 测试代码
class TestFetchUser(unittest.TestCase):

    def test_db_error_raises_service_error(self):
        with patch('myapp.db.query') as mock_query:
            mock_query.side_effect = DatabaseError("connection lost")

            with self.assertRaises(ServiceError) as ctx:
                fetch_user(42)

            exc = ctx.exception
            self.assertIn("42", str(exc))

            # 验证异常链：ServiceError 是由 DatabaseError 引起的
            self.assertIsInstance(exc.__cause__, DatabaseError)
            self.assertIn("connection lost", str(exc.__cause__))
```

---

## 9.4 测试多个异常的共同行为

```python
class TestInputValidator(unittest.TestCase):

    INVALID_INPUTS = [
        (None, TypeError, "cannot be None"),
        ("", ValueError, "cannot be empty"),
        (-1, ValueError, "must be positive"),
        (float('inf'), ValueError, "must be finite"),
    ]

    def test_invalid_inputs_raise_appropriate_errors(self):
        for value, exc_type, message_fragment in self.INVALID_INPUTS:
            with self.subTest(value=value):
                with self.assertRaisesRegex(exc_type, message_fragment):
                    validate_quantity(value)
```

---

## 9.5 测试边界条件

```python
class TestBoundaryConditions(unittest.TestCase):

    def test_zero_is_valid_quantity(self):
        """零是边界值，应该不抛出异常"""
        result = validate_quantity(0)
        self.assertEqual(result, 0)

    def test_max_quantity_is_valid(self):
        """最大值（999）应该有效"""
        result = validate_quantity(999)
        self.assertEqual(result, 999)

    def test_over_max_raises_error(self):
        """超过最大值应该抛出异常"""
        with self.assertRaisesRegex(ValueError, "999"):
            validate_quantity(1000)

    def test_exactly_at_boundary(self):
        """准确在边界值处的行为"""
        # 999 有效，1000 无效
        validate_quantity(999)   # 不抛出
        with self.assertRaises(ValueError):
            validate_quantity(1000)
```

---

## 9.6 测试异常不被吞噬

有时候最危险的 bug 不是"抛出了错误的异常"，而是"应该抛出但被捕获了"：

```python
# 问题代码：异常被过度捕获
def process_payment(amount):
    try:
        result = gateway.charge(amount)
        return result
    except Exception:
        return None   # ← 吞噬了所有异常！调用方看不到错误

# 测试暴露这个问题
class TestPaymentProcessing(unittest.TestCase):

    def test_gateway_error_propagates_to_caller(self):
        """验证网关错误不被静默吞噬"""
        with patch('myapp.payment.gateway') as mock_gateway:
            mock_gateway.charge.side_effect = NetworkError("timeout")

            # 正确的行为：异常应该传播
            with self.assertRaises((NetworkError, PaymentError)):
                process_payment(50.0)

    def test_invalid_amount_raises_not_returns_none(self):
        """验证无效金额不会静默返回 None"""
        result = process_payment(-1.0)
        # 如果返回了 None，说明异常被吞噬了
        self.assertIsNotNone(result,
            "process_payment(-1) returned None instead of raising an error")
```

---

## 9.7 测试 finally 和 cleanup 行为

```python
class TestResourceCleanup(unittest.TestCase):

    def test_connection_closed_even_if_query_fails(self):
        """即使查询失败，连接也应该被关闭"""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = QueryError("syntax error")

        with patch('myapp.db.get_connection', return_value=mock_conn):
            with self.assertRaises(QueryError):
                execute_safe_query("INVALID SQL")

            # finally 块应该确保连接被关闭
            mock_conn.close.assert_called_once()

    def test_file_closed_after_processing_error(self):
        """处理文件出错时，文件句柄应该被关闭"""
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)  # 不抑制异常

        with patch('builtins.open', return_value=mock_file):
            with self.assertRaises(ProcessingError):
                process_file("data.txt")

            # 验证 __exit__ 被调用（with 语句保证这一点）
            mock_file.__exit__.assert_called_once()
```

---

## 9.8 自定义异常的测试

```python
# 生产代码
class InsufficientFundsError(ValueError):
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        self.shortfall = amount - balance
        super().__init__(
            f"Cannot charge {amount:.2f}: balance is only {balance:.2f}"
        )


# 测试代码
class TestInsufficientFundsError(unittest.TestCase):

    def test_error_message_format(self):
        exc = InsufficientFundsError(balance=10.0, amount=50.0)
        self.assertIn("50.00", str(exc))
        self.assertIn("10.00", str(exc))

    def test_shortfall_calculation(self):
        exc = InsufficientFundsError(balance=10.0, amount=50.0)
        self.assertAlmostEqual(exc.shortfall, 40.0)

    def test_is_value_error_subclass(self):
        exc = InsufficientFundsError(10.0, 50.0)
        self.assertIsInstance(exc, ValueError)

    def test_raised_when_balance_insufficient(self):
        account = BankAccount(balance=10.0)
        with self.assertRaises(InsufficientFundsError) as ctx:
            account.charge(50.0)
        
        exc = ctx.exception
        self.assertAlmostEqual(exc.balance, 10.0)
        self.assertAlmostEqual(exc.amount, 50.0)
        self.assertAlmostEqual(exc.shortfall, 40.0)
```

---

## 9.9 测试上下文管理器的异常行为

```python
class TestTransactionContext(unittest.TestCase):

    def test_successful_transaction_commits(self):
        db = MockDatabase()
        with Transaction(db) as tx:
            tx.execute("INSERT INTO users VALUES (1, 'Alice')")
        db.commit.assert_called_once()
        db.rollback.assert_not_called()

    def test_failed_transaction_rolls_back(self):
        db = MockDatabase()
        db.execute.side_effect = IntegrityError("duplicate key")

        with self.assertRaises(IntegrityError):
            with Transaction(db) as tx:
                tx.execute("INSERT INTO users VALUES (1, 'Duplicate')")

        db.rollback.assert_called_once()
        db.commit.assert_not_called()

    def test_context_manager_suppresses_specific_errors(self):
        """某些上下文管理器会抑制特定异常"""
        with SuppressNotFound():
            raise FileNotFoundError("no such file")  # 应该被抑制
        # 如果执行到这里，说明异常被正确抑制了
```

---

## 9.10 本章小结

- `assertRaises`：验证异常类型
- `assertRaisesRegex`：同时验证异常消息
- `ctx.exception`：获取异常对象，检查属性和 `__cause__`
- `subTest`：在一个测试中验证多个异常场景
- 测试"异常不被吞噬"和"cleanup 在异常后执行"同样重要
- 自定义异常的属性也是契约，需要测试

**下一章**：参数化测试——用数据驱动测试逻辑。

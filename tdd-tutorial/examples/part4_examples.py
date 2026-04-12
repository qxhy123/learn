"""
Part 4 示例：高阶技巧 - 测试替身、异步、覆盖率
运行方式：python -m unittest examples/part4_examples.py -v
"""
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch, call


# ── 生产代码 ────────────────────────────────────────────────────────────────

# --- 测试替身示例 ---

class NotificationService:
    def send(self, user_id: int, message: str) -> bool:
        raise NotImplementedError

class UserRepository:
    def find_by_id(self, user_id: int):
        raise NotImplementedError
    def save(self, user) -> 'User':
        raise NotImplementedError
    def find_by_email(self, email: str):
        raise NotImplementedError


class User:
    def __init__(self, user_id=None, name="", email="", active=True):
        self.id = user_id
        self.name = name
        self.email = email
        self.active = active

    def __repr__(self):
        return f"User(id={self.id}, name={self.name!r})"


class FakeUserRepository:
    """Fake：真实逻辑的内存实现"""

    def __init__(self):
        self._store = {}
        self._next_id = 1

    def save(self, user: User) -> User:
        user.id = self._next_id
        self._next_id += 1
        self._store[user.id] = user
        return user

    def find_by_id(self, user_id: int):
        return self._store.get(user_id)

    def find_by_email(self, email: str):
        return next(
            (u for u in self._store.values() if u.email == email),
            None
        )

    def count(self) -> int:
        return len(self._store)


class UserService:
    def __init__(self, repo: UserRepository, notifier: NotificationService):
        self._repo = repo
        self._notifier = notifier

    def register(self, name: str, email: str) -> User:
        if not name or not email:
            raise ValueError("name and email are required")
        existing = self._repo.find_by_email(email)
        if existing:
            raise ValueError(f"Email already registered: {email}")
        user = User(name=name, email=email)
        saved = self._repo.save(user)
        self._notifier.send(saved.id, f"Welcome, {name}!")
        return saved

    def deactivate(self, user_id: int) -> User:
        user = self._repo.find_by_id(user_id)
        if user is None:
            raise LookupError(f"User {user_id} not found")
        user.active = False
        return self._repo.save(user)


# --- 异步代码 ---

class AsyncWeatherClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def fetch(self, city: str) -> dict:
        # 真实实现会调用 HTTP API
        raise NotImplementedError

    async def fetch_multiple(self, cities: list) -> list:
        tasks = [self.fetch(city) for city in cities]
        return await asyncio.gather(*tasks)


class AsyncWeatherService:
    def __init__(self, client: AsyncWeatherClient):
        self._client = client

    async def get_temperature(self, city: str) -> float:
        data = await self._client.fetch(city)
        return data["main"]["temp"]

    async def get_hottest_city(self, cities: list) -> str:
        results = await self._client.fetch_multiple(cities)
        hottest = max(results, key=lambda r: r["main"]["temp"])
        return hottest["city"]


# ── 测试替身模式 ─────────────────────────────────────────────────────────────

class TestDummyPattern(unittest.TestCase):
    """Dummy：只是占位符"""

    def test_user_name_stored_correctly(self):
        dummy_notifier = MagicMock()  # Dummy：必须传入但不关心
        fake_repo = FakeUserRepository()
        service = UserService(repo=fake_repo, notifier=dummy_notifier)

        user = service.register("Alice", "alice@example.com")
        self.assertEqual(user.name, "Alice")
        # 不对 dummy_notifier 做任何断言


class TestStubPattern(unittest.TestCase):
    """Stub：预设返回值，不验证调用"""

    def test_register_with_unique_email(self):
        # Stub：find_by_email 返回 None（邮箱未注册）
        stub_repo = MagicMock(spec=UserRepository)
        stub_repo.find_by_email.return_value = None
        stub_repo.save.return_value = User(user_id=42, name="Bob", email="bob@ex.com")

        dummy_notifier = MagicMock()
        service = UserService(repo=stub_repo, notifier=dummy_notifier)

        user = service.register("Bob", "bob@ex.com")
        self.assertEqual(user.id, 42)
        # 注意：没有对 stub_repo 的调用做断言 → 这是 Stub


class TestSpyPattern(unittest.TestCase):
    """Spy：记录调用，事后验证"""

    def test_welcome_message_sent_after_registration(self):
        fake_repo = FakeUserRepository()
        spy_notifier = MagicMock()  # MagicMock 自带 Spy 能力

        service = UserService(repo=fake_repo, notifier=spy_notifier)
        service.register("Carol", "carol@example.com")

        # 事后检查（Spy 风格）
        spy_notifier.send.assert_called_once()
        args = spy_notifier.send.call_args
        self.assertIn("Carol", args.args[1])  # 消息包含用户名


class TestMockPattern(unittest.TestCase):
    """Mock：预设期望，验证精确交互"""

    def test_deactivate_saves_user_with_active_false(self):
        mock_repo = MagicMock(spec=UserRepository)
        active_user = User(user_id=1, name="Dave", email="dave@ex.com", active=True)
        mock_repo.find_by_id.return_value = active_user
        mock_repo.save.return_value = active_user

        dummy_notifier = MagicMock()
        service = UserService(repo=mock_repo, notifier=dummy_notifier)

        service.deactivate(1)

        # Mock 验证：save 被调用，且传入的用户 active=False
        mock_repo.save.assert_called_once()
        saved_user = mock_repo.save.call_args.args[0]
        self.assertFalse(saved_user.active)


class TestFakePattern(unittest.TestCase):
    """Fake：有真实业务逻辑的内存替代"""

    def setUp(self):
        self.fake_repo = FakeUserRepository()
        self.dummy_notifier = MagicMock()
        self.service = UserService(
            repo=self.fake_repo,
            notifier=self.dummy_notifier
        )

    def test_register_assigns_unique_ids(self):
        u1 = self.service.register("Alice", "alice@ex.com")
        u2 = self.service.register("Bob", "bob@ex.com")
        self.assertNotEqual(u1.id, u2.id)

    def test_duplicate_email_raises(self):
        self.service.register("Alice", "same@ex.com")
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.service.register("Alice2", "same@ex.com")

    def test_deactivate_user(self):
        user = self.service.register("Eve", "eve@ex.com")
        self.service.deactivate(user.id)
        found = self.fake_repo.find_by_id(user.id)
        self.assertFalse(found.active)

    def test_deactivate_nonexistent_raises(self):
        with self.assertRaises(LookupError):
            self.service.deactivate(9999)


# ── 异步测试 ─────────────────────────────────────────────────────────────────

class TestAsyncWeatherService(unittest.IsolatedAsyncioTestCase):
    """第十三章：异步代码测试"""

    async def asyncSetUp(self):
        self.mock_client = AsyncMock(spec=AsyncWeatherClient)
        self.service = AsyncWeatherService(client=self.mock_client)

    async def test_get_temperature_returns_float(self):
        self.mock_client.fetch.return_value = {
            "city": "London",
            "main": {"temp": 18.5}
        }
        temp = await self.service.get_temperature("London")
        self.assertAlmostEqual(temp, 18.5)
        self.mock_client.fetch.assert_awaited_once_with("London")

    async def test_get_temperature_for_unknown_city_propagates_error(self):
        self.mock_client.fetch.side_effect = ValueError("unknown city")
        with self.assertRaises(ValueError):
            await self.service.get_temperature("Atlantis")

    async def test_get_hottest_city(self):
        self.mock_client.fetch_multiple.return_value = [
            {"city": "London",    "main": {"temp": 18.0}},
            {"city": "Dubai",     "main": {"temp": 42.0}},
            {"city": "Stockholm", "main": {"temp": 5.0}},
        ]
        hottest = await self.service.get_hottest_city(
            ["London", "Dubai", "Stockholm"]
        )
        self.assertEqual(hottest, "Dubai")

    async def test_concurrent_gather(self):
        """验证 asyncio.gather 的并发语义"""
        results = []

        async def task(n):
            await asyncio.sleep(0.01)
            results.append(n)
            return n * 2

        outputs = await asyncio.gather(task(1), task(2), task(3))
        self.assertEqual(outputs, [2, 4, 6])
        self.assertCountEqual(results, [1, 2, 3])

    async def test_timeout_raises(self):
        async def slow():
            await asyncio.sleep(10)

        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(slow(), timeout=0.05)


# ── 最佳实践演示 ────────────────────────────────────────────────────────────

class TestBestPracticesAAA(unittest.TestCase):
    """第十五章：AAA 模式与最佳实践"""

    def test_aaa_pattern_clearly_separated(self):
        # Arrange
        fake_repo = FakeUserRepository()
        notifier = MagicMock()
        service = UserService(repo=fake_repo, notifier=notifier)
        name = "Test User"
        email = "test@example.com"

        # Act
        user = service.register(name, email)

        # Assert
        self.assertIsNotNone(user.id)
        self.assertEqual(user.name, name)
        self.assertEqual(user.email, email)

    def test_one_assertion_per_behavior(self):
        """每个测试验证一个行为"""
        fake_repo = FakeUserRepository()
        service = UserService(repo=fake_repo, notifier=MagicMock())
        user = service.register("Alice", "alice@ex.com")
        # 只验证 ID 被分配
        self.assertIsNotNone(user.id)

    def test_notification_sent_on_registration(self):
        """另一个测试验证通知"""
        spy = MagicMock()
        service = UserService(repo=FakeUserRepository(), notifier=spy)
        service.register("Alice", "alice@ex.com")
        spy.send.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)

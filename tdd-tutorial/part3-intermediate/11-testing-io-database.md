# 第十一章：IO 与数据库测试

## 11.1 测试策略分层

```
        ┌─────────────────────────────────┐
        │  E2E Tests（端到端）             │  少量，慢，真实环境
        ├─────────────────────────────────┤
        │  Integration Tests（集成）       │  适量，中速，真实 I/O
        ├─────────────────────────────────┤
        │  Unit Tests（单元）             │  大量，极快，Mock I/O
        └─────────────────────────────────┘
```

选择原则：能 Mock 就 Mock（单元测试），关键路径需要真实 I/O（集成测试）。

---

## 11.2 文件 I/O 测试

### 方案一：Mock open（单元测试）

```python
import unittest
from unittest.mock import patch, mock_open

class TestConfigLoader(unittest.TestCase):

    def test_loads_config_from_file(self):
        config_content = '{"host": "localhost", "port": 5432}'

        with patch('builtins.open', mock_open(read_data=config_content)):
            config = load_config("config.json")

        self.assertEqual(config["host"], "localhost")
        self.assertEqual(config["port"], 5432)

    def test_raises_when_config_missing(self):
        with patch('builtins.open', side_effect=FileNotFoundError):
            with self.assertRaises(ConfigError):
                load_config("nonexistent.json")

    def test_writes_output_file(self):
        m = mock_open()
        with patch('builtins.open', m):
            write_report(data={"total": 100}, path="report.txt")

        # 验证文件被打开用于写入
        m.assert_called_once_with("report.txt", 'w')
        # 验证写入内容
        handle = m()
        handle.write.assert_called()
        written = ''.join(call.args[0] for call in handle.write.call_args_list)
        self.assertIn("100", written)
```

### 方案二：临时文件（集成测试）

```python
import tempfile
import os
import unittest

class TestCSVProcessor(unittest.TestCase):

    def setUp(self):
        # 创建临时文件
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(__import__('shutil').rmtree, self.temp_dir)

    def _write_csv(self, filename, content):
        path = os.path.join(self.temp_dir, filename)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def test_process_valid_csv(self):
        csv_path = self._write_csv("data.csv",
            "name,age\nAlice,30\nBob,25\n")

        result = process_csv(csv_path)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "Alice")
        self.assertEqual(result[1]["age"], 25)

    def test_process_empty_csv(self):
        csv_path = self._write_csv("empty.csv", "name,age\n")

        result = process_csv(csv_path)
        self.assertEqual(result, [])

    def test_output_written_correctly(self):
        csv_path = self._write_csv("input.csv", "name,score\nAlice,95\n")
        output_path = os.path.join(self.temp_dir, "output.json")

        process_and_export(csv_path, output_path)

        self.assertTrue(os.path.exists(output_path))
        import json
        with open(output_path) as f:
            data = json.load(f)
        self.assertEqual(data[0]["name"], "Alice")
```

### 方案三：使用 `io.StringIO` 避免真实文件

```python
import io

class TestReportGenerator(unittest.TestCase):

    def test_report_format(self):
        output = io.StringIO()
        generate_report(data=[{"name": "Alice", "score": 95}], stream=output)

        content = output.getvalue()
        self.assertIn("Alice", content)
        self.assertIn("95", content)
        self.assertIn("---", content)  # 报告分隔线
```

---

## 11.3 数据库测试策略

### 策略一：内存 SQLite（推荐，速度最快）

```python
import sqlite3
import unittest

class TestUserRepository(unittest.TestCase):

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
        self.addCleanup(self.conn.close)
        self.repo = UserRepository(self.conn)

    def _create_schema(self):
        self.conn.executescript("""
            CREATE TABLE users (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT    NOT NULL,
                email   TEXT    NOT NULL UNIQUE,
                active  INTEGER NOT NULL DEFAULT 1
            );
        """)

    def test_create_user_assigns_id(self):
        user = self.repo.create(name="Alice", email="alice@example.com")
        self.assertIsNotNone(user.id)
        self.assertGreater(user.id, 0)

    def test_find_by_email(self):
        self.repo.create(name="Bob", email="bob@example.com")
        found = self.repo.find_by_email("bob@example.com")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Bob")

    def test_find_nonexistent_returns_none(self):
        result = self.repo.find_by_email("ghost@example.com")
        self.assertIsNone(result)

    def test_duplicate_email_raises(self):
        self.repo.create(name="Alice", email="same@example.com")
        with self.assertRaises(IntegrityError):
            self.repo.create(name="Alice2", email="same@example.com")

    def test_deactivate_user(self):
        user = self.repo.create(name="Carol", email="carol@example.com")
        self.repo.deactivate(user.id)
        found = self.repo.find_by_id(user.id)
        self.assertFalse(found.active)
```

### 策略二：事务回滚（真实 DB，每测试干净）

```python
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class TestWithRealDatabase(unittest.TestCase):
    """使用真实数据库，但每个测试后回滚"""

    engine = None

    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("postgresql://localhost/test_db")
        Base.metadata.create_all(cls.engine)

    def setUp(self):
        self.connection = self.engine.connect()
        self.trans = self.connection.begin()
        Session = sessionmaker(bind=self.connection)
        self.session = Session()
        self.addCleanup(self.trans.rollback)
        self.addCleanup(self.session.close)
        self.addCleanup(self.connection.close)

    def test_complex_query(self):
        # 数据在测试结束后自动回滚，不污染数据库
        self.session.add(User(name="Alice"))
        self.session.flush()
        result = self.session.query(User).filter_by(name="Alice").first()
        self.assertIsNotNone(result)
```

### 策略三：Mock Repository（纯单元测试）

```python
class TestOrderService(unittest.TestCase):
    """Mock 掉数据库层，专注测试业务逻辑"""

    def setUp(self):
        self.user_repo = MagicMock()
        self.order_repo = MagicMock()
        self.service = OrderService(
            user_repo=self.user_repo,
            order_repo=self.order_repo
        )

    def test_place_order_saves_to_repo(self):
        self.user_repo.find_by_id.return_value = User(id=1, name="Alice")
        self.order_repo.save.return_value = Order(id=100, user_id=1)

        order = self.service.place_order(user_id=1, items=["apple"])

        self.order_repo.save.assert_called_once()
        call_args = self.order_repo.save.call_args[0][0]
        self.assertEqual(call_args.user_id, 1)

    def test_place_order_for_nonexistent_user_raises(self):
        self.user_repo.find_by_id.return_value = None

        with self.assertRaises(UserNotFoundError):
            self.service.place_order(user_id=999, items=["apple"])
```

---

## 11.4 HTTP/API 测试

### Mock requests

```python
from unittest.mock import patch, MagicMock
import requests

class TestWeatherService(unittest.TestCase):

    def setUp(self):
        self.service = WeatherService(api_key="test_key")

    def test_get_weather_returns_temperature(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "main": {"temp": 25.5},
            "weather": [{"description": "sunny"}]
        }

        with patch('requests.get', return_value=mock_response):
            weather = self.service.get_weather("London")

        self.assertAlmostEqual(weather.temperature, 25.5)
        self.assertEqual(weather.description, "sunny")

    def test_api_error_raises_service_exception(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "invalid api key"}

        with patch('requests.get', return_value=mock_response):
            with self.assertRaises(WeatherServiceError):
                self.service.get_weather("London")

    def test_network_timeout_raises_service_exception(self):
        with patch('requests.get', side_effect=requests.Timeout):
            with self.assertRaises(WeatherServiceError) as ctx:
                self.service.get_weather("London")
            self.assertIn("timeout", str(ctx.exception).lower())

    def test_correct_api_endpoint_called(self):
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {"main": {"temp": 20}, "weather": [{"description": "cloudy"}]}

        with patch('requests.get', return_value=mock_response) as mock_get:
            self.service.get_weather("Paris")

        called_url = mock_get.call_args[0][0]
        self.assertIn("Paris", called_url)
        called_params = mock_get.call_args[1].get('params', {})
        self.assertEqual(called_params.get('appid'), "test_key")
```

---

## 11.5 环境变量与配置测试

```python
import os
import unittest
from unittest.mock import patch

class TestDatabaseConfig(unittest.TestCase):

    def test_config_from_environment_variables(self):
        env = {
            "DB_HOST": "prod.example.com",
            "DB_PORT": "5432",
            "DB_NAME": "myapp",
        }
        with patch.dict(os.environ, env):
            config = DatabaseConfig.from_env()

        self.assertEqual(config.host, "prod.example.com")
        self.assertEqual(config.port, 5432)
        self.assertEqual(config.name, "myapp")

    def test_missing_required_env_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # 清除所有环境变量
            with self.assertRaises(ConfigError) as ctx:
                DatabaseConfig.from_env()
            self.assertIn("DB_HOST", str(ctx.exception))

    def test_default_port_when_not_set(self):
        env = {"DB_HOST": "localhost", "DB_NAME": "test"}
        with patch.dict(os.environ, env, clear=False):
            # 不清除其他环境变量，只添加指定的
            if "DB_PORT" in os.environ:
                del os.environ["DB_PORT"]
            config = DatabaseConfig.from_env()
        self.assertEqual(config.port, 5432)  # 默认端口
```

---

## 11.6 本章小结

| 场景 | 推荐方案 | 速度 |
|------|----------|------|
| 文件读写（单元） | `mock_open` + `patch('builtins.open')` | 极快 |
| 文件读写（集成） | `tempfile.mkdtemp` + `addCleanup` | 快 |
| 流式输出 | `io.StringIO` 注入 | 极快 |
| 数据库（单元） | Mock Repository | 极快 |
| 数据库（集成） | SQLite `:memory:` | 快 |
| 数据库（真实） | 事务回滚 | 中 |
| HTTP API | `patch('requests.get')` | 极快 |
| 环境变量 | `patch.dict(os.environ, ...)` | 极快 |

**下一部分**：高阶主题——测试替身模式、异步测试、覆盖率分析与 TDD 最佳实践。

# 第六章：测试夹具与生命周期

## 6.1 什么是测试夹具

**测试夹具（Test Fixture）**是测试运行所需的前置状态——数据库连接、配置对象、临时文件等。`unittest` 通过 `setUp/tearDown` 系列方法管理夹具的生命周期。

---

## 6.2 四个生命周期钩子

```
测试类级别：
  setUpClass()   ← 整个类运行前，仅一次
    测试方法级别：
      setUp()    ← 每个测试方法前
        test_xxx()
      tearDown() ← 每个测试方法后（即使测试失败也运行）
  tearDownClass() ← 整个类运行后，仅一次
```

### 完整示例

```python
import unittest
import tempfile
import os

class TestFileProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """类级别：创建共享的临时目录"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.config = load_config("test_config.json")
        print(f"\n[setUpClass] 临时目录: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        """类级别：清理临时目录"""
        import shutil
        shutil.rmtree(cls.temp_dir)
        print(f"\n[tearDownClass] 已清理: {cls.temp_dir}")

    def setUp(self):
        """方法级别：每个测试前准备独立文件"""
        self.test_file = os.path.join(self.temp_dir, "test_input.txt")
        with open(self.test_file, 'w') as f:
            f.write("line1\nline2\nline3\n")
        self.processor = FileProcessor(self.test_file)

    def tearDown(self):
        """方法级别：清理测试文件"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_line_count(self):
        self.assertEqual(self.processor.count_lines(), 3)

    def test_read_first_line(self):
        self.assertEqual(self.processor.first_line(), "line1")
```

---

## 6.3 setUp vs setUpClass 的选择

| 场景 | 使用 | 原因 |
|------|------|------|
| 数据库连接（重量级） | `setUpClass` | 连接建立开销大，复用同一个 |
| 测试数据（轻量级） | `setUp` | 每个测试需要干净的状态 |
| 临时文件路径 | `setUpClass` | 目录可共享 |
| 文件内容 | `setUp` | 每个测试写不同内容 |
| 全局配置 | `setUpClass` | 配置不变 |
| 被测对象实例 | `setUp` | 避免状态污染 |

---

## 6.4 tearDown 的可靠性保证

`tearDown` 在测试**通过或失败**后都会运行，但如果 `setUp` 本身抛出异常，`tearDown` 不会运行。

```python
def setUp(self):
    self.db = DatabaseConnection()   # 如果这里失败
    self.user = self.db.create_user() # tearDown 不会被调用

def tearDown(self):
    self.db.close()   # 可能导致资源泄漏
```

### 解决方案：addCleanup

```python
def setUp(self):
    self.db = DatabaseConnection()
    self.addCleanup(self.db.close)   # 无论 setUp 后面是否失败，都会清理
    
    self.user = self.db.create_user()
    self.addCleanup(self.db.delete_user, self.user.id)
```

`addCleanup` 的特性：
- 以 **LIFO（后进先出）** 顺序执行
- 即使测试失败，也会执行
- 即使前面的 cleanup 抛出异常，后面的 cleanup 仍会执行

---

## 6.5 addCleanup 的高级用法

```python
import unittest
import tempfile
import os

class TestWithCleanup(unittest.TestCase):

    def setUp(self):
        # 1. 创建临时文件，注册清理
        fd, self.temp_path = tempfile.mkstemp()
        os.close(fd)
        self.addCleanup(os.unlink, self.temp_path)

        # 2. 打开文件，注册关闭
        self.file = open(self.temp_path, 'w')
        self.addCleanup(self.file.close)

        # 3. 写入数据（依赖文件已打开）
        self.file.write("test content")
        self.file.flush()

    def test_file_content(self):
        with open(self.temp_path) as f:
            self.assertEqual(f.read(), "test content")

    # 无论 test_file_content 成功还是失败：
    # 清理顺序：file.close() → os.unlink(temp_path)
```

---

## 6.6 使用 contextlib 的现代夹具

```python
import unittest
from contextlib import contextmanager
from unittest.mock import patch

class TestEmailService(unittest.TestCase):

    def setUp(self):
        # 使用 patch 作为上下文管理器，并注册清理
        patcher = patch('myapp.email.send_smtp')
        self.mock_smtp = patcher.start()
        self.addCleanup(patcher.stop)

        # 设置返回值
        self.mock_smtp.return_value = {"status": "sent"}

    def test_welcome_email_sent(self):
        send_welcome_email("user@example.com")
        self.mock_smtp.assert_called_once()
```

---

## 6.7 数据库夹具模式

### 模式一：事务回滚（推荐）

```python
class TestUserRepository(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine)

    def setUp(self):
        """每个测试开启事务，测试后回滚"""
        self.connection = self.engine.__class__.connect(self.engine)
        self.transaction = self.connection.begin()
        self.session = Session(bind=self.connection)
        self.addCleanup(self.transaction.rollback)
        self.addCleanup(self.session.close)
        self.addCleanup(self.connection.close)
        self.repo = UserRepository(self.session)

    def test_create_user(self):
        user = self.repo.create(name="Alice", email="alice@example.com")
        self.assertIsNotNone(user.id)

    def test_find_by_email(self):
        self.repo.create(name="Bob", email="bob@example.com")
        found = self.repo.find_by_email("bob@example.com")
        self.assertEqual(found.name, "Bob")
    # 每个测试结束后自动回滚，数据不残留
```

### 模式二：内存数据库

```python
class TestWithSQLite(unittest.TestCase):

    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL
            )
        """)
        self.addCleanup(self.db.close)
```

---

## 6.8 参数化夹具

有时需要用不同配置重复运行同一组测试：

```python
class BaseStorageTest:
    """抽象基类（不继承 TestCase）"""
    
    def get_storage(self):
        raise NotImplementedError

    def test_save_and_retrieve(self):
        storage = self.get_storage()
        storage.save("key", "value")
        self.assertEqual(storage.get("key"), "value")

    def test_overwrite_existing(self):
        storage = self.get_storage()
        storage.save("key", "v1")
        storage.save("key", "v2")
        self.assertEqual(storage.get("key"), "v2")


class TestMemoryStorage(BaseStorageTest, unittest.TestCase):
    def get_storage(self):
        return MemoryStorage()


class TestFileStorage(BaseStorageTest, unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def get_storage(self):
        return FileStorage(self.temp_dir)
```

> 注意：`unittest.TestCase` 必须放在 MRO（方法解析顺序）的后面。

---

## 6.9 常见夹具模式汇总

```python
class TestPatternsDemo(unittest.TestCase):

    def setUp(self):
        # 模式1：构建测试对象
        self.sut = SystemUnderTest()   # sut = System Under Test

        # 模式2：准备输入数据
        self.valid_input = {"name": "Alice", "age": 25}
        self.invalid_input = {"name": "", "age": -1}

        # 模式3：注册清理（优于 tearDown）
        self.addCleanup(self.sut.shutdown)

        # 模式4：捕获输出（测试 print/logging）
        import io
        self.captured_output = io.StringIO()
        import sys
        self.original_stdout = sys.stdout
        sys.stdout = self.captured_output
        self.addCleanup(setattr, sys, 'stdout', self.original_stdout)
```

---

## 6.10 本章小结

- 四个生命周期钩子：`setUpClass > setUp > tearDown > tearDownClass`
- `setUp` 保证测试隔离，`setUpClass` 分摊重型资源开销
- `addCleanup` 比 `tearDown` 更可靠（LIFO 顺序，setUp 失败时也能清理）
- 数据库测试推荐事务回滚模式
- 共享测试行为用多继承 + 抽象基类

**下一章**：测试套件的组织与自动发现机制。

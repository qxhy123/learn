"""
Part 2 示例：核心概念 - 断言、夹具、套件
运行方式：python -m unittest examples/part2_examples.py -v
"""
import unittest
import tempfile
import os
import json
import io


# ── 生产代码 ────────────────────────────────────────────────────────────────

def validate_email(email: str) -> dict:
    """邮箱验证，返回 {'is_valid': bool, 'error': str|None}"""
    if not isinstance(email, str):
        return {"is_valid": False, "error": "must be a string"}
    if "@" not in email:
        return {"is_valid": False, "error": "missing @"}
    local, _, domain = email.partition("@")
    if not local:
        return {"is_valid": False, "error": "empty local part"}
    if not domain or "." not in domain:
        return {"is_valid": False, "error": "invalid domain"}
    if " " in email:
        return {"is_valid": False, "error": "contains spaces"}
    return {"is_valid": True, "error": None}


class JSONFileStore:
    """JSON 文件存储（用于夹具演示）"""

    def __init__(self, path: str):
        self.path = path
        self._data = {}
        if os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f)

    def save(self, key: str, value) -> None:
        self._data[key] = value
        with open(self.path, 'w') as f:
            json.dump(self._data, f)

    def get(self, key: str):
        return self._data.get(key)

    def all_keys(self) -> list:
        return list(self._data.keys())


def generate_report(records: list, stream=None) -> str:
    """生成文本报告，可写入 stream 或返回字符串"""
    lines = ["=== Report ==="]
    for rec in records:
        lines.append(f"  {rec['name']}: {rec['score']}")
    lines.append(f"Total: {len(records)} records")
    content = "\n".join(lines)
    if stream:
        stream.write(content)
    return content


# ── 断言演示 ────────────────────────────────────────────────────────────────

class TestAssertions(unittest.TestCase):
    """第五章：断言深度解析"""

    # 浮点数断言
    def test_float_delta(self):
        self.assertAlmostEqual(0.1 + 0.2, 0.3, delta=1e-9)

    def test_float_places(self):
        self.assertAlmostEqual(3.14159, 3.14, places=2)

    # 容器断言
    def test_dict_equal_shows_diff(self):
        actual = {"a": 1, "b": 2}
        expected = {"a": 1, "b": 2}
        self.assertDictEqual(actual, expected)

    def test_count_equal_ignores_order(self):
        self.assertCountEqual(["c", "a", "b"], ["a", "b", "c"])

    # 字符串断言
    def test_regex_match(self):
        log_line = "2024-01-15 ERROR: something failed"
        self.assertRegex(log_line, r"\d{4}-\d{2}-\d{2} ERROR")

    # 自定义断言
    def assertEmailValid(self, email):
        result = validate_email(email)
        self.assertTrue(result["is_valid"],
            f"Expected '{email}' to be valid, got error: {result['error']}")

    def assertEmailInvalid(self, email, expected_error=None):
        result = validate_email(email)
        self.assertFalse(result["is_valid"],
            f"Expected '{email}' to be invalid, but it was accepted")
        if expected_error:
            self.assertIn(expected_error, result["error"])

    def test_valid_emails(self):
        valid = ["user@example.com", "a@b.co", "user+tag@sub.domain.org"]
        for email in valid:
            with self.subTest(email=email):
                self.assertEmailValid(email)

    def test_invalid_emails(self):
        cases = [
            ("notanemail",    "missing @"),
            ("@domain.com",  "empty local"),
            ("user@",        "invalid domain"),
            ("a b@c.com",    "spaces"),
        ]
        for email, error_hint in cases:
            with self.subTest(email=email):
                self.assertEmailInvalid(email)


# ── 夹具演示 ────────────────────────────────────────────────────────────────

class TestJSONFileStore(unittest.TestCase):
    """第六章：夹具与生命周期"""

    @classmethod
    def setUpClass(cls):
        """类级别：创建共享临时目录"""
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """类级别：清理临时目录"""
        import shutil
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """方法级别：每个测试用独立的文件"""
        self.store_path = os.path.join(self.temp_dir, f"store_{id(self)}.json")
        self.store = JSONFileStore(self.store_path)
        self.addCleanup(self._cleanup_file)

    def _cleanup_file(self):
        if os.path.exists(self.store_path):
            os.remove(self.store_path)

    def test_save_and_retrieve_string(self):
        self.store.save("greeting", "hello")
        self.assertEqual(self.store.get("greeting"), "hello")

    def test_save_and_retrieve_dict(self):
        self.store.save("user", {"name": "Alice", "age": 30})
        user = self.store.get("user")
        self.assertEqual(user["name"], "Alice")

    def test_get_missing_key_returns_none(self):
        self.assertIsNone(self.store.get("nonexistent"))

    def test_overwrite_existing_key(self):
        self.store.save("key", "v1")
        self.store.save("key", "v2")
        self.assertEqual(self.store.get("key"), "v2")

    def test_all_keys_lists_saved_keys(self):
        self.store.save("a", 1)
        self.store.save("b", 2)
        self.assertCountEqual(self.store.all_keys(), ["a", "b"])

    def test_data_persists_across_instances(self):
        """验证数据真正写到了文件"""
        self.store.save("persisted", "value")
        # 创建新实例，从同一文件读取
        new_store = JSONFileStore(self.store_path)
        self.assertEqual(new_store.get("persisted"), "value")


# ── 报告生成器（stream 注入）────────────────────────────────────────────────

class TestReportGenerator(unittest.TestCase):
    """使用 io.StringIO 测试输出"""

    def test_report_contains_all_names(self):
        records = [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]
        content = generate_report(records)
        self.assertIn("Alice", content)
        self.assertIn("Bob", content)

    def test_report_shows_total_count(self):
        records = [{"name": "Alice", "score": 95}]
        content = generate_report(records)
        self.assertIn("1 records", content)

    def test_report_writes_to_stream(self):
        stream = io.StringIO()
        records = [{"name": "Carol", "score": 78}]
        generate_report(records, stream=stream)
        output = stream.getvalue()
        self.assertIn("Carol", output)
        self.assertIn("78", output)

    def test_empty_report(self):
        content = generate_report([])
        self.assertIn("0 records", content)


# ── 基类共享测试 ─────────────────────────────────────────────────────────────

class BaseStorageTest:
    """抽象基类：定义存储接口的通用测试"""

    def get_store(self):
        raise NotImplementedError

    def test_save_and_get(self):
        store = self.get_store()
        store["key"] = "value"
        self.assertEqual(store["key"], "value")

    def test_overwrite(self):
        store = self.get_store()
        store["key"] = "v1"
        store["key"] = "v2"
        self.assertEqual(store["key"], "v2")

    def test_missing_key_raises(self):
        store = self.get_store()
        with self.assertRaises(KeyError):
            _ = store["nonexistent"]


class TestDictAsStorage(BaseStorageTest, unittest.TestCase):
    def get_store(self):
        return {}


if __name__ == '__main__':
    unittest.main(verbosity=2)

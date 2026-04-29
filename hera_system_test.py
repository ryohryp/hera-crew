"""hera_system_test.py の機能テストモジュール。

本モジュールは、Hera システムの基本的な動作をテストします。
"""

import unittest

class HeraGreeter:
    """挨拶を生成するクラス。"""
    def __init__(self, name: str):
        self.name = name

    def greet(self) -> str:
        """指定された名前で挨拶を返します。"""
        return f"Hello, {self.name}!"

class TestHeraGreeter(unittest.TestCase):
    """HeraGreeter クラスの単体テストクラス."""

    def test_greeting(self):
        """Greeting メソッドの正常動作を確認します."""
        greeter = HeraGreeter("Hera")
        self.assertEqual(greeter.greet(), "Hello, Hera!")

if __name__ == "__main__":
    unittest.main()

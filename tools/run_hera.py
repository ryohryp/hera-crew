"""
HERA 直接実行用のシンプルなランナー。

使い方 (PowerShell):
  # ファイル指定 (推奨、長文に最適)
  .\venv\Scripts\python.exe tools/run_hera.py --file prompt.txt

  # コマンドライン引数で短いプロンプト
  .\venv\Scripts\python.exe tools/run_hera.py "Hello"

  # 標準入力 (短文のみ。長文は --file 推奨)
  echo "Hello" | .\venv\Scripts\python.exe tools/run_hera.py

実行後、最終応答が標準出力に出ます。
中間ログ (Step毎の進行状況) は stderr に出ます。
HTML レポートは reports/hera_report.html に保存されます。
"""
import asyncio
import os
import sys
from pathlib import Path

# Windows のコンソール文字化け対策
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.platform == "win32" and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# プロジェクトの src/ を sys.path に追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hera_crew.crew import HeraCrew
from hera_crew.utils.env_setup import setup_environment


def read_prompt() -> str:
    """
    プロンプトを取得する。優先順位:
      1. --file <path> でファイル指定 (推奨、長文に最適)
      2. コマンドライン引数 (短文用)
      3. 標準入力 (PowerShellのパイプは長文だと不安定なので非推奨)
    """
    # 1. --file <path>
    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv):
            with open(sys.argv[idx + 1], encoding="utf-8") as f:
                return f.read()
        else:
            print("ERROR: --file の後にパスを指定してください", file=sys.stderr)
            sys.exit(1)

    # 2. コマンドライン引数 (--file 以外)
    args = [a for a in sys.argv[1:] if a != "--file"]
    # --file の引数があれば除去
    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv):
            try:
                args.remove(sys.argv[idx + 1])
            except ValueError:
                pass
    if args:
        return " ".join(args)

    # 3. 標準入力 (バイナリで読んでUTF-8デコード、エンコーディング不安定対策)
    if not sys.stdin.isatty():
        try:
            return sys.stdin.buffer.read().decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            return sys.stdin.read()
    return ""


async def main() -> int:
    prompt = read_prompt().strip()
    if not prompt:
        print(__doc__, file=sys.stderr)
        print("ERROR: プロンプトが空です。引数か標準入力で渡してください。", file=sys.stderr)
        return 1

    print(f"=== HERA 直接実行 (prompt {len(prompt)} chars) ===", file=sys.stderr)
    setup_environment()
    crew = HeraCrew()
    result = await crew.run(prompt)

    # 最終応答を stdout に
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

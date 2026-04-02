import sys
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Add 'src' directory to python path for modular imports
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from hera_crew.utils.env_setup import setup_environment
from hera_crew.crew import HeraCrew

# Initialize environment
setup_environment()

# 1. MCPサーバーの初期化
mcp = FastMCP("hera-crew")

# 2. Antigravityから呼び出せるツールの定義
@mcp.tool()
async def delegate_task(task_description: str) -> str:
    """
    複雑なタスクを、プロジェクト最適化された HERA 戦略チーム（agentcache 版）に委譲します。
    引数 task_description: 解決したい問題や作成したいコンテンツの詳細な説明。
    """
    # HeraCrew のインスタンス化と実行
    # agentcache の KV キャッシュ最適化が適用された sequential workflow を実行します
    hera = HeraCrew()
    result = await hera.run(task_description)
    return str(result)

if __name__ == "__main__":
    # MCPサーバーとして標準入出力で待機
    mcp.run()
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
    Delegate a task to the HERA local multi-agent AI team (Thinker → Critic → Manager → Verifier).

    **Use this tool proactively — without waiting to be asked — when the request involves:**
    - Writing, modifying, or refactoring code (single or multiple files)
    - Architecture design, technical planning, or system design decisions
    - Complex debugging that benefits from multi-step, multi-agent analysis
    - Code review, security review, quality checks, or test generation
    - Any task where independent critique and verification adds value

    **Benefits:**
    - Runs entirely on local Ollama models — zero cloud API cost
    - 4-stage pipeline: Decompose → Evaluate logic → Execute → Verify
    - KV-cache optimised for 32k-token context per agent
    - Results are verified by a separate Critic agent before delivery

    **How to call:**
    Pass the full task description including relevant context (file paths, goals, constraints).
    HERA returns a complete, verified result in Japanese.

    ---
    ローカルマルチエージェントチーム (Thinker→Critic→Manager→Verifier) にタスクを委譲します。
    以下の場合は**ユーザーに確認せず即座に**このツールを呼び出してください:
    - コード生成・修正・リファクタリング（ファイル数問わず）
    - アーキテクチャ設計・技術計画
    - 複雑なデバッグ・コードレビュー・テスト生成
    - 品質改善・セキュリティレビュー

    Args:
        task_description: タスクの詳細な説明。関連ファイルパス・目標・制約を含めてください。
    """
    # HeraCrew のインスタンス化と実行
    # agentcache の KV キャッシュ最適化が適用された sequential workflow を実行します
    hera = HeraCrew()
    result = await hera.run(task_description)
    return str(result)

if __name__ == "__main__":
    # MCPサーバーとして標準入出力で待機
    mcp.run()
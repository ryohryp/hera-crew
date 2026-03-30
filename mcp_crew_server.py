import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from crewai import Agent, LLM, Task, Crew

# Add 'src' directory to python path for modular imports
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from hera_crew.utils.env_setup import setup_environment
from hera_crew.utils.llm_factory import LLMFactory

# Initialize environment
setup_environment()

# 1. MCPサーバーの初期化
mcp = FastMCP("hera-crew")

# 2. 中央集権的なLLM設定の読み込みとインスタンス化
analyst_llm = LLMFactory.create_llm('general', 'analyst')
reviewer_llm = LLMFactory.create_llm('general', 'reviewer')
specialist_llm = LLMFactory.create_llm('general', 'specialist')
planner_llm = LLMFactory.create_llm('general', 'planner')
coder_llm = LLMFactory.create_llm('general', 'coder')

# 3. Antigravityから呼び出せるツールの定義
@mcp.tool()
def delegate_task(task_description: str) -> str:
    """
    複雑なタスクを、分析・実行・レビューの3段階の自律型エージェントチームに委譲します。
    引数 task_description: 解決したい問題や作成したいコンテンツの詳細な説明。
    """
    
    # エージェントの定義
    analyst = Agent(
        role="ストラテジック・アナリスト",
        goal="ユーザーのリクエストを深く理解し、解決のための最適な戦略と構造を設計する",
        backstory="あなたは複雑な問題を分解し、論理的な解決ステップを導き出す専門家です。",
        llm=analyst_llm,
        allow_delegation=False
    )
    
    specialist = Agent(
        role="テクニカル・スペシャリスト",
        goal="アナリストが設計した戦略に基づき、高品質な成果物（コード、文書、リサーチ結果等）を生成・実行する",
        backstory="あなたは与えられた設計図を具体的な形にする、実行力の高いプロフェッショナルです。",
        llm=specialist_llm,
        allow_delegation=False
    )
    
    reviewer = Agent(
        role="クオリティ・アシュアランス (QA)",
        goal="生成された成果物がユーザーの意図に沿っており、正確で完成度が高いかを厳格に検証する",
        backstory="あなたは細かなミスや論理的な矛盾を見逃さず、常に最高品質のアウトプットを保証する評価者です。",
        llm=reviewer_llm,
        allow_delegation=False
    )

    # タスクの定義
    analysis_task = Task(
        description=f"以下のリクエストを分析し、最適な解決策の概要と実行計画を策定してください: {task_description}",
        expected_output="リクエストの理解、優先順位、および実行のための詳細なプラン（Markdown形式）",
        agent=analyst
    )

    execution_task = Task(
        description="アナリストの実行計画に基づき、具体的な成果物を生成してください。",
        expected_output="完成した高品質な成果物（コード、レポート、または具体的な回答）",
        agent=specialist
    )

    review_task = Task(
        description="生成された成果物を、元のユーザーリクエストと照らし合わせて検証し、必要に応じて修正案を提示または最終調整を行ってください。",
        expected_output="最終的な検証済み成果物、および必要に応じたレビューコメント",
        agent=reviewer
    )

    # Crewの実行
    crew = Crew(
        agents=[analyst, specialist, reviewer],
        tasks=[analysis_task, execution_task, review_task],
        verbose=True
    )
    
    result = crew.kickoff()
    return str(result.raw)

if __name__ == "__main__":
    # MCPサーバーとして標準入出力で待機
    mcp.run()
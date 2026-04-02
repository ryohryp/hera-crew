import asyncio
import time
from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider

# LiteLLM経由でOllamaのモデルを指定します
provider = LiteLLMSDKProvider()
MODEL = "ollama/qwen2.5-coder:14b"

# 意図的に長いシステムプロンプト（hera-crewの統合ルールを想定）
# ※キャッシュの効果をわかりやすくするため、ダミーのテキストで長さをかさ増ししています
SYSTEM_PROMPT = """
You are 'hera-crew', an autonomous development squad optimized for 16GB VRAM.
Your mission is to write high-quality, bug-free code.
Available roles:
- Tech Lead: Writes Next.js/TypeScript code.
- Critic: Reviews code and finds logical flaws.
Always think step-by-step.
""" + ("\nRule: Always strictly follow the architectural guidelines." * 50)

async def main():
    print("=== 1. 親セッションの起動 ===")
    print("（初回は長いシステムプロンプトをGPUのKVキャッシュに読み込むため、少し時間がかかります）")
    t0 = time.time()
    
    session = AgentSession(
        model=MODEL,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
    )
    # キャッシュを生成させるための最初のやり取り
    await session.respond("System initialization. Are you ready?")
    
    print(f"✅ 初回ロード＆推論時間: {time.time() - t0:.2f}秒\n")

    print("=== 2. Tech Lead タスク (Fork) ===")
    print("（システムプロンプトがキャッシュから読まれるため、推論開始が爆速になるはずです！）")
    t1 = time.time()
    
    tech_lead = await session.fork(
        prompt="Act as Tech Lead. Write a simple Python function for Fibonacci.",
        policy=ForkPolicy.cache_safe_ephemeral() # メモリを汚さず分岐
    )
    
    print(f"✅ Tech Lead 処理時間: {time.time() - t1:.2f}秒")
    # print(tech_lead.final_text) # 結果を見たい場合はコメントアウトを外す

    print("\n=== 3. Critic タスク (Fork) ===")
    print("（並行して別のタスクを投げても、同じくキャッシュが効きます）")
    t2 = time.time()
    
    critic = await session.fork(
        prompt="Act as Critic. Explain what a Fibonacci function does in 1 sentence.",
        policy=ForkPolicy.cache_safe_ephemeral()
    )
    
    print(f"✅ Critic 処理時間: {time.time() - t2:.2f}秒")

    print("\n=== キャッシュ・ステータス ===")
    # OllamaAPIの仕様上、トークン数の正確なレポートが出ないことがありますが、
    # タイムの短縮でキャッシュヒットが証明されています。
    print(session.cache_status().pretty())

if __name__ == "__main__":
    asyncio.run(main())
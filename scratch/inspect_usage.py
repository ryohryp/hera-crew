
import asyncio
import os
from hera_crew.utils.llm_factory import LLMFactory
from hera_crew.utils.usage_tracker import UsageTracker
from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider

async def main():
    tracker = UsageTracker()
    model_cfg = LLMFactory.create_llm_config("hera", "critic")
    provider = LiteLLMSDKProvider()
    
    tracker.register_litellm(model_cfg['model'])
    
    session = AgentSession(
        model=model_cfg['model'],
        provider=provider,
        system_prompt="You are a test assistant.",
    )
    
    print("Sending first request...")
    await session.respond("Hello, this is a test.")
    
    print("Forking...")
    fork = await session.fork(prompt="Repeat 'OK' and nothing else.", policy=ForkPolicy.cache_safe_ephemeral())
    
    print(f"Fork text: {fork.final_text}")
    print(f"Has usage attribute: {hasattr(fork, 'usage')}")
    if hasattr(fork, 'usage'):
        print(f"Usage: {fork.usage}")
        # Check standard attributes
        for attr in ['input_tokens', 'output_tokens', 'prompt_tokens', 'completion_tokens']:
            print(f"  {attr}: {getattr(fork.usage, attr, 'N/A')}")
            
    print(f"Fork text length: {len(fork.final_text) if fork.final_text else 0}")
    print(f"Fork attributes: {[a for a in dir(fork) if not a.startswith('_')]}")
    print(f"Session attributes: {[a for a in dir(session) if not a.startswith('_')]}")
    
    # Try common names for conversation/history
    for candidate in ['conversation', 'messages', 'history', 'responses']:
        if hasattr(session, candidate):
            print(f"Found session.{candidate}")
            
    if hasattr(fork, 'usage'):
        print(f"Manually recording usage via record_agent_usage: {fork.usage}")
        tracker.record_agent_usage(fork.usage)
        
    print(f"Tracker records count: {len(tracker._records)}")
    for r in tracker._records:
        print(f"  Record: {r}")

if __name__ == "__main__":
    asyncio.run(main())

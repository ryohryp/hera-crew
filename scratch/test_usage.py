import asyncio
from agentcache import AgentSession, LiteLLMSDKProvider, ForkPolicy
import litellm

async def main():
    provider = LiteLLMSDKProvider()
    model = "ollama/gemma4:26b"
    
    session = AgentSession(
        model=model,
        provider=provider,
        system_prompt="You are a helpful assistant.",
    )
    
    print("Sending fork request...")
    fork = await session.fork(
        prompt="Hello, this is a test for token tracking.",
        policy=ForkPolicy.cache_safe_ephemeral()
    )
    
    print(f"Fork object type: {type(fork)}")
    print(f"Fork attributes: {dir(fork)}")
    
    # Let's check if it has 'final_text' (as seen in crew.py)
    if hasattr(fork, "final_text"):
        print(f"Found final_text: {fork.final_text[:50]}...")
    
    # Look for usage in fork or its internal responses
    if hasattr(fork, "usage"):
        print(f"Found usage on fork: {fork.usage}")
    
    # Does it have a list of responses?
    if hasattr(fork, "responses"):
        print(f"Found responses: {len(fork.responses)}")
        for i, resp in enumerate(fork.responses):
            print(f"Response {i} type: {type(resp)}")
            if hasattr(resp, "usage"):
                print(f"Response {i} usage: {resp.usage}")

if __name__ == "__main__":
    asyncio.run(main())

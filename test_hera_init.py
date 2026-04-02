import asyncio
import os
import sys
from pathlib import Path

# Add src to sys.path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from hera_crew.crew import HeraCrew
from hera_crew.utils.llm_factory import LLMFactory

async def test_init():
    print("--- Testing HeraCrew Initialization ---")
    try:
        crew = HeraCrew()
        print("✅ HeraCrew initialized successfully.")
        print(f"Model Config: {crew.model_cfg}")
        print(f"System Prompt Head: {crew.shared_system_prompt[:200]}...")
        
        # Test if the unified prompt contains all roles
        for role in ['Orchestrator Manager', 'Bridge Thinker', 'The Quality Critic']:
            if role in crew.shared_system_prompt:
                print(f"✅ Role '{role}' found in unified prompt.")
            else:
                print(f"❌ Role '{role}' NOT found in unified prompt.")
                
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_init())

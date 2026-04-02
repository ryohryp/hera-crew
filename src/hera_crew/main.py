import sys
import os
import asyncio

# Add 'src' directory to python path for modular imports
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from hera_crew.utils.env_setup import setup_environment
from hera_crew.crew import HeraCrew

async def main():
    """
    Run the crew using agentcache.
    """
    # Load environment variables and optimizations
    setup_environment()
    print(f"--- Starting HERA Multi-Agent System (agentcache Optimized) ---")
    
    user_request = input("Please enter your development task (Japanese/English): ")
    if not user_request:
        user_request = "Create a simple Python script for calculating prime numbers."

    try:
        crew = HeraCrew()
        result = await crew.run(user_request)
        
        print("\n\n########################")
        print("## HERA FINAL OUTPUT ##")
        print("########################\n")
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

from hera_crew.crew import HeraCrew

async def main():
    crew = HeraCrew()
    print(f"Testing HeraCrew run...")
    # Use a tiny task
    result = await crew.run("Say 'OK' for token test.")
    print(f"Result: {result}")
    
    print("\n--- Tracker Stats ---")
    print(f"Total tokens: {crew.tracker.total_tokens}")
    print(f"Call count: {crew.tracker.call_count}")
    
    # Check history.jsonl
    history_file = Path(__file__).resolve().parent.parent / "reports" / "history.jsonl"
    with open(history_file, "r", encoding="utf-8") as f:
        last_line = f.readlines()[-1]
        print(f"Last history line: {last_line}")

if __name__ == "__main__":
    asyncio.run(main())

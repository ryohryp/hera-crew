import sys
import os

# Force UTF-8 for stdout/stderr to prevent encoding errors on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add 'src' directory to python path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(root_path, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from hera_crew.crew import HeraCrew

# Environmental overrides for testing (Local Ollama)
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OLLAMA_HOST"] = "http://localhost:11434"

import asyncio

def test_run():
    # Simple "Hello" level prompt for basic testing
    user_request = "Hello"
    
    print(f"Testing with request: {user_request}")
    try:
        # Note: HeraCrew().run() is an async method
        result = asyncio.run(HeraCrew().run(user_request))
        print("\n\n########################")
        print("## TEST RESULT ##")
        print("########################\n")
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_run()

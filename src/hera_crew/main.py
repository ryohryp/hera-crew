import sys
import os

# Force UTF-8 for stdout/stderr to prevent encoding errors on Windows (cp932)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# Add 'src' directory to python path for modular imports
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from hera_crew.crew import HeraCrew

def run():
    """
    Run the crew.
    """
    # Load environment variables
    load_dotenv()
    
    # Environment optimizations
    num_parallel = os.getenv("OLLAMA_NUM_PARALLEL", "4")
    os.environ["OLLAMA_NUM_PARALLEL"] = str(num_parallel)
    print(f"--- Starting HERA Multi-Agent System (Parallel: {num_parallel}) ---")
    
    # Using environment LLM (no explicit key requirement)

    user_request = input("Please enter your development task (Japanese/English): ")
    if not user_request:
        user_request = "Create a simple Python script for calculating prime numbers."

    inputs = {
        'user_request': user_request,
        'manifest': 'Initial decomposition needed',
        'current_subtask': 'Starting workflow'
    }
    
    try:
        result = HeraCrew().crew().kickoff(inputs=inputs)
        print("\n\n########################")
        print("## HERA FINAL OUTPUT ##")
        print("########################\n")
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run()

import sys
import os

# Add 'src' directory to python path for modular imports
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from hera_crew.utils.env_setup import setup_environment
from hera_crew.crew import HeraCrew

def run():
    """
    Run the crew.
    """
    # Load environment variables and optimizations
    setup_environment()
    print(f"--- Starting HERA Multi-Agent System (Parallel: {os.environ.get('OLLAMA_NUM_PARALLEL', '4')}) ---")
    
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

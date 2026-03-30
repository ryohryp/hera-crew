import sys
import os

# Force UTF-8 for stdout/stderr to prevent encoding errors on Windows (cp932)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add 'src' directory to python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from my_hera_crew.crew import MyHeraCrew

# Environmental overrides for testing
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OLLAMA_HOST"] = "http://localhost:11434"

def test_run():
    user_request = "宇宙物理学に基づき、一般相対性理論を考慮した人工衛星の軌道シミュレータを実装せよ。数値積分にはRunge-Kutta法を使用し、PyTorchでの高速化も検討すること。"
    
    inputs = {
        'user_request': user_request,
        'manifest': 'Deep technical analysis required',
        'current_subtask': 'Starting complex orbital mechanics task'
    }
    
    print(f"Testing with request: {user_request}")
    try:
        result = MyHeraCrew().crew().kickoff(inputs=inputs)
        print("\n\n########################")
        print("## TEST RESULT ##")
        print("########################\n")
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_run()

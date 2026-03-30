import os
from crewai import LLM

# Set environment
os.environ["OPENAI_API_KEY"] = "NA"

def test_options_dict():
    print("Testing options dict call...")
    try:
        # Testing options={} approach
        llm = LLM(
            model="ollama/qwen2.5:14b", 
            base_url="http://localhost:11434",
            options={"num_ctx": 32768}
        )
        print("Initialized. Calling...")
        res = llm.call("Hi")
        print(f"Success! Result: {res}")
    except Exception as e:
        print(f"Call error: {e}")

if __name__ == "__main__":
    test_options_dict()

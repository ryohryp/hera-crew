import os
from crewai import LLM

# Set environment
os.environ["OPENAI_API_KEY"] = "NA"

def test_extra_body_call():
    print("Testing extra_body call...")
    try:
        # Pass num_ctx via extra_body (for OpenAI client fallback mode)
        llm = LLM(
            model="ollama/qwen2.5:14b", 
            base_url="http://localhost:11434",
            extra_body={"num_ctx": 32768}
        )
        print("Initialized. Calling...")
        res = llm.call("Hi")
        print(f"Success! Result: {res}")
    except Exception as e:
        print(f"Call error: {e}")

if __name__ == "__main__":
    test_extra_body_call()

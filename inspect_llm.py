from crewai import LLM
import inspect

print("--- LLM Init Signature ---")
print(inspect.signature(LLM.__init__))

print("\n--- LLM Fields (if Pydantic) ---")
try:
    for field in LLM.__fields__:
        print(field)
except Exception:
    print("Not a standard Pydantic v1 model or __fields__ not available")

try:
    for field in LLM.model_fields:
        print(field)
except Exception:
    print("Not a standard Pydantic v2 model or model_fields not available")

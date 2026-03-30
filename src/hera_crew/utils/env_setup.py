import os
import sys
from dotenv import load_dotenv

def setup_environment():
    """
    Standard environment setup for Hera-Crew components.
    Handles UTF-8 configuration, environment variables, and LiteLLM/CrewAI settings.
    """
    # Force UTF-8 for stdout/stderr to prevent encoding errors on Windows (cp932)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Load environment variables from .env
    # We look for .env in the current directory or parent directory
    load_dotenv()

    # Prevent LiteLLM from failing due to missing OpenAI API Key and disable telemetry
    # These are required when using only local models (Ollama)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "NA"
    
    os.environ["CREWAI_TELEMETRY"] = "false"
    os.environ["OTEL_SDK_DISABLED"] = "true"

    # Optimization for Ollama parallel execution
    if "OLLAMA_NUM_PARALLEL" not in os.environ:
        os.environ["OLLAMA_NUM_PARALLEL"] = "4"

import os
import sys
from pathlib import Path
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
    # Search from this file's location upward to find the project root .env
    # This ensures the .env is found regardless of the working directory (e.g., when called as MCP server)
    _here = Path(__file__).resolve()
    for parent in [_here.parent, _here.parent.parent, _here.parent.parent.parent, _here.parent.parent.parent.parent]:
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
            break
    else:
        load_dotenv()  # fallback to default behavior

    # Prevent LiteLLM from failing due to missing OpenAI API Key and disable telemetry
    # These are required when using only local models (Ollama)
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "NA"
    
    # Disable LiteLLM's attempts to call OpenAI for model cost lookups etc.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    # Disable any background retries/callbacks that might try external endpoints
    os.environ["LITELLM_DROP_PARAMS"] = "True"

    os.environ["CREWAI_TELEMETRY"] = "false"
    os.environ["OTEL_SDK_DISABLED"] = "true"

    # Optimization for Ollama parallel execution
    if "OLLAMA_NUM_PARALLEL" not in os.environ:
        os.environ["OLLAMA_NUM_PARALLEL"] = "4"

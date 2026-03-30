import os
import yaml
from pathlib import Path
from crewai import LLM

class LLMFactory:
    """
    Centralized factory for creating CrewAI LLM instances.
    Handles configuration loading, environment overrides, and Ollama-specific logic.
    """
    
    _config = None
    _config_path = Path(__file__).parent.parent / "config" / "llms.yaml"

    @classmethod
    def _load_config(cls):
        """Lazy load the llms.yaml configuration."""
        if cls._config is None:
            if not cls._config_path.exists():
                raise FileNotFoundError(f"LLM configuration not found at {cls._config_path}")
            
            with open(cls._config_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
        return cls._config

    @classmethod
    def create_llm(cls, group: str, name: str, env_override: str = None) -> LLM:
        """
        Creates a CrewAI LLM instance based on the configuration.
        
        Args:
            group: The configuration group (e.g., 'hera' or 'general').
            name: The model identifier within the group (e.g., 'manager', 'analyst').
            env_override: Optional environment variable name to override the model name.
        """
        config = cls._load_config()
        group_config = config.get(group, {})
        model_cfg = group_config.get(name)
        
        if not model_cfg:
            raise ValueError(f"Model configuration '{name}' not found in group '{group}'")

        # Extract settings
        model = os.getenv(env_override) if env_override else None
        if not model:
            model = model_cfg.get('model')
            
        timeout = model_cfg.get('timeout', 120)
        num_ctx = model_cfg.get('num_ctx', 32768)
        base_url = os.getenv("OLLAMA_BASE_URL", config.get("default_ollama_base_url"))

        # Ollama specific handling
        if "ollama" in model.lower():
            return LLM(
                model=model, 
                base_url=base_url, 
                timeout=timeout,
                api_key="NA",
                extra_body={"num_ctx": num_ctx}
            )
        
        # General handling
        return LLM(model=model, timeout=timeout)

    @classmethod
    def get_group_llms(cls, group: str) -> dict:
        """Helper to create all LLMs in a specific group."""
        config = cls._load_config()
        group_config = config.get(group, {})
        return {name: cls.create_llm(group, name) for name in group_config.keys()}

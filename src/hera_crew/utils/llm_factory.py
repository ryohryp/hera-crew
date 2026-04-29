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
    def create_llm_config(cls, group: str, name: str, env_override: str = None) -> dict:
        """
        Returns model configuration for agentcache/LiteLLM.
        
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
            
        timeout = model_cfg.get('timeout', 600)
        num_ctx = model_cfg.get('num_ctx', 32768)
        base_url = os.getenv("OLLAMA_BASE_URL", config.get("default_ollama_base_url"))

        # Format for LiteLLM if it's Ollama
        if "ollama/" not in model.lower() and "ollama" in model.lower():
            model = f"ollama/{model}"

        return {
            "model": model,
            "base_url": base_url,
            "timeout": timeout,
            "num_ctx": num_ctx
        }

    # Default env-var override map for the 'hera' group.
    # 3役分離構成で各役のモデルを環境変数から差し替えられるようにする。
    HERA_ENV_OVERRIDES = {
        "manager": "MANAGER_MODEL",
        "thinker": "THINKER_MODEL",
        "critic": "CRITIC_MODEL",
        "tool_calling": "TOOL_CALLING_MODEL",
    }

    @classmethod
    def get_group_llms(cls, group: str, env_overrides: dict | None = None) -> dict:
        """
        Helper to create all LLM configs in a specific group.

        Args:
            group: Group name in llms.yaml (e.g. 'hera').
            env_overrides: Optional mapping {role_name: ENV_VAR_NAME}. If group is
                'hera' and this is omitted, HERA_ENV_OVERRIDES is used by default.
        """
        config = cls._load_config()
        group_config = config.get(group, {})
        if env_overrides is None and group == "hera":
            env_overrides = cls.HERA_ENV_OVERRIDES
        env_overrides = env_overrides or {}
        return {
            name: cls.create_llm_config(group, name, env_overrides.get(name))
            for name in group_config.keys()
        }

    @classmethod
    def create_crewai_llm(cls, group: str, name: str, env_override: str = None, **extra_config) -> LLM:
        """
        Creates a CrewAI LLM instance based on the YAML configuration.
        """
        cfg = cls.create_llm_config(group, name, env_override)
        
        # YAMLの設定をベースに、引数で渡された追加設定(num_ctxなど)を上書きマージする
        config_opts = {"num_ctx": cfg.get("num_ctx")}
        config_opts.update(extra_config)
        
        return LLM(
            model=cfg.get("model"),
            base_url=cfg.get("base_url"),
            timeout=cfg.get("timeout"),
            config=config_opts
        )

# Radeon環境 (ROCm/Ollama) 向けのLLM定義例 (CrewAI互換)
# モジュール import 時に Ollama へ接続しないよう、必要な時に呼び出す関数として提供する
def build_local_worker_llm() -> LLM:
    return LLMFactory.create_crewai_llm(
        group='hera',
        name='thinker',
        env_override='THINKER_MODEL',
        num_ctx=16384,
        temperature=0.2,
    )


def build_local_critic_llm() -> LLM:
    return LLMFactory.create_crewai_llm(
        group='hera',
        name='critic',
        env_override='CRITIC_MODEL',
        num_ctx=8192,
    )

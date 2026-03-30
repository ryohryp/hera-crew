import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from .tools.antigravity_delegate import AntigravityDelegateTool

# Load environment variables
load_dotenv()

# Prevent LiteLLM from failing due to missing OpenAI API Key
os.environ["OPENAI_API_KEY"] = "NA"

@CrewBase
class MyHeraCrew():
    """MyHeraCrew crew representing the HERA strategy (Full Local Edition)"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    llms_config = 'config/llms.yaml'

    def __init__(self) -> None:
        # Load centralized LLM configurations
        config_path = Path(__file__).parent / self.llms_config
        with open(config_path, 'r', encoding='utf-8') as f:
            llm_settings = yaml.safe_load(f)
        
        self.base_url = os.getenv("OLLAMA_BASE_URL", llm_settings.get("default_ollama_base_url"))
        hera_llms = llm_settings.get("hera", {})
        
        # Setup each LLM with Environment Overrides
        self.manager_llm = self._setup_llm(hera_llms.get('manager'), "MANAGER_MODEL")
        self.thinker_llm = self._setup_llm(hera_llms.get('thinker'), "THINKER_MODEL")
        self.critic_llm = self._setup_llm(hera_llms.get('critic'), "CRITIC_MODEL")
        # Tool-calling LLM for manager: uses a function-calling-capable model
        self.tool_calling_llm = self._setup_llm(hera_llms.get('tool_calling'), "TOOL_CALLING_MODEL")

        # Tools
        self.antigravity_tool = AntigravityDelegateTool()

    def _setup_llm(self, config: dict, env_var: str):
        """Helper to initialize LLM with Environment Overrides and local/cloud detection."""
        model = os.getenv(env_var, config.get('model'))
        timeout = config.get('timeout', 120)

        if "ollama" in model.lower():
            return LLM(model=model, base_url=self.base_url, timeout=timeout)
        else:
            return LLM(model=model, timeout=timeout)

    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config['manager'],
            llm=self.manager_llm,
            function_calling_llm=self.tool_calling_llm,
            verbose=True,
            allow_delegation=False,
            tools=[self.antigravity_tool]
        )

    @agent
    def thinker(self) -> Agent:
        return Agent(
            config=self.agents_config['thinker'],
            llm=self.thinker_llm,
            verbose=True,
            allow_delegation=False
        )

    @agent
    def critic(self) -> Agent:
        return Agent(
            config=self.agents_config['critic'],
            llm=self.critic_llm,
            verbose=True,
            allow_delegation=False
        )

    @task
    def task_decomposition(self) -> Task:
        return Task(
            config=self.tasks_config['task_decomposition'],
            agent=self.thinker()
        )

    @task
    def logic_evaluation(self) -> Task:
        return Task(
            config=self.tasks_config['logic_evaluation'],
            agent=self.critic()
        )

    @task
    def execution_routing(self) -> Task:
        return Task(
            config=self.tasks_config['execution_routing'],
            agent=self.thinker()
        )

    @task
    def final_verification(self) -> Task:
        return Task(
            config=self.tasks_config['final_verification'],
            agent=self.manager()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MyHeraCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

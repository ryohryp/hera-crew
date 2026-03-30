from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from hera_crew.utils.env_setup import setup_environment
from hera_crew.utils.llm_factory import LLMFactory
from .tools.antigravity_delegate import AntigravityDelegateTool

# Initialize environment
setup_environment()

@CrewBase
class HeraCrew():
    """HeraCrew crew representing the HERA strategy (Full Local Edition)"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    def __init__(self) -> None:
        # Setup each LLM using the centralized factory
        self.manager_llm = LLMFactory.create_llm('hera', 'manager', "MANAGER_MODEL")
        self.thinker_llm = LLMFactory.create_llm('hera', 'thinker', "THINKER_MODEL")
        self.critic_llm = LLMFactory.create_llm('hera', 'critic', "CRITIC_MODEL")
        self.tool_calling_llm = LLMFactory.create_llm('hera', 'tool_calling', "TOOL_CALLING_MODEL")

        # Tools
        self.antigravity_tool = AntigravityDelegateTool()

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
        """Creates the HeraCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

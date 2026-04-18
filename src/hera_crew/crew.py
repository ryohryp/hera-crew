import os
import yaml
import asyncio
import time
from pathlib import Path
from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider
from hera_crew.utils.llm_factory import LLMFactory
from .tools.antigravity_delegate import antigravity_delegate_tool, ANTIGRAVITY_DELEGATE_SPEC

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich import box


class HeraUI:
    _STEPS = [
        (1, "Task Decomposition",  "Thinker"),
        (2, "Logic Evaluation",    "Critic"),
        (3, "Execution & Routing", "Manager"),
        (4, "Final Verification",  "Manager"),
    ]

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._status: dict[int, str] = {n: "pending" for n, _, _ in self._STEPS}
        self._start_at: dict[int, float] = {}
        self._elapsed: dict[int, float] = {}
        self._output = ""
        self._current_agent = ""
        self._current_task = ""
        self._live: Live | None = None

    def __rich__(self):
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED, expand=True)
        table.add_column("", width=2, no_wrap=True)
        table.add_column("Task", min_width=26)
        table.add_column("Agent", min_width=10)
        table.add_column("Time", min_width=9, justify="right")

        for num, task_name, agent_name in self._STEPS:
            s = self._status[num]
            if s == "pending":
                icon, style, t = "⬜", "dim", "-"
            elif s == "running":
                elapsed = time.time() - self._start_at.get(num, time.time())
                icon, style, t = "⏳", "bold yellow", f"{elapsed:.1f}s"
            else:
                icon, style, t = "✅", "green", f"{self._elapsed.get(num, 0):.1f}s"
            table.add_row(icon, task_name, agent_name, t, style=style)

        title = (
            f"[bold cyan]🤖 {self._current_agent}[/] [dim]│[/] [cyan]{self._current_task}[/]"
            if self._current_agent else "[dim]Output[/]"
        )
        output_text = self._output[-2000:] if self._output else "[dim]Waiting...[/dim]"
        output_panel = Panel(output_text, title=title, border_style="cyan", padding=(1, 2))

        return Group(
            Panel(
                f"[bold yellow]Model:[/] [white]{self.model_name}[/]",
                title="[bold magenta]🤖 HERA Multi-Agent System[/]",
                border_style="magenta",
            ),
            table,
            output_panel,
        )

    def start_step(self, step_num: int):
        _, task_name, agent_name = self._STEPS[step_num - 1]
        self._status[step_num] = "running"
        self._start_at[step_num] = time.time()
        self._current_agent = agent_name
        self._current_task = task_name
        self._output = ""
        if self._live:
            self._live.update(self)

    def complete_step(self, step_num: int, output: str):
        self._status[step_num] = "done"
        self._elapsed[step_num] = time.time() - self._start_at.get(step_num, time.time())
        self._output = output
        if self._live:
            self._live.update(self)

    def __enter__(self):
        self._live = Live(
            self,
            console=Console(stderr=True),
            refresh_per_second=4,
            auto_refresh=True,
            vertical_overflow="visible",
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)


class HeraCrew:
    """HeraCrew representing the HERA strategy (agentcache Optimized Edition)"""

    def __init__(self) -> None:
        self.config_path = Path(__file__).parent / "config"
        self.agents_config = self._load_yaml("agents.yaml")
        self.tasks_config = self._load_yaml("tasks.yaml")

        self.provider = LiteLLMSDKProvider()
        self.model_cfg = LLMFactory.create_llm_config('hera', 'manager', "MANAGER_MODEL")
        self.shared_system_prompt = self._create_unified_prompt()

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_path / filename
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_unified_prompt(self) -> str:
        prompt = "You are 'hera-crew', an autonomous development squad optimized for KV cache efficiency.\n"
        prompt += "Available roles and their specifications:\n\n"
        for agent_id, config in self.agents_config.items():
            prompt += f"### ROLE: {config['role']}\n"
            prompt += f"GOAL: {config['goal']}\n"
            prompt += f"BACKSTORY: {config['backstory']}\n\n"
        prompt += "Always respond in Japanese as per the role instructions.\n"
        return prompt

    async def _tool_executor(self, tool_call_id: str, name: str, arguments: dict) -> str:
        if name == "antigravity_delegate_tool":
            return antigravity_delegate_tool(**arguments)
        return f"Tool '{name}' not found."

    async def run(self, user_request: str):
        with HeraUI(self.model_cfg['model']) as ui:
            session = AgentSession(
                model=self.model_cfg['model'],
                provider=self.provider,
                system_prompt=self.shared_system_prompt,
            )
            await session.respond("System initialization. Awaiting tasks.")

            # Step 1: Task Decomposition
            ui.start_step(1)
            task_info = self.tasks_config['task_decomposition']
            prompt = (
                f"Act as Thinker.\n"
                f"Description: {task_info['description'].format(user_request=user_request)}\n"
                f"Expected Output: {task_info['expected_output']}\n\n"
                "CRITICAL INSTRUCTION: You MUST output your response as plain text. "
                "Do NOT use any tools or function calls for this step."
            )
            fork = await session.fork(prompt=prompt, policy=ForkPolicy.cache_safe_ephemeral())
            decomposition_result = fork.final_text
            ui.complete_step(1, decomposition_result)

            # Step 2: Logic Evaluation
            ui.start_step(2)
            task_info = self.tasks_config['logic_evaluation']
            prompt = (
                f"Act as Critic.\n"
                f"Previous Step Result: {decomposition_result}\n"
                f"Description: {task_info['description']}\n"
                f"Expected Output: {task_info['expected_output']}\n\n"
                "CRITICAL INSTRUCTION: You MUST output your response as plain text. "
                "Do NOT use any tools or function calls for this step."
            )
            fork = await session.fork(prompt=prompt, policy=ForkPolicy.cache_safe_ephemeral())
            evaluation_result = fork.final_text
            ui.complete_step(2, evaluation_result)

            # Step 3: Execution & Routing
            ui.start_step(3)
            task_info = self.tasks_config['execution_routing']
            prompt = (
                f"Act as Manager/Thinker.\n"
                f"Context: {decomposition_result}\n"
                f"Evaluation: {evaluation_result}\n"
                f"Description: {task_info['description']}\n"
                f"Expected Output: {task_info['expected_output']}"
            )
            session.tools = [ANTIGRAVITY_DELEGATE_SPEC]
            execution_policy = ForkPolicy.cache_safe_ephemeral()
            execution_policy.max_turns = 3
            fork = await session.fork(
                prompt=prompt,
                tool_executor=self._tool_executor,
                policy=execution_policy,
            )
            execution_result = fork.final_text
            session.tools = []
            if not execution_result:
                fork = await session.fork(
                    prompt=(
                        "CRITICAL INSTRUCTION: The tool has executed successfully. "
                        "Now, provide a plain text summary of the final result based on "
                        "the tool output above. Do NOT use any tools."
                    ),
                    policy=ForkPolicy.cache_safe_ephemeral(),
                )
                execution_result = fork.final_text
            ui.complete_step(3, execution_result)

            # Step 4: Final Verification
            ui.start_step(4)
            task_info = self.tasks_config['final_verification']
            prompt = (
                f"Act as Orchestrator Manager.\n"
                f"Original Request: {user_request}\n"
                f"Final Result of Process: {execution_result}\n"
                f"Description: {task_info['description']}\n"
                f"Expected Output: {task_info['expected_output']}\n\n"
                "CRITICAL INSTRUCTION: You MUST output your response as plain text. "
                "Do NOT use any tools or function calls for this step."
            )
            fork = await session.fork(prompt=prompt, policy=ForkPolicy.cache_safe_ephemeral())
            final_result = fork.final_text or execution_result
            ui.complete_step(4, final_result)

            return final_result

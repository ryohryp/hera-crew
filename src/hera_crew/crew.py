import os
import yaml
import asyncio
import time
from pathlib import Path
from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider
from hera_crew.utils.llm_factory import LLMFactory
from hera_crew.utils.usage_tracker import UsageTracker
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

    def __init__(self, model_name: str | dict):
        # model_name は単一の文字列 (後方互換) または
        # {"thinker": "...", "critic": "...", "manager": "..."} の辞書を受け取る
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
            elif s == "failed":
                icon, style, t = "❌", "bold red", f"{self._elapsed.get(num, 0):.1f}s"
            else:
                icon, style, t = "✅", "green", f"{self._elapsed.get(num, 0):.1f}s"
            table.add_row(icon, task_name, agent_name, t, style=style)

        title = (
            f"[bold cyan]🤖 {self._current_agent}[/] [dim]│[/] [cyan]{self._current_task}[/]"
            if self._current_agent else "[dim]Output[/]"
        )
        output_text = self._output[-2000:] if self._output else "[dim]Waiting...[/dim]"
        output_panel = Panel(output_text, title=title, border_style="cyan", padding=(1, 2))

        if isinstance(self.model_name, dict):
            model_label = (
                f"[bold yellow]Thinker:[/] [white]{self.model_name.get('thinker', '-')}[/]\n"
                f"[bold yellow]Critic :[/] [white]{self.model_name.get('critic', '-')}[/]\n"
                f"[bold yellow]Manager:[/] [white]{self.model_name.get('manager', '-')}[/]"
            )
        else:
            model_label = f"[bold yellow]Model:[/] [white]{self.model_name}[/]"

        return Group(
            Panel(
                model_label,
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

    def fail_step(self, step_num: int, error: str):
        self._status[step_num] = "failed"
        self._elapsed[step_num] = time.time() - self._start_at.get(step_num, time.time())
        self._output = f"[bold red]ERROR:[/] {error}"
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
        self.tasks_config = self._load_yaml("tasks.yaml")

        # 3役分離: Thinker / Critic / Manager それぞれに別モデルを割り当てる
        # 環境変数 (THINKER_MODEL / CRITIC_MODEL / MANAGER_MODEL) で個別に上書き可能
        self.role_cfgs: dict[str, dict] = LLMFactory.get_group_llms("hera")
        self.thinker_cfg = self.role_cfgs["thinker"]
        self.critic_cfg = self.role_cfgs["critic"]
        self.manager_cfg = self.role_cfgs["manager"]

        # 後方互換のため model_cfg は manager を指す
        self.model_cfg = self.manager_cfg

        # provider のタイムアウトは3役の最大値を採用
        max_timeout = max(
            self.thinker_cfg.get("timeout", 1200),
            self.critic_cfg.get("timeout", 1200),
            self.manager_cfg.get("timeout", 1200),
        )
        timeout_val = float(os.getenv("LITELLM_REQUEST_TIMEOUT", str(max_timeout)))
        self.provider = LiteLLMSDKProvider(default_timeout_s=timeout_val)
        self.shared_system_prompt = self._create_unified_prompt()
        self.tracker = UsageTracker()

    async def _new_session(self, model: str) -> AgentSession:
        """Step毎に切り替える AgentSession を生成する。"""
        session = AgentSession(
            model=model,
            provider=self.provider,
            system_prompt=self.shared_system_prompt,
        )
        await session.respond("System initialization. Awaiting tasks.")
        self._record_session_usage(session, model=model)
        return session

    def _load_yaml(self, filename: str) -> dict:
        path = self.config_path / filename
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_unified_prompt(self) -> str:
        return (
            "You are 'hera-crew', an autonomous multi-agent development system running on local hardware.\n"
            "You execute tasks in the role specified in each message (Thinker / Critic / Manager).\n"
            "Always respond in Japanese."
        )

    def _record_session_usage(self, session_or_fork, model: str = ""):
        """Helper to extract and record usage from AgentSession or fork objects."""
        if not session_or_fork:
            return
        usage = getattr(session_or_fork, "usage", None)
        if usage:
            self.tracker.record_agent_usage(usage, model=model)

    async def _tool_executor(self, tool_call_id: str, name: str, arguments: dict) -> str:
        if name == "antigravity_delegate_tool":
            self.tracker.record_delegation()
            return antigravity_delegate_tool(**arguments)
        return f"Tool '{name}' not found."

    async def run(
        self,
        user_request: str,
        orchestrator_input_tokens: int = 0,
        orchestrator_output_tokens: int = 0,
        orchestrator_model: str = "",
    ):
        # tracker には3役全モデルを登録 (使用量レポートに全モデル分が出るように)
        for cfg in (self.thinker_cfg, self.critic_cfg, self.manager_cfg):
            self.tracker.register_litellm(cfg["model"])
        self.tracker.set_task(user_request)
        if orchestrator_input_tokens or orchestrator_output_tokens:
            self.tracker.record_orchestrator_usage(
                orchestrator_input_tokens,
                orchestrator_output_tokens,
                orchestrator_model,
            )
        final_result = ""
        try:
            ui_models = {
                "thinker": self.thinker_cfg["model"],
                "critic": self.critic_cfg["model"],
                "manager": self.manager_cfg["model"],
            }
            with HeraUI(ui_models) as ui:
                err = Console(stderr=True)

                # Step 1: Task Decomposition (Thinker)
                ui.start_step(1)
                self.tracker.set_step("Task Decomposition")
                err.print(f"[dim]Calling {self.thinker_cfg['model']} for Task Decomposition...[/]")
                try:
                    thinker_session = await self._new_session(self.thinker_cfg["model"])
                    task_info = self.tasks_config['task_decomposition']
                    prompt = (
                        f"Act as Thinker.\n"
                        f"Description: {task_info['description'].format(user_request=user_request)}\n"
                        f"Expected Output: {task_info['expected_output']}\n\n"
                        "CRITICAL INSTRUCTION: You MUST output your response as plain text. "
                        "Do NOT use any tools or function calls for this step."
                    )
                    fork = await thinker_session.fork(prompt=prompt, policy=ForkPolicy.cache_safe_ephemeral())
                    decomposition_result = fork.final_text
                    self._record_session_usage(fork, model=self.thinker_cfg["model"])
                    ui.complete_step(1, decomposition_result)
                except Exception as e:
                    ui.fail_step(1, str(e))
                    raise

                # Step 2: Logic Evaluation (Critic)
                ui.start_step(2)
                self.tracker.set_step("Logic Evaluation")
                err.print(f"[dim]Calling {self.critic_cfg['model']} for Logic Evaluation...[/]")
                try:
                    critic_session = await self._new_session(self.critic_cfg["model"])
                    task_info = self.tasks_config['logic_evaluation']
                    prompt = (
                        f"Act as Critic.\n"
                        f"Previous Step Result: {decomposition_result}\n"
                        f"Description: {task_info['description']}\n"
                        f"Expected Output: {task_info['expected_output']}\n\n"
                        "CRITICAL INSTRUCTION: You MUST output your response as plain text. "
                        "Do NOT use any tools or function calls for this step."
                    )
                    fork = await critic_session.fork(prompt=prompt, policy=ForkPolicy.cache_safe_ephemeral())
                    evaluation_result = fork.final_text
                    self._record_session_usage(fork, model=self.critic_cfg["model"])
                    ui.complete_step(2, evaluation_result)
                except Exception as e:
                    ui.fail_step(2, str(e))
                    raise

                # Step 3: Execution & Routing (Manager)
                ui.start_step(3)
                self.tracker.set_step("Execution & Routing")
                err.print(f"[dim]Calling {self.manager_cfg['model']} for Execution & Routing...[/]")
                try:
                    manager_session = await self._new_session(self.manager_cfg["model"])
                    task_info = self.tasks_config['execution_routing']
                    prompt = (
                        f"Act as Manager.\n"
                        f"Context: {decomposition_result}\n"
                        f"Evaluation: {evaluation_result}\n"
                        f"Description: {task_info['description']}\n"
                        f"Expected Output: {task_info['expected_output']}"
                    )
                    manager_session.tools = [ANTIGRAVITY_DELEGATE_SPEC]
                    execution_policy = ForkPolicy.cache_safe_ephemeral()
                    execution_policy.max_turns = 3
                    fork = await manager_session.fork(
                        prompt=prompt,
                        tool_executor=self._tool_executor,
                        policy=execution_policy,
                    )
                    execution_result = fork.final_text
                    self._record_session_usage(fork, model=self.manager_cfg["model"])
                    manager_session.tools = []
                    if not execution_result:
                        fork = await manager_session.fork(
                            prompt=(
                                "CRITICAL INSTRUCTION: The tool has executed successfully. "
                                "Now, provide a plain text summary of the final result based on "
                                "the tool output above. Do NOT use any tools."
                            ),
                            policy=ForkPolicy.cache_safe_ephemeral(),
                        )
                        execution_result = fork.final_text
                        self._record_session_usage(fork, model=self.manager_cfg["model"])
                    ui.complete_step(3, execution_result)
                except Exception as e:
                    ui.fail_step(3, str(e))
                    raise

                # Step 4: Final Verification (Manager)
                # Step3 の manager_session を再利用してOllama側のモデル再ロードを避ける
                ui.start_step(4)
                self.tracker.set_step("Final Verification")
                err.print(f"[dim]Calling {self.manager_cfg['model']} for Final Verification...[/]")
                try:
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
                    fork = await manager_session.fork(prompt=prompt, policy=ForkPolicy.cache_safe_ephemeral())
                    final_result = fork.final_text or execution_result
                    self._record_session_usage(fork, model=self.manager_cfg["model"])
                    ui.complete_step(4, final_result)
                except Exception as e:
                    ui.fail_step(4, str(e))
                    raise

        finally:
            self.tracker.finalize()
            try:
                report_path = self.tracker.save_html()
                err = Console(stderr=True)
                err.print(self.tracker.render_savings_panel())
                err.print(f"[dim]レポート保存 → [cyan]{report_path}[/][/]")
            except Exception as e:
                Console(stderr=True).print(f"[yellow]レポート生成失敗: {e}[/]")

        return final_result

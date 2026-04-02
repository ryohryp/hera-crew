import os
import yaml
import asyncio
from pathlib import Path
from agentcache import AgentSession, ForkPolicy, LiteLLMSDKProvider
from hera_crew.utils.llm_factory import LLMFactory
from .tools.antigravity_delegate import antigravity_delegate_tool, ANTIGRAVITY_DELEGATE_SPEC

class HeraCrew:
    """HeraCrew representing the HERA strategy (agentcache Optimized Edition)"""

    def __init__(self) -> None:
        self.config_path = Path(__file__).parent / "config"
        self.agents_config = self._load_yaml("agents.yaml")
        self.tasks_config = self._load_yaml("tasks.yaml")
        
        # Initialize provider
        self.provider = LiteLLMSDKProvider()
        
        # Get model config (default to manager model for the main session)
        self.model_cfg = LLMFactory.create_llm_config('hera', 'manager', "MANAGER_MODEL")
        
        # Create Unified System Prompt
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
        """Dispatches tool calls from the LLM to local tool functions."""
        if name == "antigravity_delegate_tool":
            return antigravity_delegate_tool(**arguments)
        return f"Tool '{name}' not found."

    def _log_agent_output(self, agent_role: str, task_name: str, text: str):
        # ANSI escape codes for colors
        CYAN = '\033[96m'
        YELLOW = '\033[93m'
        MAGENTA = '\033[95m'
        RESET = '\033[0m'
        
        print(f"\n{CYAN}=================================================={RESET}")
        print(f"{MAGENTA}🤖 [Working Agent: {agent_role}]{RESET} | {CYAN}📋 [Task: {task_name}]{RESET}")
        print(f"{CYAN}=================================================={RESET}")
        print(f"{YELLOW}{text}{RESET}\n")

    async def run(self, user_request: str):
        """
        Executes the HERA sequential workflow using agentcache forks.
        """
        print(f"Initializing session with model: {self.model_cfg['model']}...")
        
# 1. Start parent AgentSession (Pure Text Mode)
        session = AgentSession(
            model=self.model_cfg['model'],
            provider=self.provider,
            system_prompt=self.shared_system_prompt
            # toolsはここでは渡さない！
        )
        
        # Warm up the cache
        await session.respond("System initialization. Awaiting tasks.")
        print("✅ Session initialized and cache warmed up.")

        # 2. Step 1: Task Decomposition (Thinker)
        print("\n--- [Step 1: Task Decomposition] ---")
        task_info = self.tasks_config['task_decomposition']
        prompt = f"Act as Thinker.\nDescription: {task_info['description'].format(user_request=user_request)}\nExpected Output: {task_info['expected_output']}"
        prompt += "\n\nCRITICAL INSTRUCTION: You MUST output your response as plain text. Do NOT use any tools or function calls for this step."
        
        decomposition_fork = await session.fork(
            prompt=prompt,
            policy=ForkPolicy.cache_safe_ephemeral()
        )
        decomposition_result = decomposition_fork.final_text
        self._log_agent_output("Thinker", "Task Decomposition", decomposition_result)

        # 3. Step 2: Logic Evaluation (Critic)
        print("\n--- [Step 2: Logic Evaluation] ---")
        task_info = self.tasks_config['logic_evaluation']
        prompt = f"Act as Critic.\nPrevious Step Result: {decomposition_result}\nDescription: {task_info['description']}\nExpected Output: {task_info['expected_output']}"
        prompt += "\n\nCRITICAL INSTRUCTION: You MUST output your response as plain text. Do NOT use any tools or function calls for this step."

        evaluation_fork = await session.fork(
            prompt=prompt,
            policy=ForkPolicy.cache_safe_ephemeral()
        )
        evaluation_result = evaluation_fork.final_text
        self._log_agent_output("Critic", "Logic Evaluation", evaluation_result)

        # 4. Step 3: Execution Routing (Thinker/Manager)
        print("\n--- [Step 3: Execution & Routing] ---")
        task_info = self.tasks_config['execution_routing']
        prompt = f"Act as Manager/Thinker.\nContext: {decomposition_result}\nEvaluation: {evaluation_result}\nDescription: {task_info['description']}\nExpected Output: {task_info['expected_output']}"
        
        # 💡 ここで一時的にツールを解禁する
        session.tools = [ANTIGRAVITY_DELEGATE_SPEC]
        
        execution_policy = ForkPolicy.cache_safe_ephemeral()
        execution_policy.max_turns = 3
        
        execution_fork = await session.fork(
            prompt=prompt,
            tool_executor=self._tool_executor,
            policy=execution_policy
        )
        execution_result = execution_fork.final_text
        
        # ツール解禁を解除（Step 4に持ち越さない）
        session.tools = []
        
        if not execution_result:
            # 💡 フォールバック：ツール実行だけでテキストが返ってこなかった場合の強制テキスト化
            summary_fork = await session.fork(
                prompt="CRITICAL INSTRUCTION: The tool has executed successfully. Now, provide a plain text summary of the final result based on the tool output above. Do NOT use any tools.",
                policy=ForkPolicy.cache_safe_ephemeral()
            )
            execution_result = summary_fork.final_text

        self._log_agent_output("Manager/Thinker", "Execution & Routing", execution_result)

        # 5. Step 4: Final Verification (Manager)
        print("\n--- [Step 4: Final Verification] ---")
        task_info = self.tasks_config['final_verification']
        prompt = f"Act as Orchestrator Manager.\nOriginal Request: {user_request}\nFinal Result of Process: {execution_result}\nDescription: {task_info['description']}\nExpected Output: {task_info['expected_output']}"
        prompt += "\n\nCRITICAL INSTRUCTION: You MUST output your response as plain text. Do NOT use any tools or function calls for this step."

        final_fork = await session.fork(
            prompt=prompt,
            policy=ForkPolicy.cache_safe_ephemeral()
        )
        final_result = final_fork.final_text
        
        # 💡 最終防波堤：LLMが無言になったら、Step3の結果を最終出力とする
        if not final_result:
            final_result = execution_result
        
        return final_result

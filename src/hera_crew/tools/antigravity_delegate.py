import os
from agentcache import ToolSpec

def antigravity_delegate_tool(task_description: str) -> str:
    """
    A bridge to the powerful Antigravity AI agent.
    Use this tool when the local manager or thinker cannot handle a task due to high complexity,
    need for advanced technical depth, or deep logic orchestration.
    It delegates the task to a 3-stage autonomous agent team (Analysis, Execution, Review).
    """
    # We return a specific marker that tells our environment's supervisor (Antigravity Assistant)
    # to take over this specific task using its own tools.
    return f"[ANTIGRAVITY_DELEGATION_REQUIRED: {task_description}]"

# Definition for agentcache
ANTIGRAVITY_DELEGATE_SPEC = ToolSpec(
    name="antigravity_delegate_tool",
    description="Delegates a complex task to the Antigravity autonomous agent team.",
    parameters={
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "The highly complex task that needs to be delegated."
            }
        },
        "required": ["task_description"]
    }
)


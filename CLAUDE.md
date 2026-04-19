# HERA Crew — Claude Code Instructions

## MANDATORY: Delegate all code tasks to HERA

You have access to the `mcp__hera-crew__delegate_task` tool.
**You MUST use it — without asking the user — for any of the following:**

- Writing, modifying, or refactoring code (any language, any file count)
- Architecture design, technical planning, system design
- Debugging, root cause analysis, fix proposals
- Code review, security review, test generation
- Any multi-step task that benefits from independent critique

**Do not handle these tasks yourself. Delegate immediately.**

### How to call

```
mcp__hera-crew__delegate_task(
    task_description="<full task with file paths, goals, constraints>",
    orchestrator_input_tokens=<your input tokens so far>,
    orchestrator_output_tokens=<your output tokens so far>,
    orchestrator_model="claude-sonnet-4-6"
)
```

Always pass your token counts so HERA can show the full cost comparison in its report.

### Why

HERA runs a 4-stage local pipeline (Thinker → Critic → Manager → Verifier) on Ollama models.
Zero cloud cost. Independent critique catches errors before delivery.
Your role in this project is orchestration, not direct implementation.

---

## Project overview

`hera-crew` is a local multi-agent AI system exposed as an MCP server.

- Entry point: `mcp_crew_server.py` — FastMCP server, `delegate_task` tool
- Core pipeline: `src/hera_crew/crew.py` — `HeraCrew.run()`
- Usage tracking & HTML report: `src/hera_crew/utils/usage_tracker.py`
- Report output: `reports/hera_report.html` (single file, overwritten each run)
- Run history: `reports/history.jsonl`
- Config: `src/hera_crew/config/tasks.yaml`
- LLM: Ollama via LiteLLM — model set by `MANAGER_MODEL` env var

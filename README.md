# HERA: Hybrid Edge-cloud Resource Allocation

![HERA Banner](https://img.shields.io/badge/Strategy-HERA-blueviolet?style=for-the-badge)
![CrewAI](https://img.shields.io/badge/Framework-CrewAI-red?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Local_LLM-Ollama-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> [日本語版はこちら](README_ja.md)

A local-first, multi-agent AI system built on [CrewAI](https://github.com/crewAIInc/crewAI) and [Ollama](https://ollama.com/).
Run powerful 14B-class models entirely on your own GPU — no cloud API required.
When you need it, plug in Gemini or any other cloud LLM with a single `.env` change.

---

## Why HERA?

Most AI workflows default to cloud APIs for every call. HERA flips that default:

- **Thinker** (Gemma 3) — decomposes tasks and writes first drafts locally
- **Critic** (Phi-4) — reviews and catches hallucinations locally
- **Manager** (Qwen2.5 14B) — orchestrates, validates, and escalates to cloud only when needed

The result: the expensive cloud token budget is spent only on work that actually needs it.

| Concern | HERA answer |
|---|---|
| API cost | Iterative thinking stays local |
| Privacy | Drafts never leave your machine |
| Quality | 3-agent cross-review catches errors |
| Flexibility | Swap any model in one line of `.env` |

---

## Key Features

- **HERA resource strategy** — dynamic local/cloud routing per task
- **MCP server mode** — expose the crew as a tool to Claude Desktop, Cursor, etc.
- **Centralized LLM config** — one `llms.yaml` controls every model; `.env` overrides per-run
- **32k context window** — `num_ctx: 32768` applied to all Ollama calls via `extra_body`
- **Zero OpenAI dependency** — fully offline by default; no `OPENAI_API_KEY` required

---

## Architecture

```mermaid
graph TD
    subgraph "Cloud (optional)"
        A[Cloud LLM e.g. Gemini]
    end

    subgraph "Local PC"
        B[MCP Server<br/>mcp_crew_server.py]
        C[CrewAI Orchestrator]

        subgraph "Ollama 14B models"
            D[Thinker: Gemma 3]
            E[Critic: Phi-4]
            F[Manager: Qwen2.5 14B]
        end
    end

    A <-->|MCP| B
    B <--> C
    C --> D
    C --> E
    C --> F
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

---

## Project Structure

```text
my_hera_crew/
├── .env.example                # Environment variable template
├── mcp_settings_example.json   # MCP client config example
├── mcp_crew_server.py          # MCP server entry point
├── requirements.txt
├── scripts/
│   └── inspect_llm.py
├── tests/
│   ├── test_delegation.py
│   └── test_llm_syntax.py
└── src/my_hera_crew/
    ├── config/
    │   ├── agents.yaml         # Agent role definitions
    │   ├── llms.yaml           # Centralized model config
    │   └── tasks.yaml          # Task routing definitions
    ├── tools/
    │   └── antigravity_delegate.py
    ├── crew.py
    └── main.py
```

---

## Requirements

- Python 3.10 – 3.13
- [Ollama](https://ollama.com/) installed and running
- GPU recommended (VRAM 16 GB+ for all 14B models simultaneously)

---

## Setup

```bash
git clone https://github.com/ryohryp/my_hera_crew.git
cd my_hera_crew

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env if needed
```

Pull the required Ollama models:

```bash
# HERA crew (main.py)
ollama pull qwen2.5:14b         # Manager — must support function calling
ollama pull gemma3:latest       # Thinker
ollama pull phi4:latest         # Critic

# MCP server (mcp_crew_server.py)
ollama pull qwen2.5-coder:14b   # Specialist / Coder
ollama pull deepseek-r1:14b     # Reviewer
```

---

## Usage

### Standalone CLI

```bash
python src/my_hera_crew/main.py
```

Enter your task at the prompt. The crew runs sequentially:
Thinker → Critic → Manager → final output.

### MCP Server

```bash
python mcp_crew_server.py
```

Add to your MCP client config (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "my_hera_crew": {
      "command": "/absolute/path/to/venv/Scripts/python",
      "args": ["/absolute/path/to/mcp_crew_server.py"]
    }
  }
}
```

This exposes a `delegate_task(task_description)` tool that offloads complex work to your local agent team.

### Quick test

```bash
python tests/test_delegation.py
```

---

## Configuration

### Swap models via `llms.yaml`

```yaml
hera:
  manager:
    model: "ollama/qwen2.5:14b"   # ollama/ prefix required
    timeout: 300
    num_ctx: 32768
  thinker:
    model: "ollama/gemma3:latest"
  critic:
    model: "ollama/phi4:latest"
```

### Override per-run via `.env`

```ini
MANAGER_MODEL=ollama/qwen2.5:14b
THINKER_MODEL=ollama/gemma3:latest
CRITIC_MODEL=ollama/phi4:latest

# Switch to cloud:
# MANAGER_MODEL=gemini/gemini-1.5-pro
# GOOGLE_API_KEY=your_key
```

> **Note:** Always use the `ollama/` prefix for Ollama models. Without it, LiteLLM routes the request to OpenAI and fails.
> **Note:** The Manager must use a function-calling-capable model (`deepseek-r1` does not support tool calling via Ollama).

---

## License

[MIT](LICENSE)

---

*HERA: Hybrid Edge-cloud Resource Allocation for Autonomous Multi-Agent Development.*

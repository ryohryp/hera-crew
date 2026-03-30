# システムアーキテクチャ (Architecture)

本プロジェクト `my_hera_crew` は、Google Antigravity (Cloud) と Ollama (Local) を組み合わせた、プライバシー重視かつ高効率なハイブリッド自律エージェント環境です。

## 構成図 (Herbal Edge Rendering Architecture)

```mermaid
graph TD
    subgraph "Cloud (Google Antigravity / Commander)"
        A[Cloud LLM / Commander]
        G[Browser / Development Tools]
    end

    subgraph "Edge (Local PC: my_hera_crew)"
        B[MCP Server<br/>FastMCP / General_Autonomous_Crew]
        C[CrewAI Orchestrator<br/>Sequential/Manager Process]
        
        subgraph "Local LLM Experts (Ollama 14B + 32k ctx)"
            D[Planner: Gemma 3]
            E[Critic: DeepSeek-R1]
            F[Coder: Qwen2.5-Coder]
        end

        H[Local Filesystem / Codes]
    end

    A <-->|Model Context Protocol (MCP)| B
    B <--> C
    C --> D
    C --> E
    C --> F
    F --> H
    G --> H
```

## 各コンポーネントの役割

| コンポーネント | 役割 | 使用モデル / 技術 |
| :--- | :--- | :--- |
| **Commander** | 全体の方針決定、MCP経由の委譲 | Gemini 2.0 Pro / Flash (Cloud) |
| **MCP Server** | ローカルリソースとクラウドの橋渡し | Python / FastMCP |
| **Orchestrator** | ローカルでのマルチエージェント制御 | CrewAI / Manager Agent |
| **Local Experts** | 特定領域（分析、コード生成等）の高速推論 | Ollama (Qwen, DeepSeek, Gemma, Phi-4) |

## 特徴
1.  **完全遮断**: `CREWAI_TELEMETRY=false` 設定により、ローカルでの計算中に OpenAI 等の外部サーバーへデータが漏洩することはありません。
2.  **長文記憶**: `num_ctx: 32768` を全モデルに強制適用しており、複雑な議論や巨大なソースコードを「忘れる」ことなく処理できます。
3.  **ハイブリッド制御**: マネージャーエージェントが「推論」と「ツール実行（MCPなど）」のLLMを使い分けることで、高い安定性を実現しています。

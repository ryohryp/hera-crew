# AntiGravity × CrewAI Hybrid Multi-Agent System (HERA)

![HERA Banner](https://img.shields.io/badge/Strategy-HERA-blueviolet?style=for-the-badge)
![CrewAI](https://img.shields.io/badge/Framework-CrewAI-red?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Local_LLM-Gemma_3-orange?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Cloud_LLM-Gemini-blue?style=for-the-badge)

## 1. 概要 (System Overview)
本システムは、**HERA（Hybrid Edge-cloud Resource Allocation）戦略**に基づき、クラウドLLM（Gemini）とローカルLLM（Ollama上のGemma 3）を動的に使い分ける自律型AI開発チームです。

AntiGravityのクォータ制限を賢く回避しながら、ローカルPCのGPUパワー（VRAM 16GB等）を最大限に活用し、コスト効率と高度な推論能力を両立させます。

## 2. システムアーキテクチャ (HERA Strategy)
タスクの性質に応じて、以下のエージェントが連携します：

1.  **Orchestrator / Manager (Cloud: Gemini)**:
    *   全体計画の策定、タスクの委任、最終成果物の検証。
    *   高度な推論が必要な場合や進行度の後半（Late stage）で稼働。
2.  **Bridge / Thinker (Local: Gemma 3)**:
    *   タスクの細分化、翻訳、初期コードのドラフト作成、一次調査。
    *   クラウドAPIを消費せず、ローカル環境で迅速に思考を実行。
3.  **The Critic (Hybrid)**:
    *   Thinkerの出力を厳格にレビューし、ハルシネーションを防止。
    *   ローカルでの「収束」が困難な場合、Managerへのフォールバックを推奨。

## 3. ディレクトリ構成 (Project Structure)
```text
my_hera_crew/
├── .env                        # 環境変数設定（APIキー等）
├── requirements.txt            # 依存ライブラリ
├── src/
│   └── my_hera_crew/
│       ├── config/
│       │   ├── agents.yaml     # エージェント役割定義
│       │   └── tasks.yaml      # タスク・ルーティング定義
│       ├── tools/              # カスタムツール類
│       ├── crew.py             # CrewAI初期化 & LLM設定 (Ollama/Gemini)
│       └── main.py             # 実行用エントリーポイント
└── .agent_state.json           # 状態管理（自動生成）
```

## 4. セットアップ (Setup)

### 必須要件
*   Python 3.10 ~ 3.13
*   Ollama (gemma3:latest がインストールされていること)
*   AntiGravity (Gemini 1.5 Flash へのアクセス)

### インストール
```powershell
# 依存関係のインストール
pip install -r requirements.txt
```

### ローカルモデルの準備
OllamaでGemma 3を起動しておきます。
```powershell
ollama run gemma3:latest
```

## 5. 実行方法 (Usage)
以下のコマンドを実行し、開発タスクを入力してください。

```powershell
python src/my_hera_crew/main.py
```

## 6. パフォーマンス最適化 (Optimizations)
*   **Parallel Execution**: `OLLAMA_NUM_PARALLEL` を `4` に設定。
*   **Context Window**: `num_ctx` を `8192` に最適化。
*   **GPU Acceleration**: AMD Radeon RX 6800 (ROCm) を利用した高速推論。

---
**AntiGravity × CrewAI: Hybrid Resource Allocation for Next-Gen Autonomous Development.**

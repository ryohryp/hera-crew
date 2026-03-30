# HERA: Hybrid Edge-cloud Resource Allocation Multi-Agent System

![HERA Banner](https://img.shields.io/badge/Strategy-HERA-blueviolet?style=for-the-badge)
![CrewAI](https://img.shields.io/badge/Framework-CrewAI-red?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Local_LLM-Ollama-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 1. 概要 (System Overview)

本システムは、**HERA（Hybrid Edge-cloud Resource Allocation）戦略**に基づき、クラウドLLMとローカルLLM（Ollama）を動的に使い分ける自律型AIマルチエージェントチームです。

ローカルPCのGPUパワーを最大限に活用し、コスト効率と高度な推論能力を両立させます。CrewAI フレームワークを基盤とし、MCPサーバーとしても動作可能です。

## 2. システムアーキテクチャ (HERA Strategy)

タスクの性質に応じて、以下のエージェントが連携します：

1.  **Orchestrator / Manager** (例: DeepSeek-R1):
    *   全体計画の策定、タスクの委任、最終成果物の検証。
    *   高度な推論が必要な場合や進行度の後半（Late stage）で稼働。
2.  **Bridge / Thinker** (例: Gemma 3):
    *   タスクの細分化、翻訳、初期コードのドラフト作成、一次調査。
    *   クラウドAPIを消費せず、ローカル環境で迅速に思考を実行。
3.  **The Critic** (例: Phi-4):
    *   Thinkerの出力を厳格にレビューし、ハルシネーションを防止。
    *   ローカルでの「収束」が困難な場合、Managerへのフォールバックを推奨。

## 3. ディレクトリ構成 (Project Structure)

```text
my_hera_crew/
├── .env.example                # 環境変数テンプレート
├── .gitignore                  # Git除外設定
├── LICENSE                     # MITライセンス
├── README.md                   # 本ドキュメント
├── requirements.txt            # 依存ライブラリ
├── mcp_crew_server.py          # MCPサーバー（外部ツール連携用）
├── test_delegation.py          # 委譲テストスクリプト
└── src/
    └── my_hera_crew/
        ├── __init__.py
        ├── config/
        │   ├── agents.yaml     # エージェント役割定義
        │   ├── llms.yaml       # LLMモデル設定（集中管理）
        │   └── tasks.yaml      # タスク・ルーティング定義
        ├── tools/
        │   └── antigravity_delegate.py  # 外部エージェント委譲ツール
        ├── crew.py             # CrewAI初期化 & LLM設定
        └── main.py             # 実行用エントリーポイント
```

## 4. セットアップ (Setup)

### 必須要件

*   Python 3.10 ~ 3.13
*   [Ollama](https://ollama.com/) がインストールされていること
*   GPU（推奨: VRAM 16GB以上）

### インストール

```bash
# リポジトリのクローン
git clone https://github.com/ryohryp/my_hera_crew.git
cd my_hera_crew

# 仮想環境のセットアップ
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

# 依存関係のインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# 必要に応じて .env を編集
```

### ローカルモデルの準備

Ollamaで必要なモデルを取得しておきます（`llms.yaml` の設定に従って変更可能）。

```bash
ollama pull deepseek-r1:14b
ollama pull gemma3:latest
ollama pull phi4:latest
```

## 5. 実行方法 (Usage)

### スタンドアロン実行

```bash
python src/my_hera_crew/main.py
```

### MCPサーバーとして使用

```bash
python mcp_crew_server.py
```

MCP経由で `delegate_task` ツールが利用可能になります。

## 6. 設定のカスタマイズ

### LLMモデルの変更

`src/my_hera_crew/config/llms.yaml` でモデルを一元管理しています。  
環境変数（`.env`）でオーバーライドも可能です。

```yaml
# llms.yaml の例
hera:
  manager:
    model: "ollama_chat/deepseek-r1:14b"   # 任意のOllamaモデルに変更可
    timeout: 300
```

```ini
# .env でのオーバーライド
MANAGER_MODEL=gemini/gemini-1.5-flash  # クラウドLLMへの切替も可能
```

### パフォーマンス最適化

*   **Parallel Execution**: `OLLAMA_NUM_PARALLEL` を環境変数で設定（デフォルト: 4）
*   **Context Window**: `llms.yaml` でモデルごとに `timeout` を調整
*   **GPU Acceleration**: Ollama がGPUを自動検出（NVIDIA CUDA / AMD ROCm 対応）

## 7. ライセンス (License)

本プロジェクトは [MIT License](LICENSE) の下で公開されています。

---

**HERA: Hybrid Edge-cloud Resource Allocation for Autonomous Multi-Agent Development.**

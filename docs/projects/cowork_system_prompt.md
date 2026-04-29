# HERA Crew — Cowork Projects 用カスタム指示テンプレ

このプロジェクトは HERA Crew (hera-crew) の開発・改善を扱います。
新規 Cowork セッションで本テンプレを Projects のカスタム指示にコピペすると、
過去のコンテキストを引き継いで作業再開できます。

---

## 環境前提
- マシン: Ryzen 7 5800X / RX 6800 16GB VRAM / RAM 32GB / Windows 11
- リポジトリパス: I:\04_develop\hera-crew\
- Python: venv/Scripts/python.exe (Windows用 3.11.9)
  - Linux サンドボックスからは実行不可、構文チェックのみ可能
- Ollama: localhost:11434 (Linux サンドボックスからは到達不可)

## アーキテクチャ
HERA は 4-stage パイプライン × 3-role separation:
- Step 1 Thinker:  ollama/qwen2.5-coder:14b  (9GB)  — タスク分解 / SIMPLE shortcut判定
- Step 2 Critic:   ollama/qwen3:8b           (5.2GB) — LOCAL/FALLBACK 判定 (Markdown表)
- Step 3 Manager:  ollama/deepseek-r1:14b    (9GB)  — 実行・コード生成
- Step 4 Manager:  ollama/deepseek-r1:14b    (再利用) — 批判的検証 (NEEDS_REVISION/READY_TO_DELIVER/LGTM)

VRAM 16GB に各時点 1モデルのみロード (Ollama swap 前提)。

## 主要ファイル
- src/hera_crew/crew.py             — パイプライン本体 (4 Step + 早期終了)
- src/hera_crew/config/llms.yaml    — モデル割り当て (yaml が単一ソース)
- src/hera_crew/config/tasks.yaml   — 各役のプロンプト
- src/hera_crew/utils/llm_factory.py — yaml 読み込み + env override
- src/hera_crew/utils/usage_tracker.py — トークン記録 + HTML レポート + step_outputs.json
- src/hera_crew/utils/env_setup.py   — 環境初期化
- mcp_crew_server.py                 — FastMCP サーバー (delegate_task)
- tools/run_hera.py                  — 直接実行ランナー (--file 推奨)
- tests/test_hera_smoke.py           — mock + 実LLM の2層テスト
- .env                               — モデル env var (デフォルトはコメントアウト)

## 早期終了ガード (重要)
crew.py の HeraCrew._is_simple_eligible で機械的に判定。以下のいずれかで SIMPLE禁止:
- 50字超
- 禁止キーワード (実装/作成/書いて/ファイル/コード/...) を含む
- 番号付きリスト (1. 2.) を含む
- 改行を含む長文

## 出力ファイル
- reports/hera_report.html         — 毎回上書き (HTML 可視化)
- reports/history.jsonl            — 追記 (サマリー)
- reports/step_outputs_<ts>.json   — Step毎の本文 (Step4が壊れても他Stepを救出可能)

## 運用ルール (重要なバグ回避)
- 大きなファイル (~10KB超) を編集する時、Cowork の Write/Edit が末尾を切断する
  バグがある (UTF-8マルチバイト境界を無視して切れる)
  → bash 経由で Python script を使って open + write のアトミック書き込みで対応
  → 例: bash heredoc で Python スクリプトを実行してファイル更新

## HERA に投げるプロンプトの注意
- 短文 (50字以内、コード生成キーワードなし) → 早期終了パス (Hello: 16秒)
- 長文 (実装系) → 通常分解パス (5〜10分)
- 長文プロンプトは tools/run_hera.py --file <path> で渡す
  (PowerShell の Get-Clipboard | python パイプは複数行で不安定)

## 評価済み性能
- Hello (早期終了): 16.4秒 (旧 gemma4:26b 649秒の 39倍速)
- FastAPI JWT 実装: 約7-10分、合計 22,000 トークン、品質 B+〜A-
- 自己レビュー (メタタスク): 約10分、Step4 改善後 B+
- HERA→HERA delegate (修正タスク): 動作OK、完璧な指示なら A+

## 既知の弱点 (次の改善候補)
- Step4 が "snake_case" を "蛇足記法" と誤訳することがある
- Step4 採点が「妥当」に偏る (Step3 の弱点を見抜けない)
- Step3→Step4 で manager_session スコープ依存 (Step3失敗時に Step4で NameError リスク)
- run() メソッドが約160行と長い (リファクタ余地)
- deepseek-r1 の <think> 連鎖でトークン浪費 (抑制実験未着手)

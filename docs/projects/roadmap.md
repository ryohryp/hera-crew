# HERA Crew — 次回作業ロードマップ

優先度順。Cowork セッション再開時の起点。

## 🥇 Tier 1 (最高優先 — Step4 のさらなる改善)

### Task 1.1: Step4 翻訳ミス対策
- 症状: deepseek-r1 が "snake_case" を "蛇足記法" と訳す
- 対応: tasks.yaml の final_verification に「専門用語 (snake_case 等) は原語のまま使用」と追記
- 影響: 主に評価レポートの読みやすさ

### Task 1.2: Step4 採点の偏り改善
- 症状: 「妥当」3/3 で甘い採点、Step3 の `pass` 空実装などを見抜けない
- 対応案 A: 「最低1個は『不十分』判定を出すこと」を yaml に追加 (forced criticism)
- 対応案 B: Step3 の出力を構造化 (各提案にメタデータ) してから渡す
- 影響: HERA の批判性向上

## 🥈 Tier 2 (堅牢性)

### Task 2.1: Step3→Step4 manager_session スコープ依存修正
- 症状: Step3 の except で raise すると Step4 の `manager_session` が NameError
- crew.py:362 manager_session = await self._new_session(...) が Step3 内
- crew.py:414 manager_session.fork() を Step4 で参照 (スコープ越境)
- 対応: manager_session を Step3 開始前 (run()レベル) で初期化、または Step4 で再生成
- 影響: 実用上のクラッシュ防止

### Task 2.2: run() メソッドのリファクタ
- 症状: run() が約160行、Step1-4のロジック直書き
- 対応: _run_step1_thinker / _run_step2_critic / _run_step3_manager / _run_step4_verifier に分割
- 影響: テスタビリティ向上、HERA 自身がレビューでも指摘済み

## 🥉 Tier 3 (性能チューニング)

### Task 3.1: deepseek-r1 の <think> 抑制実験
- 症状: Step3 で <think>...</think> 連鎖が長く、トークン消費の半分以上を占める可能性
- 対応案 A: tasks.yaml の execution_routing に「推論チェーンは出力しないこと、結論を直接」と追記
- 対応案 B: Ollama の num_predict 制限で強制カット
- 検証: A/B 比較で品質 vs トークン削減のトレードオフ測定
- 影響: 約3-5分の時間短縮見込み

### Task 3.2: 早期終了の境界値テスト
- 対応: tests/test_simple_eligible.py 追加
  - 49字 / 50字 / 51字 (長さ境界)
  - 各キーワード単独・複数組み合わせ
  - 番号付きリスト (1. / 1) / 1： などの記号バリエーション)
  - マルチバイト文字での length 判定
- 影響: ガードロジックの堅牢性

## 🥉 Tier 3 (実用拡張)

### Task 3.3: hera_run_<ts>.html の個別保存
- 症状: hera_report.html が毎回上書きされ、過去レポートを参照できない
- 対応: usage_tracker.py の save_html() で hera_run_<ts>.html も保存
- 影響: HTML レポートの履歴

### Task 3.4: HERA 並列実行 (実験)
- 課題: agentcache fork の並列度を活用できていない
- 対応: 独立サブタスクの並列実行を試す
- 影響: 大幅な時間短縮の可能性 (要検証)

## 📚 Tier 4 (ドキュメント・知見化)

### Task 4.1: README.md 作成
- 現状 README.md なし (CLAUDE.md のみ)
- 内容: セットアップ手順、環境変数、tools/run_hera.py 使い方、性能ベンチ

### Task 4.2: ベンチマーク標準化
- 標準タスクセット (Hello / FastAPI JWT / 自己レビュー) を benchmarks/ に配置
- 実行スクリプトで各タスクを連続実行、結果を表形式でまとめ
- llms.yaml 変更時の回帰検証用

## 開始の起点
次回 Cowork セッションでこのドキュメントを開いた時、Tier 1 から順に進めることを推奨。
ただし当日のテーマ次第で柔軟に。

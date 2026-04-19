from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import litellm
from rich import box
from rich.panel import Panel
from rich.table import Table


@dataclass
class _CallRecord:
    step: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _StepSummary:
    name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_s: float = 0.0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ── task keyword classifier ───────────────────────────────────────────────────

_CLOUD_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6":         (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-opus-4-7":           (15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-haiku-4-5":          (0.80 / 1_000_000, 4.00 / 1_000_000),
    "gpt-4o":                    (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-4o-mini":               (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gemini-2.0-flash":          (0.10 / 1_000_000, 0.40 / 1_000_000),
}
_DEFAULT_ORCHESTRATOR_PRICING = (3.00 / 1_000_000, 15.00 / 1_000_000)  # Claude Sonnet 4.6

_TASK_CATEGORIES: list[tuple[str, list[str]]] = [
    ("コード生成",   ["作成", "書いて", "実装", "generate", "create", "write", "implement"]),
    ("リファクタ",   ["リファクタ", "整理", "refactor", "clean", "restructure"]),
    ("デバッグ",     ["バグ", "エラー", "直して", "修正", "bug", "fix", "error", "debug"]),
    ("コードレビュー", ["レビュー", "確認", "チェック", "review", "check", "audit"]),
    ("設計・計画",   ["設計", "アーキテクチャ", "計画", "design", "architect", "plan"]),
    ("テスト",       ["テスト", "test", "spec", "pytest", "unittest"]),
    ("説明・調査",   ["説明", "教えて", "調査", "explain", "what", "how", "why"]),
]

def _classify_task(task: str) -> str:
    tl = task.lower()
    for label, keywords in _TASK_CATEGORIES:
        if any(k in tl for k in keywords):
            return label
    return "その他"


class UsageTracker:
    _REF_MODEL = "GPT-4o"
    _INPUT_PRICE = 2.50 / 1_000_000
    _OUTPUT_PRICE = 10.00 / 1_000_000

    def __init__(self) -> None:
        self._records: list[_CallRecord] = []
        self._delegations: int = 0
        self._current_step: str = "init"
        self._run_start: float = time.time()
        self._model_name: str = ""
        self._task: str = ""
        self._step_start_times: dict[str, float] = {}
        self._step_elapsed: dict[str, float] = {}
        self._orch_input_tokens: int = 0
        self._orch_output_tokens: int = 0
        self._orch_model: str = ""

    def register_litellm(self, model_name: str = "") -> None:
        self._model_name = model_name
        self._run_start = time.time()
        tracker = self

        def _on_success(kwargs, response_obj, start_time, end_time):
            usage = getattr(response_obj, "usage", None)
            if not usage:
                return
            tracker._records.append(_CallRecord(
                step=tracker._current_step,
                model=kwargs.get("model", "unknown"),
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ))

        litellm.success_callback.append(_on_success)

    def set_task(self, description: str) -> None:
        self._task = description[:120]

    def set_step(self, name: str) -> None:
        now = time.time()
        if self._current_step != "init":
            prev = self._current_step
            self._step_elapsed[prev] = (
                self._step_elapsed.get(prev, 0.0)
                + now - self._step_start_times.get(prev, now)
            )
        self._current_step = name
        self._step_start_times[name] = now

    def finalize(self) -> None:
        """Close the last step's elapsed timer."""
        now = time.time()
        if self._current_step != "init":
            s = self._current_step
            self._step_elapsed[s] = (
                self._step_elapsed.get(s, 0.0)
                + now - self._step_start_times.get(s, now)
            )

    def record_delegation(self) -> None:
        self._delegations += 1

    def record_orchestrator_usage(self, input_tokens: int, output_tokens: int, model: str = "") -> None:
        self._orch_input_tokens = input_tokens
        self._orch_output_tokens = output_tokens
        self._orch_model = model or "claude-sonnet-4-6"

    # ── aggregated stats ──────────────────────────────────────────────────────

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self._records)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def call_count(self) -> int:
        return len(self._records)

    @property
    def estimated_cloud_savings_usd(self) -> float:
        return (
            self.total_prompt_tokens * self._INPUT_PRICE
            + self.total_completion_tokens * self._OUTPUT_PRICE
        )

    @property
    def orchestrator_cost_usd(self) -> float:
        if not (self._orch_input_tokens or self._orch_output_tokens):
            return 0.0
        key = self._orch_model.lower()
        in_price, out_price = _CLOUD_PRICING.get(key, _DEFAULT_ORCHESTRATOR_PRICING)
        return self._orch_input_tokens * in_price + self._orch_output_tokens * out_price

    @property
    def has_orchestrator_data(self) -> bool:
        return bool(self._orch_input_tokens or self._orch_output_tokens)

    @property
    def estimated_savings_vs_orchestrator(self) -> float:
        """HERAの節約額をオーケストレーターモデルと同じレートで計算"""
        key = self._orch_model.lower() if self._orch_model else ""
        in_p, out_p = _CLOUD_PRICING.get(key, _DEFAULT_ORCHESTRATOR_PRICING)
        return self.total_prompt_tokens * in_p + self.total_completion_tokens * out_p

    @property
    def output_ratio(self) -> float:
        """出力トークン / 合計トークン (高いほど応答が充実)"""
        return self.total_completion_tokens / self.total_tokens if self.total_tokens else 0.0

    def _step_summaries(self) -> list[_StepSummary]:
        steps: dict[str, _StepSummary] = {}
        for r in self._records:
            if r.step not in steps:
                steps[r.step] = _StepSummary(name=r.step)
            steps[r.step].prompt_tokens += r.prompt_tokens
            steps[r.step].completion_tokens += r.completion_tokens
        for name, s in steps.items():
            s.elapsed_s = round(self._step_elapsed.get(name, 0.0), 1)
        return list(steps.values())

    # ── Rich rendering ────────────────────────────────────────────────────────

    def render_savings_panel(self) -> Panel:
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1), expand=True)
        table.add_column("", style="dim", min_width=28)
        table.add_column("", justify="right", min_width=16)

        table.add_row("LLM 呼び出し数", str(self.call_count))
        table.add_row("入力トークン", f"{self.total_prompt_tokens:,}")
        table.add_row("出力トークン", f"{self.total_completion_tokens:,}")
        table.add_row("[bold]合計トークン[/]", f"[bold]{self.total_tokens:,}[/]")
        table.add_row("出力効率", f"{self.output_ratio:.1%}")
        table.add_row("", "")
        table.add_row(
            f"クラウド相当コスト ({self._REF_MODEL})",
            f"[bold red]${self.estimated_cloud_savings_usd:.4f}[/]",
        )
        table.add_row("実際のコスト（ローカル）", "[bold green]$0.0000[/]")
        table.add_row(
            "[bold]節約額[/]",
            f"[bold green]${self.estimated_cloud_savings_usd:.4f} 💰[/]",
        )
        table.add_row("", "")
        if self._delegations > 0:
            table.add_row("クラウド委譲", f"[bold yellow]{self._delegations} 件[/]")
        else:
            table.add_row("クラウド委譲", "[bold green]なし（100% ローカル）[/]")

        return Panel(
            table,
            title="[bold green]💰 クラウドクォータ節約レポート[/]",
            border_style="green",
        )

    # ── history persistence ───────────────────────────────────────────────────

    def _append_history(self, ts: str, elapsed: float, output_dir: Path) -> None:
        entry = {
            "ts": ts,
            "task": self._task,
            "category": _classify_task(self._task),
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "output_ratio": round(self.output_ratio, 4),
            "savings_usd": round(self.estimated_cloud_savings_usd, 6),
            "elapsed_s": round(elapsed, 1),
            "call_count": self.call_count,
            "delegations": self._delegations,
            "model": self._model_name,
            "step_elapsed": {k: round(v, 1) for k, v in self._step_elapsed.items()},
            "orch_input_tokens": self._orch_input_tokens,
            "orch_output_tokens": self._orch_output_tokens,
            "orch_model": self._orch_model,
            "orch_cost_usd": round(self.orchestrator_cost_usd, 6),
        }
        with open(output_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _load_history(self, output_dir: Path) -> list[dict]:
        p = output_dir / "history.jsonl"
        if not p.exists():
            return []
        runs: list[dict] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return runs[-30:]

    # ── SVG helpers ───────────────────────────────────────────────────────────

    def _svg_bar_chart(self, history: list[dict], metric: str, color: str, fmt: str = "{:.0f}") -> str:
        vals = [h[metric] for h in history]
        max_val = max(vals) or 1
        n = len(vals)
        W, H = 560, 110
        PL, PR, PT, PB = 8, 8, 8, 24
        inner_w = W - PL - PR
        bar_w = max(2, inner_w / n - 2)

        bars = ""
        for i, v in enumerate(vals):
            bh = max(2, (v / max_val) * (H - PT - PB))
            x = PL + i * (inner_w / n)
            y = H - PB - bh
            tip = history[i]["ts"][:16].replace("_", " ")
            bars += (
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}"'
                f' fill="{color}bb" rx="2"><title>{tip}: {fmt.format(v)}</title></rect>'
            )

        labels = ""
        step = max(1, n // 5)
        for i in range(0, n, step):
            x = PL + i * (inner_w / n) + bar_w / 2
            labels += (
                f'<text x="{x:.1f}" y="{H-6}" text-anchor="middle"'
                f' font-size="8" fill="#475569">{history[i]["ts"][5:10]}</text>'
            )
        if vals:
            last_x = PL + (n - 1) * (inner_w / n) + bar_w / 2
            last_bh = max(2, (vals[-1] / max_val) * (H - PT - PB))
            last_y = H - PB - last_bh - 4
            labels += (
                f'<text x="{last_x:.1f}" y="{last_y:.1f}" text-anchor="middle"'
                f' font-size="8" fill="{color}" font-weight="bold">{fmt.format(vals[-1])}</text>'
            )
        return f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;">{bars}{labels}</svg>'

    def _svg_area_chart(self, history: list[dict], metric: str, color: str, fmt: str = "{:.4f}") -> str:
        vals = [h[metric] for h in history]
        max_val = max(vals) or 1
        n = len(vals)
        if n < 2:
            return self._svg_bar_chart(history, metric, color, fmt)
        W, H = 560, 110
        PL, PR, PT, PB = 8, 8, 8, 24

        def px(i: int) -> float:
            return PL + i * (W - PL - PR) / (n - 1)
        def py(v: float) -> float:
            return PT + (1 - v / max_val) * (H - PT - PB)

        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        area = pts + f" {px(n-1):.1f},{H-PB} {px(0):.1f},{H-PB}"

        labels = ""
        step = max(1, n // 5)
        for i in range(0, n, step):
            labels += (
                f'<text x="{px(i):.1f}" y="{H-6}" text-anchor="middle"'
                f' font-size="8" fill="#475569">{history[i]["ts"][5:10]}</text>'
            )
        lx, ly = px(n - 1), py(vals[-1]) - 4
        labels += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="end"'
            f' font-size="8" fill="{color}" font-weight="bold">{fmt.format(vals[-1])}</text>'
        )
        return (
            f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;">'
            f'<polygon points="{area}" fill="{color}22"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>'
            f'{labels}</svg>'
        )

    def _svg_hbar(self, items: list[tuple[str, float]], color: str, fmt: str = "{:.1f}s") -> str:
        """Horizontal bar chart for step breakdown."""
        if not items:
            return ""
        max_val = max(v for _, v in items) or 1
        W, H_ROW = 560, 22
        H = H_ROW * len(items) + 4
        PL, PR = 110, 60

        bars = ""
        for i, (label, val) in enumerate(items):
            y = i * H_ROW + 2
            bw = max(2, (val / max_val) * (W - PL - PR))
            bars += (
                f'<text x="{PL-6}" y="{y+15}" text-anchor="end" font-size="9" fill="#94a3b8">{label}</text>'
                f'<rect x="{PL}" y="{y+4}" width="{W-PL-PR}" height="14" fill="#2d2d44" rx="3"/>'
                f'<rect x="{PL}" y="{y+4}" width="{bw:.1f}" height="14" fill="{color}cc" rx="3"/>'
                f'<text x="{PL+bw+4:.1f}" y="{y+15}" font-size="9" fill="{color}">{fmt.format(val)}</text>'
            )
        return f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;">{bars}</svg>'

    def _svg_freq_chart(self, history: list[dict]) -> str:
        """Runs-per-day bar chart."""
        counts: dict[str, int] = defaultdict(int)
        for h in history:
            counts[h["ts"][:10]] += 1
        if not counts:
            return ""
        dates = sorted(counts)
        return self._svg_bar_chart(
            [{"ts": d + "_00-00-00", metric: counts[d]} for d in dates],
            metric := "count",
            "#a78bfa",
            "{:.0f}回",
        ) if (metric := "count") else ""

    def _svg_category_chart(self, history: list[dict]) -> str:
        """Horizontal bar chart of task categories."""
        counts: dict[str, int] = defaultdict(int)
        for h in history:
            counts[h.get("category", "その他")] += 1
        if not counts:
            return ""
        items = sorted(counts.items(), key=lambda x: -x[1])
        return self._svg_hbar([(k, v) for k, v in items], "#f59e0b", "{:.0f}件")

    # ── HTML infographic ──────────────────────────────────────────────────────

    def save_html(self, output_dir: Path | None = None) -> Path:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent.parent.parent / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        elapsed = time.time() - self._run_start

        self._append_history(ts, elapsed, output_dir)
        history = self._load_history(output_dir)

        path = output_dir / "hera_report.html"
        path.write_text(self._render_html(ts, elapsed, history), encoding="utf-8")
        return path

    def _effectiveness_score(self, history: list[dict]) -> tuple[int, str]:
        """0–100のスコアと評価コメントを返す"""
        if not history:
            return 0, "データなし"
        last = history[-1]
        if not last.get("total_tokens", 0):
            return 0, "データなし"
        score = 0
        ratio = last.get("output_ratio", 0)
        score += int(min(ratio / 0.20, 1.0) * 30)
        score += 20 if last.get("delegations", 0) == 0 else 0
        score += int(max(0, 1 - last.get("elapsed_s", 999) / 120) * 20)
        recent_dates = {h["ts"][:10] for h in history[-20:] if h.get("total_tokens", 0) > 0}
        score += min(int(len(recent_dates) / 3 * 30), 30)

        if score >= 80:
            comment = "非常に効果的"
        elif score >= 60:
            comment = "概ね良好"
        elif score >= 40:
            comment = "改善の余地あり"
        else:
            comment = "利用を増やしましょう"
        return min(score, 100), comment

    def _render_html(self, ts: str, elapsed: float, history: list[dict]) -> str:
        steps = self._step_summaries()
        max_tokens = max((s.total for s in steps), default=1)
        out_ratio = self.output_ratio
        score, score_comment = self._effectiveness_score(history)

        # ── タイムスタンプ表示用 ──────────────────────────────────────────
        ts_display = ts[:10] + " " + ts[11:].replace("-", ":")

        # ── コスト計算（④ 参照モデルの統一） ─────────────────────────────
        orch_cost = self.orchestrator_cost_usd
        if self.has_orchestrator_data:
            hera_savings = self.estimated_savings_vs_orchestrator
            savings_ref_label = self._orch_model or "Claude Sonnet 4.6"
            in_p, out_p = _CLOUD_PRICING.get(savings_ref_label.lower(), _DEFAULT_ORCHESTRATOR_PRICING)
            savings_ref_price_str = f"入力 ${in_p*1_000_000:.2f} / 出力 ${out_p*1_000_000:.2f} per 1M tok"
        else:
            hera_savings = self.estimated_cloud_savings_usd
            savings_ref_label = self._REF_MODEL
            savings_ref_price_str = "入力 $2.50 / 出力 $10.00 per 1M tok"
        full_cloud_cost = orch_cost + hera_savings
        cost_reduction_pct = (hera_savings / full_cloud_cost * 100) if full_cloud_cost > 0 else 0

        # ── ⑥ 1行サマリー ────────────────────────────────────────────────
        if self.has_orchestrator_data:
            summary_html = f"""<div class="summary-callout">
  クラウドLLM <span style="color:#f87171;font-weight:600;">${orch_cost:.4f}</span> 使用 &nbsp;·&nbsp;
  HERAで <span style="color:#34d399;font-weight:600;">${hera_savings:.4f}</span> 節約 &nbsp;·&nbsp;
  コスト削減率 <span style="color:#a78bfa;font-weight:600;">{cost_reduction_pct:.0f}%</span>
</div>"""
        elif hera_savings > 0:
            summary_html = f"""<div class="summary-callout">
  HERAで <span style="color:#34d399;font-weight:600;">${hera_savings:.4f}</span> 節約（vs {savings_ref_label}）&nbsp;·&nbsp;
  <span style="color:#64748b;">クラウドLLMのトークン使用量は未報告</span>
</div>"""
        else:
            summary_html = ""

        delegation_badge = (
            f'<span class="badge badge-warn">クラウド委譲 {self._delegations} 件</span>'
            if self._delegations > 0
            else '<span class="badge badge-ok">100% ローカル処理</span>'
        )

        # ── ステップ別トークン棒グラフ ─────────────────────────────────────
        step_bars = ""
        colors = ["#6366f1", "#22d3ee", "#f59e0b", "#34d399"]
        for i, s in enumerate(steps):
            pct_p = int(s.prompt_tokens / max_tokens * 100)
            pct_c = int(s.completion_tokens / max_tokens * 100)
            col = colors[i % len(colors)]
            step_bars += f"""
            <div class="bar-row">
              <div class="bar-label">{s.name}</div>
              <div class="bar-track">
                <div class="bar-seg" style="width:{pct_p}%;background:{col}cc;"
                     title="入力: {s.prompt_tokens:,}"></div>
                <div class="bar-seg" style="width:{pct_c}%;background:{col}55;"
                     title="出力: {s.completion_tokens:,}"></div>
              </div>
              <div class="bar-val">{s.total:,} tok</div>
            </div>"""

        # ── ② 空セクションを非表示 ───────────────────────────────────────
        step_token_section = f"""
<hr class="divider">
<p class="section-title">ステップ別トークン使用量</p>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#6366f1cc;"></div>入力トークン</div>
  <div class="legend-item"><div class="legend-dot" style="background:#6366f155;"></div>出力トークン</div>
</div>
{step_bars}""" if step_bars.strip() else ""

        step_time_items = [(s.name, s.elapsed_s) for s in steps if s.elapsed_s > 0]
        step_time_svg = self._svg_hbar(step_time_items, "#22d3ee")
        step_time_section = f"""
<hr class="divider">
<p class="section-title">ステップ別所要時間</p>
<div class="ts-card" style="max-width:560px;">{step_time_svg}</div>""" if step_time_svg else ""

        # ── ③④⑦ コスト内訳セクション ──────────────────────────────────
        if self.has_orchestrator_data:
            orch_model_label = self._orch_model or "Claude Sonnet 4.6"
            orch_tokens_total = self._orch_input_tokens + self._orch_output_tokens
            bar_max = max(full_cloud_cost, 0.0001)
            orch_pct = int(orch_cost / bar_max * 100)
            hera_pct = int(hera_savings / bar_max * 100)
            zero_savings_note = (
                '<p style="font-size:.75rem;color:#64748b;margin-top:.6rem;">'
                '⚠ 今回はHERAへの委譲タスクがなく、クラウドLLMのみ使用しました。</p>'
            ) if hera_savings == 0 else ""
            cost_section = f"""
<hr class="divider">
<p class="section-title">コスト内訳  <span style="font-weight:400;color:#334155">（クラウド実費 + ローカル節約）</span></p>
<div class="grid" style="grid-template-columns:1fr 1fr 1fr 1fr;">
  <div class="card" style="border-color:#f87171;">
    <div class="card-label">クラウドLLM実費</div>
    <div class="card-value" style="color:#f87171;">${orch_cost:.4f}</div>
    <div class="ref">{orch_model_label} · {orch_tokens_total:,} tok</div>
    <div class="ref">入力: {self._orch_input_tokens:,} / 出力: {self._orch_output_tokens:,}</div>
  </div>
  <div class="card savings">
    <div class="card-label">HERA節約額</div>
    <div class="card-value">${hera_savings:.4f}</div>
    <div class="ref">vs {orch_model_label}（同レートで換算）</div>
  </div>
  <div class="card" style="border-color:#6366f1;">
    <div class="card-label">全クラウド換算（参考）</div>
    <div class="card-value" style="color:#6366f1;">${full_cloud_cost:.4f}</div>
    <div class="ref">すべてクラウドで処理した場合</div>
  </div>
  <div class="card" style="border-color:#a78bfa;">
    <div class="card-label">コスト削減率（ROI）</div>
    <div class="card-value" style="color:#a78bfa;">{cost_reduction_pct:.0f}%</div>
    <div class="ref">全クラウド比 {cost_reduction_pct:.0f}% 安く実行</div>
  </div>
</div>
<div class="ts-card" style="max-width:560px;margin-bottom:1rem;">
  <div class="ts-label">コスト構成比（全クラウド換算を100%とした場合）</div>
  <svg viewBox="0 0 560 44" style="width:100%;height:auto;">
    <rect x="0" y="10" width="560" height="24" fill="#2d2d44" rx="4"/>
    <rect x="0" y="10" width="{orch_pct * 5.6:.1f}" height="24" fill="#f87171cc" rx="4">
      <title>クラウドLLM: ${orch_cost:.4f} ({orch_pct}%)</title></rect>
    <rect x="{orch_pct * 5.6:.1f}" y="10" width="{hera_pct * 5.6:.1f}" height="24" fill="#34d39988" rx="0">
      <title>HERA節約: ${hera_savings:.4f} ({hera_pct}%)</title></rect>
    <text x="{orch_pct * 5.6 / 2:.1f}" y="26" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">クラウド {orch_pct}%</text>
    <text x="{orch_pct * 5.6 + hera_pct * 5.6 / 2:.1f}" y="26" text-anchor="middle" font-size="9" fill="#fff" font-weight="bold">HERA節約 {hera_pct}%</text>
    <text x="4" y="42" font-size="8" fill="#f87171">${orch_cost:.4f}</text>
    <text x="556" y="42" text-anchor="end" font-size="8" fill="#34d399">節約 ${hera_savings:.4f}</text>
  </svg>
  {zero_savings_note}
</div>"""
        else:
            cost_section = f"""
<hr class="divider">
<p class="section-title">コスト内訳</p>
<div class="ts-card" style="max-width:480px;margin-bottom:1rem;">
  <div style="color:#64748b;font-size:.82rem;">
    ⚠️ クラウドLLMのトークン使用量が未報告です。<br>
    <code style="font-size:.75rem;color:#94a3b8;">delegate_task</code> 呼び出し時に
    <code style="font-size:.75rem;color:#94a3b8;">orchestrator_input_tokens</code> /
    <code style="font-size:.75rem;color:#94a3b8;">orchestrator_output_tokens</code> を渡すとコスト比較が表示されます。
  </div>
  <div style="margin-top:.8rem;">
    <span class="badge badge-ok">HERA節約額: ${hera_savings:.4f}（vs {savings_ref_label}）</span>
  </div>
</div>"""

        # ── 時系列データ ──────────────────────────────────────────────────
        cum_history: list[dict] = []
        cumsum = 0.0
        for h in history:
            cumsum += h["savings_usd"]
            cum_history.append({**h, "cumulative_savings": round(cumsum, 6)})

        ratio_history = [{**h, "output_ratio_pct": round(h.get("output_ratio", 0) * 100, 1)} for h in history]

        date_counts: dict[str, int] = defaultdict(int)
        for h in history:
            date_counts[h["ts"][:10]] += 1
        freq_history = [{"ts": d + "_00-00-00", "count": date_counts[d]} for d in sorted(date_counts)]

        timeseries_html = ""
        if len(history) >= 2:
            tokens_chart = self._svg_bar_chart(history, "total_tokens", "#6366f1", "{:.0f}")
            savings_chart = self._svg_area_chart(cum_history, "cumulative_savings", "#34d399", "${:.4f}")
            ratio_chart = self._svg_area_chart(ratio_history, "output_ratio_pct", "#f59e0b", "{:.1f}%")
            freq_chart = self._svg_bar_chart(freq_history, "count", "#a78bfa", "{:.0f}回")
            category_chart = self._svg_category_chart(history)
            orch_history = [h for h in history if h.get("orch_cost_usd", 0) > 0]

            recent = history[-10:][::-1]
            rows = ""
            for h in recent:
                cat = h.get("category", "その他")
                task_short = (h.get("task") or "")[:40] + ("…" if len(h.get("task","")) > 40 else "")
                deleg = f'<span style="color:#f59e0b">{h["delegations"]} 件</span>' if h["delegations"] else '<span style="color:#34d399">なし</span>'
                ratio_pct = f'{h.get("output_ratio",0)*100:.0f}%'
                oc = h.get("orch_cost_usd", 0)
                orch_cell = f'<span style="color:#f87171">${oc:.4f}</span>' if oc else '<span style="color:#475569">—</span>'
                h_ts = h["ts"]
                h_ts_display = h_ts[:10] + " " + h_ts[11:].replace("-", ":")
                rows += f"""<tr>
                  <td>{h_ts_display}</td>
                  <td><span class="cat-badge">{cat}</span></td>
                  <td title="{h.get('task','')}">{task_short}</td>
                  <td>{h["total_tokens"]:,}</td>
                  <td>{ratio_pct}</td>
                  <td style="color:#34d399">${h["savings_usd"]:.4f}</td>
                  <td>{orch_cell}</td>
                  <td>{h["elapsed_s"]}s</td>
                  <td>{deleg}</td>
                </tr>"""

            orch_cost_chart = (
                f'<div class="ts-card"><div class="ts-label">クラウドLLM実費の推移 (USD)</div>'
                f'{self._svg_area_chart(orch_history, "orch_cost_usd", "#f87171", "${:.4f}")}</div>'
            ) if len(orch_history) >= 2 else ""

            timeseries_html = f"""
<hr class="divider">
<p class="section-title">時系列推移  <span style="font-weight:400;color:#334155">（{len(history)} 回分）</span></p>
<div class="ts-grid">
  <div class="ts-card"><div class="ts-label">実行ごとのトークン数</div>{tokens_chart}</div>
  <div class="ts-card"><div class="ts-label">クラウド節約額の累計 (USD)</div>{savings_chart}</div>
  <div class="ts-card"><div class="ts-label">出力効率の推移 (%)</div>{ratio_chart}</div>
  <div class="ts-card"><div class="ts-label">1日あたりの利用回数</div>{freq_chart}</div>
</div>
{"<div class='ts-grid'>" + orch_cost_chart + "</div>" if orch_cost_chart else ""}

<hr class="divider">
<p class="section-title">タスク種別の分布</p>
<div class="ts-card" style="max-width:480px;">{category_chart}</div>

<hr class="divider">
<p class="section-title">直近の実行履歴</p>
<div style="overflow-x:auto;">
<table class="run-table">
  <thead><tr>
    <th>日時</th><th>カテゴリ</th><th>タスク概要</th>
    <th>トークン</th><th>出力効率</th><th>HERA節約</th><th>クラウド実費</th><th>時間</th><th>委譲</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>"""

        # ── 効率スコアゲージ ──────────────────────────────────────────────
        score_color = "#34d399" if score >= 70 else "#f59e0b" if score >= 40 else "#f87171"
        gauge_dash = 2 * 3.14159 * 40
        gauge_fill = gauge_dash * score / 100
        ratio_color = "#34d399" if out_ratio >= 0.15 else "#f59e0b" if out_ratio >= 0.08 else "#f87171"

        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>HERA 実行レポート – {ts_display}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', 'Noto Sans JP', system-ui, sans-serif;
          background: #0f0f18; color: #e2e8f0; min-height: 100vh; padding: 2rem; }}
  h1 {{ font-size: 1.5rem; font-weight: 700; color: #a78bfa; letter-spacing: .05em; }}
  .meta {{ font-size: .8rem; color: #64748b; margin-top: .3rem; }}
  .task-text {{ font-size: .85rem; color: #94a3b8; margin-top: .6rem;
               background: #1e1e2e; border-left: 3px solid #6366f1;
               padding: .4rem .8rem; border-radius: 0 6px 6px 0; }}
  .summary-callout {{ background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 8px;
                      padding: .6rem 1rem; margin-top: .8rem; font-size: .85rem; color: #94a3b8; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }}
  .card {{ background: #1e1e2e; border: 1px solid #2d2d44; border-radius: 12px; padding: 1.2rem 1.4rem; }}
  .card-label {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #64748b; }}
  .card-value {{ font-size: 1.8rem; font-weight: 800; margin-top: .3rem; }}
  .savings .card-value {{ color: #34d399; }}
  .tokens  .card-value {{ color: #6366f1; }}
  .timec   .card-value {{ color: #f59e0b; }}
  .ratio   .card-value {{ color: {ratio_color}; }}
  .score-wrap {{ display: flex; align-items: center; gap: 1.5rem; }}
  .score-text {{ font-size: 2.2rem; font-weight: 900; color: {score_color}; }}
  .score-comment {{ font-size: .8rem; color: #64748b; margin-top: .2rem; }}
  .section-title {{ font-size: .82rem; text-transform: uppercase; letter-spacing: .1em;
                    color: #64748b; margin-bottom: .8rem; }}
  .bar-row {{ display: flex; align-items: center; gap: .8rem; margin-bottom: .7rem; }}
  .bar-label {{ width: 10rem; font-size: .82rem; color: #94a3b8; flex-shrink: 0; text-align: right; }}
  .bar-track {{ flex: 1; display: flex; height: 18px; border-radius: 4px; overflow: hidden; background: #2d2d44; }}
  .bar-seg {{ height: 100%; min-width: 0; }}
  .bar-val {{ width: 5.5rem; font-size: .8rem; color: #64748b; flex-shrink: 0; }}
  .legend {{ display: flex; gap: 1.5rem; margin-bottom: 1rem; }}
  .legend-item {{ display: flex; align-items: center; gap: .4rem; font-size: .78rem; color: #94a3b8; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
  .badge {{ display: inline-block; font-size: .72rem; font-weight: 600;
            padding: .2rem .6rem; border-radius: 20px; margin-top: .6rem; }}
  .badge-ok   {{ background: #064e3b; color: #34d399; }}
  .badge-warn {{ background: #451a03; color: #f59e0b; }}
  .cat-badge {{ font-size: .7rem; background: #1e293b; color: #94a3b8;
                padding: .1rem .5rem; border-radius: 10px; white-space: nowrap; }}
  .ref {{ font-size: .72rem; color: #475569; margin-top: .3rem; }}
  .divider {{ border: none; border-top: 1px solid #2d2d44; margin: 1.8rem 0; }}
  .ts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.2rem; }}
  .ts-card {{ background: #1e1e2e; border: 1px solid #2d2d44; border-radius: 10px; padding: .9rem 1.1rem; }}
  .ts-label {{ font-size: .72rem; color: #64748b; margin-bottom: .5rem;
               text-transform: uppercase; letter-spacing: .06em; }}
  .run-table {{ width: 100%; border-collapse: collapse; font-size: .8rem; }}
  .run-table th {{ text-align: left; padding: .45rem .7rem; color: #64748b; font-weight: 600;
                   border-bottom: 1px solid #2d2d44; font-size: .72rem; text-transform: uppercase; }}
  .run-table td {{ padding: .45rem .7rem; border-bottom: 1px solid #1a1a2e; color: #94a3b8; }}
  .run-table tr:first-child td {{ color: #e2e8f0; }}
  .footer {{ font-size: .7rem; color: #334155; text-align: center; margin-top: 2rem; }}
</style>
</head>
<body>

<h1>🤖 HERA 実行レポート</h1>
<p class="meta">生成日時: {ts_display} &nbsp;·&nbsp; モデル: {self._model_name}</p>
{f'<p class="task-text">📋 {self._task}</p>' if self._task else ""}
{summary_html}

<div class="grid">
  <div class="card savings">
    <div class="card-label">今回の節約額</div>
    <div class="card-value">${hera_savings:.4f}</div>
    <div class="ref">vs {savings_ref_label}（{savings_ref_price_str}）</div>
    {delegation_badge}
  </div>
  <div class="card tokens">
    <div class="card-label">使用トークン数</div>
    <div class="card-value">{self.total_tokens:,}</div>
    <div class="ref">入力: {self.total_prompt_tokens:,} / 出力: {self.total_completion_tokens:,}</div>
  </div>
  <div class="card ratio">
    <div class="card-label">出力効率</div>
    <div class="card-value">{out_ratio:.1%}</div>
    <div class="ref">出力トークン / 合計トークン（目安: 15〜30%）</div>
  </div>
  <div class="card timec">
    <div class="card-label">実行時間</div>
    <div class="card-value">{elapsed:.1f}s</div>
    <div class="ref">{self.call_count} 回の LLM 呼び出し · すべて Ollama</div>
  </div>
</div>

<hr class="divider">

<p class="section-title">利用効率スコア</p>
<div class="card" style="max-width:340px; margin-bottom:1.5rem;">
  <div class="score-wrap">
    <svg width="90" height="90" viewBox="0 0 90 90">
      <circle cx="45" cy="45" r="40" fill="none" stroke="#2d2d44" stroke-width="8"/>
      <circle cx="45" cy="45" r="40" fill="none" stroke="{score_color}" stroke-width="8"
        stroke-dasharray="{gauge_fill:.1f} {gauge_dash:.1f}"
        stroke-linecap="round" transform="rotate(-90 45 45)"/>
      <text x="45" y="50" text-anchor="middle" font-size="20" font-weight="900" fill="{score_color}">{score if score > 0 else "—"}</text>
    </svg>
    <div>
      <div class="score-text">{score_comment}</div>
      <div class="score-comment">出力効率 · ローカル率 · 速度 · 継続利用</div>
    </div>
  </div>
</div>

{step_token_section}

{step_time_section}

{cost_section}

{timeseries_html}

<div class="footer">Generated by HERA · ローカルファーストのマルチエージェントシステム</div>
</body>
</html>"""

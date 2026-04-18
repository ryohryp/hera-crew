from __future__ import annotations

import json
import time
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

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


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

    def set_step(self, name: str) -> None:
        self._current_step = name

    def record_delegation(self) -> None:
        self._delegations += 1

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

    def _step_summaries(self) -> list[_StepSummary]:
        steps: dict[str, _StepSummary] = {}
        for r in self._records:
            if r.step not in steps:
                steps[r.step] = _StepSummary(name=r.step)
            steps[r.step].prompt_tokens += r.prompt_tokens
            steps[r.step].completion_tokens += r.completion_tokens
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
            table.add_row(
                "クラウド委譲",
                f"[bold yellow]{self._delegations} 件[/]",
            )
        else:
            table.add_row("クラウド委譲", "[bold green]なし（100% ローカル）[/]")

        return Panel(
            table,
            title="[bold green]💰 Cloud Quota Savings[/]",
            border_style="green",
        )

    # ── history persistence ───────────────────────────────────────────────────

    def _append_history(self, ts: str, elapsed: float, output_dir: Path) -> None:
        entry = {
            "ts": ts,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "savings_usd": round(self.estimated_cloud_savings_usd, 6),
            "elapsed_s": round(elapsed, 1),
            "call_count": self.call_count,
            "delegations": self._delegations,
            "model": self._model_name,
        }
        with open(output_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

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
        return runs[-30:]  # keep last 30 runs for chart

    # ── SVG chart helpers ─────────────────────────────────────────────────────

    def _svg_bar_chart(
        self, history: list[dict], metric: str, color: str, fmt: str = "{:.0f}"
    ) -> str:
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
            bars += (
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}"'
                f' fill="{color}bb" rx="2">'
                f'<title>{history[i]["ts"][:16].replace("_"," ")}: {fmt.format(v)}</title>'
                f'</rect>'
            )

        # x-axis date labels (show ~5 evenly spaced)
        labels = ""
        step = max(1, n // 5)
        for i in range(0, n, step):
            x = PL + i * (inner_w / n) + bar_w / 2
            labels += (
                f'<text x="{x:.1f}" y="{H - 6}" text-anchor="middle"'
                f' font-size="8" fill="#475569">{history[i]["ts"][5:10]}</text>'
            )

        # top value label on last bar
        if vals:
            last_x = PL + (n - 1) * (inner_w / n) + bar_w / 2
            last_bh = max(2, (vals[-1] / max_val) * (H - PT - PB))
            last_y = H - PB - last_bh - 4
            labels += (
                f'<text x="{last_x:.1f}" y="{last_y:.1f}" text-anchor="middle"'
                f' font-size="8" fill="{color}" font-weight="bold">'
                f'{fmt.format(vals[-1])}</text>'
            )

        return (
            f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;">'
            f'{bars}{labels}'
            f'</svg>'
        )

    def _svg_area_chart(
        self, history: list[dict], metric: str, color: str, fmt: str = "{:.4f}"
    ) -> str:
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
        area_pts = pts + f" {px(n-1):.1f},{H-PB} {px(0):.1f},{H-PB}"

        # x-axis labels
        labels = ""
        step = max(1, n // 5)
        for i in range(0, n, step):
            labels += (
                f'<text x="{px(i):.1f}" y="{H - 6}" text-anchor="middle"'
                f' font-size="8" fill="#475569">{history[i]["ts"][5:10]}</text>'
            )

        # latest value label
        lx, ly = px(n - 1), py(vals[-1]) - 4
        labels += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="end"'
            f' font-size="8" fill="{color}" font-weight="bold">'
            f'{fmt.format(vals[-1])}</text>'
        )

        return (
            f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;">'
            f'<polygon points="{area_pts}" fill="{color}22"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>'
            f'{labels}'
            f'</svg>'
        )

    # ── HTML infographic ──────────────────────────────────────────────────────

    def save_html(self, output_dir: Path | None = None) -> Path:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent.parent.parent / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        elapsed = time.time() - self._run_start

        self._append_history(ts, elapsed, output_dir)
        history = self._load_history(output_dir)

        path = output_dir / f"hera_run_{ts}.html"
        path.write_text(self._render_html(ts, elapsed, history), encoding="utf-8")
        return path

    def _render_html(self, ts: str, elapsed: float, history: list[dict]) -> str:
        steps = self._step_summaries()
        max_tokens = max((s.total for s in steps), default=1)
        savings = self.estimated_cloud_savings_usd
        delegation_badge = (
            f'<span class="badge badge-warn">クラウド委譲 {self._delegations} 件</span>'
            if self._delegations > 0
            else '<span class="badge badge-ok">100% ローカル処理</span>'
        )

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
                     title="Prompt: {s.prompt_tokens:,}"></div>
                <div class="bar-seg" style="width:{pct_c}%;background:{col}55;"
                     title="Completion: {s.completion_tokens:,}"></div>
              </div>
              <div class="bar-val">{s.total:,} tok</div>
            </div>"""

        # cumulative savings for area chart
        cum_history = []
        cumsum = 0.0
        for h in history:
            cumsum += h["savings_usd"]
            cum_history.append({**h, "cumulative_savings": round(cumsum, 6)})

        timeseries_section = ""
        if len(history) >= 2:
            tokens_chart = self._svg_bar_chart(history, "total_tokens", "#6366f1", "{:.0f}")
            savings_chart = self._svg_area_chart(cum_history, "cumulative_savings", "#34d399", "${:.4f}")

            recent = history[-10:][::-1]
            rows = ""
            for h in recent:
                deleg = f'<span style="color:#f59e0b">{h["delegations"]} 件</span>' if h["delegations"] else '<span style="color:#34d399">なし</span>'
                rows += f"""<tr>
                  <td>{h["ts"].replace("_"," ")}</td>
                  <td>{h["total_tokens"]:,}</td>
                  <td style="color:#34d399">${h["savings_usd"]:.4f}</td>
                  <td>{h["elapsed_s"]}s</td>
                  <td>{deleg}</td>
                </tr>"""

            timeseries_section = f"""
<hr class="divider">
<p class="section-title">時系列推移  <span style="font-weight:400;color:#334155">({len(history)} 回分)</span></p>
<div class="ts-grid">
  <div class="ts-card">
    <div class="ts-label">実行ごとのトークン数</div>
    {tokens_chart}
  </div>
  <div class="ts-card">
    <div class="ts-label">クラウド節約額の累計 (USD)</div>
    {savings_chart}
  </div>
</div>

<hr class="divider">
<p class="section-title">直近の実行履歴</p>
<table class="run-table">
  <thead><tr>
    <th>日時</th><th>トークン数</th><th>節約額</th><th>実行時間</th><th>クラウド委譲</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>"""

        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>HERA Run Report – {ts}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f0f18; color: #e2e8f0;
    min-height: 100vh; padding: 2rem;
  }}
  h1 {{ font-size: 1.5rem; font-weight: 700; color: #a78bfa; letter-spacing: .05em; }}
  .meta {{ font-size: .8rem; color: #64748b; margin-top: .3rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.2rem; margin: 2rem 0; }}
  .card {{ background: #1e1e2e; border: 1px solid #2d2d44; border-radius: 12px; padding: 1.4rem 1.6rem; }}
  .card-label {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; color: #64748b; }}
  .card-value {{ font-size: 2rem; font-weight: 800; margin-top: .4rem; }}
  .savings .card-value {{ color: #34d399; }}
  .tokens  .card-value {{ color: #6366f1; }}
  .time    .card-value {{ color: #f59e0b; }}
  .section-title {{
    font-size: .85rem; text-transform: uppercase; letter-spacing: .1em;
    color: #64748b; margin-bottom: 1rem;
  }}
  .bar-row {{ display: flex; align-items: center; gap: .8rem; margin-bottom: .75rem; }}
  .bar-label {{ width: 10rem; font-size: .82rem; color: #94a3b8; flex-shrink: 0; text-align: right; }}
  .bar-track {{ flex: 1; display: flex; height: 18px; border-radius: 4px; overflow: hidden; background: #2d2d44; }}
  .bar-seg {{ height: 100%; min-width: 0; }}
  .bar-val {{ width: 5.5rem; font-size: .8rem; color: #64748b; flex-shrink: 0; }}
  .legend {{ display: flex; gap: 1.5rem; margin-bottom: 1.2rem; }}
  .legend-item {{ display: flex; align-items: center; gap: .4rem; font-size: .78rem; color: #94a3b8; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
  .badge {{ display: inline-block; font-size: .75rem; font-weight: 600; padding: .25rem .75rem; border-radius: 20px; margin-top: .8rem; }}
  .badge-ok   {{ background: #064e3b; color: #34d399; }}
  .badge-warn {{ background: #451a03; color: #f59e0b; }}
  .ref {{ font-size: .75rem; color: #475569; margin-top: .4rem; }}
  .divider {{ border: none; border-top: 1px solid #2d2d44; margin: 2rem 0; }}
  .ts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; margin-bottom: 1.5rem; }}
  .ts-card {{ background: #1e1e2e; border: 1px solid #2d2d44; border-radius: 12px; padding: 1rem 1.2rem; }}
  .ts-label {{ font-size: .75rem; color: #64748b; margin-bottom: .6rem; text-transform: uppercase; letter-spacing: .06em; }}
  .run-table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
  .run-table th {{ text-align: left; padding: .5rem .8rem; color: #64748b; font-weight: 600;
                   border-bottom: 1px solid #2d2d44; font-size: .75rem; text-transform: uppercase; }}
  .run-table td {{ padding: .5rem .8rem; border-bottom: 1px solid #1a1a2e; color: #94a3b8; }}
  .run-table tr:first-child td {{ color: #e2e8f0; }}
  .footer {{ font-size: .72rem; color: #334155; text-align: center; margin-top: 2rem; }}
</style>
</head>
<body>

<h1>🤖 HERA Run Report</h1>
<p class="meta">Generated: {ts.replace("_", " ")} &nbsp;·&nbsp; Model: {self._model_name}</p>

<div class="grid">
  <div class="card savings">
    <div class="card-label">今回の節約額（クラウド相当）</div>
    <div class="card-value">${savings:.4f}</div>
    <div class="ref">vs {self._REF_MODEL}（入力 $2.50 / 出力 $10.00 per 1M tok）</div>
    {delegation_badge}
  </div>
  <div class="card tokens">
    <div class="card-label">使用トークン数（ローカル処理）</div>
    <div class="card-value">{self.total_tokens:,}</div>
    <div class="ref">入力: {self.total_prompt_tokens:,} &nbsp;/&nbsp; 出力: {self.total_completion_tokens:,}</div>
  </div>
  <div class="card time">
    <div class="card-label">実行時間</div>
    <div class="card-value">{elapsed:.1f}s</div>
    <div class="ref">{self.call_count} 回の LLM 呼び出し · すべて Ollama</div>
  </div>
</div>

<hr class="divider">

<p class="section-title">ステップ別トークン使用量</p>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#6366f1cc;"></div>入力トークン</div>
  <div class="legend-item"><div class="legend-dot" style="background:#6366f155;"></div>出力トークン</div>
</div>
{step_bars}
{timeseries_section}

<div class="footer">Generated by HERA · ローカルファーストのマルチエージェントシステム</div>
</body>
</html>"""

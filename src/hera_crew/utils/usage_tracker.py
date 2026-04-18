from __future__ import annotations

import time
from dataclasses import dataclass, field
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

        table.add_row("LLM Calls (Local)", str(self.call_count))
        table.add_row("Prompt Tokens", f"{self.total_prompt_tokens:,}")
        table.add_row("Completion Tokens", f"{self.total_completion_tokens:,}")
        table.add_row("[bold]Total Tokens[/]", f"[bold]{self.total_tokens:,}[/]")
        table.add_row("", "")
        table.add_row(
            f"Est. Cloud Cost  ({self._REF_MODEL})",
            f"[bold red]${self.estimated_cloud_savings_usd:.4f}[/]",
        )
        table.add_row("Actual Local Cost", "[bold green]$0.0000[/]")
        table.add_row(
            "[bold]Savings[/]",
            f"[bold green]${self.estimated_cloud_savings_usd:.4f} 💰[/]",
        )
        table.add_row("", "")
        if self._delegations > 0:
            table.add_row(
                "Cloud Delegations",
                f"[bold yellow]{self._delegations} task(s) escalated[/]",
            )
        else:
            table.add_row("Cloud Delegations", "[bold green]0  (100% local)[/]")

        return Panel(
            table,
            title="[bold green]💰 Cloud Quota Savings[/]",
            border_style="green",
        )

    # ── HTML infographic ──────────────────────────────────────────────────────

    def save_html(self, output_dir: Path | None = None) -> Path:
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent.parent / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = output_dir / f"hera_run_{ts}.html"
        path.write_text(self._render_html(ts), encoding="utf-8")
        return path

    def _render_html(self, ts: str) -> str:
        elapsed = time.time() - self._run_start
        steps = self._step_summaries()
        max_tokens = max((s.total for s in steps), default=1)
        savings = self.estimated_cloud_savings_usd
        delegation_badge = (
            f'<span class="badge badge-warn">{self._delegations} Cloud Escalation(s)</span>'
            if self._delegations > 0
            else '<span class="badge badge-ok">100% Local</span>'
        )

        step_bars = ""
        colors = ["#6366f1", "#22d3ee", "#f59e0b", "#34d399"]
        for i, s in enumerate(steps):
            pct_prompt = int(s.prompt_tokens / max_tokens * 100)
            pct_comp = int(s.completion_tokens / max_tokens * 100)
            color = colors[i % len(colors)]
            step_bars += f"""
            <div class="bar-row">
              <div class="bar-label">{s.name}</div>
              <div class="bar-track">
                <div class="bar-seg bar-prompt"
                     style="width:{pct_prompt}%; background:{color}cc;"
                     title="Prompt: {s.prompt_tokens:,}"></div>
                <div class="bar-seg bar-comp"
                     style="width:{pct_comp}%; background:{color}66;"
                     title="Completion: {s.completion_tokens:,}"></div>
              </div>
              <div class="bar-val">{s.total:,} tok</div>
            </div>"""

        return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>HERA Run Report – {ts}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f0f18;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 2rem;
  }}
  h1 {{ font-size: 1.5rem; font-weight: 700; color: #a78bfa; letter-spacing: .05em; }}
  .meta {{ font-size: .8rem; color: #64748b; margin-top: .3rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.2rem; margin: 2rem 0; }}
  .card {{
    background: #1e1e2e;
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
  }}
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
  .bar-seg {{
    height: 100%;
    transition: width .6s cubic-bezier(.4,0,.2,1);
    min-width: 0;
  }}
  .bar-val {{ width: 5.5rem; font-size: .8rem; color: #64748b; flex-shrink: 0; }}
  .legend {{ display: flex; gap: 1.5rem; margin-bottom: 1.2rem; }}
  .legend-item {{ display: flex; align-items: center; gap: .4rem; font-size: .78rem; color: #94a3b8; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
  .badge {{
    display: inline-block; font-size: .75rem; font-weight: 600;
    padding: .25rem .75rem; border-radius: 20px; margin-top: .8rem;
  }}
  .badge-ok   {{ background: #064e3b; color: #34d399; }}
  .badge-warn {{ background: #451a03; color: #f59e0b; }}
  .ref {{ font-size: .75rem; color: #475569; margin-top: .4rem; }}
  .divider {{ border: none; border-top: 1px solid #2d2d44; margin: 2rem 0; }}
  .footer {{ font-size: .72rem; color: #334155; text-align: center; margin-top: 2rem; }}
</style>
</head>
<body>

<h1>🤖 HERA Run Report</h1>
<p class="meta">Generated: {ts.replace("_", " ")} &nbsp;·&nbsp; Model: {self._model_name}</p>

<div class="grid">
  <div class="card savings">
    <div class="card-label">Cloud Quota Saved</div>
    <div class="card-value">${savings:.4f}</div>
    <div class="ref">vs {self._REF_MODEL} (input $2.50/output $10.00 per 1M tok)</div>
    {delegation_badge}
  </div>
  <div class="card tokens">
    <div class="card-label">Total Tokens (Local)</div>
    <div class="card-value">{self.total_tokens:,}</div>
    <div class="ref">Prompt: {self.total_prompt_tokens:,} &nbsp;/&nbsp; Completion: {self.total_completion_tokens:,}</div>
  </div>
  <div class="card time">
    <div class="card-label">Run Duration</div>
    <div class="card-value">{elapsed:.1f}s</div>
    <div class="ref">{self.call_count} LLM call(s) · All on Ollama</div>
  </div>
</div>

<hr class="divider">

<p class="section-title">Token Usage per Step</p>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#6366f1cc;"></div>Prompt tokens</div>
  <div class="legend-item"><div class="legend-dot" style="background:#6366f166;"></div>Completion tokens</div>
</div>
{step_bars}

<div class="footer">Generated by HERA · Local-first multi-agent system</div>
</body>
</html>"""

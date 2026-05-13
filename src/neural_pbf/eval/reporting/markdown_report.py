"""Markdown report builder for evaluation results."""

from __future__ import annotations

from pathlib import Path

from neural_pbf.eval.rollout.autoregressive import ar_summary
from neural_pbf.eval.rollout.engine import RolloutResult


def build_markdown_report(
    result: RolloutResult,
    figure_paths: list[Path] | None = None,
) -> str:
    """Render a RolloutResult as a Markdown document.

    Sections: header, summary table, figures (optional), per-step metric table.
    """
    summary = ar_summary(result)
    n_eval = len(result.per_step_metrics)
    div_note = (
        f"Yes (step {result.diverged_at_step})"
        if result.diverged_at_step is not None
        else "No"
    )
    mean_latency_ms = (
        1000.0 * sum(result.latencies_s) / n_eval if n_eval > 0 else float("nan")
    )

    lines: list[str] = [
        f"# Evaluation Report — {result.stepper_name}",
        "",
        f"**Mode**: {result.mode}  ",
        f"**Steps evaluated**: {n_eval}  ",
        f"**Diverged**: {div_note}  ",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean MAE [K] | {summary.get('mean_mae', float('nan')):.4f} |",
        f"| Final MAE [K] | {summary.get('final_mae', float('nan')):.4f} |",
        f"| Peak MAE [K] | {summary.get('peak_mae', float('nan')):.4f} |",
        f"| Peak VRAM [MB] | {result.vram_peak_bytes / 1e6:.1f} |",
        f"| Mean latency [ms] | {mean_latency_ms:.2f} |",
        "",
    ]

    if figure_paths:
        lines += ["## Figures", ""]
        for fp in figure_paths:
            lines.append(f"![{fp.stem}]({fp.name})")
        lines.append("")

    if result.per_step_metrics:
        lines += [
            "## Per-Step Metrics",
            "",
            "| Step | MAE Global [K] | Max Error [K] | Melt IoU |",
            "|---|---|---|---|",
        ]
        for i, m in enumerate(result.per_step_metrics):
            val_str = (
                f"| {i} | {m['mae_global']:.4f} | "
                f"{m['max_error']:.4f} | {m['iou_melt']:.4f} |"
            )
            lines.append(val_str)

    return "\n".join(lines)


def save_markdown_report(
    result: RolloutResult,
    output_dir: Path | str,
    figure_paths: list[Path] | None = None,
) -> Path:
    """Write the report to ``<output_dir>/report.md`` and return its path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = build_markdown_report(result, figure_paths)
    report_path = out / "report.md"
    report_path.write_text(text, encoding="utf-8")
    return report_path

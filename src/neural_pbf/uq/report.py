"""Generates docs/uq_report.md. Every number comes from the saved draws at report time."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_pbf.uq.analysis import QUANTITIES, evaluate, point_errors, second_seed_table, sensitivity_table
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.guard import atomic_write_text

# Fixed by method, validated with the dataviz palette checker; markers add a non-color cue.
_STYLE = {
    "absolute": ("#2a78d6", "o"),
    "normalized": ("#eb6834", "s"),
    "constant_single_draw": ("#1baf7a", "^"),
    "raw_ensemble": ("#6250d6", "D"),
}
_INK, _MUTED = "#0b0b0b", "#52514e"


def _fmt(value: Any) -> str:
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    if isinstance(value, float | np.floating):
        if np.isinf(value):
            return "inf"
        return f"{value:.3f}" if abs(value) < 100 else f"{value:.0f}"
    return str(value)


def _table(df: Any) -> str:
    if df.empty:
        return "_No samples in this group._\n"
    lines = ["| " + " | ".join(df.columns) + " |", "|" + " --- |" * len(df.columns)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _coverage_cols(df: Any) -> Any:
    return df.assign(
        Wilson_95=[f"{a:.3f} to {b:.3f}" for a, b in zip(df.binom_lo, df.binom_hi, strict=True)],
        run_bootstrap_95=[f"{a:.3f} to {b:.3f}" for a, b in zip(df.boot_lo, df.boot_hi, strict=True)],
    )


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=_INK, loc="left")
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=_MUTED, labelsize=8)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(fig)


def _fig_nominal(df: Any, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, quantity in zip(axes, QUANTITIES, strict=True):
        sub = df[(df.stratum_type == "all") & (df.quantity == quantity)]
        ax.plot([0.7, 1], [0.7, 1], color=_MUTED, lw=1, ls="--", label="nominal")
        for method, (color, marker) in _STYLE.items():
            m = sub[sub.method == method].sort_values("alpha", ascending=False)
            ax.plot(1 - m.alpha, m.coverage, color=color, marker=marker, ms=6, lw=2, label=method)
        _style(ax, quantity, "nominal coverage", "empirical coverage")
    axes[-1].legend(fontsize=7, frameon=False)
    _save(fig, path)


def _fig_width_error(test: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for i, (ax, quantity) in enumerate(zip(axes, QUANTITIES, strict=True)):
        spread = test["draws"][:, :, i].std(axis=1, ddof=1)
        error = np.abs(test["draws"][:, :, i].mean(axis=1) - test["truth"][:, i])
        ax.scatter(spread, error, s=9, alpha=0.5, color=_STYLE["normalized"][0], linewidths=0)
        _style(ax, quantity, "ensemble spread (sigma)", "absolute error of ensemble mean")
    _save(fig, path)


def _fig_material(df: Any, primary: str, alpha: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, quantity in zip(axes, QUANTITIES, strict=True):
        sub = df[(df.stratum_type == "material") & (df.quantity == quantity)]
        sub = sub[(sub.alpha == alpha) & (sub.method == primary)]
        x = np.arange(len(sub))
        ax.errorbar(
            x, sub.coverage,
            yerr=[sub.coverage - sub.boot_lo, sub.boot_hi - sub.coverage],
            fmt="o", ms=7, color=_STYLE[primary][0], capsize=3, lw=1.5,
        )  # fmt: skip
        ax.axhline(1 - alpha, color=_MUTED, lw=1, ls="--")
        ax.set_xticks(x, sub.stratum)
        ax.set_ylim(0, 1.05)
        _style(ax, quantity, "", f"coverage (dashed: {1 - alpha:.2f})")
    _save(fig, path)


def _model_section(result: dict[str, Any], cfg: UQConfig, fig_dir: Path, report_dir: Path) -> tuple[str, pd.DataFrame]:
    prereg = cfg.prereg
    df: Any = evaluate(result["cal"], result["test"], prereg, allow_shared_runs=result["role"] == "leakage_demo")
    slug = "".join(c if c.isalnum() else "_" for c in result["name"]).lower()
    y = result["test"]["truth"]
    head = [
        f"### {result['name']}\n",
        f"**{result['label']}**\n",
        f"n_cal = {y_n(result['cal'])} ({n_runs(result['cal'])} runs), "
        f"n_test = {y_n(result['test'])} ({n_runs(result['test'])} runs), K = {result['cal']['draws'].shape[1]}, "
        f"sampling seed = {prereg.sampling_seed}.\n",
        "Point error of the ensemble mean on the test runs:\n",
        _table(point_errors(result["test"])),
        "Coverage over all test samples:\n",
    ]
    overall = _coverage_cols(df[df.stratum_type == "all"]).assign(
        support=lambda d: np.where(d.run_level_supported, "supported", "not supported at run level")
    )
    head.append(_table(overall[["quantity", "alpha", "method", "n", "coverage", "Wilson_95", "run_bootstrap_95",
                                "mean_width", "median_width", "interval_score", "support"]]))  # fmt: skip

    primary: Any = df[(df.method == prereg.primary_method) & (df.alpha == prereg.primary_alpha)]
    columns = ["quantity", "stratum", "n", "n_runs", "coverage", "Wilson_95", "run_bootstrap_95", "mean_width"]
    for title, kind in (("material", "material"), ("pattern", "pattern")):
        head += [f"\nCoverage per {title} ({prereg.primary_method}, alpha = {prereg.primary_alpha}):\n",
                 _table(_coverage_cols(primary[primary.stratum_type == kind])[columns])]  # fmt: skip

    runs = primary[primary.stratum_type == "run"].groupby("quantity")["coverage"]
    spread = runs.agg(["min", "median", "max", "count"]).reset_index()
    head += ["\nPer-run coverage spread (each test run counted once):\n", _table(spread)]

    depth = y[:, QUANTITIES.index("depth_vox")]
    nz = result["test"]["nz"]
    head += [
        f"\nDepth: zero-depth samples are {(depth == 0).mean():.3f} of the test set; domain-limited samples "
        f"(depth reaches Nz) are {(depth >= nz).mean():.3f}. A zero-depth sample is an easy hit and a "
        "domain-limited one is censored, so read these strata separately:\n",
        _table(_coverage_cols(primary[primary.stratum_type == "depth_stratum"])[columns]),
        "\nT_max coverage by true temperature range:\n",
        _table(_coverage_cols(primary[primary.stratum_type == "T_max_bin"])[columns]),
    ]

    nominal, width, material = (fig_dir / f"{slug}_{n}.png" for n in ("nominal", "width", "material"))
    _fig_nominal(df, nominal)
    _fig_width_error(result["test"], width)
    _fig_material(df, prereg.primary_method, prereg.primary_alpha, material)
    for image, alt in ((nominal, "Nominal against empirical coverage"), (width, "Ensemble spread against error"),
                       (material, "Coverage per material")):  # fmt: skip
        head.append(f"\n![{alt}]({os.path.relpath(image, report_dir)})\n")

    if result["role"] == "primary":
        head += ["\n#### Sensitivity (same frozen calibration data, no new sampling)\n",
                 "Coverage and width against K (first K draws):\n",
                 _table(sensitivity_table(result["cal"], result["test"], prereg))]  # fmt: skip
        if result.get("second") is not None:
            head += [f"\nSecond base seed ({prereg.second_sampling_seed}) on a subset of the test inputs:\n",
                     _table(second_seed_table(result["cal"], result["test"], result["second"], prereg))]  # fmt: skip
    return "\n".join(head), df


def y_n(part: dict[str, np.ndarray]) -> int:
    return int(part["truth"].shape[0])


def n_runs(part: dict[str, np.ndarray]) -> int:
    return int(np.unique(part["run_id"]).size)


def build_report(
    results: list[dict[str, Any]], meta: dict[str, Any], cfg: UQConfig, out_path: str | Path, fig_dir: str | Path
) -> pd.DataFrame:
    """Write the report; returns all coverage rows for MLflow logging."""
    out_path, fig_dir = Path(out_path), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prereg = cfg.prereg
    n_eval = sum(e["event"] == "evaluation" for e in meta["access_events"])
    sections, frames = [], []
    for result in results:
        text, df = _model_section(result, cfg, fig_dir, out_path.parent)
        sections.append((result["role"], text))
        frames.append(df.assign(model=result["name"], label=result["label"]))

    primary_df = frames[0]
    canonical = {"coverage_nominal_vs_empirical": lambda p: _fig_nominal(primary_df, p),
                 "width_vs_error": lambda p: _fig_width_error(results[0]["test"], p),
                 "coverage_per_material": lambda p: _fig_material(
                     primary_df, prereg.primary_method, prereg.primary_alpha, p)}  # fmt: skip
    for name, draw in canonical.items():
        draw(fig_dir / f"{name}.png")

    smoke = "\n> **SMOKE RUN.** Tiny grid and few epochs. These numbers only prove the chain runs.\n" if (
        cfg.mode == "smoke") else ""  # fmt: skip
    parts = [
        f"# Uncertainty quantification report\n{smoke}",
        "Generated by the UQ pipeline. Do not edit numbers by hand.\n",
        "## Setup\n",
        f"- Split type: {meta['split_type']}, manifest hash `{meta['manifest_hash']}`.",
        f"- Runs per set: {meta['runs_per_set']}.",
        f"- Pre-registered config hash `{meta['prereg_hash']}`: K = {prereg.K}, alphas = {list(prereg.alphas)}, "
        f"primary method = {prereg.primary_method}, primary alpha = {prereg.primary_alpha}, seeds = "
        f"{prereg.sampling_seed}, {prereg.second_sampling_seed}, {prereg.bootstrap_seed}.",
        f"- Training: {meta['training']}.",
        "- Confidence intervals: Wilson 95% (assumes independent samples, too narrow for within-run data) and "
        f"run-level percentile bootstrap 95% ({prereg.n_boot} resamples, seed {prereg.bootstrap_seed}). "
        "Prefer the bootstrap.",
        "- Methods: `absolute` (conformal on residuals of the ensemble mean), `normalized` (residual / (spread + "
        "eps)), `constant_single_draw` (constant-width baseline: conformal on one draw, no ensemble), and "
        "`raw_ensemble` (uncalibrated mean +- z sigma, reference only).",
        f"- Test-set access: evaluated {n_eval} times"
        + (" (re-evaluation forced with --allow-reeval; treat results as no longer pristine)." if n_eval > 1 else ".")
        + "\n",
        "Conformal coverage assumes exchangeable samples. Samples within a run are correlated, so the guarantee "
        "holds only loosely here. An alpha marked `not supported at run level` lacks enough calibration runs for a "
        "finite quantile when each run counts once. Its sample-level number is shown but carries no run-level "
        "guarantee.\n",
        "Models are never ranked across different run-disjoint labels.\n",
        "## Results\n",
        *[text for role, text in sections if role == "primary"],
    ]
    demos = [text for role, text in sections if role == "leakage_demo"]
    if demos:
        parts += ["\n## Leakage demonstration\n",
                  "The legacy Baseline (post-step mask, 3 training runs) under the same pipeline, labelled by actual "
                  "overlap with its training data from `dataset_cfg`. The held-out n is an illustration, not a "
                  "measurement.\n", *demos]  # fmt: skip
    parts += [
        "\n## Known limitations\n",
        "- Results are on simulated data only, with no real-world validation cells.",
        "- There is a small number of runs, so run-level intervals are wide.",
        "- All legacy results use the post-step mask, which is derived from the predicted step. The new Baseline "
        "uses the causal `mask_in`.",
        "- Melt-pool functionals use each sample's own T_l. Depth is capped by the domain height.",
        "- Temperatures are stored as float16, which limits T_max resolution at high temperatures.",
        "- The model does not see T_in, so thermal history reaches it only through mask_in and conditioning.",
    ]
    atomic_write_text(out_path, "\n".join(parts) + "\n")
    return pd.concat(frames, ignore_index=True)

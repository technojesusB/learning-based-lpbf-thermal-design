"""Generates docs/uq_report.md. Every number comes from the saved draws at report time."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neural_pbf.eval.viz import uq as viz
from neural_pbf.uq.analysis import QUANTITIES, point_errors, second_seed_table
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.guard import atomic_write_text

REGIMES_FILE = "regime_rows.csv"
_NOTICE_START, _NOTICE_END = "<!-- missing-figures -->", "<!-- /missing-figures -->"


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


def _overall_block(df: Any) -> list[str]:
    overall = _coverage_cols(df[df.stratum_type == "all"]).assign(
        support=lambda d: np.where(d.run_level_supported, "supported", "not supported at run level")
    )
    columns = ["quantity", "alpha", "method", "n", "coverage", "Wilson_95", "run_bootstrap_95", "mean_width",
               "median_width", "interval_score", "support"]  # fmt: skip
    return ["Coverage over all test samples:\n", _table(overall[columns])]


def _strata_block(primary: Any, test: dict[str, np.ndarray], label: str) -> list[str]:
    columns = ["quantity", "stratum", "n", "n_runs", "coverage", "Wilson_95", "run_bootstrap_95", "mean_width"]

    def stratum_table(kind: str) -> str:
        return _table(_coverage_cols(primary[primary.stratum_type == kind])[columns])

    runs = primary[primary.stratum_type == "run"].groupby("quantity")["coverage"]
    spread = runs.agg(["min", "median", "max", "count"]).reset_index()
    depth = test["truth"][:, QUANTITIES.index("depth_vox")]
    volume = test["truth"][:, QUANTITIES.index("volume_vox")]
    return [
        f"\nCoverage per material ({label}):\n", stratum_table("material"),
        f"\nCoverage per pattern ({label}):\n", stratum_table("pattern"),
        "\nPer-run coverage spread (each test run counted once):\n", _table(spread),
        f"\nDepth: zero-depth samples are {(depth == 0).mean():.3f} of the test set; domain-limited samples "
        f"(depth reaches Nz) are {(depth >= test['nz']).mean():.3f}. A zero-depth sample is an easy hit and a "
        "domain-limited one is censored, so read these strata separately:\n",
        stratum_table("depth_stratum"),
        f"\nVolume: zero-volume samples are {(volume == 0).mean():.3f} of the test set; patch-limited samples "
        f"(melt touches a lateral patch face or the bottom) are {test['volume_limited'].mean():.3f}. Their true "
        "volume is larger than the patch can show, so read these strata separately:\n",
        stratum_table("volume_stratum"),
        "\nT_max coverage by true temperature range:\n", stratum_table("T_max_bin"),
    ]  # fmt: skip


def _image(path: Path, alt: str, report_dir: Path) -> str:
    return f"\n![{alt}]({os.path.relpath(path, report_dir)})\n"


def _figure_block(name: str, fig_dir: Path, report_dir: Path) -> list[str]:
    alts = {"nominal": "Nominal against empirical coverage", "spread": "Ensemble spread against error",
            "material": "Coverage per material"}  # fmt: skip
    return [_image(fig_dir / viz.model_figure_name(name, kind), alts[kind], report_dir) for kind in viz.MODEL_FIGURES]


def _sensitivity_block(
    result: dict[str, Any], prereg: Any, sensitivity: pd.DataFrame, fig_dir: Path, report_dir: Path
) -> list[str]:
    block = ["\n#### Sensitivity (same frozen calibration data, no new sampling)\n",
             "Coverage and width against K (first K draws):\n",
             _table(sensitivity),
             _image(fig_dir / viz.SENSITIVITY_FIG_FILE, "Coverage and width against K", report_dir)]  # fmt: skip
    if result.get("second") is not None:
        block += [f"\nSecond base seed ({prereg.second_sampling_seed}) on a subset of the test inputs:\n",
                  _table(second_seed_table(result["cal"], result["test"], result["second"], prereg))]  # fmt: skip
    return block


def _regime_block(model: str, tables: dict[str, pd.DataFrame], cfg: UQConfig) -> list[str]:
    """Exploratory coverage by thermal regime for one model, at the primary alpha and method."""
    from neural_pbf.uq.regimes import criterion

    regimes: Any = tables.get("regimes")
    if regimes is None or regimes.empty:
        return []
    p = cfg.prereg
    df = regimes[(regimes.model == model) & (regimes.alpha == p.primary_alpha) & (regimes.method == p.primary_method)]
    columns = ["quantity", "regime", "n_cal", "n_cal_runs", "n", "coverage", "run_bootstrap_95", "mean_width"]

    def view(variant: str) -> str:
        part = df[df.variant == variant].copy()
        part["run_bootstrap_95"] = [
            "n/a" if np.isnan(a) else f"{a:.3f} to {b:.3f}" for a, b in zip(part.boot_lo, part.boot_hi, strict=True)
        ]
        return _table(part[columns])

    lines = [
        f"\n#### {model}: coverage by thermal regime (exploratory)\n",
        f"{p.primary_method}, alpha = {p.primary_alpha}. Each regime has its own calibration (Mondrian conformal).\n",
        "Regimes from the **predicted** T_max (usable at prediction time):\n",
        view("predicted"),
        "\nRegimes from the **true** T_max (diagnostic only; unknown at prediction time):\n",
        view("true"),
    ]
    if model == _primary_name(tables):
        verdict = criterion(df.to_dict("records"), p, cfg.exploratory, model)
        e = cfg.exploratory
        checks = "; ".join(
            f"{q}: " + (f"coverage {c['coverage']:.3f}, mean width {c['mean_width']:.1f} (limit {c['max_width']:g})"
                        if "coverage" in c else c["reason"])
            for q, c in verdict["checks"].items()
        )  # fmt: skip
        lines.append(
            f"\n**Usefulness criterion (fixed before run 2): {'met' if verdict['passed'] else 'not met'}.** "
            f"Melt regime, predicted regimes, coverage at least {e.criterion_min_coverage:g} and width below the "
            f"limit for every listed quantity. {checks}.\n"
        )
    return lines


def _primary_name(tables: dict[str, pd.DataFrame]) -> str:
    coverage: Any = tables["coverage"]
    return str(coverage[coverage.role == "primary"].model.iloc[0])


def _model_section(
    result: dict[str, Any], cfg: UQConfig, fig_dir: Path, report_dir: Path, tables: dict[str, pd.DataFrame]
) -> str:
    prereg = cfg.prereg
    coverage: Any = tables["coverage"]
    df: Any = coverage[coverage.model == result["name"]]
    primary = df[(df.method == prereg.primary_method) & (df.alpha == prereg.primary_alpha)]
    label = f"{prereg.primary_method}, alpha = {prereg.primary_alpha}"
    parts = [
        f"### {result['name']}\n",
        f"**{result['label']}**\n",
        f"n_cal = {y_n(result['cal'])} ({n_runs(result['cal'])} runs), "
        f"n_test = {y_n(result['test'])} ({n_runs(result['test'])} runs), K = {result['cal']['draws'].shape[1]}, "
        f"base sampling seed = {prereg.sampling_seed}.\n",
        "Point error of the ensemble mean on the test runs:\n",
        _table(point_errors(result["test"])),
        *_overall_block(df),
        *_strata_block(primary, result["test"], label),
        *_figure_block(result["name"], fig_dir, report_dir),
        *_regime_block(result["name"], tables, cfg),
    ]
    if result["role"] == "primary":
        parts.append(_image(fig_dir / viz.RUN_SPREAD_FILE, "Coverage per run", report_dir))
        parts += _sensitivity_block(result, prereg, tables["sensitivity"], fig_dir, report_dir)
    return "\n".join(parts)


def y_n(part: dict[str, np.ndarray]) -> int:
    return int(part["truth"].shape[0])


def n_runs(part: dict[str, np.ndarray]) -> int:
    return int(np.unique(part["run_id"]).size)


def _setup_lines(meta: dict[str, Any], prereg: Any) -> list[str]:
    n_eval = sum(e["event"] == "evaluation" for e in meta["access_events"])
    forced = " (re-evaluation forced with --allow-reeval; treat results as no longer pristine)." if n_eval > 1 else "."
    return [
        "## Setup\n",
        f"- Split type: {meta['split_type']}, manifest hash `{meta['manifest_hash']}`.",
        f"- Runs per set: {meta['runs_per_set']}.",
        f"- Pre-registered config hash `{meta['prereg_hash']}`: K = {prereg.K}, alphas = {list(prereg.alphas)}, "
        f"primary method = {prereg.primary_method}, primary alpha = {prereg.primary_alpha}, seeds = "
        f"{prereg.sampling_seed}, {prereg.second_sampling_seed}, {prereg.bootstrap_seed}.",
        "- Seed derivation: `SeedSequence([base seed, crc32(set name)])` per set, then "
        "`SeedSequence([set seed, input index])` per input.",
        f"- Training: {meta['training']}.",
        "- Confidence intervals: Wilson 95% (assumes independent samples, too narrow for within-run data) and "
        f"run-level percentile bootstrap 95% ({prereg.n_boot} resamples, seed {prereg.bootstrap_seed}). "
        "Prefer the bootstrap.",
        "- Methods: `absolute` (conformal on residuals of the ensemble mean), `normalized` (residual / (spread + "
        "eps)), `constant_single_draw` (constant-width baseline: conformal on one draw, no ensemble), and "
        "`raw_ensemble` (uncalibrated mean +- z sigma, reference only).",
        f"- Test-set access: evaluated {n_eval} time{'s' if n_eval != 1 else ''}{forced}",
        "- Deviations from the pre-registration, made after the first test evaluation: the primary model now "
        "also gets T_in (the temperature before the step) as an input channel and exposure_time as conditioning, "
        "because run 1 lacked both (see the appendix); the coverage-by-regime analysis and its usefulness "
        "criterion are exploratory. Methods, alphas, K, seeds, manifest, and calibration and test runs are "
        "unchanged.\n",
        "Conformal coverage assumes exchangeable samples. Samples within a run are correlated, so the guarantee "
        "holds only loosely here. An alpha marked `not supported at run level` lacks enough calibration runs for a "
        "finite quantile when each run counts once. Its sample-level number is shown but carries no run-level "
        "guarantee.\n",
        "Models are never ranked across different run-disjoint labels.\n",
    ]


_LIMITATIONS = [
    "\n## Known limitations\n",
    "- Results are on simulated data only, with no real-world validation cells.",
    "- There is a small number of runs, so run-level intervals are wide.",
    "- All legacy results use the post-step mask, which is derived from the predicted step. The new Baseline "
    "uses the causal `mask_in`.",
    "- Melt-pool functionals use each sample's own T_l. Depth is capped by the domain height.",
    "- Melt-pool volume is capped by the patch: a pool wider than the 64-voxel patch is truncated, so large "
    "volumes are lower bounds. The patch-limited stratum shows how often this happens.",
    "- Temperatures are stored as float16, which limits T_max resolution at high temperatures.",
    "- The solver has no evaporation, recoil pressure, or keyhole physics. Above the boiling point its temperatures "
    "are unphysical (up to about 17 000 K); the keyhole regime marks those samples as outside its validity.",
    "- Boiling points of alloys are model values from LPBF simulation literature, not sharp measured points.",
    "- The solver grid is 2 um; grid convergence against 1 um is not checked, so intervals hold against this "
    "solver, not against the physics.",
    "- Material data share one absorptivity (0.4) for every alloy; the Ti64 cp table dips between 300 and 373 K.",
    "- No physics loss: the available PDE residual does not yet match the solver (fixed dt, scalar k).",
]


def save_tables(tables: dict[str, pd.DataFrame], directory: str | Path) -> None:
    """Save everything the figures need; these files are their only input."""
    directory = Path(directory)
    names = {"coverage": viz.COVERAGE_FILE, "sensitivity": viz.SENSITIVITY_FILE, "errors": viz.ERRORS_FILE,
             "points": viz.POINTS_FILE}  # fmt: skip
    if "regimes" in tables:
        names["regimes"] = REGIMES_FILE
    for key, name in names.items():
        atomic_write_text(directory / name, tables[key].to_csv(index=False))


def update_missing_figures(path: str | Path, failures: dict[str, str], command: str) -> None:
    """Write, replace, or clear the missing-figures notice under the first line of a Markdown file."""
    path = Path(path)
    pattern = re.escape(_NOTICE_START) + r".*?" + re.escape(_NOTICE_END) + r"\n"
    text = re.sub(pattern, "", path.read_text(), flags=re.DOTALL)
    if failures:
        items = "\n".join(f"> - `{name}`: {(error.splitlines() or [''])[0]}" for name, error in failures.items())
        notice = (
            f"{_NOTICE_START}\n> **MISSING FIGURES.** These figures failed to render. The numbers in this "
            f"document are not affected. Run `{command}` to render them.\n>\n{items}\n{_NOTICE_END}\n"
        )
        first, newline, rest = text.partition("\n")
        text = f"{first}{newline}{notice}{rest}"
    atomic_write_text(path, text)


def build_report(
    results: list[dict[str, Any]],
    meta: dict[str, Any],
    cfg: UQConfig,
    out_path: str | Path,
    fig_dir: str | Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    """Write the report from computed tables. Figures are rendered separately from the saved tables."""
    out_path, fig_dir = Path(out_path), Path(fig_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = out_path.parent
    sections = [(r["role"], _model_section(r, cfg, fig_dir, report_dir, tables)) for r in results]

    smoke = "\n> **SMOKE RUN.** Tiny grid and few epochs. These numbers only prove the chain runs.\n"
    parts = [
        f"# Uncertainty quantification report\n{smoke if cfg.mode == 'smoke' else ''}",
        "Generated by the UQ pipeline. Do not edit numbers by hand.\n",
        *_setup_lines(meta, cfg.prereg),
        "## Results\n",
        *[text for role, text in sections if role == "primary"],
    ]
    demos = [text for role, text in sections if role == "leakage_demo"]
    if demos:
        parts += ["\n## Leakage demonstration\n",
                  "The legacy Baseline (post-step mask, 3 training runs) under the same pipeline, labelled by actual "
                  "overlap with its training data from `dataset_cfg`. The held-out n is an illustration, not a "
                  "measurement.\n", _image(fig_dir / viz.LEAKAGE_FILE, "Coverage and error per model", report_dir),
                  *demos]  # fmt: skip
    appendix = [text for role, text in sections if role == "appendix"]
    if appendix:
        parts += ["\n## Appendix: run 1\n",
                  "The first run trained the same Baseline without T_in and without exposure_time, so the model saw "
                  "neither the heat left by earlier tracks nor the deposited energy. It is kept as evidence of what "
                  "those inputs change; it is not a candidate model. Same manifest, same calibration and test runs.\n",
                  *appendix]  # fmt: skip
    atomic_write_text(out_path, "\n".join([*parts, *_LIMITATIONS]) + "\n")

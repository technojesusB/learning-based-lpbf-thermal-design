"""Whole-chain pause tests on the real smoke pipeline. Needs a GPU and a clean tree; takes minutes.

Deselected by default; run with ``pytest -m slow``. A dirty tree fails instead of skipping, so a run
that was asked for cannot look green without having happened.

GPU training is not bitwise reproducible (docs/uq.md), so a continued chain is compared with a normal
one on S0 to S3, the data, and everything of the legacy model, and on table structure for the new model.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from neural_pbf.uq.stages import STAGE_IDS, dirty_code_paths

SCRIPT = "scripts/run_uq_pipeline.sh"
LEGACY = "Legacy"
TABLES = ("coverage_rows.csv", "point_errors.csv", "sample_points.csv")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


@pytest.fixture(scope="module")
def clean_tree():
    tree = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if tree or dirty_code_paths():
        pytest.fail("S0 refuses a dirty tree; commit first")


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "UQ_SMOKE_ROOT": str(root)}
    return subprocess.run([SCRIPT, "--smoke", *args], env=env, capture_output=True, text=True, timeout=900)


def _status(root: Path) -> dict:
    return json.loads((root / "status.json").read_text())


def _hashes(root: Path, *patterns: str) -> dict[str, str]:
    files = sorted(p for pattern in patterns for p in root.glob(pattern))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def _marker(root: Path, stage: str) -> dict:
    return json.loads((root / "markers" / f"{stage}.done.json").read_text())


def _table(root: Path, name: str) -> list[dict[str, str]]:
    with (root / name).open() as f:
        return list(csv.DictReader(f))


def _assert_equal_where_reproducible(a: Path, b: Path) -> None:
    patterns = ("data/run_*.h5", "manifest.json", "prereg.json", "draws/legacy_baseline/*/chunk_*.npz")
    expected = _hashes(a, *patterns)
    assert expected and _hashes(b, *patterns) == expected
    for stage in STAGE_IDS[:4]:
        assert _marker(a, stage)["input_hash"] == _marker(b, stage)["input_hash"]
        assert _marker(a, stage)["output_hash"] == _marker(b, stage)["output_hash"]
    new_model = "draws/baseline_pre/*/chunk_*.npz"
    assert _hashes(a, new_model).keys() == _hashes(b, new_model).keys()
    for name in TABLES:
        rows_a, rows_b = _table(a, name), _table(b, name)
        assert len(rows_a) == len(rows_b) and list(rows_a[0]) == list(rows_b[0])
        legacy = [(x, y) for x, y in zip(rows_a, rows_b, strict=True) if x["model"].startswith(LEGACY)]
        assert legacy and all(x == y for x, y in legacy), f"{name}: legacy numbers differ"
    labels = ("quantity", "alpha", "method", "model")
    rows = [[[r[k] for k in labels] for r in _table(root, "coverage_rows.csv")] for root in (a, b)]
    assert rows[0] == rows[1]


@pytest.fixture(scope="module")
def normal_chain(clean_tree, tmp_path_factory):
    """Every chain runs at the same path, because the manifest and the draws store absolute paths."""
    base = tmp_path_factory.mktemp("chains")
    result = _run(base / "run")
    assert result.returncode == 0, result.stdout + result.stderr
    return (base / "run").rename(base / "normal")


@pytest.fixture
def root(normal_chain):
    path = normal_chain.parent / "run"
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_stop_after_s2_then_continue_matches_a_normal_chain(normal_chain, root):
    paused = _run(root, "--stop-after-stage", "S2")
    assert paused.returncode == 0, paused.stdout + paused.stderr
    status = _status(root)
    assert status["state"] == "paused: --stop-after-stage S2" and status["next_stage"] == "S3"
    assert all((root / "markers" / f"{s}.done.json").exists() for s in ("S0", "S1", "S2"))
    assert not (root / "markers" / "S3.done.json").exists() and not (root / "ckpt").exists()
    assert f"{SCRIPT} --smoke" in paused.stdout

    resumed = _run(root, "--from-stage", "S3", "--stop-after-stage", "S7")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _status(root)["state"] == "finished"
    assert [a["stage"] for a in _status(root)["attempts"]] == list(STAGE_IDS), "no stage may run twice"
    _assert_equal_where_reproducible(normal_chain, root)


def test_stop_at_time_with_a_short_offset_pauses_and_resumes(normal_chain, root):
    soon = datetime.datetime.now() + datetime.timedelta(seconds=30)
    paused = _run(root, "--stop-at-time", soon.isoformat(timespec="seconds"))
    assert paused.returncode == 0, paused.stdout + paused.stderr
    status = _status(root)
    assert status["state"].startswith("paused: --stop-at-time") and status["next_stage"] in STAGE_IDS
    assert not (root / "markers" / "S7.done.json").exists()
    assert not list(root.rglob("*.tmp")) and not list(root.rglob("*.partial.h5"))

    resumed = _run(root)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _status(root)["state"] == "finished"
    _assert_equal_where_reproducible(normal_chain, root)

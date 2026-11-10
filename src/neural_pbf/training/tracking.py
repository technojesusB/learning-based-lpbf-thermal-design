"""Shared MLflow tracking of every training run, so every script logs the same keys.

A standalone script run goes to ``lpbf_experiments`` with ``pinned=false`` and the script name as run name. The
ablation layer (``training/ablation.py``, which nothing here imports) passes ``RunHooks`` that pin the run: MLflow
experiment ``lpbf_ablation_rerun``, the experiment name as run name, and the record of the ablation experiment as
provenance key ``ablation_experiment``, params ``ablation.*`` and tags. ``step`` and ``epoch`` refuse a metrics dict
that misses a required key, so no script can log less than the others. ``provenance`` is the same record for the
checkpoint, so a checkpoint is traceable without MLflow.

A paused run ends as KILLED with the tag ``state=paused``; ``RunTracker.resume`` reopens the same MLflow run, but
only for the same name, arguments (infrastructure aside), manifest, data and, for pinned runs, git commit. Step
metrics a hard-killed run logged after its last saved epoch are logged again on resume, so MLflow shows two values at
those steps.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import mlflow

from neural_pbf.data.material_dataset import resolve_key
from neural_pbf.data.splits import load_manifest
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.stages import dirty_code_paths, git_commit

STANDALONE_EXPERIMENT = "lpbf_experiments"
ABLATION_EXPERIMENT = "lpbf_ablation_rerun"
DEBUG_EXPERIMENT = "lpbf_debug"
# Arguments that do not change what is trained: a resumed run may change them.
INFRA_ARGS = frozenset({"num_workers", "mlflow_uri", "checkpoint_dir", "uq_root", "log_every", "snapshot_every"})
REQUIRED_STEP = ("loss_fm", "loss_phys", "loss_total", "lambda_raw", "lambda_phys", "cap_hit", "skipped", "lr",
                 "grad_norm")  # fmt: skip
REQUIRED_EPOCH = (
    "train_loss_fm", "train_loss_phys", "train_loss_total", "val_loss_fm", "val_loss_phys", "val_loss_total",
    "val_phys_tin", "val_phys_tin_melt", "val_phys_coupled", "val_phys_coupled_melt",
    "epoch_time_s", "peak_gpu_mem_mb", "throughput_samples_s",
)  # fmt: skip
_MAX_PARAM_CHARS = 6000  # MLflow's limit per parameter value
_PARAMS_PER_CALL = 100  # MLflow's limit per log_params call


def split_runs(manifest_path: str | Path, train_ds: Any, val_ds: Any) -> dict[str, list[str]]:
    """The manifest run ids behind the train and val datasets; refuses any sample of another set.

    Every training run uses the manifest's train split and validates on its val split only; cal and test belong to the
    UQ pipeline.
    """
    manifest = load_manifest(manifest_path)
    where = {
        (p, k): (name, run["run_id"]) for name, runs in manifest["sets"].items() for run in runs
        for p, k in run["sample_keys"]
    }  # fmt: skip
    out = {}
    for name, ds in (("train", train_ds), ("val", val_ds)):
        seen = set()
        for i in range(len(ds)):
            key = resolve_key(ds, i)
            if key not in where:
                raise ValueError(f"the {name} data holds {key}, which is not in the manifest")
            found, run_id = where[key]
            if found != name:
                raise ValueError(f"the {name} data holds {key} of the {found} set")
            seen.add(run_id)
        out[f"{name}_runs"] = [r["run_id"] for r in manifest["sets"][name] if r["run_id"] in seen]
    return out


def _flatten(prefix: str, obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_flatten(f"{prefix}.{k}" if prefix else str(k), v))
        return out
    return {prefix: obj}


@dataclass(frozen=True)
class RunHooks:
    """How a layer above the scripts runs one: the default is a standalone, unpinned run.

    ``validate(args, fixed)`` raises ValueError when the parsed arguments, with the values the script fixes, are not
    the run the caller asked for. ``record`` is added to the provenance and as ``ablation.*`` params; its scalar
    entries become tags.
    """

    name: str | None = None  # MLflow run name and default checkpoint dir; None: the script name
    mlflow_experiment: str = STANDALONE_EXPERIMENT
    pinned: bool = False
    record: Mapping[str, Any] = field(default_factory=dict)
    validate: Callable[[argparse.Namespace, Mapping[str, Any]], None] | None = None

    def run_name(self, script: str | Path) -> str:
        return self.name or Path(script).stem


def _data_hash(uq_root: Path, pinned: bool) -> str:
    marker = uq_root / "markers" / "S1.done.json"
    if marker.exists():
        return str(json.loads(marker.read_text())["output_hash"])
    if pinned:
        raise ValueError(f"{marker} is missing: a pinned run needs the dataset hash from the UQ pipeline's S1")
    return "unknown"


class RunTracker:
    """One MLflow run of one training run. Use ``RunTracker.start(...)`` as a context manager on the main process."""

    def __init__(self, run_id: str, experiment: str, provenance: dict[str, Any]) -> None:
        self.run_id = run_id
        self.experiment = experiment
        self.provenance = provenance
        self._paused = False

    @staticmethod
    def _params(
        args: argparse.Namespace, hooks: RunHooks, configs: Mapping[str, Any], preflight: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {**_flatten("", vars(args)), **_flatten("ablation", dict(hooks.record)), **_flatten("", dict(configs)),
                **_flatten("", dict(preflight))}  # fmt: skip

    @classmethod
    def start(
        cls, args: argparse.Namespace, hooks: RunHooks, *, script: str, split: Mapping[str, list[str]],
        configs: Mapping[str, Any], preflight: Mapping[str, Any], uq_root: str | Path | None = None,
    ) -> RunTracker:  # fmt: skip
        """Open the run and log params, tags and the resolved config.

        ``configs``: the model and dataset configs (``fm_cfg``, ``dataset_cfg``, ...) and anything else that varies:
        normalization constants, conditioning keys, seeds.
        """
        pinned, name, experiment = hooks.pinned, hooks.run_name(script), hooks.mlflow_experiment
        root = Path(uq_root) if uq_root is not None else Path(UQConfig().root)
        manifest = load_manifest(args.manifest)
        dirty = dirty_code_paths()
        params = cls._params(args, hooks, configs, preflight)
        provenance: dict[str, Any] = {
            "pinned": pinned, "name": name, **dict(hooks.record), "script": script, "experiment": experiment,
            "git_commit": git_commit(), "git_dirty": bool(dirty), "git_dirty_paths": dirty,
            "manifest_path": str(args.manifest), "manifest_hash": manifest["hash"],
            "data_hash": _data_hash(root, pinned), "train_runs": list(split["train_runs"]),
            "val_runs": list(split["val_runs"]), "preflight": str(preflight.get("preflight", "")), "params": params,
        }  # fmt: skip

        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=name)
        provenance["mlflow_run_id"] = run.info.run_id
        items = [(k, str(v)[:_MAX_PARAM_CHARS]) for k, v in params.items()]
        for i in range(0, len(items), _PARAMS_PER_CALL):
            mlflow.log_params(dict(items[i : i + _PARAMS_PER_CALL]))
        tag_keys = ("script", "git_commit", "manifest_path", "manifest_hash", "data_hash", "preflight")
        mlflow.set_tags({
            **{k: str(provenance[k]) for k in tag_keys}, "state": "running",
            **{k: str(v) for k, v in hooks.record.items() if isinstance(v, str | int | float | bool)},
            "pinned": str(pinned).lower(), "git_dirty": str(bool(dirty)).lower(),
            "train_runs": json.dumps(provenance["train_runs"]), "val_runs": json.dumps(provenance["val_runs"]),
        })  # fmt: skip
        mlflow.log_dict({k: v for k, v in provenance.items()}, "resolved_config.json")
        return cls(run.info.run_id, experiment, provenance)

    @classmethod
    def resume(
        cls, args: argparse.Namespace, hooks: RunHooks, *, provenance: Mapping[str, Any], epoch: int, script: str,
        configs: Mapping[str, Any], preflight: Mapping[str, Any], uq_root: str | Path | None = None,
    ) -> RunTracker:  # fmt: skip
        """Reopen the MLflow run of ``provenance`` (from latest.pt) at ``epoch``; ValueError if the run changed."""
        pinned, name = hooks.pinned, hooks.run_name(script)
        root = Path(uq_root) if uq_root is not None else Path(UQConfig().root)
        prov = dict(provenance)
        if prov["pinned"] != pinned:
            raise ValueError(f"latest.pt is from a run with pinned={prov['pinned']}; this run has pinned={pinned}")
        if prov["name"] != name:
            raise ValueError(f"latest.pt belongs to the run {prov['name']!r}, not {name!r}")
        now = cls._params(args, hooks, configs, preflight)
        keys = [k for k in _flatten("", vars(args)) if k.split(".")[0] not in INFRA_ARGS]
        keys += [k for k in now if k.startswith("ablation.")]
        old_keys = [k for k in prov["params"] if k.startswith("ablation.")]
        changed = sorted({k for k in [*keys, *old_keys] if prov["params"].get(k) != now.get(k)})
        if changed:
            raise ValueError(f"refuse to resume: the run changed since latest.pt: {changed}")
        manifest_hash = load_manifest(args.manifest)["hash"]
        if prov["manifest_hash"] != manifest_hash:
            raise ValueError(f"refuse to resume: manifest hash {manifest_hash} differs from {prov['manifest_hash']}")
        data_hash = _data_hash(root, pinned)
        if prov["data_hash"] != data_hash:
            raise ValueError(f"refuse to resume: data hash {data_hash} differs from {prov['data_hash']}")
        commit = git_commit()
        if pinned and commit != prov["git_commit"]:
            raise ValueError(f"refuse to resume a pinned run at commit {commit}; it was trained at "
                             f"{prov['git_commit']} (run the queue from a worktree at that commit)")  # fmt: skip

        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(prov["experiment"])
        mlflow.start_run(run_id=prov["mlflow_run_id"])
        tags = {"resumed_at_epoch": str(epoch), "state": "running"}
        if commit != prov["git_commit"]:
            tags["resumed_git_commit"] = commit
        mlflow.set_tags(tags)
        return cls(prov["mlflow_run_id"], prov["experiment"], prov)

    def __enter__(self) -> RunTracker:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is not None:
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run(status="KILLED" if self._paused else "FINISHED")

    def pause(self) -> None:
        """End the run as paused (KILLED, ``state=paused``); ``resume`` continues it."""
        self._paused = True
        mlflow.set_tag("state", "paused")

    @staticmethod
    def _check(metrics: Mapping[str, float], required: tuple[str, ...], what: str) -> None:
        missing = [k for k in required if k not in metrics]
        if missing:
            raise ValueError(f"{what} metrics missing: {missing}")

    def step(self, metrics: Mapping[str, float], step: int) -> None:
        """Per-step metrics (every ``--log_every`` steps): losses, weights, cap, lr, grad norm."""
        self._check(metrics, REQUIRED_STEP, "step")
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def epoch(self, metrics: Mapping[str, float], epoch: int) -> None:
        """Per-epoch metrics: train and val loss per term, val physics residual, time, memory, throughput."""
        self._check(metrics, REQUIRED_EPOCH, "epoch")
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=epoch)

    def finish(self, best: str | Path, final: str | Path) -> None:
        mlflow.set_tags({"best_checkpoint": str(best), "final_checkpoint": str(final), "state": "finished"})

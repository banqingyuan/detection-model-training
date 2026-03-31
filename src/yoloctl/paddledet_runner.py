from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .config import RunConfig
from .datasets import DatasetVersionRecord
from .exceptions import YoloCtlError
from .paddledet_dataset import prepare_paddledet_dataset
from .paths import project_root


def _quote(command: list[str], env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    body = shlex.join(command)
    return f"{prefix} {body}".strip()


def _resolve_device(run_config: RunConfig) -> str | None:
    if run_config.paddledet is None:
        return None
    if run_config.paddledet.device:
        return run_config.paddledet.device
    if run_config.train.get("device") is not None:
        return str(run_config.train.get("device"))
    return None


def _validate_backend(run_config: RunConfig, engine: str) -> None:
    if not run_config.is_paddledet or run_config.paddledet is None:
        raise YoloCtlError("PaddleDet runner requires a run config with backend=paddledet")
    if engine != "auto":
        raise YoloCtlError("PaddleDet backend only supports the default subprocess engine. Omit --engine.")
    if not run_config.paddledet.root.exists():
        raise YoloCtlError(f"PaddleDetection root does not exist: {run_config.paddledet.root}")
    if not run_config.paddledet.config.exists():
        raise YoloCtlError(f"PaddleDetection config does not exist: {run_config.paddledet.config}")


def _build_env(device: str | None) -> dict[str, str]:
    if not device or device.lower() == "cpu":
        return {}
    return {"CUDA_VISIBLE_DEVICES": device}


def _override_args(overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in overrides.items():
        if value is None:
            continue
        args.append(f"{key}={value}")
    return args


def _distributed_train_prefix(device: str | None) -> list[str]:
    if not device or device.lower() == "cpu":
        return ["python"]
    if "," not in device:
        return ["python"]
    return ["python", "-m", "paddle.distributed.launch", "--gpus", device]


def _default_output_dir(run_config: RunConfig) -> str:
    return str((project_root() / run_config.default_project_dir() / run_config.default_run_name()).resolve())


def _paddledet_overrides(
    *,
    run_config: RunConfig,
    dataset_summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = {
        "TrainDataset.dataset_dir": dataset_summary["materialized_root"],
        "TrainDataset.image_dir": dataset_summary["train_image_dir"],
        "TrainDataset.anno_path": dataset_summary["train_annotation_path"],
    }
    if dataset_summary.get("val_annotation_path"):
        overrides.update(
            {
                "EvalDataset.dataset_dir": dataset_summary["materialized_root"],
                "EvalDataset.image_dir": dataset_summary["val_image_dir"],
                "EvalDataset.anno_path": dataset_summary["val_annotation_path"],
            }
        )
    if run_config.paddledet:
        overrides.update(run_config.paddledet.overrides)
    if extra:
        overrides.update(extra)
    return overrides


def run_paddledet_train(
    *,
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    dataset_yaml: Path,
    weights: str | None = None,
    engine: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    _validate_backend(run_config, engine)
    require_val = bool(run_config.paddledet and run_config.paddledet.eval)
    dataset_summary = prepare_paddledet_dataset(
        record=dataset_record,
        dataset_yaml=dataset_yaml,
        run_id=run_config.run_id,
        require_val=require_val,
        dry_run=dry_run,
    )
    extra_overrides: dict[str, Any] = {}
    pretrain_weights = weights or (run_config.paddledet.pretrain_weights if run_config.paddledet else None)
    if pretrain_weights:
        extra_overrides["pretrain_weights"] = pretrain_weights
    overrides = _paddledet_overrides(run_config=run_config, dataset_summary=dataset_summary, extra=extra_overrides)
    device = _resolve_device(run_config)
    env = _build_env(device)
    command = [
        *_distributed_train_prefix(device),
        "tools/train.py",
        "-c",
        str(run_config.paddledet.config),
        "--output_dir",
        _default_output_dir(run_config),
    ]
    if run_config.paddledet and run_config.paddledet.eval:
        command.append("--eval")
    override_args = _override_args(overrides)
    if override_args:
        command.extend(["-o", *override_args])
    quoted_command = _quote(command, env)
    if dry_run:
        return {
            "engine": "preview",
            "backend": "paddledet",
            "command": quoted_command,
            "working_dir": str(run_config.paddledet.root),
            "derived_dataset_dir": dataset_summary["derived_dataset_dir"],
            "config_overrides": overrides,
        }
    completed = subprocess.run(
        command,
        cwd=run_config.paddledet.root,
        env=None if not env else {**os.environ, **env},
        text=True,
        capture_output=True,
        check=True,
    )
    return {
        "engine": "subprocess",
        "backend": "paddledet",
        "command": quoted_command,
        "working_dir": str(run_config.paddledet.root),
        "derived_dataset_dir": dataset_summary["derived_dataset_dir"],
        "config_overrides": overrides,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_dir": _default_output_dir(run_config),
    }


def run_paddledet_validation(
    *,
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    dataset_yaml: Path,
    weights: str | None,
    engine: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    _validate_backend(run_config, engine)
    if not weights:
        weights = run_config.paddledet.eval_weights if run_config.paddledet else None
    if not weights:
        raise YoloCtlError(
            "PaddleDet validation requires --weights or paddledet.eval_weights in the run config."
        )
    dataset_summary = prepare_paddledet_dataset(
        record=dataset_record,
        dataset_yaml=dataset_yaml,
        run_id=run_config.run_id,
        require_val=True,
        dry_run=dry_run,
    )
    overrides = _paddledet_overrides(
        run_config=run_config,
        dataset_summary=dataset_summary,
        extra={"weights": weights},
    )
    device = _resolve_device(run_config)
    env = _build_env(device)
    command = ["python", "tools/eval.py", "-c", str(run_config.paddledet.config)]
    override_args = _override_args(overrides)
    if override_args:
        command.extend(["-o", *override_args])
    quoted_command = _quote(command, env)
    if dry_run:
        return {
            "engine": "preview",
            "backend": "paddledet",
            "command": quoted_command,
            "working_dir": str(run_config.paddledet.root),
            "derived_dataset_dir": dataset_summary["derived_dataset_dir"],
            "config_overrides": overrides,
            "weights": weights,
        }
    completed = subprocess.run(
        command,
        cwd=run_config.paddledet.root,
        env=None if not env else {**os.environ, **env},
        text=True,
        capture_output=True,
        check=True,
    )
    return {
        "engine": "subprocess",
        "backend": "paddledet",
        "command": quoted_command,
        "working_dir": str(run_config.paddledet.root),
        "derived_dataset_dir": dataset_summary["derived_dataset_dir"],
        "config_overrides": overrides,
        "weights": weights,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

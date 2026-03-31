from __future__ import annotations

import importlib
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import RunConfig
from .datasets import DatasetVersionRecord
from .exceptions import YoloCtlError
from .quantization import (
    archive_export_artifact,
    assessment_settings,
    build_artifact_metadata,
    prepare_calibration_dataset,
    resolve_export_source,
)


def _quote(command: list[str]) -> str:
    return shlex.join(command)


def _import_ultralytics() -> Any:
    try:
        return importlib.import_module("ultralytics")
    except ImportError as exc:
        raise YoloCtlError(
            "ultralytics is not installed. Install project dependencies inside the Ubuntu training environment."
        ) from exc


def resolve_engine(preferred: str = "auto") -> str:
    if preferred == "auto":
        try:
            _import_ultralytics()
            return "python"
        except YoloCtlError:
            if shutil.which("yolo"):
                return "cli"
            raise
    if preferred == "cli" and shutil.which("yolo") is None:
        raise YoloCtlError("Requested CLI engine but 'yolo' is not available in PATH")
    if preferred == "python":
        _import_ultralytics()
    return preferred


def _serialize_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _build_cli_key_values(payload: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in payload.items():
        if value is None:
            continue
        args.append(f"{key}={_serialize_cli_value(value)}")
    return args


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict())
        except Exception:
            return str(value)
    return str(value)


def train_kwargs(run_config: RunConfig, dataset_yaml: Path, resume: bool = False) -> dict[str, Any]:
    payload = dict(run_config.train)
    payload.setdefault("data", str(dataset_yaml))
    payload.setdefault("project", run_config.default_project_dir())
    payload.setdefault("name", run_config.default_run_name())
    payload.setdefault("pretrained", run_config.model.pretrained)
    payload.setdefault("amp", True)
    payload.setdefault("optimizer", "auto")
    if resume:
        payload["resume"] = True
    return payload


def val_kwargs(run_config: RunConfig, dataset_yaml: Path) -> dict[str, Any]:
    payload = {
        "data": str(dataset_yaml),
        "imgsz": run_config.train.get("imgsz"),
        "batch": run_config.train.get("batch"),
        "device": run_config.train.get("device"),
        "project": run_config.default_project_dir(),
        "name": f"{run_config.default_run_name()}-val",
    }
    return payload


def model_val_kwargs(
    dataset_yaml: Path,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    project: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "data": str(dataset_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
    }
    return payload


def export_jobs(run_config: RunConfig, dataset_yaml: Path) -> list[dict[str, Any]]:
    export = dict(run_config.export)
    formats = export.get("formats", ["torchscript", "onnx", "engine"])
    precision_targets = export.get("precision_targets", [])
    if not precision_targets:
        if export.get("half"):
            precision_targets.append("fp16")
        if export.get("int8"):
            precision_targets.append("int8")
    common = {
        "imgsz": export.get("imgsz", run_config.train.get("imgsz")),
        "batch": export.get("batch", 1),
        "device": export.get("device", run_config.train.get("device")),
        "dynamic": export.get("dynamic", False),
        "simplify": export.get("simplify", True),
        "workspace": export.get("workspace"),
        "nms": export.get("nms", False),
    }
    calibration = export.get("calibration", {})
    jobs: list[dict[str, Any]] = []
    end2end_modes = [None]
    if run_config.model.family == "yolo26":
        end2end_modes = export.get("end2end_modes", [True, False])
    for fmt in formats:
        for end2end in end2end_modes:
            job = dict(common)
            job["format"] = fmt
            job["half"] = False
            job["int8"] = False
            if fmt in {"torchscript", "onnx"} and "fp16" in precision_targets:
                job["half"] = True
            elif fmt in {"engine", "openvino"} and "int8" in precision_targets:
                job["int8"] = True
                job["data"] = str(dataset_yaml)
                job["fraction"] = calibration.get("fraction", 1.0)
            elif fmt in {"engine", "openvino"} and "fp16" in precision_targets:
                job["half"] = True
            if end2end is not None:
                job["end2end"] = end2end
            jobs.append(job)
    return jobs


def build_train_command(run_config: RunConfig, dataset_yaml: Path, weights: str | None = None, resume: bool = False) -> list[str]:
    payload = train_kwargs(run_config, dataset_yaml, resume=resume)
    command = ["yolo", run_config.task, "train"]
    if resume:
        model_value = weights or "path/to/last.pt"
        command.append("resume")
        payload = {k: v for k, v in payload.items() if k != "resume"}
    else:
        model_value = weights or run_config.model.weights_name
    payload["model"] = model_value
    command.extend(_build_cli_key_values(payload))
    return command


def build_val_command(run_config: RunConfig, dataset_yaml: Path, weights: str) -> list[str]:
    payload = val_kwargs(run_config, dataset_yaml)
    payload["model"] = weights
    command = ["yolo", run_config.task, "val"]
    command.extend(_build_cli_key_values(payload))
    return command


def build_model_val_command(
    model_path: str,
    dataset_yaml: Path,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    project: str | None = None,
    name: str | None = None,
) -> list[str]:
    payload = model_val_kwargs(
        dataset_yaml=dataset_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )
    payload["model"] = model_path
    command = ["yolo", "detect", "val"]
    command.extend(_build_cli_key_values(payload))
    return command


def build_export_command(weights: str, payload: dict[str, Any]) -> list[str]:
    args = dict(payload)
    args["model"] = weights
    command = ["yolo", "export"]
    command.extend(_build_cli_key_values(args))
    return command


def build_benchmark_command(model_path: str, dataset_yaml: Path, imgsz: int | None = None, device: str | None = None) -> list[str]:
    payload: dict[str, Any] = {"model": model_path, "data": str(dataset_yaml)}
    if imgsz is not None:
        payload["imgsz"] = imgsz
    if device is not None:
        payload["device"] = device
    command = ["yolo", "benchmark"]
    command.extend(_build_cli_key_values(payload))
    return command


def _run_cli(command: list[str]) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        raise YoloCtlError(f"Required command not found in PATH: {command[0]}")
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return {"engine": "cli", "command": _quote(command), "stdout": completed.stdout, "stderr": completed.stderr}


def _prepare_export_jobs(
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    dataset_yaml: Path,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    calibration_summary: dict[str, Any] | None = None
    prepared_yaml = dataset_yaml
    requires_int8 = "int8" in run_config.export.get("precision_targets", []) or bool(run_config.export.get("int8"))
    if requires_int8:
        calibration_summary = prepare_calibration_dataset(
            record=dataset_record,
            run_id=run_config.run_id,
            export_config=run_config.export,
            dataset_yaml=dataset_yaml,
            dry_run=dry_run,
        )
        if calibration_summary.get("mode") == "materialized_subset":
            prepared_yaml = Path(calibration_summary["dataset_yaml"])
    jobs = export_jobs(run_config, prepared_yaml)
    if calibration_summary and calibration_summary.get("mode") == "materialized_subset":
        for job in jobs:
            if job.get("int8"):
                job["data"] = str(prepared_yaml)
                job["fraction"] = 1.0
    if calibration_summary and calibration_summary.get("mode") == "fraction_fallback":
        fraction = float(calibration_summary["fraction"])
        for job in jobs:
            if job.get("int8"):
                job["data"] = str(dataset_yaml)
                job["fraction"] = fraction
    return jobs, calibration_summary


def run_train(
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    dataset_yaml: Path,
    weights: str | None = None,
    engine: str = "auto",
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    command = build_train_command(run_config, dataset_yaml, weights=weights, resume=resume)
    if dry_run:
        return {"engine": "preview", "command": _quote(command), "dataset": dataset_record.dataset_key}
    resolved = resolve_engine(engine)
    if resolved == "cli":
        return _run_cli(command)
    ultralytics = _import_ultralytics()
    model_path = weights or run_config.model.weights_name
    model = ultralytics.YOLO(model_path)
    results = model.train(**train_kwargs(run_config, dataset_yaml, resume=resume))
    return {
        "engine": "python",
        "command": _quote(command),
        "save_dir": str(getattr(results, "save_dir", "")),
        "results": getattr(results, "results_dict", None),
    }


def run_val(
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    dataset_yaml: Path,
    weights: str,
    engine: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    command = build_val_command(run_config, dataset_yaml, weights=weights)
    if dry_run:
        return {"engine": "preview", "command": _quote(command), "dataset": dataset_record.dataset_key}
    return run_model_validation(
        model_path=weights,
        dataset_yaml=dataset_yaml,
        imgsz=run_config.train.get("imgsz"),
        batch=run_config.train.get("batch"),
        device=run_config.train.get("device"),
        project=run_config.default_project_dir(),
        name=f"{run_config.default_run_name()}-val",
        engine=engine,
        dry_run=dry_run,
    )


def run_model_validation(
    model_path: str,
    dataset_yaml: Path,
    imgsz: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    project: str | None = None,
    name: str | None = None,
    engine: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    command = build_model_val_command(
        model_path=model_path,
        dataset_yaml=dataset_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )
    if dry_run:
        return {"engine": "preview", "command": _quote(command), "model_path": model_path}
    resolved = resolve_engine(engine)
    if resolved == "cli":
        return _run_cli(command)
    ultralytics = _import_ultralytics()
    model = ultralytics.YOLO(model_path)
    results = model.val(
        **model_val_kwargs(
            dataset_yaml=dataset_yaml,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
        )
    )
    return {
        "engine": "python",
        "command": _quote(command),
        "metrics": _json_safe(getattr(results, "results_dict", None)),
        "speed": _json_safe(getattr(results, "speed", None)),
        "save_dir": str(getattr(results, "save_dir", "")),
    }


def run_export(
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    dataset_yaml: Path,
    weights: str,
    engine: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    jobs, calibration_summary = _prepare_export_jobs(
        run_config=run_config,
        dataset_record=dataset_record,
        dataset_yaml=dataset_yaml,
        dry_run=dry_run,
    )
    previews = [_quote(build_export_command(weights, job)) for job in jobs]
    if dry_run:
        return {
            "engine": "preview",
            "commands": previews,
            "dataset": dataset_record.dataset_key,
            "calibration": calibration_summary,
        }
    resolved = resolve_engine(engine)
    outputs: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    required = bool(assessment_settings(run_config.export)["required"])
    if resolved == "cli":
        for job in jobs:
            cli_result = _run_cli(build_export_command(weights, job))
            source_path = resolve_export_source(weights, job)
            if source_path is None:
                raise YoloCtlError(
                    f"Export for format '{job['format']}' completed but no artifact could be located near {weights}."
                )
            archived_path = archive_export_artifact(run_config.run_id, weights, job, source_path)
            artifact_meta = build_artifact_metadata(
                run_id=run_config.run_id,
                weights_path=weights,
                job=job,
                engine="cli",
                required=required,
                source_path=source_path,
                archived_path=archived_path,
            )
            outputs.append({"job": job, "artifact": artifact_meta, "raw": cli_result})
            artifacts.append(artifact_meta)
        return {"engine": "cli", "outputs": outputs, "artifacts": artifacts, "calibration": calibration_summary}
    ultralytics = _import_ultralytics()
    model = ultralytics.YOLO(weights)
    for job in jobs:
        raw_artifact = model.export(**job)
        source_path = resolve_export_source(weights, job, raw_artifact=raw_artifact)
        if source_path is None:
            raise YoloCtlError(
                f"Export for format '{job['format']}' completed but no artifact could be located near {weights}."
            )
        archived_path = archive_export_artifact(run_config.run_id, weights, job, source_path)
        artifact_meta = build_artifact_metadata(
            run_id=run_config.run_id,
            weights_path=weights,
            job=job,
            engine="python",
            required=required,
            source_path=source_path,
            archived_path=archived_path,
        )
        outputs.append({"job": job, "artifact": artifact_meta, "raw": _json_safe(raw_artifact)})
        artifacts.append(artifact_meta)
    return {
        "engine": "python",
        "commands": previews,
        "outputs": outputs,
        "artifacts": artifacts,
        "calibration": calibration_summary,
    }


def run_benchmark(
    model_path: str,
    dataset_yaml: Path,
    imgsz: int | None = None,
    device: str | None = None,
    engine: str = "auto",
    dry_run: bool = False,
) -> dict[str, Any]:
    command = build_benchmark_command(model_path=model_path, dataset_yaml=dataset_yaml, imgsz=imgsz, device=device)
    if dry_run:
        return {"engine": "preview", "command": _quote(command)}
    resolved = resolve_engine(engine)
    if resolved == "cli":
        return _run_cli(command)
    ultralytics = _import_ultralytics()
    model = ultralytics.YOLO(model_path)
    results = model.benchmark(data=str(dataset_yaml), imgsz=imgsz, device=device)
    return {
        "engine": "python",
        "command": _quote(command),
        "results": _json_safe(results),
    }

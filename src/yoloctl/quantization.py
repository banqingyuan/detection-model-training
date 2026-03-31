from __future__ import annotations

from collections import Counter
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

from .datasets import DatasetVersionRecord, IMAGE_SUFFIXES
from .exceptions import YoloCtlError
from .paths import (
    assessment_reports_dir,
    calibration_runs_dir,
    ensure_parent,
    export_artifacts_dir,
    manifests_dir,
    project_root,
)
from .yamlio import dump_yaml, write_json

CALIBRATION_DEFAULTS = {
    "strategy": "class_balanced",
    "split": "train",
    "fraction": 0.25,
    "min_images": 128,
    "max_images": 1024,
    "seed": 42,
    "allow_fraction_fallback": False,
}

ASSESSMENT_DEFAULTS = {
    "required": True,
    "primary_metric": "map50_95",
    "max_drop": {
        "fp16": 0.2,
        "int8": 0.5,
    },
    "warn_metrics": ["map50", "recall"],
    "fail_on_reject": True,
}

FORMAT_SUFFIXES = {
    "torchscript": ".torchscript",
    "onnx": ".onnx",
    "engine": ".engine",
}


def _resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def calibration_settings(export_config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(CALIBRATION_DEFAULTS)
    calibration = dict(export_config.get("calibration", {}))
    payload["strategy"] = str(calibration.get("strategy", payload["strategy"]))
    payload["split"] = str(calibration.get("split", payload["split"]))
    payload["fraction"] = max(0.0, min(float(calibration.get("fraction", payload["fraction"])), 1.0))
    payload["min_images"] = max(1, int(calibration.get("min_images", payload["min_images"])))
    payload["max_images"] = max(payload["min_images"], int(calibration.get("max_images", payload["max_images"])))
    payload["seed"] = int(calibration.get("seed", payload["seed"]))
    payload["allow_fraction_fallback"] = bool(
        calibration.get("allow_fraction_fallback", payload["allow_fraction_fallback"])
    )
    if payload["strategy"] not in {"class_balanced", "uniform"}:
        raise YoloCtlError(
            f"Unsupported calibration.strategy '{payload['strategy']}'. "
            "Supported values: class_balanced, uniform."
        )
    return payload


def assessment_settings(export_config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(ASSESSMENT_DEFAULTS)
    assessment = dict(export_config.get("assessment", {}))
    max_drop = dict(ASSESSMENT_DEFAULTS["max_drop"])
    max_drop.update({str(key): float(value) for key, value in dict(assessment.get("max_drop", {})).items()})
    warn_metrics = [str(item) for item in assessment.get("warn_metrics", ASSESSMENT_DEFAULTS["warn_metrics"])]
    payload.update(
        {
            "required": bool(assessment.get("required", payload["required"])),
            "primary_metric": "map50_95",
            "max_drop": {
                "fp16": float(max_drop.get("fp16", ASSESSMENT_DEFAULTS["max_drop"]["fp16"])),
                "int8": float(max_drop.get("int8", ASSESSMENT_DEFAULTS["max_drop"]["int8"])),
            },
            "warn_metrics": warn_metrics or list(ASSESSMENT_DEFAULTS["warn_metrics"]),
            "fail_on_reject": bool(assessment.get("fail_on_reject", payload["fail_on_reject"])),
        }
    )
    return payload


def export_precision(job: dict[str, Any]) -> str:
    if job.get("int8"):
        return "int8"
    if job.get("half"):
        return "fp16"
    return "fp32"


def export_variant_name(job: dict[str, Any]) -> str:
    parts = [str(job.get("format", "artifact")), export_precision(job)]
    if job.get("end2end") is not None:
        parts.append(f"end2end-{str(job['end2end']).lower()}")
    return "-".join(parts)


def _expected_artifact_name(weights_path: str | Path, job: dict[str, Any]) -> str:
    weights = Path(weights_path)
    stem = weights.stem
    fmt = str(job.get("format", "artifact"))
    if fmt == "openvino":
        return f"{stem}_openvino_model"
    suffix = FORMAT_SUFFIXES.get(fmt, f".{fmt}")
    return f"{stem}{suffix}"


def candidate_export_sources(weights_path: str | Path, job: dict[str, Any]) -> list[Path]:
    weights = Path(weights_path)
    fmt = str(job.get("format", "artifact"))
    direct = weights.parent / _expected_artifact_name(weights, job)
    candidates = [direct]
    if fmt == "openvino":
        candidates.extend(sorted(weights.parent.glob(f"{weights.stem}*openvino*")))
    else:
        suffix = FORMAT_SUFFIXES.get(fmt, f".{fmt}")
        candidates.extend(sorted(weights.parent.glob(f"{weights.stem}*{suffix}")))
    return candidates


def resolve_export_source(weights_path: str | Path, job: dict[str, Any], raw_artifact: Any = None) -> Path | None:
    if isinstance(raw_artifact, (str, Path)):
        candidate = Path(raw_artifact)
        if candidate.exists():
            return candidate
    for candidate in candidate_export_sources(weights_path, job):
        if candidate.exists():
            return candidate
    weights = Path(weights_path)
    fmt = str(job.get("format", "artifact"))
    search_root = weights.parent
    if not search_root.exists():
        return None
    if fmt == "openvino":
        discovered = [
            path
            for path in search_root.rglob("*")
            if path.is_dir() and path.name.endswith("openvino_model")
        ]
    else:
        suffix = FORMAT_SUFFIXES.get(fmt, f".{fmt}")
        discovered = [path for path in search_root.rglob(f"*{suffix}") if path.is_file()]
    if not discovered:
        return None
    discovered.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return discovered[0]


def _artifact_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def archive_export_artifact(run_id: str, weights_path: str | Path, job: dict[str, Any], source_path: Path) -> Path:
    variant = export_variant_name(job)
    destination = export_artifacts_dir(run_id) / variant / source_path.name
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    ensure_parent(destination)
    if source_path.is_dir():
        shutil.copytree(source_path, destination)
    else:
        shutil.copy2(source_path, destination)
    return destination


def build_artifact_metadata(
    run_id: str,
    weights_path: str | Path,
    job: dict[str, Any],
    engine: str,
    required: bool,
    source_path: Path | None = None,
    archived_path: Path | None = None,
) -> dict[str, Any]:
    artifact_path = archived_path or (
        export_artifacts_dir(run_id) / export_variant_name(job) / _expected_artifact_name(weights_path, job)
    )
    size_mb = round(_artifact_size_bytes(artifact_path) / (1024.0 * 1024.0), 6) if artifact_path.exists() else None
    return {
        "format": str(job.get("format", "artifact")),
        "precision": export_precision(job),
        "end2end": job.get("end2end"),
        "path": str(artifact_path),
        "source_path": str(source_path) if source_path else None,
        "size_mb": size_mb,
        "engine": engine,
        "required": required,
        "variant": export_variant_name(job),
    }


def _label_path_for_image(image_path: Path, source_images_dir: Path, source_labels_dir: Path) -> Path:
    relative = image_path.relative_to(source_images_dir)
    return (source_labels_dir / relative).with_suffix(".txt")


def _parse_label_classes(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    classes: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            classes.append(int(float(parts[0])))
        except ValueError:
            continue
    return classes


def _sample_target(available: int, settings: dict[str, Any]) -> int:
    if available <= 0:
        return 0
    fraction_target = int(round(available * float(settings["fraction"])))
    if float(settings["fraction"]) > 0 and fraction_target == 0:
        fraction_target = 1
    target = max(fraction_target, int(settings["min_images"]))
    target = min(target, int(settings["max_images"]), available)
    return target


def _class_balanced_selection(examples: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    class_frequency: Counter[int] = Counter()
    for example in shuffled:
        class_frequency.update(example["class_ids"])
    if not class_frequency:
        return shuffled[:target]

    scored = list(shuffled)
    scored.sort(
        key=lambda item: (
            sum(1.0 / max(class_frequency[class_id], 1) for class_id in item["class_set"]),
            len(item["class_set"]),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for class_id, _ in sorted(class_frequency.items(), key=lambda item: (item[1], item[0])):
        for example in scored:
            if example["id"] in seen_ids or class_id not in example["class_set"]:
                continue
            selected.append(example)
            seen_ids.add(example["id"])
            break
        if len(selected) >= target:
            return selected[:target]

    for example in scored:
        if example["id"] in seen_ids:
            continue
        selected.append(example)
        seen_ids.add(example["id"])
        if len(selected) >= target:
            break
    return selected[:target]


def _uniform_selection(examples: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return shuffled[:target]


def _link_or_copy(source: Path, destination: Path) -> None:
    ensure_parent(destination)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def prepare_calibration_dataset(
    record: DatasetVersionRecord,
    run_id: str,
    export_config: dict[str, Any],
    dataset_yaml: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    settings = calibration_settings(export_config)
    materialized = _resolve_repo_path(record.materialized_path)
    if materialized is None or not materialized.exists():
        if settings["allow_fraction_fallback"]:
            return {
                "mode": "fraction_fallback",
                "strategy": settings["strategy"],
                "source_split": settings["split"],
                "dataset_yaml": str(dataset_yaml),
                "fraction": settings["fraction"],
                "seed": settings["seed"],
                "selected_count": None,
                "available_count": record.split_counts.get(settings["split"], {}).get("images"),
                "required": False,
            }
        raise YoloCtlError(
            "INT8 export requires a materialized dataset for deterministic calibration. "
            "Run dataset prepare first or set export.calibration.allow_fraction_fallback=true."
        )

    split = str(settings["split"])
    if split not in record.splits:
        raise YoloCtlError(
            f"Calibration split '{split}' is not declared in dataset version '{record.version_id}'."
        )

    source_images_dir = materialized / record.splits[split]
    source_labels_relative = (
        record.splits[split].replace("images/", "labels/", 1)
        if record.splits[split].startswith("images/")
        else record.splits[split]
    )
    source_labels_dir = materialized / source_labels_relative
    if not source_images_dir.exists():
        if settings["allow_fraction_fallback"]:
            return {
                "mode": "fraction_fallback",
                "strategy": settings["strategy"],
                "source_split": split,
                "dataset_yaml": str(dataset_yaml),
                "fraction": settings["fraction"],
                "seed": settings["seed"],
                "selected_count": None,
                "available_count": record.split_counts.get(split, {}).get("images"),
                "required": False,
            }
        raise YoloCtlError(
            f"Calibration source split does not exist: {source_images_dir}. "
            "Run dataset prepare/validate before INT8 export."
        )

    examples: list[dict[str, Any]] = []
    for image_path in sorted(
        path for path in source_images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image_path.relative_to(source_images_dir)
        label_path = _label_path_for_image(image_path, source_images_dir, source_labels_dir)
        class_ids = _parse_label_classes(label_path)
        examples.append(
            {
                "id": relative.as_posix(),
                "relative_path": relative,
                "image_path": image_path,
                "label_path": label_path,
                "class_ids": class_ids,
                "class_set": set(class_ids),
            }
        )

    if not examples:
        raise YoloCtlError(f"No calibration images found under {source_images_dir}")

    target = _sample_target(len(examples), settings)
    if settings["strategy"] == "class_balanced":
        selected = _class_balanced_selection(examples, target, int(settings["seed"]))
    else:
        selected = _uniform_selection(examples, target, int(settings["seed"]))

    calibration_root = calibration_runs_dir() / run_id / f"{split}-{settings['strategy']}-seed{settings['seed']}"
    dataset_yaml_path = calibration_root / "detect.yaml"
    summary_path = calibration_root / "summary.json"

    histogram: Counter[int] = Counter()
    for example in selected:
        histogram.update(example["class_ids"])

    summary = {
        "mode": "materialized_subset",
        "strategy": settings["strategy"],
        "source_split": split,
        "dataset_yaml": str(dataset_yaml_path),
        "summary_path": str(summary_path),
        "seed": settings["seed"],
        "fraction": settings["fraction"],
        "min_images": settings["min_images"],
        "max_images": settings["max_images"],
        "available_count": len(examples),
        "selected_count": len(selected),
        "selected_images": [example["id"] for example in selected],
        "class_histogram": {
            record.class_names[class_id] if 0 <= class_id < len(record.class_names) else str(class_id): count
            for class_id, count in sorted(histogram.items())
        },
    }
    if dry_run:
        summary["planned"] = True
        return summary

    subset_images_dir = calibration_root / "images" / "train"
    subset_labels_dir = calibration_root / "labels" / "train"
    for example in selected:
        image_target = subset_images_dir / example["relative_path"]
        _link_or_copy(example["image_path"], image_target)
        label_target = (subset_labels_dir / example["relative_path"]).with_suffix(".txt")
        if example["label_path"].exists():
            _link_or_copy(example["label_path"], label_target)
        else:
            ensure_parent(label_target)
            label_target.write_text("", encoding="utf-8")

    dump_yaml(
        dataset_yaml_path,
        {
            "path": str(calibration_root),
            "train": "images/train",
            "val": "images/train",
            "names": record.names_mapping(),
        },
    )
    write_json(summary_path, summary)
    return summary


def flatten_mapping(payload: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        flattened[name] = value
        if isinstance(value, dict):
            flattened.update(flatten_mapping(value, prefix=name))
    return flattened


def normalize_metrics(payload: Any) -> dict[str, float | None]:
    flattened = flatten_mapping(payload if isinstance(payload, dict) else {})
    numeric_items = [
        (re.sub(r"[^a-z0-9]+", "", key.lower()), value)
        for key, value in flattened.items()
        if isinstance(value, (int, float))
    ]
    map50_95 = None
    map50 = None
    recall = None
    precision = None
    for key, value in numeric_items:
        if map50_95 is None and "map5095" in key:
            map50_95 = float(value)
        if map50 is None and "map50" in key and "map5095" not in key:
            map50 = float(value)
        if recall is None and "recall" in key:
            recall = float(value)
        if precision is None and "precision" in key:
            precision = float(value)
    return {
        "map50_95": map50_95,
        "map50": map50,
        "recall": recall,
        "precision": precision,
    }


def normalize_performance(validation_speed: Any = None, benchmark_payload: Any = None) -> dict[str, float | None]:
    latency_ms = None
    fps = None
    for payload in (benchmark_payload, validation_speed):
        flattened = flatten_mapping(payload if isinstance(payload, dict) else {})
        for key, value in flattened.items():
            if not isinstance(value, (int, float)):
                continue
            normalized = re.sub(r"[^a-z0-9]+", "", key.lower())
            if latency_ms is None and ("inference" in normalized or "latency" in normalized):
                latency_ms = float(value)
            if fps is None and "fps" in normalized:
                fps = float(value)
        if latency_ms is not None and fps is not None:
            break
    if fps is None and latency_ms not in (None, 0):
        fps = round(1000.0 / float(latency_ms), 3)
    return {
        "latency_ms": latency_ms,
        "fps": fps,
    }


def compare_against_baseline(
    baseline_metrics: dict[str, float | None],
    candidate_metrics: dict[str, float | None],
    precision: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    primary_metric = str(settings["primary_metric"])
    baseline_value = baseline_metrics.get(primary_metric)
    candidate_value = candidate_metrics.get(primary_metric)
    reasons: list[str] = []
    warnings: list[str] = []
    baseline_delta: dict[str, float | None] = {}
    for metric_name in {primary_metric, *settings.get("warn_metrics", [])}:
        baseline_metric = baseline_metrics.get(metric_name)
        candidate_metric = candidate_metrics.get(metric_name)
        if baseline_metric is None or candidate_metric is None:
            baseline_delta[metric_name] = None
            continue
        delta = round(float(candidate_metric) - float(baseline_metric), 6)
        baseline_delta[metric_name] = delta
        if metric_name != primary_metric and delta < 0:
            warnings.append(f"{metric_name} dropped by {abs(delta):.4f}")

    if baseline_value is None or candidate_value is None:
        reasons.append(f"Missing primary metric '{primary_metric}' for gate evaluation")
    else:
        drop = float(baseline_value) - float(candidate_value)
        threshold = float(settings["max_drop"].get(precision, settings["max_drop"]["int8"]))
        if drop > threshold:
            reasons.append(
                f"{primary_metric} dropped by {drop:.4f}, exceeding the allowed {threshold:.4f} for {precision}"
            )

    status = "accepted" if not reasons else "rejected"
    return {
        "status": status,
        "accepted": status == "accepted",
        "reasons": reasons,
        "warnings": warnings,
        "baseline_delta": baseline_delta,
    }


def load_export_manifest(run_id: str) -> dict[str, Any]:
    path = manifests_dir() / run_id / "export.json"
    if not path.exists():
        raise YoloCtlError(
            f"Export assessment requires a prior export manifest at {path}. "
            "Run 'yoloctl export run' first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_assessment_report(run_id: str, payload: dict[str, Any]) -> Path:
    output = assessment_reports_dir() / f"{run_id}.json"
    return write_json(output, payload)

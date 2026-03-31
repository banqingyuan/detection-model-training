from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    explicit = os.getenv("YOLOCTL_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return project_root() / "configs"


def artifacts_dir() -> Path:
    return project_root() / "artifacts"


def dataset_registry_dir() -> Path:
    return artifacts_dir() / "registry" / "datasets"


def dataset_index_path() -> Path:
    return dataset_registry_dir() / "index.yaml"


def dataset_manifest_path(dataset_key: str, version_id: str) -> Path:
    return dataset_registry_dir() / dataset_key / f"{version_id}.yaml"


def default_dataset_yaml_path(dataset_key: str, version_id: str) -> Path:
    return configs_dir() / "datasets" / f"{dataset_key}--{version_id}.yaml"


def manifests_dir() -> Path:
    return artifacts_dir() / "manifests"


def reports_dir() -> Path:
    return artifacts_dir() / "reports"


def reviews_dir() -> Path:
    return artifacts_dir() / "reviews"


def review_dataset_dir(dataset_key: str, version_id: str) -> Path:
    return reviews_dir() / dataset_key / version_id


def review_session_dir(dataset_key: str, version_id: str, session_id: str) -> Path:
    return review_dataset_dir(dataset_key, version_id) / session_id


def export_artifacts_dir(run_id: str) -> Path:
    return artifacts_dir() / "exports" / run_id


def calibration_runs_dir() -> Path:
    return artifacts_dir() / "calibration"


def paddledet_datasets_dir() -> Path:
    return artifacts_dir() / "paddledet_datasets"


def assessment_reports_dir() -> Path:
    return reports_dir() / "assessments"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

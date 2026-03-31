from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import YoloCtlError
from .models import ModelSpec, validate_model
from .paths import project_root
from .yamlio import load_yaml

SUPPORTED_BACKENDS = {"ultralytics", "paddledet"}


@dataclass
class VastProfile:
    name: str
    search: dict[str, Any] = field(default_factory=dict)
    launch: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: Path) -> "VastProfile":
        payload = load_yaml(path)
        name = payload.get("name") or path.stem
        return cls(
            name=name,
            search=payload.get("search", {}),
            launch=payload.get("launch", {}),
            storage=payload.get("storage", {}),
            source_path=path.resolve(),
        )


@dataclass
class PaddleDetConfig:
    root: Path
    config: Path
    pretrain_weights: str | None = None
    eval_weights: str | None = None
    device: str | None = None
    eval: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    run_id: str
    task: str
    dataset_key: str
    dataset_version_id: str | None
    dataset_version_label: str | None
    model: ModelSpec | None
    train: dict[str, Any]
    export: dict[str, Any]
    tracking: dict[str, Any]
    source_path: Path
    backend: str = "ultralytics"
    paddledet: PaddleDetConfig | None = None

    @classmethod
    def from_file(cls, path: Path) -> "RunConfig":
        payload = load_yaml(path)
        run_id = payload.get("run_id") or path.stem
        task = payload.get("task", "detect")
        backend = str(payload.get("backend", "ultralytics"))
        if backend not in SUPPORTED_BACKENDS:
            raise YoloCtlError(f"Unsupported backend '{backend}'. Supported: {sorted(SUPPORTED_BACKENDS)}")

        model: ModelSpec | None = None
        paddledet: PaddleDetConfig | None = None
        if backend == "ultralytics":
            model_payload = payload.get("model", {})
            family = model_payload.get("family")
            size = model_payload.get("size")
            if not family or not size:
                raise YoloCtlError("Run config requires model.family and model.size")
            model = validate_model(family=family, size=size, task=task)
        else:
            paddledet_payload = payload.get("paddledet", {})
            root_value = paddledet_payload.get("root")
            config_value = paddledet_payload.get("config")
            if not root_value or not config_value:
                raise YoloCtlError("PaddleDet run config requires paddledet.root and paddledet.config")
            root = Path(root_value).expanduser()
            if not root.is_absolute():
                root = (project_root() / root).resolve()
            config_path = Path(config_value).expanduser()
            if not config_path.is_absolute():
                config_path = (root / config_path).resolve()
            paddledet = PaddleDetConfig(
                root=root,
                config=config_path,
                pretrain_weights=str(paddledet_payload["pretrain_weights"]) if paddledet_payload.get("pretrain_weights") else None,
                eval_weights=str(paddledet_payload["eval_weights"]) if paddledet_payload.get("eval_weights") else None,
                device=str(paddledet_payload["device"]) if paddledet_payload.get("device") else None,
                eval=bool(paddledet_payload.get("eval", True)),
                overrides=dict(paddledet_payload.get("overrides", {})),
            )
        dataset_key = payload.get("dataset_key") or payload.get("dataset_id")
        if not dataset_key:
            raise YoloCtlError("Run config requires dataset_key")
        return cls(
            run_id=run_id,
            task=task,
            dataset_key=dataset_key,
            dataset_version_id=payload.get("dataset_version_id"),
            dataset_version_label=payload.get("dataset_version_label") or payload.get("dataset_version"),
            model=model,
            train=payload.get("train", {}),
            export=payload.get("export", {}),
            tracking=payload.get("tracking", {}),
            source_path=path.resolve(),
            backend=backend,
            paddledet=paddledet,
        )

    def default_project_dir(self) -> str:
        return str(self.train.get("project", "artifacts/runs"))

    def default_run_name(self) -> str:
        return str(self.train.get("name", self.run_id))

    @property
    def is_ultralytics(self) -> bool:
        return self.backend == "ultralytics"

    @property
    def is_paddledet(self) -> bool:
        return self.backend == "paddledet"

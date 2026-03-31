from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RunConfig
from .datasets import DatasetVersionRecord
from .exceptions import YoloCtlError
from .ultralytics_runner import _import_ultralytics, train_kwargs
from .yamlio import load_yaml


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass
class QatConfig:
    experiment_id: str
    base_run_config: Path
    weights: str | None = None
    project: str = "artifacts/runs/qat"
    name: str | None = None
    train: dict[str, Any] = field(default_factory=dict)
    qat: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: Path) -> "QatConfig":
        payload = load_yaml(path)
        base_run_config = payload.get("base_run_config") or payload.get("run_config")
        if not base_run_config:
            raise YoloCtlError("QAT config requires base_run_config")
        return cls(
            experiment_id=str(payload.get("experiment_id") or path.stem),
            base_run_config=(path.parent / str(base_run_config)).resolve()
            if not Path(str(base_run_config)).is_absolute()
            else Path(str(base_run_config)),
            weights=str(payload["weights"]) if payload.get("weights") else None,
            project=str(payload.get("project") or "artifacts/runs/qat"),
            name=str(payload["name"]) if payload.get("name") else None,
            train=dict(payload.get("train", {})),
            qat=dict(payload.get("qat", {})),
            source_path=path.resolve(),
        )

    def run_name(self) -> str:
        return self.name or self.experiment_id


def _apply_experimental_qat(model: Any, backend: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise YoloCtlError("QAT requires torch to be installed in the training environment.") from exc

    if not hasattr(torch, "ao") or not hasattr(torch.ao, "quantization"):
        raise YoloCtlError("QAT requires torch.ao.quantization support.")

    torch.backends.quantized.engine = backend
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    patched_modules = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.qconfig = qconfig
            patched_modules += 1
    if patched_modules == 0:
        raise YoloCtlError("No Conv2d/Linear modules were found for experimental QAT preparation.")
    model.train()
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return {"backend": backend, "patched_modules": patched_modules}


def run_qat_experiment(
    qat_config: QatConfig,
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    dataset_yaml: Path,
    weights: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    if run_config.task != "detect":
        raise YoloCtlError("The experimental QAT lane only supports detect runs.")
    train_payload = train_kwargs(run_config, dataset_yaml, resume=False)
    train_payload.update(qat_config.train)
    train_payload["data"] = str(dataset_yaml)
    train_payload["project"] = qat_config.project
    train_payload["name"] = qat_config.run_name()
    train_payload["pretrained"] = False
    train_payload.setdefault("amp", False)
    train_payload.setdefault("optimizer", "auto")
    backend = str(qat_config.qat.get("backend", "fbgemm"))

    if dry_run:
        return {
            "engine": "preview",
            "experiment_id": qat_config.experiment_id,
            "base_run_config": str(qat_config.base_run_config),
            "dataset": dataset_record.dataset_key,
            "weights": weights,
            "train": train_payload,
            "qat": {
                "backend": backend,
                "experimental": True,
            },
        }

    ultralytics = _import_ultralytics()
    try:
        import torch
    except ImportError as exc:
        raise YoloCtlError("QAT requires torch to be installed in the training environment.") from exc

    float_model = ultralytics.YOLO(weights)
    qat_model = ultralytics.YOLO(weights)
    qat_setup = _apply_experimental_qat(qat_model.model, backend=backend)
    results = qat_model.train(**train_payload)
    save_dir = Path(str(getattr(results, "save_dir", train_payload["project"])))
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    missing, unexpected = float_model.model.load_state_dict(qat_model.model.state_dict(), strict=False)
    exportable_checkpoint = weights_dir / "qat_best.pt"
    checkpoint_payload = {
        "model": copy.deepcopy(float_model.model).cpu(),
        "qat": {
            "backend": backend,
            "experimental": True,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        },
        "train_args": train_payload,
    }
    torch.save(checkpoint_payload, exportable_checkpoint)
    return {
        "engine": "python",
        "experiment_id": qat_config.experiment_id,
        "base_run_config": str(qat_config.base_run_config),
        "dataset": dataset_record.dataset_key,
        "weights": weights,
        "save_dir": str(save_dir),
        "checkpoint_path": str(exportable_checkpoint),
        "results": _json_safe(getattr(results, "results_dict", None)),
        "qat": {
            **qat_setup,
            "experimental": True,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        },
    }

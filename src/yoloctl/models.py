from __future__ import annotations

from dataclasses import dataclass

from .exceptions import YoloCtlError

SUPPORTED_TASKS = {"detect"}
SUPPORTED_MODEL_FAMILIES = {
    "yolo11": {"sizes": {"n", "s", "m", "l", "x"}},
    "yolo26": {"sizes": {"n", "s", "m", "l", "x"}},
}


@dataclass(frozen=True)
class ModelSpec:
    family: str
    size: str
    task: str = "detect"
    pretrained: bool = True

    @property
    def weights_name(self) -> str:
        suffix = "" if self.task == "detect" else f"-{self.task}"
        return f"{self.family}{self.size}{suffix}.pt"


def validate_model(family: str, size: str, task: str = "detect") -> ModelSpec:
    if task not in SUPPORTED_TASKS:
        raise YoloCtlError(f"Unsupported task '{task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}")
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise YoloCtlError(
            f"Unsupported model family '{family}'. Supported: {sorted(SUPPORTED_MODEL_FAMILIES)}"
        )
    if size not in SUPPORTED_MODEL_FAMILIES[family]["sizes"]:
        raise YoloCtlError(
            f"Unsupported size '{size}' for {family}. Supported: {sorted(SUPPORTED_MODEL_FAMILIES[family]['sizes'])}"
        )
    return ModelSpec(family=family, size=size, task=task)


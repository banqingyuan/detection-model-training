from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import manifests_dir
from .yamlio import write_json


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_manifest(
    stage: str,
    run_id: str,
    command_preview: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "created_at": now_utc_iso(),
        "stage": stage,
        "run_id": run_id,
        "command_preview": command_preview,
    }
    manifest.update(payload)
    return manifest


def write_manifest(stage: str, run_id: str, manifest: dict[str, Any]) -> Path:
    output = manifests_dir() / run_id / f"{stage}.json"
    return write_json(output, manifest)


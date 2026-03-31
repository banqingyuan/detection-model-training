from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .exceptions import YoloCtlError
from .paths import reports_dir
from .vast import invoice_snapshot, lifecycle_action, volume_action
from .yamlio import write_json


def snapshot_costs(
    output_path: Path | None = None,
    dry_run: bool = False,
    use_sdk: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
    only_charges: bool = False,
    only_credits: bool = False,
    instance_label: str | None = None,
) -> dict[str, Any]:
    payload = invoice_snapshot(
        dry_run=dry_run,
        use_sdk=use_sdk,
        start_date=start_date,
        end_date=end_date,
        only_charges=only_charges,
        only_credits=only_credits,
        instance_label=instance_label,
    )
    if output_path is None and not dry_run:
        output_path = reports_dir() / "costs" / "latest-invoices.json"
    if output_path and not dry_run:
        write_json(output_path, payload)
    return payload


def summarize_snapshots(directory: Path | None = None) -> dict[str, Any]:
    base = directory or reports_dir() / "costs"
    if not base.exists():
        raise YoloCtlError(f"Cost snapshot directory does not exist: {base}")
    files = sorted(base.glob("*.json"))
    summary = {
        "snapshot_count": len(files),
        "files": [str(path) for path in files],
        "engines": Counter(),
    }
    for path in files:
        data = path.read_text(encoding="utf-8")
        if '"engine": "sdk"' in data:
            summary["engines"]["sdk"] += 1
        elif '"engine": "cli"' in data:
            summary["engines"]["cli"] += 1
        elif '"engine": "cli-preview"' in data:
            summary["engines"]["cli-preview"] += 1
        else:
            summary["engines"]["unknown"] += 1
    summary["engines"] = dict(summary["engines"])
    return summary


def cleanup_resources(
    instance_id: int | None = None,
    volume_id: int | None = None,
    destroy_instance: bool = False,
    delete_volume: bool = False,
    dry_run: bool = True,
    use_sdk: bool = True,
) -> dict[str, Any]:
    if not destroy_instance and not delete_volume:
        raise YoloCtlError("Cleanup requires at least one action flag")
    results: dict[str, Any] = {}
    if destroy_instance:
        if instance_id is None:
            raise YoloCtlError("Destroying an instance requires --instance-id")
        results["instance"] = lifecycle_action("destroy", instance_id=instance_id, dry_run=dry_run, use_sdk=use_sdk)
    if delete_volume:
        if volume_id is None:
            raise YoloCtlError("Deleting a volume requires --volume-id")
        results["volume"] = volume_action("delete", volume_id=volume_id, dry_run=dry_run, use_sdk=use_sdk)
    return results


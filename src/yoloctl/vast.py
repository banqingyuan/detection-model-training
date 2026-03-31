from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import asdict
from typing import Any

from .config import VastProfile
from .exceptions import YoloCtlError


def _load_vast_client() -> Any:
    try:
        from vastai_sdk import VastAI  # type: ignore
    except ImportError as exc:
        raise YoloCtlError(
            "vastai-sdk is not installed. Install project dependencies inside the Ubuntu training environment."
        ) from exc
    api_key = os.getenv("VAST_API_KEY")
    return VastAI(api_key=api_key, raw=True, quiet=False)


def _maybe_parse_json(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _quote(command: list[str]) -> str:
    return shlex.join(command)


def build_search_query(profile: VastProfile) -> str:
    launch = profile.launch
    search = profile.search
    clauses = list(search.get("query_clauses", []))
    if launch.get("gpu_name") and not any("gpu_name" in clause for clause in clauses):
        clauses.insert(0, f"gpu_name={launch['gpu_name']}")
    if launch.get("num_gpus") and not any("num_gpus" in clause for clause in clauses):
        clauses.insert(1 if clauses else 0, f"num_gpus={launch['num_gpus']}")
    if not clauses:
        raise YoloCtlError("Vast profile must define search.query_clauses or launch.gpu_name/launch.num_gpus")
    return " ".join(clauses)


def build_search_command(profile: VastProfile, limit: int | None = None) -> list[str]:
    command = ["vastai", "search", "offers", build_search_query(profile)]
    order = profile.search.get("order")
    if order:
        command.extend(["-o", str(order)])
    effective_limit = limit or profile.search.get("limit")
    if effective_limit:
        command.extend(["--limit", str(effective_limit)])
    storage = profile.search.get("storage")
    if storage is not None:
        command.extend(["--storage", str(storage)])
    return command


def build_launch_command(profile: VastProfile, label: str | None = None) -> list[str]:
    launch = profile.launch
    gpu_name = launch.get("gpu_name")
    num_gpus = launch.get("num_gpus")
    image = launch.get("image")
    if not gpu_name or not num_gpus or not image:
        raise YoloCtlError("Vast launch requires launch.gpu_name, launch.num_gpus, and launch.image")
    command = [
        "vastai",
        "launch",
        "instance",
        "--gpu-name",
        str(gpu_name),
        "--num-gpus",
        str(num_gpus),
        "--image",
        str(image),
    ]
    for key in ("region", "disk", "limit", "order", "login", "onstart", "onstart_cmd", "entrypoint", "jupyter_dir", "env", "extra", "template_hash"):
        value = launch.get(key)
        if value is not None:
            command.extend([f"--{key.replace('_', '-')}", str(value)])
    if label or launch.get("label"):
        command.extend(["--label", str(label or launch.get("label"))])
    for flag in ("ssh", "jupyter", "direct", "jupyter_lab", "lang_utf8", "python_utf8", "force", "cancel_unavail", "explain", "raw"):
        if launch.get(flag):
            command.append(f"--{flag.replace('_', '-')}")
    args = launch.get("args") or []
    if args:
        command.append("--args")
        command.extend([str(item) for item in args])
    return command


def build_volume_command(action: str, **kwargs: Any) -> list[str]:
    if action == "show":
        command = ["vastai", "show", "volumes"]
        if kwargs.get("type"):
            command.extend(["--type", str(kwargs["type"])])
        return command
    if action == "search":
        query = kwargs.get("query") or ""
        command = ["vastai", "search", "volumes", query]
        if kwargs.get("storage") is not None:
            command.extend(["--storage", str(kwargs["storage"])])
        if kwargs.get("order"):
            command.extend(["--order", str(kwargs["order"])])
        return command
    if action == "create":
        if kwargs.get("offer_id") is None:
            raise YoloCtlError("Volume creation requires an offer ID")
        command = ["vastai", "create", "volume", str(kwargs["offer_id"])]
        if kwargs.get("size") is not None:
            command.extend(["--size", str(kwargs["size"])])
        if kwargs.get("name"):
            command.extend(["--name", str(kwargs["name"])])
        return command
    if action == "delete":
        if kwargs.get("volume_id") is None:
            raise YoloCtlError("Volume deletion requires a volume ID")
        return ["vastai", "delete", "volume", str(kwargs["volume_id"])]
    raise YoloCtlError(f"Unsupported volume action: {action}")


def execute_vast_command(command: list[str], dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return {"engine": "cli-preview", "command": _quote(command)}
    if shutil.which(command[0]) is None:
        raise YoloCtlError(f"Required command not found in PATH: {command[0]}")
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return {"engine": "cli", "command": _quote(command), "stdout": completed.stdout, "stderr": completed.stderr}


def search_offers(profile: VastProfile, dry_run: bool = False, use_sdk: bool = True, limit: int | None = None) -> dict[str, Any]:
    if dry_run:
        return {"engine": "cli-preview", "command": _quote(build_search_command(profile, limit=limit))}
    query = build_search_query(profile)
    if use_sdk:
        client = _load_vast_client()
        response = client.search_offers(
            limit=limit or profile.search.get("limit"),
            order=profile.search.get("order"),
            storage=profile.search.get("storage"),
            query=query,
        )
        return {
            "engine": "sdk",
            "query": query,
            "response": _maybe_parse_json(response),
        }
    return execute_vast_command(build_search_command(profile, limit=limit), dry_run=False)


def launch_instance(profile: VastProfile, dry_run: bool = False, use_sdk: bool = True, label: str | None = None) -> dict[str, Any]:
    if dry_run:
        return {"engine": "cli-preview", "command": _quote(build_launch_command(profile, label=label))}
    if use_sdk:
        client = _load_vast_client()
        params = dict(profile.launch)
        if label:
            params["label"] = label
        response = client.launch_instance(**params)
        return {"engine": "sdk", "params": params, "response": _maybe_parse_json(response)}
    return execute_vast_command(build_launch_command(profile, label=label), dry_run=False)


def lifecycle_action(action: str, instance_id: int, dry_run: bool = False, use_sdk: bool = True) -> dict[str, Any]:
    cli_command = ["vastai", action, "instance", str(instance_id)]
    if dry_run:
        return {"engine": "cli-preview", "command": _quote(cli_command)}
    if use_sdk:
        client = _load_vast_client()
        method = getattr(client, f"{action}_instance")
        response = method(id=instance_id)
        return {"engine": "sdk", "response": _maybe_parse_json(response), "instance_id": instance_id}
    return execute_vast_command(cli_command, dry_run=False)


def show_instances(dry_run: bool = False, use_sdk: bool = True) -> dict[str, Any]:
    cli_command = ["vastai", "show", "instances"]
    if dry_run:
        return {"engine": "cli-preview", "command": _quote(cli_command)}
    if use_sdk:
        client = _load_vast_client()
        response = client.show_instances(quiet=False)
        return {"engine": "sdk", "response": _maybe_parse_json(response)}
    return execute_vast_command(cli_command, dry_run=False)


def invoice_snapshot(
    dry_run: bool = False,
    use_sdk: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
    only_charges: bool = False,
    only_credits: bool = False,
    instance_label: str | None = None,
) -> dict[str, Any]:
    cli_command = ["vastai", "show", "invoices"]
    for key, value in (
        ("start_date", start_date),
        ("end_date", end_date),
        ("instance_label", instance_label),
    ):
        if value:
            cli_command.extend([f"--{key.replace('_', '-')}", str(value)])
    if only_charges:
        cli_command.append("--only-charges")
    if only_credits:
        cli_command.append("--only-credits")
    if dry_run:
        return {"engine": "cli-preview", "command": _quote(cli_command)}
    if use_sdk:
        client = _load_vast_client()
        response = client.show_invoices(
            quiet=False,
            start_date=start_date,
            end_date=end_date,
            only_charges=only_charges,
            only_credits=only_credits,
            instance_label=instance_label,
        )
        return {"engine": "sdk", "response": _maybe_parse_json(response)}
    return execute_vast_command(cli_command, dry_run=False)


def volume_action(
    action: str,
    dry_run: bool = False,
    use_sdk: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    command = build_volume_command(action, **kwargs)
    if dry_run:
        return {"engine": "cli-preview", "command": _quote(command)}
    if use_sdk:
        client = _load_vast_client()
        if action == "show":
            response = client.show_volumes(type=kwargs.get("type", "all"))
        elif action == "search":
            response = client.search_volumes(
                query=kwargs.get("query"),
                storage=kwargs.get("storage", 1.0),
                order=kwargs.get("order", "score-"),
            )
        elif action == "create":
            response = client.create_volume(
                id=kwargs["offer_id"],
                size=kwargs.get("size", 15),
                name=kwargs.get("name"),
            )
        elif action == "delete":
            response = client.delete_volume(id=kwargs["volume_id"])
        else:
            raise YoloCtlError(f"Unsupported volume action: {action}")
        return {"engine": "sdk", "response": _maybe_parse_json(response)}
    return execute_vast_command(command, dry_run=False)

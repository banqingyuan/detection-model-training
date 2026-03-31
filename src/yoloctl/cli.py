from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import webbrowser

from .advisory import recommend_offers, summarize_market_estimate
from .config import RunConfig, VastProfile
from .cost import cleanup_resources, snapshot_costs, summarize_snapshots
from .datasets import (
    DatasetVersionRecord,
    build_dataset_summary,
    create_dataset_record,
    create_merge_draft,
    get_dataset_record,
    import_release_manifest,
    lineage_for_record,
    load_dataset_index,
    merge_draft_from_release,
    prepare_dataset_version,
    read_detect_yaml,
    save_dataset_record,
    validate_dataset_layout,
    write_dataset_yaml,
)
from .exceptions import YoloCtlError
from .manifests import build_manifest, now_utc_iso, write_manifest
from .oss import (
    build_oss_cp_command,
    build_oss_stat_command,
    download_oss_object,
    format_command,
    run_command,
    stat_oss_object,
    upload_oss_object,
)
from .paddledet_runner import run_paddledet_train, run_paddledet_validation
from .paths import dataset_index_path, dataset_manifest_path, project_root, review_session_dir
from .qat import QatConfig, run_qat_experiment
from .quantization import (
    assessment_settings,
    compare_against_baseline,
    load_export_manifest,
    normalize_metrics,
    normalize_performance,
    write_assessment_report,
)
from .review import build_session_summary, create_or_resume_review_session, finalize_review_session, list_review_sessions
from .review_app import create_review_app
from .ultralytics_runner import run_benchmark, run_export, run_model_validation, run_train, run_val
from .vast import lifecycle_action, launch_instance, search_offers, show_instances, volume_action
from .yamlio import dump_yaml, load_yaml


def _print(data: Any) -> None:
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, ensure_ascii=True, sort_keys=False))
    else:
        print(data)


def _build_cloud_paths(prefix: str, dataset_key: str, version_id: str) -> tuple[str, str]:
    clean_prefix = prefix.rstrip("/")
    return (
        f"{clean_prefix}/{dataset_key}/{version_id}/source.zip",
        f"{clean_prefix}/{dataset_key}/{version_id}/manifest.yaml",
    )


def _resolve_cloud_prefix(explicit: str | None) -> str:
    prefix = explicit or os.getenv("YOLOCTL_OSS_DATASET_PREFIX")
    if not prefix:
        raise YoloCtlError(
            "Cloud sync requires --cloud-prefix or YOLOCTL_OSS_DATASET_PREFIX, "
            "for example oss://your-bucket/datasets"
        )
    if not prefix.startswith("oss://"):
        raise YoloCtlError("Cloud prefix must start with oss://")
    return prefix


def _collect_class_names(args: argparse.Namespace) -> list[str]:
    class_names = list(args.class_name or [])
    if getattr(args, "class_names", None):
        class_names.extend([item.strip() for item in args.class_names.split(",") if item.strip()])
    return class_names


def _resolve_dataset_yaml(record: DatasetVersionRecord) -> Path:
    if not record.dataset_yaml_path:
        raise YoloCtlError(f"Dataset version '{record.version_id}' has no dataset_yaml_path")
    dataset_yaml = Path(record.dataset_yaml_path)
    if not dataset_yaml.is_absolute():
        dataset_yaml = project_root() / dataset_yaml
    if not dataset_yaml.exists():
        write_dataset_yaml(record, output_path=dataset_yaml)
    return dataset_yaml


def _load_record_from_args(args: argparse.Namespace) -> DatasetVersionRecord:
    dataset_key = getattr(args, "dataset_key", None)
    if not dataset_key:
        raise YoloCtlError("A dataset_key is required")
    return get_dataset_record(
        dataset_key=dataset_key,
        version_id=getattr(args, "version_id", None),
        version_label=getattr(args, "version_label", None),
    )


def _default_weights_path(run_config: RunConfig) -> str:
    if not run_config.is_ultralytics:
        raise YoloCtlError("Default weights path is only defined for the Ultralytics backend.")
    return str(Path(run_config.default_project_dir()) / run_config.default_run_name() / "weights" / "best.pt")


def _require_ultralytics_backend(run_config: RunConfig, action: str) -> None:
    if not run_config.is_ultralytics:
        raise YoloCtlError(f"{action} is not supported for backend '{run_config.backend}' in this release.")


def _resolve_profile_path(explicit_profile: str | None, run_config: RunConfig | None = None) -> Path:
    candidate = explicit_profile
    if candidate is None and run_config is not None:
        candidate = run_config.tracking.get("vast_profile")
    if not candidate:
        raise YoloCtlError(
            "A Vast profile is required. Provide --profile or set tracking.vast_profile in the run config."
        )
    profile_path = Path(candidate)
    if not profile_path.is_absolute():
        profile_path = project_root() / profile_path
    return profile_path


def handle_dataset_register(args: argparse.Namespace) -> int:
    detect_yaml = Path(args.detect_yaml) if args.detect_yaml else None
    dataset_meta = read_detect_yaml(detect_yaml) if detect_yaml else {}
    materialized_path = args.materialized_path or (str(detect_yaml.parent) if detect_yaml else None)
    class_names = dataset_meta.get("class_names") or _collect_class_names(args)
    splits = dataset_meta.get("splits") or {
        "train": args.train,
        "val": args.val,
        "test": args.test,
    }
    source_archive = args.source_archive or args.bootstrap_archive
    source_type = "bootstrap_archive" if args.bootstrap_archive else ("local_archive" if source_archive else "materialized_local")
    record = create_dataset_record(
        dataset_key=args.dataset_key,
        version_label=args.version_label,
        class_names=class_names,
        splits=splits,
        source_archive_path=source_archive,
        materialized_path=materialized_path,
        source_type=source_type,
        aliases=list(args.alias or []),
        metadata={
            key: value
            for key, value in {
                "notes": args.notes,
                "source_detect_yaml": str(detect_yaml) if detect_yaml else None,
                "bootstrap_archive": args.bootstrap_archive,
            }.items()
            if value
        },
    )
    if args.cloud_prefix:
        cloud_uri, cloud_manifest_uri = _build_cloud_paths(_resolve_cloud_prefix(args.cloud_prefix), record.dataset_key, record.version_id)
        record.cloud_uri = cloud_uri
        record.cloud_manifest_uri = cloud_manifest_uri
    save_dataset_record(record)
    if record.materialized_path:
        dataset_yaml = _resolve_dataset_yaml(record)
    else:
        dataset_yaml = None
    _print(
        {
            "dataset_index": str(dataset_index_path()),
            "dataset_key": record.dataset_key,
            "version_id": record.version_id,
            "version_label": record.version_label,
            "sync_state": record.sync_state,
            "dataset_yaml": str(dataset_yaml) if dataset_yaml else None,
        }
    )
    return 0


def handle_dataset_list(_: argparse.Namespace) -> int:
    _print(load_dataset_index())
    return 0


def handle_dataset_status(args: argparse.Namespace) -> int:
    if not args.dataset_key:
        _print(load_dataset_index())
        return 0
    _print(build_dataset_summary(dataset_key=args.dataset_key, version_id=args.version_id))
    return 0


def handle_dataset_sync_push(args: argparse.Namespace) -> int:
    record = _load_record_from_args(args)
    if not record.source_archive_path:
        raise YoloCtlError(
            f"Dataset version '{record.version_id}' has no source_archive_path. "
            "Zip-first cloud sync requires a local archive."
        )
    prefix = _resolve_cloud_prefix(args.cloud_prefix) if (args.cloud_prefix or os.getenv("YOLOCTL_OSS_DATASET_PREFIX")) else None
    cloud_uri = record.cloud_uri
    cloud_manifest_uri = record.cloud_manifest_uri
    if prefix:
        cloud_uri, cloud_manifest_uri = _build_cloud_paths(prefix, record.dataset_key, record.version_id)
    if not cloud_uri or not cloud_manifest_uri:
        raise YoloCtlError("Dataset version has no cloud target configured. Provide --cloud-prefix.")
    archive_path = Path(record.source_archive_path)
    if not archive_path.is_absolute():
        archive_path = project_root() / archive_path
    if not archive_path.exists():
        raise YoloCtlError(f"Source archive does not exist: {archive_path}")
    preview = {
        "dataset_key": record.dataset_key,
        "version_id": record.version_id,
        "source_archive": str(archive_path),
        "cloud_uri": cloud_uri,
        "cloud_manifest_uri": cloud_manifest_uri,
        "archive_upload_command": format_command(build_oss_cp_command(archive_path, cloud_uri)),
        "manifest_upload_command": format_command(build_oss_cp_command(dataset_manifest_path(record.dataset_key, record.version_id), cloud_manifest_uri)),
        "next_sync_state": "cloud_synced",
    }
    if args.dry_run:
        _print(preview)
        return 0

    record.cloud_uri = cloud_uri
    record.cloud_manifest_uri = cloud_manifest_uri
    record.cloud_sha256 = record.source_archive_sha256
    record.sync_state = "syncing"
    save_dataset_record(record)

    upload_oss_object(archive_path, cloud_uri)
    cloud_synced_at = now_utc_iso()
    final_record = deepcopy(record)
    final_record.sync_state = "cloud_synced"
    final_record.cloud_synced_at = cloud_synced_at
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / f"{record.version_id}.yaml"
        dump_yaml(manifest_path, final_record.to_mapping())
        upload_oss_object(manifest_path, cloud_manifest_uri)

    record.sync_state = "cloud_synced"
    record.cloud_synced_at = cloud_synced_at
    save_dataset_record(record)
    _print(
        {
            "dataset_key": record.dataset_key,
            "version_id": record.version_id,
            "sync_state": record.sync_state,
            "cloud_uri": record.cloud_uri,
            "cloud_manifest_uri": record.cloud_manifest_uri,
        }
    )
    return 0


def handle_dataset_sync_verify(args: argparse.Namespace) -> int:
    record = _load_record_from_args(args)
    if not record.cloud_uri or not record.cloud_manifest_uri:
        raise YoloCtlError(f"Dataset version '{record.version_id}' has no cloud targets to verify")
    preview = {
        "dataset_key": record.dataset_key,
        "version_id": record.version_id,
        "archive_stat_command": format_command(build_oss_stat_command(record.cloud_uri)),
        "manifest_stat_command": format_command(build_oss_stat_command(record.cloud_manifest_uri)),
    }
    if args.dry_run:
        _print(preview)
        return 0

    archive_stat = stat_oss_object(record.cloud_uri)
    manifest_stat = stat_oss_object(record.cloud_manifest_uri)
    if archive_stat.returncode != 0 or manifest_stat.returncode != 0:
        record.sync_state = "cloud_stale"
        save_dataset_record(record)
        _print(
            {
                "dataset_key": record.dataset_key,
                "version_id": record.version_id,
                "sync_state": record.sync_state,
                "archive_returncode": archive_stat.returncode,
                "manifest_returncode": manifest_stat.returncode,
            }
        )
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        local_manifest = Path(tmpdir) / "manifest.yaml"
        download_oss_object(record.cloud_manifest_uri, local_manifest)
        remote_manifest = DatasetVersionRecord.from_mapping(load_yaml(local_manifest))

    remote_sha = remote_manifest.cloud_sha256 or remote_manifest.source_archive_sha256
    local_sha = record.source_archive_sha256
    if remote_manifest.version_id != record.version_id or remote_sha != local_sha:
        record.sync_state = "cloud_verify_failed"
    else:
        record.sync_state = "cloud_synced"
    save_dataset_record(record)
    _print(
        {
            "dataset_key": record.dataset_key,
            "version_id": record.version_id,
            "sync_state": record.sync_state,
            "remote_version_id": remote_manifest.version_id,
            "remote_sha256": remote_sha,
            "local_sha256": local_sha,
        }
    )
    return 0


def handle_dataset_prepare(args: argparse.Namespace) -> int:
    record = _load_record_from_args(args)
    prepared = prepare_dataset_version(
        record,
        output_dir=args.output_dir,
        output_yaml=args.output_yaml,
        allow_cloud=args.allow_cloud,
    )
    _print(
        {
            "dataset_key": prepared.dataset_key,
            "version_id": prepared.version_id,
            "materialized_path": prepared.materialized_path,
            "dataset_yaml_path": prepared.dataset_yaml_path,
            "split_counts": prepared.split_counts,
        }
    )
    return 0


def handle_dataset_validate(args: argparse.Namespace) -> int:
    record = _load_record_from_args(args)
    results = validate_dataset_layout(record, materialized_override=args.materialized_path)
    payload = {
        "dataset_key": record.dataset_key,
        "version_id": record.version_id,
        "version_label": record.version_label,
        "dataset_index": str(dataset_index_path()),
        "dataset_yaml_path": str(_resolve_dataset_yaml(record)),
        "sync_state": record.sync_state,
        "paths": results,
    }
    _print(payload)
    return 0


def handle_dataset_lineage(args: argparse.Namespace) -> int:
    record = _load_record_from_args(args)
    _print(
        {
            "dataset_key": record.dataset_key,
            "version_id": record.version_id,
            "lineage": lineage_for_record(record),
        }
    )
    return 0


def handle_dataset_import_release(args: argparse.Namespace) -> int:
    record = import_release_manifest(
        args.manifest_uri,
        version_label=args.version_label,
        dataset_key=args.dataset_key,
        output_dir=args.output_dir,
        archive_path=args.archive_path,
    )
    _print(
        {
            "dataset_key": record.dataset_key,
            "version_id": record.version_id,
            "version_label": record.version_label,
            "source_type": record.source_type,
            "dataset_yaml_path": record.dataset_yaml_path,
            "release_type": record.metadata.get("release_type"),
            "release_manifest_uri": record.metadata.get("release_manifest_uri"),
        }
    )
    return 0


def handle_dataset_merge_draft(args: argparse.Namespace) -> int:
    if getattr(args, "from_release", None):
        record = merge_draft_from_release(
            args.from_release,
            version_label=args.version_label,
            dataset_key=args.dataset_key,
            merge_note=args.merge_note,
            extra_parent_version_ids=list(args.parent_version_id or []),
        )
    else:
        if not args.dataset_key:
            raise YoloCtlError("dataset merge draft requires --dataset-key unless --from-release is provided")
        record = create_merge_draft(
            dataset_key=args.dataset_key,
            version_label=args.version_label,
            parent_version_ids=list(args.parent_version_id or []),
            merge_note=args.merge_note,
        )
    save_dataset_record(record)
    _print(
        {
            "dataset_key": record.dataset_key,
            "version_id": record.version_id,
            "version_label": record.version_label,
            "lineage_type": record.lineage_type,
            "parents": record.parents,
            "sync_state": record.sync_state,
        }
    )
    return 0


def _review_splits(record: DatasetVersionRecord, explicit: str | None) -> list[str]:
    if not explicit:
        return list(record.splits)
    selected = [item.strip() for item in explicit.split(",") if item.strip()]
    invalid = [item for item in selected if item not in record.splits]
    if invalid:
        raise YoloCtlError(
            f"Review splits {invalid} are not present in dataset version '{record.version_id}'. "
            f"Available splits: {sorted(record.splits)}"
        )
    return selected


def handle_dataset_review_open(args: argparse.Namespace) -> int:
    record = _load_record_from_args(args)
    session_dir, session = create_or_resume_review_session(
        record=record,
        selected_splits=_review_splits(record, args.split),
        weights=args.weights,
        issues_report=args.issues_report,
    )
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - depends on runtime install
        raise YoloCtlError(f"dataset review open requires uvicorn: {exc}") from exc

    app = create_review_app(session_dir)
    url = f"http://{args.host}:{args.port}/"
    _print(
        {
            "dataset_key": record.dataset_key,
            "base_version_id": record.version_id,
            "session_id": session.session_id,
            "session_dir": str(session_dir),
            "url": url,
        }
    )
    try:  # pragma: no cover - best-effort local UX
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="warning")
    return 0


def handle_dataset_review_sessions(args: argparse.Namespace) -> int:
    record = _load_record_from_args(args)
    payload = []
    for session in list_review_sessions(record.dataset_key, record.version_id):
        session_dir = review_session_dir(record.dataset_key, record.version_id, session.session_id)
        summary = build_session_summary(session_dir)
        payload.append(
            {
                "session_id": session.session_id,
                "state": session.state,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "weights": session.weights,
                "issues_report": session.issues_report,
                "selected_splits": session.selected_splits,
                "summary": summary["counts"],
                "session_dir": str(session_dir),
                "finalized_version_id": session.finalized_version_id,
            }
        )
    _print({"dataset_key": record.dataset_key, "base_version_id": record.version_id, "sessions": payload})
    return 0


def handle_dataset_review_finalize(args: argparse.Namespace) -> int:
    record = _load_record_from_args(args)
    session_dir = review_session_dir(record.dataset_key, record.version_id, args.session_id)
    if not session_dir.exists():
        raise YoloCtlError(f"Review session does not exist: {session_dir}")
    finalized = finalize_review_session(
        session_dir=session_dir,
        new_version_label=args.new_version_label,
        merge_note=args.merge_note,
    )
    _print(
        {
            "dataset_key": finalized.dataset_key,
            "version_id": finalized.version_id,
            "version_label": finalized.version_label,
            "materialized_path": finalized.materialized_path,
            "dataset_yaml_path": finalized.dataset_yaml_path,
            "parents": finalized.parents,
            "session_id": args.session_id,
        }
    )
    return 0


def handle_vast_search(args: argparse.Namespace) -> int:
    profile = VastProfile.from_file(Path(args.profile))
    result = search_offers(profile, dry_run=args.dry_run, use_sdk=not args.use_cli, limit=args.limit)
    _print(result)
    return 0


def handle_vast_launch(args: argparse.Namespace) -> int:
    profile = VastProfile.from_file(Path(args.profile))
    result = launch_instance(profile, dry_run=args.dry_run, use_sdk=not args.use_cli, label=args.label)
    _print(result)
    return 0


def handle_vast_lifecycle(args: argparse.Namespace) -> int:
    result = lifecycle_action(args.action, instance_id=args.instance_id, dry_run=args.dry_run, use_sdk=not args.use_cli)
    _print(result)
    return 0


def handle_vast_instances(args: argparse.Namespace) -> int:
    _print(show_instances(dry_run=args.dry_run, use_sdk=not args.use_cli))
    return 0


def handle_vast_advise(args: argparse.Namespace) -> int:
    run_config = RunConfig.from_file(Path(args.config))
    _require_ultralytics_backend(run_config, "vast advise")
    record = get_dataset_record(
        dataset_key=run_config.dataset_key,
        version_id=run_config.dataset_version_id,
        version_label=run_config.dataset_version_label,
    )
    profile = VastProfile.from_file(_resolve_profile_path(args.profile, run_config=run_config))
    if args.dry_run:
        _print(
            {
                "config": str(run_config.source_path),
                "profile": str(profile.source_path) if profile.source_path else profile.name,
                "search_preview": search_offers(profile, dry_run=True, use_sdk=True, limit=args.limit),
                "estimate": summarize_market_estimate(run_config, record, profile, offers=None),
            }
        )
        return 0

    result = search_offers(profile, dry_run=False, use_sdk=True, limit=args.limit)
    offers = result.get("response")
    if not isinstance(offers, list):
        raise YoloCtlError("Cost-aware Vast advice requires SDK search results")
    _print(
        {
            "config": str(run_config.source_path),
            "profile": str(profile.source_path) if profile.source_path else profile.name,
            "market_query": result.get("query"),
            "advice": recommend_offers(run_config, record, profile, offers),
        }
    )
    return 0


def handle_vast_volumes(args: argparse.Namespace) -> int:
    kwargs = {
        "type": args.type,
        "query": args.query,
        "storage": args.storage,
        "order": args.order,
        "offer_id": args.offer_id,
        "size": args.size,
        "name": args.name,
        "volume_id": args.volume_id,
    }
    _print(volume_action(args.volume_action, dry_run=args.dry_run, use_sdk=not args.use_cli, **kwargs))
    return 0


def _load_run_context(config_path: str) -> tuple[RunConfig, DatasetVersionRecord, Path]:
    run_config = RunConfig.from_file(Path(config_path))
    record = get_dataset_record(
        dataset_key=run_config.dataset_key,
        version_id=run_config.dataset_version_id,
        version_label=run_config.dataset_version_label,
    )
    dataset_yaml = _resolve_dataset_yaml(record)
    return run_config, record, dataset_yaml


def _load_qat_context(config_path: str) -> tuple[QatConfig, RunConfig, DatasetVersionRecord, Path]:
    qat_config = QatConfig.from_file(Path(config_path))
    run_config = RunConfig.from_file(qat_config.base_run_config)
    record = get_dataset_record(
        dataset_key=run_config.dataset_key,
        version_id=run_config.dataset_version_id,
        version_label=run_config.dataset_version_label,
    )
    dataset_yaml = _resolve_dataset_yaml(record)
    return qat_config, run_config, record, dataset_yaml


def _artifact_records_from_export_manifest(run_id: str) -> list[dict[str, Any]]:
    manifest = load_export_manifest(run_id)
    result = manifest.get("result", {})
    artifacts = result.get("artifacts", [])
    if artifacts:
        return [dict(item) for item in artifacts]
    outputs = result.get("outputs", [])
    derived: list[dict[str, Any]] = []
    for item in outputs:
        artifact = item.get("artifact")
        if isinstance(artifact, dict) and artifact.get("path"):
            derived.append(dict(artifact))
    if derived:
        return derived
    raise YoloCtlError(
        f"Export manifest for run '{run_id}' does not include structured artifact metadata. "
        "Re-run 'yoloctl export run' with the updated tooling."
    )


def _safe_benchmark(model_path: str, dataset_yaml: Path, run_config: RunConfig, name_suffix: str) -> dict[str, Any]:
    try:
        return run_benchmark(
            model_path=model_path,
            dataset_yaml=dataset_yaml,
            imgsz=run_config.train.get("imgsz"),
            device=run_config.train.get("device"),
            engine="python",
            dry_run=False,
        )
    except Exception as exc:
        return {
            "engine": "python",
            "error": str(exc),
            "model_path": model_path,
            "name_suffix": name_suffix,
        }


def _assess_export_artifacts(
    run_config: RunConfig,
    record: DatasetVersionRecord,
    dataset_yaml: Path,
    weights: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    assessment = assessment_settings(run_config.export)
    artifacts = _artifact_records_from_export_manifest(run_config.run_id)
    if dry_run:
        baseline_validation = run_model_validation(
            model_path=weights,
            dataset_yaml=dataset_yaml,
            imgsz=run_config.train.get("imgsz"),
            batch=run_config.train.get("batch"),
            device=run_config.train.get("device"),
            project=run_config.default_project_dir(),
            name=f"{run_config.default_run_name()}-baseline-val",
            engine="python",
            dry_run=True,
        )
        baseline_benchmark = run_benchmark(
            model_path=weights,
            dataset_yaml=dataset_yaml,
            imgsz=run_config.train.get("imgsz"),
            device=run_config.train.get("device"),
            engine="python",
            dry_run=True,
        )
        previews = []
        for artifact in artifacts:
            previews.append(
                {
                    "artifact": artifact,
                    "validation": run_model_validation(
                        model_path=str(artifact["path"]),
                        dataset_yaml=dataset_yaml,
                        imgsz=run_config.train.get("imgsz"),
                        batch=run_config.train.get("batch"),
                        device=run_config.train.get("device"),
                        project=run_config.default_project_dir(),
                        name=f"{run_config.default_run_name()}-{artifact.get('variant', 'artifact')}-val",
                        engine="python",
                        dry_run=True,
                    ),
                    "benchmark": run_benchmark(
                        model_path=str(artifact["path"]),
                        dataset_yaml=dataset_yaml,
                        imgsz=run_config.train.get("imgsz"),
                        device=run_config.train.get("device"),
                        engine="python",
                        dry_run=True,
                    ),
                }
            )
        return {
            "engine": "preview",
            "dataset": record.to_mapping(),
            "weights": weights,
            "assessment": assessment,
            "baseline": {
                "validation": baseline_validation,
                "benchmark": baseline_benchmark,
            },
            "artifacts": previews,
        }

    baseline_validation = run_model_validation(
        model_path=weights,
        dataset_yaml=dataset_yaml,
        imgsz=run_config.train.get("imgsz"),
        batch=run_config.train.get("batch"),
        device=run_config.train.get("device"),
        project=run_config.default_project_dir(),
        name=f"{run_config.default_run_name()}-baseline-val",
        engine="python",
        dry_run=False,
    )
    baseline_benchmark = _safe_benchmark(weights, dataset_yaml, run_config, "baseline")
    baseline_metrics = normalize_metrics(baseline_validation.get("metrics"))
    baseline_performance = normalize_performance(
        baseline_validation.get("speed"),
        baseline_benchmark.get("results"),
    )

    assessed_artifacts: list[dict[str, Any]] = []
    required_failures = 0
    warnings: list[str] = []
    for artifact in artifacts:
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.exists():
            raise YoloCtlError(
                f"Export artifact is missing for assessment: {artifact_path}. "
                "Re-run 'yoloctl export run' before assessment."
            )
        try:
            validation = run_model_validation(
                model_path=str(artifact_path),
                dataset_yaml=dataset_yaml,
                imgsz=run_config.train.get("imgsz"),
                batch=run_config.train.get("batch"),
                device=run_config.train.get("device"),
                project=run_config.default_project_dir(),
                name=f"{run_config.default_run_name()}-{artifact.get('variant', 'artifact')}-val",
                engine="python",
                dry_run=False,
            )
            metrics = normalize_metrics(validation.get("metrics"))
            benchmark = _safe_benchmark(str(artifact_path), dataset_yaml, run_config, artifact.get("variant", "artifact"))
            performance = normalize_performance(validation.get("speed"), benchmark.get("results"))
            gate = compare_against_baseline(
                baseline_metrics=baseline_metrics,
                candidate_metrics=metrics,
                precision=str(artifact.get("precision", "fp32")),
                settings=assessment,
            )
        except Exception as exc:
            validation = {"error": str(exc), "model_path": str(artifact_path)}
            benchmark = {"skipped": True}
            metrics = {"map50_95": None, "map50": None, "recall": None, "precision": None}
            performance = {"latency_ms": None, "fps": None}
            gate = {
                "status": "rejected",
                "accepted": False,
                "reasons": [f"Validation failed: {exc}"],
                "warnings": [],
                "baseline_delta": {"map50_95": None, "map50": None, "recall": None},
            }
        if gate.get("warnings"):
            warnings.extend([f"{artifact.get('variant', 'artifact')}: {message}" for message in gate["warnings"]])
        if artifact.get("required", assessment["required"]) and not gate["accepted"]:
            required_failures += 1
        assessed_artifacts.append(
            {
                "artifact": artifact,
                "metrics": metrics,
                "performance": performance,
                "gate": gate,
                "validation": validation,
                "benchmark": benchmark,
            }
        )

    summary = {
        "accepted": sum(1 for item in assessed_artifacts if item["gate"]["accepted"]),
        "rejected": sum(1 for item in assessed_artifacts if not item["gate"]["accepted"]),
        "required_failures": required_failures,
        "warnings": warnings,
        "failed": bool(required_failures and assessment["fail_on_reject"]),
    }
    return {
        "engine": "python",
        "dataset": record.to_mapping(),
        "weights": weights,
        "assessment": assessment,
        "baseline": {
            "model_path": weights,
            "metrics": baseline_metrics,
            "performance": baseline_performance,
            "validation": baseline_validation,
            "benchmark": baseline_benchmark,
        },
        "artifacts": assessed_artifacts,
        "summary": summary,
    }


def handle_train_run(args: argparse.Namespace) -> int:
    run_config, record, dataset_yaml = _load_run_context(args.config)
    if run_config.is_paddledet:
        result = run_paddledet_train(
            run_config=run_config,
            dataset_record=record,
            dataset_yaml=dataset_yaml,
            weights=args.weights,
            engine=args.engine,
            dry_run=args.dry_run,
        )
    else:
        result = run_train(
            run_config=run_config,
            dataset_record=record,
            dataset_yaml=dataset_yaml,
            weights=args.weights,
            engine=args.engine,
            resume=False,
            dry_run=args.dry_run,
        )
    manifest = build_manifest(
        stage="train",
        run_id=run_config.run_id,
        command_preview=result.get("command", ""),
        payload={"run_config": str(run_config.source_path), "dataset": record.to_mapping(), "result": result},
    )
    if not args.dry_run:
        write_manifest("train", run_config.run_id, manifest)
    _print(manifest)
    return 0


def handle_train_resume(args: argparse.Namespace) -> int:
    run_config, record, dataset_yaml = _load_run_context(args.config)
    _require_ultralytics_backend(run_config, "train resume")
    checkpoint = args.checkpoint or _default_weights_path(run_config).replace("best.pt", "last.pt")
    result = run_train(
        run_config=run_config,
        dataset_record=record,
        dataset_yaml=dataset_yaml,
        weights=checkpoint,
        engine=args.engine,
        resume=True,
        dry_run=args.dry_run,
    )
    manifest = build_manifest(
        stage="resume",
        run_id=run_config.run_id,
        command_preview=result.get("command", ""),
        payload={"checkpoint": checkpoint, "dataset": record.to_mapping(), "result": result},
    )
    if not args.dry_run:
        write_manifest("resume", run_config.run_id, manifest)
    _print(manifest)
    return 0


def handle_train_val(args: argparse.Namespace) -> int:
    run_config, record, dataset_yaml = _load_run_context(args.config)
    if run_config.is_paddledet:
        weights = args.weights or (run_config.paddledet.eval_weights if run_config.paddledet else None)
        result = run_paddledet_validation(
            run_config=run_config,
            dataset_record=record,
            dataset_yaml=dataset_yaml,
            weights=weights,
            engine=args.engine,
            dry_run=args.dry_run,
        )
    else:
        weights = args.weights or _default_weights_path(run_config)
        result = run_val(
            run_config=run_config,
            dataset_record=record,
            dataset_yaml=dataset_yaml,
            weights=weights,
            engine=args.engine,
            dry_run=args.dry_run,
        )
    manifest = build_manifest(
        stage="val",
        run_id=run_config.run_id,
        command_preview=result.get("command", ""),
        payload={"weights": weights, "dataset": record.to_mapping(), "result": result},
    )
    if not args.dry_run:
        write_manifest("val", run_config.run_id, manifest)
    _print(manifest)
    return 0


def handle_export_run(args: argparse.Namespace) -> int:
    run_config, record, dataset_yaml = _load_run_context(args.config)
    _require_ultralytics_backend(run_config, "export run")
    weights = args.weights or _default_weights_path(run_config)
    result = run_export(
        run_config=run_config,
        dataset_record=record,
        dataset_yaml=dataset_yaml,
        weights=weights,
        engine=args.engine,
        dry_run=args.dry_run,
    )
    preview = result.get("commands", result.get("command", ""))
    manifest = build_manifest(
        stage="export",
        run_id=run_config.run_id,
        command_preview=preview[0] if isinstance(preview, list) and preview else str(preview),
        payload={"weights": weights, "dataset": record.to_mapping(), "result": result},
    )
    if not args.dry_run:
        write_manifest("export", run_config.run_id, manifest)
    _print(manifest)
    return 0


def handle_export_benchmark(args: argparse.Namespace) -> int:
    run_config, record, dataset_yaml = _load_run_context(args.config)
    _require_ultralytics_backend(run_config, "export benchmark")
    model_path = args.model or _default_weights_path(run_config)
    result = run_benchmark(
        model_path=model_path,
        dataset_yaml=dataset_yaml,
        imgsz=args.imgsz or run_config.train.get("imgsz"),
        device=args.device or run_config.train.get("device"),
        engine="auto",
        dry_run=args.dry_run,
    )
    manifest = build_manifest(
        stage="benchmark",
        run_id=run_config.run_id,
        command_preview=result.get("command", ""),
        payload={"model_path": model_path, "dataset": record.to_mapping(), "result": result},
    )
    if not args.dry_run:
        write_manifest("benchmark", run_config.run_id, manifest)
    _print(manifest)
    return 0


def handle_export_assess(args: argparse.Namespace) -> int:
    run_config, record, dataset_yaml = _load_run_context(args.config)
    _require_ultralytics_backend(run_config, "export assess")
    weights = args.weights or _default_weights_path(run_config)
    result = _assess_export_artifacts(
        run_config=run_config,
        record=record,
        dataset_yaml=dataset_yaml,
        weights=weights,
        dry_run=args.dry_run,
    )
    preview = result.get("baseline", {}).get("validation", {}).get("command", "")
    manifest = build_manifest(
        stage="assessment",
        run_id=run_config.run_id,
        command_preview=str(preview),
        payload={"weights": weights, "dataset": record.to_mapping(), "result": result},
    )
    if not args.dry_run:
        write_manifest("assessment", run_config.run_id, manifest)
        report_path = write_assessment_report(run_config.run_id, result)
        manifest["report_path"] = str(report_path)
        write_manifest("assessment", run_config.run_id, manifest)
    _print(manifest)
    if not args.dry_run and result.get("summary", {}).get("failed"):
        return 1
    return 0


def handle_quant_qat_run(args: argparse.Namespace) -> int:
    qat_config, run_config, record, dataset_yaml = _load_qat_context(args.config)
    _require_ultralytics_backend(run_config, "quant qat run")
    weights = args.weights or qat_config.weights or _default_weights_path(run_config)
    result = run_qat_experiment(
        qat_config=qat_config,
        run_config=run_config,
        dataset_record=record,
        dataset_yaml=dataset_yaml,
        weights=weights,
        dry_run=args.dry_run,
    )
    preview = result.get("train", {}) if result.get("engine") == "preview" else result.get("save_dir", "")
    manifest = build_manifest(
        stage="qat_train",
        run_id=qat_config.experiment_id,
        command_preview=json.dumps(preview, ensure_ascii=True, sort_keys=True) if isinstance(preview, dict) else str(preview),
        payload={
            "qat_config": str(qat_config.source_path) if qat_config.source_path else args.config,
            "base_run_config": str(qat_config.base_run_config),
            "weights": weights,
            "dataset": record.to_mapping(),
            "result": result,
        },
    )
    if not args.dry_run:
        write_manifest("qat_train", qat_config.experiment_id, manifest)
    _print(manifest)
    return 0


def handle_cost_snapshot(args: argparse.Namespace) -> int:
    output = Path(args.output) if args.output else None
    result = snapshot_costs(
        output_path=output,
        dry_run=args.dry_run,
        use_sdk=not args.use_cli,
        start_date=args.start_date,
        end_date=args.end_date,
        only_charges=args.only_charges,
        only_credits=args.only_credits,
        instance_label=args.instance_label,
    )
    _print(result)
    return 0


def handle_cost_report(args: argparse.Namespace) -> int:
    directory = Path(args.directory) if args.directory else None
    _print(summarize_snapshots(directory))
    return 0


def handle_cost_cleanup(args: argparse.Namespace) -> int:
    if not args.yes and not args.dry_run:
        raise YoloCtlError("Refusing destructive cleanup without --yes")
    result = cleanup_resources(
        instance_id=args.instance_id,
        volume_id=args.volume_id,
        destroy_instance=args.destroy_instance,
        delete_volume=args.delete_volume,
        dry_run=args.dry_run or not args.yes,
        use_sdk=not args.use_cli,
    )
    _print(result)
    return 0


def handle_cost_estimate(args: argparse.Namespace) -> int:
    run_config = RunConfig.from_file(Path(args.config))
    _require_ultralytics_backend(run_config, "cost estimate")
    record = get_dataset_record(
        dataset_key=run_config.dataset_key,
        version_id=run_config.dataset_version_id,
        version_label=run_config.dataset_version_label,
    )
    profile: VastProfile | None = None
    if args.profile or run_config.tracking.get("vast_profile"):
        profile = VastProfile.from_file(_resolve_profile_path(args.profile, run_config=run_config))

    if args.dry_run:
        payload = summarize_market_estimate(run_config, record, profile, offers=None)
        if profile is not None:
            payload["search_preview"] = search_offers(profile, dry_run=True, use_sdk=True, limit=args.limit)
        _print(payload)
        return 0

    offers: list[dict[str, Any]] | None = None
    if profile is not None:
        result = search_offers(profile, dry_run=False, use_sdk=True, limit=args.limit)
        if not isinstance(result.get("response"), list):
            raise YoloCtlError("Cost estimate requires SDK search results when a profile is provided")
        offers = result["response"]
    _print(summarize_market_estimate(run_config, record, profile, offers=offers))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yoloctl", description="Object detection training control plane")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", help="Manage productized dataset governance")
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)

    register = dataset_sub.add_parser("register", help="Register a local dataset version and create an immutable manifest")
    register.add_argument("--dataset-key", "--dataset-id", dest="dataset_key", required=True)
    register.add_argument("--version-label", "--version", dest="version_label", required=True)
    register.add_argument("--source-archive")
    register.add_argument("--bootstrap-archive")
    register.add_argument("--detect-yaml")
    register.add_argument("--materialized-path")
    register.add_argument("--cloud-prefix")
    register.add_argument("--alias", action="append")
    register.add_argument("--class-name", action="append")
    register.add_argument("--class-names")
    register.add_argument("--train", default="images/train")
    register.add_argument("--val", default="images/val")
    register.add_argument("--test")
    register.add_argument("--notes")
    register.set_defaults(func=handle_dataset_register)

    list_datasets = dataset_sub.add_parser("list", help="Show the dataset index")
    list_datasets.set_defaults(func=handle_dataset_list)

    import_release = dataset_sub.add_parser("import-release", help="Import a published release manifest into a managed dataset version")
    import_release.add_argument("--manifest-uri", required=True)
    import_release.add_argument("--dataset-key")
    import_release.add_argument("--version-label")
    import_release.add_argument("--output-dir")
    import_release.add_argument("--archive-path")
    import_release.set_defaults(func=handle_dataset_import_release)

    status = dataset_sub.add_parser("status", help="Show dataset status and version summaries")
    status.add_argument("--dataset-key")
    status.add_argument("--version-id")
    status.set_defaults(func=handle_dataset_status)

    review = dataset_sub.add_parser("review", help="Review, edit, and finalize local dataset annotations")
    review_sub = review.add_subparsers(dest="review_command", required=True)

    review_open = review_sub.add_parser("open", help="Launch the local annotation review web UI")
    review_open.add_argument("--dataset-key", required=True)
    review_open.add_argument("--version-id")
    review_open.add_argument("--version-label")
    review_open.add_argument("--weights")
    review_open.add_argument("--issues-report")
    review_open.add_argument("--split")
    review_open.add_argument("--host", default="127.0.0.1")
    review_open.add_argument("--port", default=8765, type=int)
    review_open.set_defaults(func=handle_dataset_review_open)

    review_sessions = review_sub.add_parser("sessions", help="List review sessions for a dataset version")
    review_sessions.add_argument("--dataset-key", required=True)
    review_sessions.add_argument("--version-id")
    review_sessions.add_argument("--version-label")
    review_sessions.set_defaults(func=handle_dataset_review_sessions)

    review_finalize = review_sub.add_parser("finalize", help="Create a new dataset version from a review session draft")
    review_finalize.add_argument("--dataset-key", required=True)
    review_finalize.add_argument("--version-id")
    review_finalize.add_argument("--version-label")
    review_finalize.add_argument("--session-id", required=True)
    review_finalize.add_argument("--new-version-label", required=True)
    review_finalize.add_argument("--merge-note")
    review_finalize.set_defaults(func=handle_dataset_review_finalize)

    sync = dataset_sub.add_parser("sync", help="Sync dataset versions to OSS and verify cloud state")
    sync_sub = sync.add_subparsers(dest="sync_command", required=True)

    sync_push = sync_sub.add_parser("push", help="Upload the source zip and version manifest to OSS")
    sync_push.add_argument("--dataset-key", required=True)
    sync_push.add_argument("--version-id")
    sync_push.add_argument("--version-label")
    sync_push.add_argument("--cloud-prefix")
    sync_push.add_argument("--dry-run", action="store_true")
    sync_push.set_defaults(func=handle_dataset_sync_push)

    sync_verify = sync_sub.add_parser("verify", help="Verify that cloud objects exist and match the uploaded manifest checksum")
    sync_verify.add_argument("--dataset-key", required=True)
    sync_verify.add_argument("--version-id")
    sync_verify.add_argument("--version-label")
    sync_verify.add_argument("--dry-run", action="store_true")
    sync_verify.set_defaults(func=handle_dataset_sync_verify)

    prepare = dataset_sub.add_parser("prepare", help="Materialize a dataset version and generate a training YAML")
    prepare.add_argument("--dataset-key", required=True)
    prepare.add_argument("--version-id")
    prepare.add_argument("--version-label")
    prepare.add_argument("--output-dir")
    prepare.add_argument("--output-yaml")
    prepare.add_argument("--allow-cloud", action="store_true")
    prepare.set_defaults(func=handle_dataset_prepare)

    validate = dataset_sub.add_parser("validate", help="Validate a prepared dataset version")
    validate.add_argument("--dataset-key", required=True)
    validate.add_argument("--version-id")
    validate.add_argument("--version-label")
    validate.add_argument("--materialized-path")
    validate.set_defaults(func=handle_dataset_validate)

    lineage = dataset_sub.add_parser("lineage", help="Show dataset version lineage")
    lineage.add_argument("--dataset-key", required=True)
    lineage.add_argument("--version-id")
    lineage.add_argument("--version-label")
    lineage.set_defaults(func=handle_dataset_lineage)

    merge = dataset_sub.add_parser("merge", help="Draft metadata-only merge versions")
    merge_sub = merge.add_subparsers(dest="merge_command", required=True)

    merge_draft = merge_sub.add_parser("draft", help="Create a merge-draft version manifest without merging files")
    merge_draft.add_argument("--dataset-key")
    merge_draft.add_argument("--version-label", required=True)
    merge_draft.add_argument("--parent-version-id", action="append")
    merge_draft.add_argument("--from-release")
    merge_draft.add_argument("--merge-note")
    merge_draft.set_defaults(func=handle_dataset_merge_draft)

    merge_draft_alias = dataset_sub.add_parser("merge-draft", help="Create a merge draft directly from an imported release dataset version")
    merge_draft_alias.add_argument("--dataset-key")
    merge_draft_alias.add_argument("--version-label", required=True)
    merge_draft_alias.add_argument("--parent-version-id", action="append")
    merge_draft_alias.add_argument("--from-release", required=True)
    merge_draft_alias.add_argument("--merge-note")
    merge_draft_alias.set_defaults(func=handle_dataset_merge_draft)

    vast = subparsers.add_parser("vast", help="Manage Vast.ai resources")
    vast_sub = vast.add_subparsers(dest="vast_command", required=True)

    search = vast_sub.add_parser("search", help="Search Vast offers")
    search.add_argument("--profile", required=True)
    search.add_argument("--limit", type=int)
    search.add_argument("--dry-run", action="store_true")
    search.add_argument("--use-cli", action="store_true")
    search.set_defaults(func=handle_vast_search)

    advise = vast_sub.add_parser("advise", help="Rank Vast offers by estimated total training cost for a run config")
    advise.add_argument("--config", required=True)
    advise.add_argument("--profile")
    advise.add_argument("--limit", type=int, default=10)
    advise.add_argument("--dry-run", action="store_true")
    advise.set_defaults(func=handle_vast_advise)

    launch = vast_sub.add_parser("launch", help="Launch the top matching Vast instance")
    launch.add_argument("--profile", required=True)
    launch.add_argument("--label")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--use-cli", action="store_true")
    launch.set_defaults(func=handle_vast_launch)

    for action in ("start", "stop", "destroy"):
        lifecycle = vast_sub.add_parser(action, help=f"{action.capitalize()} a Vast instance")
        lifecycle.add_argument("--instance-id", required=True, type=int)
        lifecycle.add_argument("--dry-run", action="store_true")
        lifecycle.add_argument("--use-cli", action="store_true")
        lifecycle.set_defaults(func=handle_vast_lifecycle, action=action)

    instances = vast_sub.add_parser("instances", help="Show current Vast instances")
    instances.add_argument("--dry-run", action="store_true")
    instances.add_argument("--use-cli", action="store_true")
    instances.set_defaults(func=handle_vast_instances)

    volumes = vast_sub.add_parser("volumes", help="Search, show, create, or delete Vast volumes")
    volumes.add_argument("volume_action", choices=["show", "search", "create", "delete"])
    volumes.add_argument("--type", default="all")
    volumes.add_argument("--query")
    volumes.add_argument("--storage", type=float)
    volumes.add_argument("--order", default="score-")
    volumes.add_argument("--offer-id", type=int)
    volumes.add_argument("--size", type=float)
    volumes.add_argument("--name")
    volumes.add_argument("--volume-id", type=int)
    volumes.add_argument("--dry-run", action="store_true")
    volumes.add_argument("--use-cli", action="store_true")
    volumes.set_defaults(func=handle_vast_volumes)

    train = subparsers.add_parser("train", help="Run training and validation for supported backends")
    train_sub = train.add_subparsers(dest="train_command", required=True)

    train_run = train_sub.add_parser("run", help="Run training from a run config")
    train_run.add_argument("--config", required=True)
    train_run.add_argument("--weights")
    train_run.add_argument("--engine", choices=["auto", "python", "cli"], default="auto")
    train_run.add_argument("--dry-run", action="store_true")
    train_run.set_defaults(func=handle_train_run)

    train_resume = train_sub.add_parser("resume", help="Resume training from a checkpoint")
    train_resume.add_argument("--config", required=True)
    train_resume.add_argument("--checkpoint")
    train_resume.add_argument("--engine", choices=["auto", "python", "cli"], default="auto")
    train_resume.add_argument("--dry-run", action="store_true")
    train_resume.set_defaults(func=handle_train_resume)

    train_val = train_sub.add_parser("val", help="Validate a trained checkpoint")
    train_val.add_argument("--config", required=True)
    train_val.add_argument("--weights")
    train_val.add_argument("--engine", choices=["auto", "python", "cli"], default="auto")
    train_val.add_argument("--dry-run", action="store_true")
    train_val.set_defaults(func=handle_train_val)

    export = subparsers.add_parser("export", help="Run Ultralytics export and benchmark flows")
    export_sub = export.add_subparsers(dest="export_command", required=True)

    export_run = export_sub.add_parser("run", help="Export trained weights to multiple formats")
    export_run.add_argument("--config", required=True)
    export_run.add_argument("--weights")
    export_run.add_argument("--engine", choices=["auto", "python", "cli"], default="auto")
    export_run.add_argument("--dry-run", action="store_true")
    export_run.set_defaults(func=handle_export_run)

    benchmark = export_sub.add_parser("benchmark", help="Run the official Ultralytics benchmark command")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--model")
    benchmark.add_argument("--imgsz", type=int)
    benchmark.add_argument("--device")
    benchmark.add_argument("--dry-run", action="store_true")
    benchmark.set_defaults(func=handle_export_benchmark)

    assess = export_sub.add_parser("assess", help="Validate exported artifacts against the baseline checkpoint")
    assess.add_argument("--config", required=True)
    assess.add_argument("--weights")
    assess.add_argument("--dry-run", action="store_true")
    assess.set_defaults(func=handle_export_assess)

    quant = subparsers.add_parser("quant", help="Run experimental quantization workflows")
    quant_sub = quant.add_subparsers(dest="quant_command", required=True)

    qat = quant_sub.add_parser("qat", help="Experimental QAT workflows")
    qat_sub = qat.add_subparsers(dest="qat_command", required=True)

    qat_run = qat_sub.add_parser("run", help="Run an experimental QAT fine-tuning lane")
    qat_run.add_argument("--config", required=True)
    qat_run.add_argument("--weights")
    qat_run.add_argument("--dry-run", action="store_true")
    qat_run.set_defaults(func=handle_quant_qat_run)

    cost = subparsers.add_parser("cost", help="Capture cost snapshots and guarded cleanup commands")
    cost_sub = cost.add_subparsers(dest="cost_command", required=True)

    snapshot = cost_sub.add_parser("snapshot", help="Capture a Vast invoice snapshot")
    snapshot.add_argument("--output")
    snapshot.add_argument("--start-date")
    snapshot.add_argument("--end-date")
    snapshot.add_argument("--instance-label")
    snapshot.add_argument("--only-charges", action="store_true")
    snapshot.add_argument("--only-credits", action="store_true")
    snapshot.add_argument("--dry-run", action="store_true")
    snapshot.add_argument("--use-cli", action="store_true")
    snapshot.set_defaults(func=handle_cost_snapshot)

    estimate = cost_sub.add_parser("estimate", help="Estimate training workload and compare likely market cost")
    estimate.add_argument("--config", required=True)
    estimate.add_argument("--profile")
    estimate.add_argument("--limit", type=int, default=10)
    estimate.add_argument("--dry-run", action="store_true")
    estimate.set_defaults(func=handle_cost_estimate)

    report = cost_sub.add_parser("report", help="Summarize saved cost snapshots")
    report.add_argument("--directory")
    report.set_defaults(func=handle_cost_report)

    cleanup = cost_sub.add_parser("cleanup", help="Destroy instances and delete volumes with an explicit guard")
    cleanup.add_argument("--instance-id", type=int)
    cleanup.add_argument("--volume-id", type=int)
    cleanup.add_argument("--destroy-instance", action="store_true")
    cleanup.add_argument("--delete-volume", action="store_true")
    cleanup.add_argument("--dry-run", action="store_true")
    cleanup.add_argument("--yes", action="store_true")
    cleanup.add_argument("--use-cli", action="store_true")
    cleanup.set_defaults(func=handle_cost_cleanup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except YoloCtlError as exc:
        parser.error(str(exc))
    return 2

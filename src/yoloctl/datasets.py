from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.parse import urlparse
import zipfile

from .exceptions import YoloCtlError
from .paths import (
    dataset_index_path,
    dataset_manifest_path,
    dataset_registry_dir,
    default_dataset_yaml_path,
    ensure_parent,
    project_root,
)
from .yamlio import dump_yaml, load_yaml

SYNC_STATES = {
    "local_only",
    "syncing",
    "cloud_synced",
    "cloud_verify_failed",
    "cloud_stale",
}
SOURCE_TYPES = {
    "bootstrap_archive",
    "local_archive",
    "materialized_local",
    "merge_draft",
    "release_manifest",
}
LINEAGE_TYPES = {
    "root",
    "merge",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetVersionRecord:
    dataset_key: str
    version_id: str
    version_label: str
    task: str
    source_type: str
    sync_state: str
    class_names: list[str]
    splits: dict[str, str]
    split_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    source_archive_path: str | None = None
    source_archive_sha256: str | None = None
    source_archive_size: int | None = None
    cloud_uri: str | None = None
    cloud_manifest_uri: str | None = None
    cloud_sha256: str | None = None
    cloud_synced_at: str | None = None
    materialized_path: str | None = None
    dataset_yaml_path: str | None = None
    parents: list[str] = field(default_factory=list)
    lineage_type: str = "root"
    merge_note: str | None = None
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasetVersionRecord":
        return cls(
            dataset_key=str(payload["dataset_key"]),
            version_id=str(payload["version_id"]),
            version_label=str(payload["version_label"]),
            task=str(payload.get("task", "detect")),
            source_type=str(payload.get("source_type", "local_archive")),
            sync_state=str(payload.get("sync_state", "local_only")),
            class_names=list(payload.get("class_names", [])),
            splits=dict(payload.get("splits", {})),
            split_counts={
                str(split): {str(key): int(value) for key, value in counts.items()}
                for split, counts in dict(payload.get("split_counts", {})).items()
            },
            checksums=dict(payload.get("checksums", {})),
            source_archive_path=str(payload["source_archive_path"]) if payload.get("source_archive_path") else None,
            source_archive_sha256=str(payload["source_archive_sha256"]) if payload.get("source_archive_sha256") else None,
            source_archive_size=int(payload["source_archive_size"]) if payload.get("source_archive_size") is not None else None,
            cloud_uri=str(payload["cloud_uri"]) if payload.get("cloud_uri") else None,
            cloud_manifest_uri=str(payload["cloud_manifest_uri"]) if payload.get("cloud_manifest_uri") else None,
            cloud_sha256=str(payload["cloud_sha256"]) if payload.get("cloud_sha256") else None,
            cloud_synced_at=str(payload["cloud_synced_at"]) if payload.get("cloud_synced_at") else None,
            materialized_path=str(payload["materialized_path"]) if payload.get("materialized_path") else None,
            dataset_yaml_path=str(payload["dataset_yaml_path"]) if payload.get("dataset_yaml_path") else None,
            parents=[str(item) for item in payload.get("parents", [])],
            lineage_type=str(payload.get("lineage_type", "root")),
            merge_note=str(payload["merge_note"]) if payload.get("merge_note") else None,
            aliases=[str(item) for item in payload.get("aliases", [])],
            metadata=dict(payload.get("metadata", {})),
        )

    def validate(self) -> None:
        if self.task != "detect":
            raise YoloCtlError("Only detect datasets are supported in v1")
        if self.sync_state not in SYNC_STATES:
            raise YoloCtlError(f"Unsupported sync state '{self.sync_state}'")
        if self.source_type not in SOURCE_TYPES:
            raise YoloCtlError(f"Unsupported source type '{self.source_type}'")
        if self.lineage_type not in LINEAGE_TYPES:
            raise YoloCtlError(f"Unsupported lineage type '{self.lineage_type}'")
        if not self.dataset_key:
            raise YoloCtlError("Dataset version must include dataset_key")
        if not self.version_id:
            raise YoloCtlError("Dataset version must include version_id")
        if not self.version_label:
            raise YoloCtlError("Dataset version must include version_label")
        if not self.class_names:
            raise YoloCtlError("Dataset version must include at least one class name")
        required_splits = {"val"}
        if self.metadata.get("dataset_role") != "holdout_test":
            required_splits.add("train")
        for split_name in sorted(required_splits):
            if split_name not in self.splits:
                raise YoloCtlError(f"Dataset splits must include '{split_name}'")
        for uri in (self.cloud_uri, self.cloud_manifest_uri):
            if uri and not uri.startswith("oss://"):
                raise YoloCtlError("Cloud URIs must start with oss://")
        if self.source_type == "merge_draft" and not self.parents:
            raise YoloCtlError("Merge draft versions must declare parent version IDs")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def names_mapping(self) -> dict[int, str]:
        return {index: name for index, name in enumerate(self.class_names)}


def _slug(value: str) -> str:
    parts = []
    for char in value.strip().lower():
        if char.isalnum():
            parts.append(char)
        elif parts and parts[-1] != "-":
            parts.append("-")
    slug = "".join(parts).strip("-")
    if not slug:
        raise YoloCtlError("Expected a non-empty slug-compatible value")
    return slug


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def generate_version_id(version_label: str) -> str:
    return f"dsv-{_utc_stamp()}-{_slug(version_label)}"


def _portable_path(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    root = project_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _absolute_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_detect_yaml(path: Path) -> dict[str, Any]:
    payload = load_yaml(path)
    names = payload.get("names", {})
    if isinstance(names, dict):
        ordered_names = [str(name) for _, name in sorted(names.items(), key=lambda item: int(item[0]))]
    elif isinstance(names, list):
        ordered_names = [str(name) for name in names]
    else:
        raise YoloCtlError(f"Unsupported names payload in detect yaml: {path}")
    splits = {
        "train": str(payload["train"]),
        "val": str(payload["val"]),
    }
    if payload.get("test"):
        splits["test"] = str(payload["test"])
    return {
        "class_names": ordered_names,
        "splits": splits,
    }


def count_split_entries(materialized_path: Path, splits: dict[str, str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split_name, relative in splits.items():
        split_path = materialized_path / relative
        image_count = 0
        label_count = 0
        if split_path.exists():
            image_count = sum(1 for path in split_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        label_relative = relative.replace("images/", "labels/", 1) if relative.startswith("images/") else relative
        label_path = materialized_path / label_relative
        if label_path.exists():
            label_count = sum(1 for path in label_path.rglob("*.txt") if path.is_file())
        counts[split_name] = {
            "images": image_count,
            "labels": label_count,
        }
    return counts


def empty_dataset_index() -> dict[str, Any]:
    return {"version": 1, "datasets": {}}


def version_manifest_paths() -> list[Path]:
    registry = dataset_registry_dir()
    if not registry.exists():
        return []
    return sorted(
        path
        for path in registry.glob("*/*.yaml")
        if path.name != "index.yaml"
    )


def load_dataset_index() -> dict[str, Any]:
    path = dataset_index_path()
    if not path.exists():
        return rebuild_dataset_index(save=False)
    data = load_yaml(path)
    data.setdefault("version", 1)
    data.setdefault("datasets", {})
    return data


def save_dataset_index(data: dict[str, Any]) -> Path:
    return dump_yaml(dataset_index_path(), data)


def load_dataset_record_from_path(path: Path) -> DatasetVersionRecord:
    return DatasetVersionRecord.from_mapping(load_yaml(path))


def rebuild_dataset_index(save: bool = True) -> dict[str, Any]:
    index = empty_dataset_index()
    grouped: dict[str, list[DatasetVersionRecord]] = {}
    for path in version_manifest_paths():
        record = load_dataset_record_from_path(path)
        grouped.setdefault(record.dataset_key, []).append(record)
    for dataset_key, records in sorted(grouped.items()):
        records.sort(key=lambda item: item.version_id)
        latest = records[-1]
        index["datasets"][dataset_key] = {
            "dataset_key": dataset_key,
            "aliases": list(latest.aliases),
            "latest_version_id": latest.version_id,
            "latest_version_label": latest.version_label,
            "latest_sync_state": latest.sync_state,
            "version_ids": [record.version_id for record in records],
            "version_labels": [record.version_label for record in records],
            "version_count": len(records),
        }
    if save:
        save_dataset_index(index)
    return index


def save_dataset_record(record: DatasetVersionRecord) -> Path:
    record.validate()
    path = dataset_manifest_path(record.dataset_key, record.version_id)
    ensure_parent(path)
    dump_yaml(path, record.to_mapping())
    rebuild_dataset_index(save=True)
    return path


def list_dataset_records(dataset_key: str | None = None) -> list[DatasetVersionRecord]:
    records = [load_dataset_record_from_path(path) for path in version_manifest_paths()]
    if dataset_key:
        records = [record for record in records if record.dataset_key == dataset_key]
    return sorted(records, key=lambda item: (item.dataset_key, item.version_id))


def get_dataset_record(
    dataset_key: str,
    version_id: str | None = None,
    version_label: str | None = None,
) -> DatasetVersionRecord:
    matches = [record for record in list_dataset_records(dataset_key) if record.dataset_key == dataset_key]
    if not matches:
        raise YoloCtlError(f"Dataset '{dataset_key}' is not registered")
    if version_id:
        for record in matches:
            if record.version_id == version_id:
                return record
        raise YoloCtlError(f"Dataset '{dataset_key}' does not include version_id '{version_id}'")
    if version_label:
        filtered = [record for record in matches if record.version_label == version_label]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            raise YoloCtlError(
                f"Dataset '{dataset_key}' has multiple versions with label '{version_label}'. "
                "Use dataset_version_id instead."
            )
        raise YoloCtlError(f"Dataset '{dataset_key}' does not include version_label '{version_label}'")
    matches.sort(key=lambda item: item.version_id)
    return matches[-1]


def find_dataset_record_by_version_id(version_id: str) -> DatasetVersionRecord:
    for record in list_dataset_records():
        if record.version_id == version_id:
            return record
    raise YoloCtlError(f"No dataset version found for version_id '{version_id}'")


def build_dataset_summary(dataset_key: str | None = None, version_id: str | None = None) -> dict[str, Any]:
    index = load_dataset_index()
    if dataset_key is None:
        return index
    if dataset_key not in index["datasets"]:
        raise YoloCtlError(f"Dataset '{dataset_key}' is not registered")
    summary = {
        "summary": index["datasets"][dataset_key],
        "versions": [record.to_mapping() for record in list_dataset_records(dataset_key)],
    }
    if version_id:
        summary["version"] = get_dataset_record(dataset_key, version_id=version_id).to_mapping()
    return summary


def create_dataset_record(
    dataset_key: str,
    version_label: str,
    class_names: list[str],
    splits: dict[str, str],
    version_id: str | None = None,
    source_archive_path: str | None = None,
    materialized_path: str | None = None,
    dataset_yaml_path: str | None = None,
    source_type: str = "local_archive",
    aliases: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    cloud_uri: str | None = None,
    cloud_manifest_uri: str | None = None,
    parents: list[str] | None = None,
    lineage_type: str = "root",
    merge_note: str | None = None,
) -> DatasetVersionRecord:
    normalized_key = _slug(dataset_key)
    version_id = version_id or generate_version_id(version_label)
    archive_path = _absolute_path(source_archive_path)
    materialized = _absolute_path(materialized_path)
    if archive_path and not archive_path.exists():
        raise YoloCtlError(f"Source archive does not exist: {archive_path}")
    if materialized is None and archive_path:
        materialized = project_root() / "artifacts" / "datasets" / normalized_key / version_id
    counts = count_split_entries(materialized, splits) if materialized and materialized.exists() else {}
    yaml_path = dataset_yaml_path or default_dataset_yaml_path(normalized_key, version_id)
    record = DatasetVersionRecord(
        dataset_key=normalized_key,
        version_id=version_id,
        version_label=version_label,
        task="detect",
        source_type=source_type,
        sync_state="local_only",
        class_names=class_names,
        splits=splits,
        split_counts=counts,
        source_archive_path=_portable_path(archive_path),
        source_archive_sha256=file_sha256(archive_path) if archive_path else None,
        source_archive_size=archive_path.stat().st_size if archive_path else None,
        cloud_uri=cloud_uri,
        cloud_manifest_uri=cloud_manifest_uri,
        materialized_path=_portable_path(materialized),
        dataset_yaml_path=_portable_path(yaml_path),
        parents=parents or [],
        lineage_type=lineage_type,
        merge_note=merge_note,
        aliases=aliases or [],
        metadata=metadata or {},
    )
    record.validate()
    return record


def write_dataset_yaml(
    record: DatasetVersionRecord,
    materialized_override: str | None = None,
    output_path: Path | None = None,
) -> Path:
    materialized = _absolute_path(materialized_override or record.materialized_path)
    if materialized is None:
        raise YoloCtlError("Dataset version has no materialized_path. Run dataset prepare first.")
    path = output_path or _absolute_path(record.dataset_yaml_path)
    if path is None:
        raise YoloCtlError("Dataset version has no dataset_yaml_path")
    payload: dict[str, Any] = {
        "path": str(materialized),
        "val": record.splits["val"],
        "names": record.names_mapping(),
    }
    if record.splits.get("train"):
        payload["train"] = record.splits["train"]
    if record.splits.get("test"):
        payload["test"] = record.splits["test"]
    ensure_parent(path)
    return dump_yaml(path, payload)


def validate_dataset_layout(
    record: DatasetVersionRecord,
    materialized_override: str | None = None,
) -> dict[str, bool]:
    materialized = _absolute_path(materialized_override or record.materialized_path)
    if materialized is None:
        return {split: False for split in record.splits}
    results: dict[str, bool] = {}
    for split_name, relative in record.splits.items():
        results[split_name] = (materialized / relative).exists()
    return results


def prepare_dataset_version(
    record: DatasetVersionRecord,
    output_dir: str | None = None,
    output_yaml: str | None = None,
    allow_cloud: bool = False,
) -> DatasetVersionRecord:
    materialized = _absolute_path(output_dir or record.materialized_path)
    if materialized is None:
        materialized = project_root() / "artifacts" / "datasets" / record.dataset_key / record.version_id
    archive_path = _absolute_path(record.source_archive_path)
    if archive_path and archive_path.exists():
        extract_archive(archive_path, materialized)
    elif allow_cloud and record.cloud_uri:
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = Path(tmpdir) / "source.zip"
            from .oss import download_oss_object

            download_oss_object(record.cloud_uri, downloaded)
            extract_archive(downloaded, materialized)
    elif not materialized.exists():
        raise YoloCtlError(
            f"Dataset version '{record.version_id}' has no local archive to prepare from and materialized data is missing."
        )
    record.materialized_path = _portable_path(materialized)
    record.split_counts = count_split_entries(materialized, record.splits)
    record.dataset_yaml_path = _portable_path(output_yaml or record.dataset_yaml_path or default_dataset_yaml_path(record.dataset_key, record.version_id))
    write_dataset_yaml(record, output_path=_absolute_path(record.dataset_yaml_path))
    save_dataset_record(record)
    return record


def extract_archive(source_archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_archive) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts
            if "__MACOSX" in parts:
                continue
            archive.extract(member, destination)
    macosx = destination / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)
    return destination


def lineage_for_record(record: DatasetVersionRecord) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    visited: set[str] = set()

    def visit(version_record: DatasetVersionRecord, depth: int) -> None:
        if version_record.version_id in visited:
            return
        visited.add(version_record.version_id)
        lineage.append(
            {
                "depth": depth,
                "dataset_key": version_record.dataset_key,
                "version_id": version_record.version_id,
                "version_label": version_record.version_label,
                "lineage_type": version_record.lineage_type,
                "sync_state": version_record.sync_state,
                "parents": list(version_record.parents),
            }
        )
        for parent_id in version_record.parents:
            visit(find_dataset_record_by_version_id(parent_id), depth + 1)

    visit(record, 0)
    return lineage


def create_merge_draft(
    dataset_key: str,
    version_label: str,
    parent_version_ids: list[str],
    merge_note: str | None = None,
) -> DatasetVersionRecord:
    if not parent_version_ids:
        raise YoloCtlError("Merge draft requires at least one parent version ID")
    normalized_key = _slug(dataset_key)
    parents = [find_dataset_record_by_version_id(version_id) for version_id in parent_version_ids]
    base = parents[0]
    for parent in parents[1:]:
        if parent.dataset_key != normalized_key:
            raise YoloCtlError("Merge draft currently requires all parents to belong to the same dataset_key")
        if parent.class_names != base.class_names:
            raise YoloCtlError("Merge draft currently requires matching class_names across parents")
        if parent.splits != base.splits:
            raise YoloCtlError("Merge draft currently requires matching splits across parents")
    record = create_dataset_record(
        dataset_key=normalized_key,
        version_label=version_label,
        class_names=list(base.class_names),
        splits=dict(base.splits),
        source_type="merge_draft",
        materialized_path=None,
        parents=parent_version_ids,
        lineage_type="merge",
        merge_note=merge_note,
        metadata={
            "merge_inputs": [
                {
                    "version_id": parent.version_id,
                    "version_label": parent.version_label,
                    "sync_state": parent.sync_state,
                    "split_counts": parent.split_counts,
                }
                for parent in parents
            ]
        },
    )
    record.sync_state = "local_only"
    return record


def _release_manifest_path(manifest_uri: str) -> Path:
    if manifest_uri.startswith("oss://"):
        with tempfile.TemporaryDirectory() as tmpdir:
            from .oss import download_oss_object

            downloaded = Path(tmpdir) / Path(urlparse(manifest_uri).path).name
            download_oss_object(manifest_uri, downloaded)
            payload = downloaded.read_text(encoding="utf-8")
        target = Path(tempfile.mkdtemp(prefix="release_manifest_")) / "manifest.yaml"
        target.write_text(payload, encoding="utf-8")
        return target
    path = Path(manifest_uri).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        raise YoloCtlError(f"Release manifest does not exist: {path}")
    return path


def load_release_manifest(manifest_uri: str) -> dict[str, Any]:
    manifest_path = _release_manifest_path(manifest_uri)
    raw = manifest_path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return load_yaml(manifest_path)


def _copy_release_image(source_uri: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_uri.startswith("oss://"):
        from .oss import download_oss_object

        download_oss_object(source_uri, destination)
        return
    source_path = Path(source_uri).expanduser()
    if not source_path.is_absolute():
        source_path = project_root() / source_path
    if not source_path.exists():
        raise YoloCtlError(f"Release image source does not exist: {source_path}")
    shutil.copy2(source_path, destination)


def _yolo_label_line(detection: dict[str, Any], class_names: list[str]) -> str:
    class_name = str(detection.get("class_name", ""))
    class_id = detection.get("class_id", -1)
    if class_id in (None, -1):
        try:
            class_id = class_names.index(class_name)
        except ValueError as exc:
            raise YoloCtlError(f"Detection class '{class_name}' is not present in class_names") from exc
    bbox = dict(detection.get("bbox", {}))
    return " ".join(
        [
            str(int(class_id)),
            f"{float(bbox.get('x', 0.0)):.6f}",
            f"{float(bbox.get('y', 0.0)):.6f}",
            f"{float(bbox.get('width', 0.0)):.6f}",
            f"{float(bbox.get('height', 0.0)):.6f}",
        ]
    )


def archive_directory(source_dir: Path, archive_path: Path) -> Path:
    ensure_parent(archive_path)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))
    return archive_path


def import_release_manifest(
    manifest_uri: str,
    *,
    version_label: str | None = None,
    dataset_key: str | None = None,
    output_dir: str | None = None,
    archive_path: str | None = None,
    cloud_uri: str | None = None,
    cloud_manifest_uri: str | None = None,
) -> DatasetVersionRecord:
    manifest = load_release_manifest(manifest_uri)
    release_id = str(manifest.get("release_id") or manifest.get("release_version") or "release")
    release_type = str(manifest.get("release_type") or "incremental_train")
    class_names = [str(item) for item in manifest.get("class_names", [])]
    if not class_names:
        raise YoloCtlError("Release manifest must include class_names")

    normalized_key = _slug(dataset_key or str(manifest.get("dataset_key") or "release-dataset"))
    resolved_label = version_label or str(manifest.get("release_version") or release_id)
    version_id = generate_version_id(resolved_label)
    materialized = Path(output_dir) if output_dir else project_root() / "artifacts" / "datasets" / normalized_key / version_id
    if not materialized.is_absolute():
        materialized = project_root() / materialized
    if materialized.exists():
        shutil.rmtree(materialized)
    materialized.mkdir(parents=True, exist_ok=True)

    images_root = materialized / "images"
    labels_root = materialized / "labels"
    split_names: set[str] = set()

    items = list(manifest.get("items", []))
    for index, item in enumerate(items):
        split = str(item.get("split") or ("val" if release_type == "holdout_test" else "train"))
        split_names.add(split)
        source_uri = str(item.get("oss_uri") or item.get("source_uri") or item.get("image_path") or "")
        if not source_uri:
            raise YoloCtlError("Each release manifest item must include oss_uri or source_uri")
        source_name = Path(urlparse(source_uri).path).name if source_uri.startswith("oss://") else Path(source_uri).name
        suffix = Path(source_name).suffix or ".jpg"
        target_name = f"{item.get('image_id', index)}{suffix}"
        image_path = images_root / split / target_name
        label_path = labels_root / split / f"{Path(target_name).stem}.txt"
        _copy_release_image(source_uri, image_path)
        ensure_parent(label_path)
        label_lines = [
            _yolo_label_line(detection, class_names)
            for detection in item.get("detections", [])
        ]
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

    splits = {}
    if release_type == "holdout_test":
        splits["val"] = "images/val"
        if "test" in split_names:
            splits["test"] = "images/test"
        else:
            splits["test"] = "images/val"
    else:
        splits["train"] = "images/train"
        splits["val"] = "images/val" if "val" in split_names else "images/train"
        if "test" in split_names:
            splits["test"] = "images/test"

    archive_target = Path(archive_path) if archive_path else project_root() / "artifacts" / "datasets" / "archives" / normalized_key / f"{version_id}.zip"
    if not archive_target.is_absolute():
        archive_target = project_root() / archive_target
    archive_directory(materialized, archive_target)

    record = create_dataset_record(
        dataset_key=normalized_key,
        version_label=resolved_label,
        version_id=version_id,
        class_names=class_names,
        splits=splits,
        source_archive_path=str(archive_target),
        materialized_path=str(materialized),
        source_type="release_manifest",
        cloud_uri=cloud_uri,
        cloud_manifest_uri=cloud_manifest_uri,
        metadata={
            "dataset_role": release_type,
            "release_id": release_id,
            "release_type": release_type,
            "release_manifest_uri": manifest_uri,
            "parent_dataset_version_id": manifest.get("parent_version_id"),
            "source_case_ids": manifest.get("source_case_ids", []),
            "source_ground_truth_ids": manifest.get("source_ground_truth_ids", []),
            "source_sync_batch_ids": manifest.get("source_sync_batch_ids", []),
        },
    )
    write_dataset_yaml(record)
    save_dataset_record(record)
    return record


def merge_draft_from_release(
    release_version_id: str,
    *,
    version_label: str,
    dataset_key: str | None = None,
    merge_note: str | None = None,
    extra_parent_version_ids: list[str] | None = None,
) -> DatasetVersionRecord:
    release_record = find_dataset_record_by_version_id(release_version_id)
    parent_ids: list[str] = []
    parent_from_manifest = release_record.metadata.get("parent_dataset_version_id")
    if parent_from_manifest:
        parent_ids.append(str(parent_from_manifest))
    parent_ids.append(release_record.version_id)
    for parent_id in extra_parent_version_ids or []:
        if parent_id not in parent_ids:
            parent_ids.append(parent_id)
    return create_merge_draft(
        dataset_key=dataset_key or release_record.dataset_key,
        version_label=version_label,
        parent_version_ids=parent_ids,
        merge_note=merge_note,
    )

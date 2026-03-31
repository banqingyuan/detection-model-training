from __future__ import annotations

import base64
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any
from json import JSONDecodeError

from PIL import Image

from .datasets import (
    DatasetVersionRecord,
    IMAGE_SUFFIXES,
    create_dataset_record,
    get_dataset_record,
    save_dataset_record,
    write_dataset_yaml,
)
from .exceptions import YoloCtlError
from .paths import project_root, review_dataset_dir, review_session_dir, reviews_dir
from .yamlio import dump_yaml

REVIEW_STATUSES = {"todo", "reviewed_ok", "fixed", "needs_followup"}
DEFAULT_REVIEW_STATUS = "todo"
PREDICTION_CONFIDENCE = 0.25
PREDICTION_IOU = 0.7
MATCH_IOU = 0.5
FOCUS_BOX_PADDING = 24.0
PREDICTION_CACHE_VERSION = 2
MODEL_CACHE: dict[str, Any] = {}
MODEL_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    parts: list[str] = []
    for char in value.strip().lower():
        if char.isalnum():
            parts.append(char)
        elif parts and parts[-1] != "-":
            parts.append("-")
    slug = "".join(parts).strip("-")
    if not slug:
        raise YoloCtlError("Expected a non-empty slug-compatible value")
    return slug


def _portable_path(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    root = project_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _absolute_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser() if not isinstance(path_value, Path) else path_value.expanduser()
    if path.is_absolute():
        return path
    return project_root() / path


def encode_image_id(split: str, relative_path: Path) -> str:
    payload = f"{split}|{relative_path.as_posix()}".encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_image_id(image_id: str) -> tuple[str, Path]:
    padding = "=" * ((4 - len(image_id) % 4) % 4)
    raw = base64.urlsafe_b64decode(f"{image_id}{padding}".encode("ascii")).decode("utf-8")
    split, relative = raw.split("|", 1)
    return split, Path(relative)


@dataclass
class ReviewSession:
    session_id: str
    dataset_key: str
    base_version_id: str
    base_version_label: str
    created_at: str
    updated_at: str
    selected_splits: list[str]
    state: str = "active"
    weights: str | None = None
    issues_report: str | None = None
    prediction_index: dict[str, Any] = field(default_factory=dict)
    finalized_version_id: str | None = None
    finalized_at: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ReviewSession":
        return cls(
            session_id=str(payload["session_id"]),
            dataset_key=str(payload["dataset_key"]),
            base_version_id=str(payload["base_version_id"]),
            base_version_label=str(payload.get("base_version_label", "")),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            selected_splits=[str(item) for item in payload.get("selected_splits", [])],
            state=str(payload.get("state", "active")),
            weights=str(payload["weights"]) if payload.get("weights") else None,
            issues_report=str(payload["issues_report"]) if payload.get("issues_report") else None,
            prediction_index=dict(payload.get("prediction_index", {})),
            finalized_version_id=str(payload["finalized_version_id"]) if payload.get("finalized_version_id") else None,
            finalized_at=str(payload["finalized_at"]) if payload.get("finalized_at") else None,
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewItem:
    image_id: str
    split: str
    image_path: str
    label_path: str
    draft_label_path: str
    status: str = DEFAULT_REVIEW_STATUS
    note: str = ""
    has_edits: bool = False
    issue_summary: dict[str, Any] = field(default_factory=dict)
    baseline_issue_summary: dict[str, Any] = field(default_factory=dict)
    draft_issue_summary: dict[str, Any] = field(default_factory=dict)
    gt_class_names: list[str] = field(default_factory=list)
    pred_class_names: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ReviewItem":
        baseline_issue_summary = dict(
            payload.get("baseline_issue_summary")
            or payload.get("issue_summary")
            or _default_issue_summary()
        )
        draft_issue_summary = dict(payload.get("draft_issue_summary") or baseline_issue_summary)
        return cls(
            image_id=str(payload["image_id"]),
            split=str(payload["split"]),
            image_path=str(payload["image_path"]),
            label_path=str(payload["label_path"]),
            draft_label_path=str(payload["draft_label_path"]),
            status=str(payload.get("status", DEFAULT_REVIEW_STATUS)),
            note=str(payload.get("note", "")),
            has_edits=bool(payload.get("has_edits", False)),
            issue_summary=dict(baseline_issue_summary),
            baseline_issue_summary=baseline_issue_summary,
            draft_issue_summary=draft_issue_summary,
            gt_class_names=[str(item) for item in payload.get("gt_class_names", [])],
            pred_class_names=[str(item) for item in payload.get("pred_class_names", [])],
            updated_at=str(payload.get("updated_at", _now_iso())),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issue_summary"] = dict(self.baseline_issue_summary or self.issue_summary or _default_issue_summary())
        return payload


def _session_id() -> str:
    return datetime.now(timezone.utc).strftime("rsv-%Y%m%dT%H%M%SZ")


def _default_issue_summary() -> dict[str, Any]:
    return {"fp": 0, "fn": 0, "cls": 0, "severity": 0, "types": ["no_issue"]}


def _copy_issue_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    return dict(summary or _default_issue_summary())


def _session_file(session_dir: Path) -> Path:
    return session_dir / "session.yaml"


def _items_file(session_dir: Path) -> Path:
    return session_dir / "items.jsonl"


def _draft_root(session_dir: Path) -> Path:
    return session_dir / "draft" / "labels"


def _cache_root(session_dir: Path) -> Path:
    return session_dir / "cache"


def _prediction_cache_path(session_dir: Path, image_id: str) -> Path:
    return _cache_root(session_dir) / "predictions" / f"{image_id}.json"


def _thumbnail_cache_path(session_dir: Path, image_id: str) -> Path:
    return _cache_root(session_dir) / "thumbnails" / f"{image_id}.jpg"


def load_review_session(session_dir: Path) -> ReviewSession:
    session_path = _session_file(session_dir)
    if not session_path.exists():
        raise YoloCtlError(f"Review session is missing session.yaml: {session_dir}")
    from .yamlio import load_yaml

    return ReviewSession.from_mapping(load_yaml(session_path))


def save_review_session(session_dir: Path, session: ReviewSession) -> Path:
    session.updated_at = _now_iso()
    return dump_yaml(_session_file(session_dir), session.to_mapping())


def load_review_items(session_dir: Path) -> list[ReviewItem]:
    items_path = _items_file(session_dir)
    if not items_path.exists():
        return []
    items: list[ReviewItem] = []
    with items_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except JSONDecodeError as exc:
                # A torn final line can be left behind if another process rewrites the file.
                if items and (not line.startswith("{") or not line.endswith("}")):
                    continue
                raise YoloCtlError(
                    f"Review session items.jsonl is corrupted at line {line_number}: {exc}"
                ) from exc
            items.append(ReviewItem.from_mapping(payload))
    return items


def save_review_items(session_dir: Path, items: list[ReviewItem]) -> Path:
    path = _items_file(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_mapping(), ensure_ascii=True, sort_keys=False))
            handle.write("\n")
    os.replace(temp_path, path)
    return path


def _find_item(items: list[ReviewItem], image_id: str) -> ReviewItem:
    for item in items:
        if item.image_id == image_id:
            return item
    raise YoloCtlError(f"No review item found for image_id '{image_id}'")


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _label_path_for_image(record: DatasetVersionRecord, split: str, relative_image_path: Path) -> Path:
    materialized = _absolute_path(record.materialized_path)
    if materialized is None:
        raise YoloCtlError(f"Dataset version '{record.version_id}' has no materialized_path")
    images_dir = materialized / record.splits[split]
    relative = relative_image_path.relative_to(Path())
    return (images_dir.parent.parent / "labels" / split / relative).with_suffix(".txt")


def _parse_label_class_names(label_path: Path, class_names: list[str]) -> list[str]:
    if not label_path.exists():
        return []
    resolved: list[str] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if 0 <= class_id < len(class_names):
            resolved.append(class_names[class_id])
    return sorted(set(resolved))


def _sort_issue_types(summary: dict[str, Any]) -> list[str]:
    issue_types = []
    for key in ("cls", "fn", "fp"):
        if int(summary.get(key, 0)) > 0:
            issue_types.append(key)
    if not issue_types:
        issue_types.append("no_issue")
    return issue_types


def _issue_summary_from_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    fp = int(payload.get("fp", 0))
    fn = int(payload.get("fn", 0))
    cls = int(payload.get("cls", payload.get("confusions", 0)))
    summary = {
        "fp": fp,
        "fn": fn,
        "cls": cls,
        "severity": int(2 * cls + fp + fn),
    }
    summary["types"] = _sort_issue_types(summary)
    return summary


def _normalize_issue_report_row(row: dict[str, Any]) -> tuple[str | None, dict[str, Any], list[str], list[str]]:
    image_path = row.get("image_path") or row.get("path") or row.get("image")
    if not image_path:
        return None, _default_issue_summary(), [], []
    fp_preds = [str(item.get("pred")) for item in row.get("fp_details", []) if item.get("pred")]
    fn_gts = [str(item.get("gt")) for item in row.get("fn_details", []) if item.get("gt")]
    cls_gts = [str(item.get("gt")) for item in row.get("confusion_details", []) if item.get("gt")]
    cls_preds = [str(item.get("pred")) for item in row.get("confusion_details", []) if item.get("pred")]
    summary = _issue_summary_from_mapping(row)
    return str(Path(image_path).expanduser().resolve()), summary, sorted(set(fn_gts + cls_gts)), sorted(set(fp_preds + cls_preds))


def load_issues_report(path_value: str | None) -> dict[str, dict[str, Any]]:
    if not path_value:
        return {}
    path = _absolute_path(path_value)
    if path is None or not path.exists():
        raise YoloCtlError(f"Issues report does not exist: {path_value}")

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("items") or payload.get("errors") or payload.get("top_errors") or []
        else:
            rows = []

    normalized: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        image_key, summary, gt_class_names, pred_class_names = _normalize_issue_report_row(row)
        if image_key is None:
            continue
        normalized[image_key] = {
            "issue_summary": summary,
            "gt_class_names": gt_class_names,
            "pred_class_names": pred_class_names,
        }
    return normalized


def _build_review_items(
    record: DatasetVersionRecord,
    session_dir: Path,
    selected_splits: list[str],
    imported_issues: dict[str, dict[str, Any]],
) -> list[ReviewItem]:
    materialized = _absolute_path(record.materialized_path)
    if materialized is None or not materialized.exists():
        raise YoloCtlError(
            f"Dataset version '{record.version_id}' must be materialized locally before review can start."
        )

    items: list[ReviewItem] = []
    for split in selected_splits:
        split_dir = materialized / record.splits[split]
        if not split_dir.exists():
            continue
        for image_path in sorted(
            path
            for path in split_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            relative = image_path.relative_to(split_dir)
            image_key = str(image_path.resolve())
            draft_label_path = (_draft_root(session_dir) / split / relative).with_suffix(".txt")
            label_path = (materialized / "labels" / split / relative).with_suffix(".txt")
            imported = imported_issues.get(image_key, {})
            gt_class_names = imported.get("gt_class_names") or _parse_label_class_names(label_path, record.class_names)
            pred_class_names = imported.get("pred_class_names", [])
            baseline_issue_summary = _copy_issue_summary(imported.get("issue_summary"))
            item = ReviewItem(
                image_id=encode_image_id(split, relative),
                split=split,
                image_path=str(image_path),
                label_path=str(label_path),
                draft_label_path=str(draft_label_path),
                issue_summary=dict(baseline_issue_summary),
                baseline_issue_summary=dict(baseline_issue_summary),
                draft_issue_summary=dict(baseline_issue_summary),
                gt_class_names=list(gt_class_names),
                pred_class_names=list(pred_class_names),
            )
            items.append(item)
    return _sort_items(items, order_by="issue_severity")


def list_review_sessions(dataset_key: str, base_version_id: str) -> list[ReviewSession]:
    base_dir = review_dataset_dir(dataset_key, base_version_id)
    if not base_dir.exists():
        return []
    sessions: list[ReviewSession] = []
    for session_path in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        try:
            sessions.append(load_review_session(session_path))
        except Exception:
            continue
    sessions.sort(key=lambda item: item.updated_at, reverse=True)
    return sessions


def create_or_resume_review_session(
    record: DatasetVersionRecord,
    selected_splits: list[str] | None = None,
    weights: str | None = None,
    issues_report: str | None = None,
) -> tuple[Path, ReviewSession]:
    splits = selected_splits or list(record.splits)
    resolved_weights = _portable_path(weights)
    resolved_report = _portable_path(issues_report)
    for existing in list_review_sessions(record.dataset_key, record.version_id):
        if existing.state != "active":
            continue
        if existing.selected_splits == splits and existing.weights == resolved_weights and existing.issues_report == resolved_report:
            return review_session_dir(record.dataset_key, record.version_id, existing.session_id), existing

    session_id = _session_id()
    session_dir = review_session_dir(record.dataset_key, record.version_id, session_id)
    imported_issues = load_issues_report(issues_report)
    prediction_state = {
        "state": "pending" if resolved_weights and not resolved_report else ("ready" if resolved_report else "disabled"),
        "processed": 0,
        "total": 0,
        "message": None,
    }
    session = ReviewSession(
        session_id=session_id,
        dataset_key=record.dataset_key,
        base_version_id=record.version_id,
        base_version_label=record.version_label,
        created_at=_now_iso(),
        updated_at=_now_iso(),
        selected_splits=splits,
        weights=resolved_weights,
        issues_report=resolved_report,
        prediction_index=prediction_state,
    )
    items = _build_review_items(record, session_dir, splits, imported_issues)
    session.prediction_index["total"] = len(items)
    save_review_session(session_dir, session)
    save_review_items(session_dir, items)
    return session_dir, session


def _load_model(weights_path: str) -> Any:
    with MODEL_LOCK:
        if weights_path not in MODEL_CACHE:
            try:
                from ultralytics import YOLO
            except Exception as exc:  # pragma: no cover - depends on local env
                raise YoloCtlError(f"Ultralytics is required for review predictions: {exc}") from exc
            MODEL_CACHE[weights_path] = YOLO(str(_absolute_path(weights_path)))
        return MODEL_CACHE[weights_path]


def load_objects_from_label_file(
    label_path: Path,
    image_size: tuple[int, int],
    class_names: list[str],
    source: str,
) -> list[dict[str, Any]]:
    width, height = image_size
    if not label_path.exists():
        return []
    objects: list[dict[str, Any]] = []
    for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = [float(item) for item in parts[1:]]
        except ValueError:
            continue
        x1 = (x - w / 2.0) * width
        y1 = (y - h / 2.0) * height
        x2 = (x + w / 2.0) * width
        y2 = (y + h / 2.0) * height
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        objects.append(
            {
                "id": f"{source}-{index}",
                "class_id": class_id,
                "class_name": class_name,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "source": source,
            }
        )
    return objects


def _normalize_box(value: float, max_value: int) -> float:
    return max(0.0, min(float(value), float(max_value)))


def validate_review_objects(
    objects: list[dict[str, Any]],
    image_size: tuple[int, int],
    class_names: list[str],
) -> list[dict[str, Any]]:
    width, height = image_size
    normalized: list[dict[str, Any]] = []
    for index, obj in enumerate(objects):
        try:
            class_id = int(obj["class_id"])
            x1 = _normalize_box(float(obj["x1"]), width)
            y1 = _normalize_box(float(obj["y1"]), height)
            x2 = _normalize_box(float(obj["x2"]), width)
            y2 = _normalize_box(float(obj["y2"]), height)
        except (KeyError, TypeError, ValueError) as exc:
            raise YoloCtlError(f"Invalid review object at index {index}: {exc}") from exc
        if class_id < 0 or class_id >= len(class_names):
            raise YoloCtlError(f"Object class_id {class_id} is outside dataset class range")
        if x2 <= x1 or y2 <= y1:
            raise YoloCtlError("Bounding boxes must have positive area")
        normalized.append(
            {
                "id": str(obj.get("id") or f"draft-{index}"),
                "class_id": class_id,
                "class_name": class_names[class_id],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "source": str(obj.get("source") or "draft"),
                "confidence": obj.get("confidence"),
            }
        )
    return normalized


def write_objects_to_label_file(
    label_path: Path,
    objects: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> Path:
    width, height = image_size
    lines: list[str] = []
    for obj in objects:
        x1, y1, x2, y2 = float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"])
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        box_w = (x2 - x1) / width
        box_h = (y2 - y1) / height
        lines.append(f"{int(obj['class_id'])} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return label_path


def _read_prediction_cache(session_dir: Path, image_id: str) -> dict[str, Any] | None:
    cache_path = _prediction_cache_path(session_dir, image_id)
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _prediction_objects_from_cache(session_dir: Path, image_id: str) -> list[dict[str, Any]] | None:
    payload = _read_prediction_cache(session_dir, image_id)
    if payload is None:
        return None
    return list(payload.get("objects", []))


def _write_prediction_cache(session_dir: Path, image_id: str, payload: dict[str, Any]) -> None:
    cache_path = _prediction_cache_path(session_dir, image_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=False)
        handle.write("\n")
    os.replace(temp_path, cache_path)


def predict_item_objects(
    session_dir: Path,
    session: ReviewSession,
    image_path: Path,
    image_id: str,
    class_names: list[str],
    force: bool = False,
) -> list[dict[str, Any]]:
    if not session.weights:
        return []
    if not force:
        cached = _prediction_objects_from_cache(session_dir, image_id)
        if cached is not None:
            return cached
    model = _load_model(session.weights)
    results = model.predict(
        source=str(image_path),
        conf=PREDICTION_CONFIDENCE,
        iou=PREDICTION_IOU,
        verbose=False,
        max_det=300,
    )
    result = results[0]
    objects: list[dict[str, Any]] = []
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        for index, (box, class_id, score) in enumerate(zip(xyxy, cls, conf)):
            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
            objects.append(
                {
                    "id": f"pred-{index}",
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "source": "pred",
                    "confidence": float(score),
                }
            )
    _write_prediction_cache(
        session_dir,
        image_id,
        {"version": PREDICTION_CACHE_VERSION, "objects": objects, "updated_at": _now_iso()},
    )
    return objects


def _box_lookup(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(obj["id"]): obj for obj in objects}


def _best_matching_object_id(
    source: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    *,
    same_class: bool | None = None,
    min_iou: float = 0.3,
) -> str | None:
    if source is None:
        return None
    best_id: str | None = None
    best_iou = 0.0
    for candidate in candidates:
        if same_class is True and candidate["class_id"] != source["class_id"]:
            continue
        if same_class is False and candidate["class_id"] == source["class_id"]:
            continue
        iou = _box_iou(source, candidate)
        if iou < min_iou or iou <= best_iou:
            continue
        best_iou = iou
        best_id = str(candidate["id"])
    return best_id


def _expanded_focus_box(
    issue: dict[str, Any],
    image_size: tuple[int, int],
    gt_lookup: dict[str, dict[str, Any]],
    draft_lookup: dict[str, dict[str, Any]],
    pred_lookup: dict[str, dict[str, Any]],
) -> dict[str, float]:
    related: list[dict[str, Any]] = []
    for key, lookup in (
        ("gt_object_id", gt_lookup),
        ("draft_object_id", draft_lookup),
        ("pred_object_id", pred_lookup),
    ):
        object_id = issue.get(key)
        if object_id and object_id in lookup:
            related.append(lookup[object_id])
    if not related:
        width, height = image_size
        return {"x1": 0.0, "y1": 0.0, "x2": float(width), "y2": float(height)}
    x1 = min(float(obj["x1"]) for obj in related)
    y1 = min(float(obj["y1"]) for obj in related)
    x2 = max(float(obj["x2"]) for obj in related)
    y2 = max(float(obj["y2"]) for obj in related)
    width, height = image_size
    return {
        "x1": max(0.0, x1 - FOCUS_BOX_PADDING),
        "y1": max(0.0, y1 - FOCUS_BOX_PADDING),
        "x2": min(float(width), x2 + FOCUS_BOX_PADDING),
        "y2": min(float(height), y2 + FOCUS_BOX_PADDING),
    }


def _enrich_issue_items(
    issue_items: list[dict[str, Any]],
    gt_objects: list[dict[str, Any]],
    draft_objects: list[dict[str, Any]],
    prediction_objects: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    gt_lookup = _box_lookup(gt_objects)
    draft_lookup = _box_lookup(draft_objects)
    pred_lookup = _box_lookup(prediction_objects)
    enriched: list[dict[str, Any]] = []
    for issue in issue_items:
        next_issue = dict(issue)
        gt_object = gt_lookup.get(str(next_issue.get("gt_object_id"))) if next_issue.get("gt_object_id") else None
        draft_object = draft_lookup.get(str(next_issue.get("draft_object_id"))) if next_issue.get("draft_object_id") else None
        pred_object = pred_lookup.get(str(next_issue.get("pred_object_id"))) if next_issue.get("pred_object_id") else None
        if gt_object is not None and draft_object is None:
            next_issue["draft_object_id"] = _best_matching_object_id(gt_object, draft_objects, same_class=True)
            draft_object = draft_lookup.get(str(next_issue.get("draft_object_id"))) if next_issue.get("draft_object_id") else None
        if draft_object is not None and gt_object is None:
            next_issue["gt_object_id"] = _best_matching_object_id(draft_object, gt_objects, same_class=True)
            gt_object = gt_lookup.get(str(next_issue.get("gt_object_id"))) if next_issue.get("gt_object_id") else None
        if pred_object is not None and gt_object is None:
            next_issue["gt_object_id"] = _best_matching_object_id(pred_object, gt_objects, same_class=None)
            gt_object = gt_lookup.get(str(next_issue.get("gt_object_id"))) if next_issue.get("gt_object_id") else None
        if pred_object is not None and draft_object is None:
            next_issue["draft_object_id"] = _best_matching_object_id(pred_object, draft_objects, same_class=None)
            draft_object = draft_lookup.get(str(next_issue.get("draft_object_id"))) if next_issue.get("draft_object_id") else None
        if pred_object is None and gt_object is not None:
            next_issue["pred_object_id"] = _best_matching_object_id(gt_object, prediction_objects, same_class=None)
            pred_object = pred_lookup.get(str(next_issue.get("pred_object_id"))) if next_issue.get("pred_object_id") else None
        if pred_object is None and draft_object is not None:
            next_issue["pred_object_id"] = _best_matching_object_id(draft_object, prediction_objects, same_class=None)
            pred_object = pred_lookup.get(str(next_issue.get("pred_object_id"))) if next_issue.get("pred_object_id") else None
        if gt_object is not None and next_issue.get("gt_label") is None:
            next_issue["gt_label"] = gt_object["class_name"]
        if draft_object is not None and next_issue.get("gt_label") is None:
            next_issue["gt_label"] = draft_object["class_name"]
        if pred_object is not None and next_issue.get("pred_label") is None:
            next_issue["pred_label"] = pred_object["class_name"]
        if pred_object is not None and next_issue.get("confidence") is None:
            next_issue["confidence"] = pred_object.get("confidence")
        if next_issue.get("iou") is None:
            reference_object = draft_object if draft_object is not None else gt_object
            if reference_object is not None and pred_object is not None:
                next_issue["iou"] = _box_iou(reference_object, pred_object)
        next_issue["focus_box"] = _expanded_focus_box(next_issue, image_size, gt_lookup, draft_lookup, pred_lookup)
        enriched.append(next_issue)
    return enriched


def _box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_x1, left_y1, left_x2, left_y2 = left["x1"], left["y1"], left["x2"], left["y2"]
    right_x1, right_y1, right_x2, right_y2 = right["x1"], right["y1"], right["x2"], right["y2"]
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _greedy_match(
    predictions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    same_class: bool,
    used_predictions: set[int] | None = None,
    used_labels: set[int] | None = None,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    used_predictions = set() if used_predictions is None else set(used_predictions)
    used_labels = set() if used_labels is None else set(used_labels)
    candidates: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        if pred_index in used_predictions:
            continue
        for label_index, label in enumerate(labels):
            if label_index in used_labels:
                continue
            if same_class and prediction["class_id"] != label["class_id"]:
                continue
            if not same_class and prediction["class_id"] == label["class_id"]:
                continue
            iou = _box_iou(prediction, label)
            if iou >= MATCH_IOU:
                candidates.append((iou, pred_index, label_index))
    candidates.sort(reverse=True)
    matches: list[tuple[int, int, float]] = []
    for iou, pred_index, label_index in candidates:
        if pred_index in used_predictions or label_index in used_labels:
            continue
        used_predictions.add(pred_index)
        used_labels.add(label_index)
        matches.append((pred_index, label_index, iou))
    return matches, used_predictions, used_labels


def compare_label_objects_to_predictions(
    label_objects: list[dict[str, Any]],
    prediction_objects: list[dict[str, Any]],
    *,
    label_role: str = "gt",
    gt_objects: list[dict[str, Any]] | None = None,
    draft_objects: list[dict[str, Any]] | None = None,
    image_size: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    same_matches, used_predictions, used_labels = _greedy_match(prediction_objects, label_objects, same_class=True)
    confusion_matches, used_predictions, used_labels = _greedy_match(
        prediction_objects,
        label_objects,
        same_class=False,
        used_predictions=used_predictions,
        used_labels=used_labels,
    )
    issue_items: list[dict[str, Any]] = []
    next_id = 1
    for pred_index, label_index, iou in confusion_matches:
        prediction = prediction_objects[pred_index]
        label = label_objects[label_index]
        next_issue = {
            "id": next_id,
            "type": "cls",
            "gt_label": label["class_name"] if label_role == "gt" else None,
            "pred_label": prediction["class_name"],
            "confidence": prediction.get("confidence"),
            "iou": iou,
            "gt_object_id": label["id"] if label_role == "gt" else None,
            "draft_object_id": label["id"] if label_role == "draft" else None,
            "pred_object_id": prediction["id"],
            "source": "pred",
        }
        issue_items.append(next_issue)
        next_id += 1
    for label_index, label in enumerate(label_objects):
        if label_index in used_labels:
            continue
        issue_items.append(
            {
                "id": next_id,
                "type": "fn",
                "gt_label": label["class_name"] if label_role == "gt" else None,
                "pred_label": None,
                "confidence": None,
                "iou": None,
                "gt_object_id": label["id"] if label_role == "gt" else None,
                "draft_object_id": label["id"] if label_role == "draft" else None,
                "pred_object_id": None,
                "source": label_role,
            }
        )
        next_id += 1
    for pred_index, prediction in enumerate(prediction_objects):
        if pred_index in used_predictions:
            continue
        issue_items.append(
            {
                "id": next_id,
                "type": "fp",
                "gt_label": None,
                "pred_label": prediction["class_name"],
                "confidence": prediction.get("confidence"),
                "iou": None,
                "gt_object_id": None,
                "draft_object_id": None,
                "pred_object_id": prediction["id"],
                "source": "pred",
            }
        )
        next_id += 1
    summary = _issue_summary_from_mapping(
        {
            "fp": sum(1 for item in issue_items if item["type"] == "fp"),
            "fn": sum(1 for item in issue_items if item["type"] == "fn"),
            "cls": sum(1 for item in issue_items if item["type"] == "cls"),
        }
    )
    enriched_issue_items = issue_items
    if image_size is not None:
        enriched_issue_items = _enrich_issue_items(
            issue_items,
            gt_objects or ([] if label_role != "gt" else label_objects),
            draft_objects or ([] if label_role != "draft" else label_objects),
            prediction_objects,
            image_size,
        )
    return summary, enriched_issue_items


def _item_payload(
    session_dir: Path,
    session: ReviewSession,
    record: DatasetVersionRecord,
    item: ReviewItem,
    force_predict: bool = False,
) -> dict[str, Any]:
    image_path = Path(item.image_path)
    image_size = _image_size(image_path)
    gt_objects = load_objects_from_label_file(Path(item.label_path), image_size, record.class_names, source="gt")
    draft_path = Path(item.draft_label_path)
    if item.has_edits and draft_path.exists():
        draft_objects = load_objects_from_label_file(draft_path, image_size, record.class_names, source="draft")
    else:
        draft_objects = [
            {
                **obj,
                "id": f"draft-{index}",
                "source": "draft",
            }
            for index, obj in enumerate(gt_objects)
        ]
    prediction_objects = predict_item_objects(
        session_dir=session_dir,
        session=session,
        image_path=image_path,
        image_id=item.image_id,
        class_names=record.class_names,
        force=force_predict,
    )
    prediction_cache = _read_prediction_cache(session_dir, item.image_id) or {}
    if prediction_objects:
        baseline_issue_summary = _copy_issue_summary(prediction_cache.get("baseline_issue_summary"))
        baseline_issue_items = list(prediction_cache.get("baseline_issue_items") or [])
        if prediction_cache.get("version") != PREDICTION_CACHE_VERSION or not baseline_issue_items:
            baseline_issue_summary, baseline_issue_items = compare_label_objects_to_predictions(
                gt_objects,
                prediction_objects,
                label_role="gt",
                gt_objects=gt_objects,
                draft_objects=draft_objects,
                image_size=image_size,
            )
            prediction_cache = {
                **prediction_cache,
                "version": PREDICTION_CACHE_VERSION,
                "objects": prediction_objects,
                "baseline_issue_summary": baseline_issue_summary,
                "baseline_issue_items": baseline_issue_items,
                "updated_at": _now_iso(),
            }
            _write_prediction_cache(session_dir, item.image_id, prediction_cache)
        draft_issue_summary, draft_issue_items = compare_label_objects_to_predictions(
            draft_objects,
            prediction_objects,
            label_role="draft",
            gt_objects=gt_objects,
            draft_objects=draft_objects,
            image_size=image_size,
        )
    else:
        baseline_issue_summary = _copy_issue_summary(item.baseline_issue_summary or item.issue_summary)
        baseline_issue_items = []
        draft_issue_summary = _copy_issue_summary(item.draft_issue_summary or baseline_issue_summary)
        draft_issue_items = []
    return {
        "item": item.to_mapping(),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "gt_objects": gt_objects,
        "draft_objects": draft_objects,
        "prediction_objects": prediction_objects,
        "issue_summary": baseline_issue_summary,
        "issue_items": baseline_issue_items,
        "baseline_issue_summary": baseline_issue_summary,
        "baseline_issue_items": baseline_issue_items,
        "draft_issue_summary": draft_issue_summary,
        "draft_issue_items": draft_issue_items,
    }


def get_item_payload(session_dir: Path, image_id: str, force_predict: bool = False) -> dict[str, Any]:
    session = load_review_session(session_dir)
    record = get_dataset_record(session.dataset_key, version_id=session.base_version_id)
    items = load_review_items(session_dir)
    item = _find_item(items, image_id)
    payload = _item_payload(session_dir, session, record, item, force_predict=force_predict)
    if payload["prediction_objects"]:
        item.pred_class_names = sorted({obj["class_name"] for obj in payload["prediction_objects"]})
        item.baseline_issue_summary = _copy_issue_summary(payload["baseline_issue_summary"])
        item.draft_issue_summary = _copy_issue_summary(payload["draft_issue_summary"])
        item.issue_summary = _copy_issue_summary(item.baseline_issue_summary)
        item.updated_at = _now_iso()
        save_review_items(session_dir, items)
    payload["item"] = item.to_mapping()
    return payload


def _sort_items(items: list[ReviewItem], order_by: str) -> list[ReviewItem]:
    if order_by == "filename":
        return sorted(items, key=lambda item: Path(item.image_path).name.lower())
    return sorted(
        items,
        key=lambda item: (
            -int(item.issue_summary.get("severity", 0)),
            Path(item.image_path).name.lower(),
        ),
    )


def filter_review_items(
    session_dir: Path,
    split: str | None = None,
    status: str | None = None,
    issue_type: str | None = None,
    gt_class: str | None = None,
    pred_class: str | None = None,
    edited_only: bool = False,
    query: str | None = None,
    order_by: str = "issue_severity",
) -> list[dict[str, Any]]:
    items = _sort_items(load_review_items(session_dir), order_by=order_by)
    query_value = (query or "").strip().lower()
    filtered: list[dict[str, Any]] = []
    for item in items:
        if split and item.split != split:
            continue
        if status and item.status != status:
            continue
        if edited_only and not item.has_edits:
            continue
        if issue_type and issue_type not in item.issue_summary.get("types", ["no_issue"]):
            continue
        if gt_class and gt_class not in item.gt_class_names:
            continue
        if pred_class and pred_class not in item.pred_class_names:
            continue
        if query_value and query_value not in Path(item.image_path).name.lower():
            continue
        filtered.append(item.to_mapping())
    return filtered


def update_item_status(
    session_dir: Path,
    image_id: str,
    status: str,
    note: str | None = None,
) -> ReviewItem:
    if status not in REVIEW_STATUSES:
        raise YoloCtlError(f"Unsupported review status '{status}'")
    session = load_review_session(session_dir)
    items = load_review_items(session_dir)
    item = _find_item(items, image_id)
    item.status = status
    if note is not None:
        item.note = note
    item.updated_at = _now_iso()
    save_review_items(session_dir, items)
    save_review_session(session_dir, session)
    return item


def save_item_draft(
    session_dir: Path,
    image_id: str,
    draft_objects: list[dict[str, Any]],
    status: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    session = load_review_session(session_dir)
    record = get_dataset_record(session.dataset_key, version_id=session.base_version_id)
    items = load_review_items(session_dir)
    item = _find_item(items, image_id)
    image_path = Path(item.image_path)
    image_size = _image_size(image_path)
    normalized_objects = validate_review_objects(draft_objects, image_size, record.class_names)
    write_objects_to_label_file(Path(item.draft_label_path), normalized_objects, image_size)
    item.has_edits = True
    if status:
        if status not in REVIEW_STATUSES:
            raise YoloCtlError(f"Unsupported review status '{status}'")
        item.status = status
    if note is not None:
        item.note = note
    prediction_objects = _prediction_objects_from_cache(session_dir, image_id) or []
    if prediction_objects:
        item.draft_issue_summary, _ = compare_label_objects_to_predictions(
            normalized_objects,
            prediction_objects,
            label_role="draft",
            gt_objects=load_objects_from_label_file(Path(item.label_path), image_size, record.class_names, source="gt"),
            draft_objects=normalized_objects,
            image_size=image_size,
        )
        item.pred_class_names = sorted({obj["class_name"] for obj in prediction_objects})
    else:
        item.draft_issue_summary = _copy_issue_summary(item.baseline_issue_summary or item.issue_summary)
    item.issue_summary = _copy_issue_summary(item.baseline_issue_summary or item.issue_summary)
    item.updated_at = _now_iso()
    save_review_items(session_dir, items)
    save_review_session(session_dir, session)
    return get_item_payload(session_dir, image_id)


def apply_prediction_to_draft(
    session_dir: Path,
    image_id: str,
    prediction_id: str,
    mode: str = "append",
    target_draft_id: str | None = None,
) -> dict[str, Any]:
    payload = get_item_payload(session_dir, image_id)
    draft_objects = list(payload["draft_objects"])
    prediction = next((obj for obj in payload["prediction_objects"] if obj["id"] == prediction_id), None)
    if prediction is None:
        raise YoloCtlError(f"No prediction found for id '{prediction_id}'")
    if mode not in {"append", "replace_selected", "class_only", "box_only"}:
        raise YoloCtlError(f"Unsupported prediction apply mode '{mode}'")
    next_index = len(draft_objects)
    replacement = {
        "id": f"draft-{next_index}",
        "class_id": prediction["class_id"],
        "class_name": prediction["class_name"],
        "x1": prediction["x1"],
        "y1": prediction["y1"],
        "x2": prediction["x2"],
        "y2": prediction["y2"],
        "source": "draft",
        "confidence": prediction.get("confidence"),
    }
    if mode == "append":
        draft_objects.append(replacement)
        return {"draft_objects": draft_objects}

    if not target_draft_id:
        raise YoloCtlError(f"Prediction apply mode '{mode}' requires target_draft_id")
    target_index = next((index for index, obj in enumerate(draft_objects) if obj["id"] == target_draft_id), None)
    if target_index is None:
        raise YoloCtlError(f"No draft object found for id '{target_draft_id}'")
    target = dict(draft_objects[target_index])
    if mode == "replace_selected":
        replacement["id"] = target["id"]
        draft_objects[target_index] = replacement
    elif mode == "class_only":
        draft_objects[target_index] = {
            **target,
            "class_id": prediction["class_id"],
            "class_name": prediction["class_name"],
            "confidence": prediction.get("confidence"),
        }
    elif mode == "box_only":
        draft_objects[target_index] = {
            **target,
            "x1": prediction["x1"],
            "y1": prediction["y1"],
            "x2": prediction["x2"],
            "y2": prediction["y2"],
            "confidence": prediction.get("confidence"),
        }
    return {"draft_objects": draft_objects}


def replace_draft_with_predictions(session_dir: Path, image_id: str) -> dict[str, Any]:
    payload = get_item_payload(session_dir, image_id)
    replacement = [
        {
            "id": f"draft-{index}",
            "class_id": obj["class_id"],
            "class_name": obj["class_name"],
            "x1": obj["x1"],
            "y1": obj["y1"],
            "x2": obj["x2"],
            "y2": obj["y2"],
            "source": "draft",
            "confidence": obj.get("confidence"),
        }
        for index, obj in enumerate(payload["prediction_objects"])
    ]
    return {"draft_objects": replacement}


def build_session_summary(session_dir: Path) -> dict[str, Any]:
    session = load_review_session(session_dir)
    items = load_review_items(session_dir)
    status_counts = {status: 0 for status in REVIEW_STATUSES}
    edited_count = 0
    total_severity = 0
    issue_type_counts = {"fp": 0, "fn": 0, "cls": 0, "no_issue": 0}
    for item in items:
        status_counts[item.status] += 1
        if item.has_edits:
            edited_count += 1
        total_severity += int(item.issue_summary.get("severity", 0))
        for issue_type in item.issue_summary.get("types", ["no_issue"]):
            issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
    return {
        "session": session.to_mapping(),
        "counts": {
            "total": len(items),
            "edited": edited_count,
            "statuses": status_counts,
            "issue_types": issue_type_counts,
            "total_severity": total_severity,
        },
    }


def generate_thumbnail(session_dir: Path, image_id: str, width: int = 320) -> Path:
    cache_path = _thumbnail_cache_path(session_dir, image_id)
    if cache_path.exists():
        return cache_path
    session = load_review_session(session_dir)
    record = get_dataset_record(session.dataset_key, version_id=session.base_version_id)
    item = _find_item(load_review_items(session_dir), image_id)
    image_path = Path(item.image_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image.thumbnail((width, width * 2))
        image.convert("RGB").save(cache_path, format="JPEG", quality=85)
    return cache_path


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if destination.exists():
            destination.unlink()
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_tree_with_links(source: Path, destination: Path) -> None:
    for root, dirs, files in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        target_root = destination / relative
        target_root.mkdir(parents=True, exist_ok=True)
        for directory in dirs:
            (target_root / directory).mkdir(parents=True, exist_ok=True)
        for file_name in files:
            _link_or_copy(root_path / file_name, target_root / file_name)


def finalize_review_session(
    session_dir: Path,
    new_version_label: str,
    merge_note: str | None = None,
) -> DatasetVersionRecord:
    session = load_review_session(session_dir)
    if session.state != "active":
        raise YoloCtlError(f"Review session '{session.session_id}' is already finalized")
    base_record = get_dataset_record(session.dataset_key, version_id=session.base_version_id)
    materialized = _absolute_path(base_record.materialized_path)
    if materialized is None or not materialized.exists():
        raise YoloCtlError(f"Base dataset version '{base_record.version_id}' is missing materialized data")

    review_items = load_review_items(session_dir)
    edited_items = [item for item in review_items if item.has_edits and Path(item.draft_label_path).exists()]
    if not edited_items:
        raise YoloCtlError("Review session has no draft label edits to finalize")

    version_id = f"dsv-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{_slug(new_version_label)}"
    target_materialized = project_root() / "artifacts" / "datasets" / base_record.dataset_key / version_id
    if target_materialized.exists():
        raise YoloCtlError(f"Target review dataset already exists: {target_materialized}")

    _copy_tree_with_links(materialized, target_materialized)
    for item in edited_items:
        base_label_path = Path(item.label_path)
        relative = base_label_path.relative_to(materialized)
        _link_or_copy(Path(item.draft_label_path), target_materialized / relative)

    record = create_dataset_record(
        dataset_key=base_record.dataset_key,
        version_label=new_version_label,
        version_id=version_id,
        class_names=list(base_record.class_names),
        splits=dict(base_record.splits),
        source_archive_path=None,
        materialized_path=str(target_materialized),
        source_type="materialized_local",
        cloud_uri=None,
        cloud_manifest_uri=None,
        parents=[base_record.version_id],
        lineage_type="merge",
        merge_note=merge_note,
        metadata={
            **dict(base_record.metadata),
            "review_session_id": session.session_id,
            "review_source_version_id": base_record.version_id,
            "edited_label_count": len(edited_items),
            "review_summary": build_session_summary(session_dir)["counts"],
        },
    )
    save_dataset_record(record)
    write_dataset_yaml(record, output_path=_absolute_path(record.dataset_yaml_path))

    session.state = "finalized"
    session.finalized_version_id = record.version_id
    session.finalized_at = _now_iso()
    save_review_session(session_dir, session)
    return record


def build_prediction_issue_index(session_dir: Path) -> None:
    session = load_review_session(session_dir)
    if not session.weights:
        return
    record = get_dataset_record(session.dataset_key, version_id=session.base_version_id)
    items = load_review_items(session_dir)
    session.prediction_index = {
        **dict(session.prediction_index),
        "state": "running",
        "processed": 0,
        "total": len(items),
        "message": None,
    }
    save_review_session(session_dir, session)
    for index, item in enumerate(items, start=1):
        payload = _item_payload(session_dir, session, record, item, force_predict=False)
        if payload["prediction_objects"]:
            item.pred_class_names = sorted({obj["class_name"] for obj in payload["prediction_objects"]})
            item.baseline_issue_summary = _copy_issue_summary(payload["baseline_issue_summary"])
            item.draft_issue_summary = _copy_issue_summary(payload["draft_issue_summary"])
            item.issue_summary = _copy_issue_summary(item.baseline_issue_summary)
            item.updated_at = _now_iso()
        session.prediction_index["processed"] = index
        if index % 10 == 0 or index == len(items):
            save_review_items(session_dir, items)
            save_review_session(session_dir, session)
    session.prediction_index["state"] = "ready"
    save_review_items(session_dir, items)
    save_review_session(session_dir, session)

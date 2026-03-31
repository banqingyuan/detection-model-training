from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .datasets import DatasetVersionRecord, IMAGE_SUFFIXES
from .exceptions import YoloCtlError
from .paths import paddledet_datasets_dir, project_root
from .yamlio import load_yaml, write_json


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _materialized_root(record: DatasetVersionRecord, dataset_yaml: Path) -> Path:
    payload = load_yaml(dataset_yaml)
    dataset_root = payload.get("path")
    if dataset_root:
        return _resolve_path(dataset_yaml.parent, str(dataset_root))
    if record.materialized_path:
        materialized = Path(record.materialized_path)
        if not materialized.is_absolute():
            materialized = project_root() / materialized
        return materialized.resolve()
    raise YoloCtlError(
        f"Dataset version '{record.version_id}' has no materialized dataset root. "
        "Run dataset prepare or register a materialized path first."
    )


def _split_mapping(record: DatasetVersionRecord, dataset_yaml: Path) -> dict[str, str]:
    payload = load_yaml(dataset_yaml)
    splits = {}
    for split_name in ("train", "val", "test"):
        value = payload.get(split_name) or record.splits.get(split_name)
        if value:
            splits[split_name] = str(value)
    return splits


def _label_path(materialized_root: Path, image_relative_path: Path) -> Path:
    parts = list(image_relative_path.parts)
    if not parts:
        raise YoloCtlError("Image path cannot be empty")
    if "images" in parts:
        image_index = parts.index("images")
        parts[image_index] = "labels"
        relative = Path(*parts).with_suffix(".txt")
    else:
        relative = image_relative_path.with_suffix(".txt")
    return materialized_root / relative


def _iter_split_images(materialized_root: Path, split_relative: str) -> list[Path]:
    image_root = _resolve_path(materialized_root, split_relative)
    if not image_root.exists():
        raise YoloCtlError(f"Dataset split path does not exist: {image_root}")
    return sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _load_label_rows(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _convert_bbox(image_width: int, image_height: int, row: str, class_count: int) -> tuple[int, list[float], float]:
    parts = row.split()
    if len(parts) < 5:
        raise YoloCtlError(f"Invalid YOLO label row '{row}'. Expected at least 5 columns.")
    class_id = int(parts[0])
    if class_id < 0 or class_id >= class_count:
        raise YoloCtlError(
            f"YOLO label class_id {class_id} is out of range for {class_count} classes."
        )
    center_x = float(parts[1]) * image_width
    center_y = float(parts[2]) * image_height
    box_width = float(parts[3]) * image_width
    box_height = float(parts[4]) * image_height
    x_min = center_x - (box_width / 2.0)
    y_min = center_y - (box_height / 2.0)
    bbox = [
        round(x_min, 4),
        round(y_min, 4),
        round(box_width, 4),
        round(box_height, 4),
    ]
    area = round(box_width * box_height, 4)
    return class_id + 1, bbox, area


def _build_coco_for_split(
    *,
    record: DatasetVersionRecord,
    materialized_root: Path,
    split_name: str,
    split_relative: str,
) -> dict[str, Any]:
    images_payload: list[dict[str, Any]] = []
    annotations_payload: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, image_path in enumerate(_iter_split_images(materialized_root, split_relative), start=1):
        relative_image_path = image_path.relative_to(materialized_root)
        with Image.open(image_path) as image:
            width, height = image.size
        images_payload.append(
            {
                "id": image_id,
                "file_name": str(relative_image_path).replace("\\", "/"),
                "width": width,
                "height": height,
            }
        )
        label_rows = _load_label_rows(_label_path(materialized_root, relative_image_path))
        for row in label_rows:
            category_id, bbox, area = _convert_bbox(width, height, row, len(record.class_names))
            annotations_payload.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images_payload,
        "annotations": annotations_payload,
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(record.class_names)],
        "split": split_name,
    }


def prepare_paddledet_dataset(
    *,
    record: DatasetVersionRecord,
    dataset_yaml: Path,
    run_id: str,
    require_val: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    materialized_root = _materialized_root(record, dataset_yaml)
    splits = _split_mapping(record, dataset_yaml)
    if "train" not in splits:
        raise YoloCtlError(f"Dataset version '{record.version_id}' does not include a train split")
    if require_val and "val" not in splits:
        raise YoloCtlError(
            f"Dataset version '{record.version_id}' does not include a val split required for PicoDet evaluation."
        )

    output_root = paddledet_datasets_dir() / record.dataset_key / record.version_id / run_id
    annotations_dir = output_root / "annotations"
    generated: dict[str, Any] = {
        "backend": "paddledet",
        "dataset_key": record.dataset_key,
        "dataset_version_id": record.version_id,
        "materialized_root": str(materialized_root),
        "derived_dataset_dir": str(output_root),
        "train_image_dir": splits["train"],
        "train_annotation_path": str((annotations_dir / "instances_train.json").resolve()),
    }
    train_payload = _build_coco_for_split(
        record=record,
        materialized_root=materialized_root,
        split_name="train",
        split_relative=splits["train"],
    )
    generated["train_images"] = len(train_payload["images"])
    generated["train_annotations"] = len(train_payload["annotations"])
    if not dry_run:
        write_json(annotations_dir / "instances_train.json", train_payload)

    if "val" in splits:
        val_payload = _build_coco_for_split(
            record=record,
            materialized_root=materialized_root,
            split_name="val",
            split_relative=splits["val"],
        )
        generated["val_image_dir"] = splits["val"]
        generated["val_annotation_path"] = str((annotations_dir / "instances_val.json").resolve())
        generated["val_images"] = len(val_payload["images"])
        generated["val_annotations"] = len(val_payload["annotations"])
        if not dry_run:
            write_json(annotations_dir / "instances_val.json", val_payload)

    return generated

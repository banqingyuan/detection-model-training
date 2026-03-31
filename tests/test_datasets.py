from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import zipfile
import json

from yoloctl.datasets import (
    build_dataset_summary,
    create_dataset_record,
    create_merge_draft,
    get_dataset_record,
    import_release_manifest,
    merge_draft_from_release,
    read_detect_yaml,
    save_dataset_record,
)
from yoloctl.paths import dataset_index_path, dataset_manifest_path
from yoloctl.yamlio import dump_yaml


class DatasetTests(unittest.TestCase):
    def test_register_and_load_dataset_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            detect_dir = root / "materialized"
            (detect_dir / "images" / "train").mkdir(parents=True)
            (detect_dir / "images" / "val").mkdir(parents=True)
            (detect_dir / "labels" / "train").mkdir(parents=True)
            (detect_dir / "labels" / "val").mkdir(parents=True)
            archive_path = root / "demo.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("images/train/sample.jpg", b"demo")
            record = create_dataset_record(
                dataset_key="demo",
                version_label="v1",
                class_names=["card", "chip"],
                splits={"train": "images/train", "val": "images/val"},
                source_archive_path=str(archive_path),
                materialized_path=str(detect_dir),
            )
            self.assertTrue(record.version_id.startswith("dsv-"))

    def test_summary_and_lookup_are_backed_by_manifest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            detect_dir = root / "materialized"
            (detect_dir / "images" / "train").mkdir(parents=True)
            (detect_dir / "images" / "val").mkdir(parents=True)
            (detect_dir / "labels" / "train").mkdir(parents=True)
            (detect_dir / "labels" / "val").mkdir(parents=True)
            archive_path = root / "demo.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("images/train/sample.jpg", b"demo")
            record = create_dataset_record(
                dataset_key="demo",
                version_label="v1",
                class_names=["card", "chip"],
                splits={"train": "images/train", "val": "images/val"},
                source_archive_path=str(archive_path),
                materialized_path=str(detect_dir),
            )
            old_root = Path.cwd()
            try:
                os_root = root / "repo"
                os_root.mkdir()
                (os_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
                (os_root / "configs" / "datasets").mkdir(parents=True)
                # Switch root through env so path helpers resolve inside the temp repo.
                import os

                os.environ["YOLOCTL_ROOT"] = str(os_root)
                save_dataset_record(record)
                loaded = get_dataset_record("demo", version_id=record.version_id)
                summary = build_dataset_summary("demo", version_id=record.version_id)
                self.assertEqual(loaded.version_label, "v1")
                self.assertEqual(summary["summary"]["latest_version_id"], record.version_id)
                self.assertTrue(dataset_index_path().exists())
                self.assertTrue(dataset_manifest_path("demo", record.version_id).exists())
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)
                os.chdir(old_root)

    def test_read_detect_yaml_and_merge_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            detect_yaml = root / "detect.yaml"
            dump_yaml(
                detect_yaml,
                {
                    "train": "images/train",
                    "val": "images/val",
                    "names": {0: "card", 1: "chip"},
                },
            )
            parsed = read_detect_yaml(detect_yaml)
            self.assertEqual(parsed["class_names"], ["card", "chip"])
            self.assertEqual(parsed["splits"]["train"], "images/train")

            repo_root = root / "repo"
            repo_root.mkdir()
            (repo_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            archive_path = root / "demo.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("images/train/sample.jpg", b"demo")
            import os

            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                base = create_dataset_record(
                    dataset_key="demo",
                    version_label="v1",
                    class_names=["card", "chip"],
                    splits={"train": "images/train", "val": "images/val"},
                    source_archive_path=str(archive_path),
                    materialized_path=str(root),
                )
                save_dataset_record(base)
                draft = create_merge_draft("demo", "merge-v2", [base.version_id], merge_note="candidate")
                self.assertEqual(draft.parents, [base.version_id])
                self.assertEqual(draft.lineage_type, "merge")
                self.assertEqual(draft.source_type, "merge_draft")
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_import_release_manifest_and_merge_from_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            (repo_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            import os

            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                base_materialized = root / "base"
                (base_materialized / "images" / "train").mkdir(parents=True)
                (base_materialized / "images" / "val").mkdir(parents=True)
                (base_materialized / "labels" / "train").mkdir(parents=True)
                (base_materialized / "labels" / "val").mkdir(parents=True)
                archive_path = root / "base.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("images/train/sample.jpg", b"demo")
                base = create_dataset_record(
                    dataset_key="demo",
                    version_label="v1",
                    class_names=["card"],
                    splits={"train": "images/train", "val": "images/val"},
                    source_archive_path=str(archive_path),
                    materialized_path=str(base_materialized),
                )
                save_dataset_record(base)

                image_dir = root / "release-src"
                image_dir.mkdir()
                source_image = image_dir / "sample.jpg"
                source_image.write_bytes(b"fake-image")
                manifest_path = root / "release.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "release_id": "rel-001",
                            "release_version": "inc-001",
                            "dataset_key": "demo",
                            "release_type": "merge_candidate",
                            "parent_version_id": base.version_id,
                            "class_names": ["card"],
                            "items": [
                                {
                                    "image_id": "img-001",
                                    "source_uri": str(source_image),
                                    "split": "train",
                                    "detections": [
                                        {
                                            "class_id": 0,
                                            "class_name": "card",
                                            "bbox": {"x": 0.5, "y": 0.5, "width": 0.4, "height": 0.4},
                                        }
                                    ],
                                },
                                {
                                    "image_id": "img-002",
                                    "source_uri": str(source_image),
                                    "split": "val",
                                    "detections": [],
                                },
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                imported = import_release_manifest(str(manifest_path))
                self.assertEqual(imported.source_type, "release_manifest")
                self.assertEqual(imported.metadata["parent_dataset_version_id"], base.version_id)
                loaded = get_dataset_record("demo", version_id=imported.version_id)
                self.assertEqual(loaded.version_id, imported.version_id)

                draft = merge_draft_from_release(imported.version_id, version_label="merge-v2")
                self.assertEqual(draft.parents[0], base.version_id)
                self.assertEqual(draft.parents[1], imported.version_id)
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)


if __name__ == "__main__":
    unittest.main()

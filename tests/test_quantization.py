from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from yoloctl.cli import _assess_export_artifacts
from yoloctl.config import RunConfig
from yoloctl.datasets import DatasetVersionRecord
from yoloctl.exceptions import YoloCtlError
from yoloctl.quantization import (
    assessment_settings,
    build_artifact_metadata,
    compare_against_baseline,
    load_export_manifest,
    prepare_calibration_dataset,
)
from yoloctl.ultralytics_runner import run_export
from yoloctl.yamlio import dump_yaml


class QuantizationTests(unittest.TestCase):
    def _record_for_root(self, root: Path) -> DatasetVersionRecord:
        return DatasetVersionRecord(
            dataset_key="demo",
            version_id="dsv-demo-v1",
            version_label="v1",
            task="detect",
            source_type="materialized_local",
            sync_state="local_only",
            class_names=["card", "chip"],
            splits={"train": "images/train", "val": "images/val"},
            split_counts={"train": {"images": 4, "labels": 4}, "val": {"images": 1, "labels": 1}},
            materialized_path=str(root / "materialized"),
            dataset_yaml_path=str(root / "configs" / "datasets" / "demo.yaml"),
        )

    def _build_materialized_dataset(self, root: Path) -> None:
        for split in ("train", "val"):
            (root / "materialized" / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "materialized" / "labels" / split).mkdir(parents=True, exist_ok=True)
        image_names = ["sample-0.jpg", "sample-1.jpg", "sample-2.jpg", "sample-3.jpg"]
        labels = {
            "sample-0.txt": "0 0.5 0.5 0.2 0.2\n",
            "sample-1.txt": "0 0.5 0.5 0.2 0.2\n",
            "sample-2.txt": "0 0.5 0.5 0.2 0.2\n",
            "sample-3.txt": "1 0.5 0.5 0.2 0.2\n",
        }
        for image_name in image_names:
            (root / "materialized" / "images" / "train" / image_name).write_bytes(b"image")
        for label_name, payload in labels.items():
            (root / "materialized" / "labels" / "train" / label_name).write_text(payload, encoding="utf-8")
        (root / "materialized" / "images" / "val" / "val-0.jpg").write_bytes(b"image")
        (root / "materialized" / "labels" / "val" / "val-0.txt").write_text("", encoding="utf-8")

    def _run_config(self, root: Path) -> RunConfig:
        config_path = root / "configs" / "runs" / "run.yaml"
        dump_yaml(
            config_path,
            {
                "run_id": "demo-run",
                "task": "detect",
                "dataset_key": "demo",
                "dataset_version_id": "dsv-demo-v1",
                "model": {"family": "yolo26", "size": "n"},
                "train": {"imgsz": 640, "batch": 4, "device": "0", "project": "artifacts/runs", "name": "demo"},
                "export": {
                    "formats": ["engine"],
                    "precision_targets": ["int8"],
                    "assessment": {"required": True},
                },
                "tracking": {},
            },
        )
        return RunConfig.from_file(config_path)

    def test_prepare_calibration_dataset_is_deterministic_and_writes_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                self._build_materialized_dataset(repo_root)
                record = self._record_for_root(repo_root)
                dataset_yaml = Path(record.dataset_yaml_path)
                dump_yaml(dataset_yaml, {"path": str(repo_root / "materialized"), "train": "images/train", "val": "images/val", "names": {0: "card", 1: "chip"}})
                export_config = {
                    "calibration": {
                        "strategy": "class_balanced",
                        "fraction": 0.5,
                        "min_images": 2,
                        "max_images": 2,
                        "seed": 7,
                    }
                }
                first = prepare_calibration_dataset(record, "demo-run-a", export_config, dataset_yaml, dry_run=False)
                second = prepare_calibration_dataset(record, "demo-run-b", export_config, dataset_yaml, dry_run=True)
                self.assertEqual(first["selected_images"], second["selected_images"])
                self.assertTrue(Path(first["dataset_yaml"]).exists())
                self.assertIn("chip", first["class_histogram"])
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_prepare_calibration_dataset_can_fallback_to_fraction(self) -> None:
        record = DatasetVersionRecord(
            dataset_key="demo",
            version_id="dsv-demo-v1",
            version_label="v1",
            task="detect",
            source_type="materialized_local",
            sync_state="local_only",
            class_names=["card"],
            splits={"train": "images/train", "val": "images/val"},
            split_counts={"train": {"images": 100, "labels": 100}},
            materialized_path=None,
        )
        result = prepare_calibration_dataset(
            record=record,
            run_id="demo-run",
            export_config={"calibration": {"allow_fraction_fallback": True, "fraction": 0.1}},
            dataset_yaml=Path("configs/datasets/example-detect.yaml"),
            dry_run=False,
        )
        self.assertEqual(result["mode"], "fraction_fallback")
        self.assertEqual(result["fraction"], 0.1)

    def test_run_export_dry_run_uses_fraction_one_with_materialized_calibration_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            repo_root = root / "repo"
            repo_root.mkdir()
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "runs").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                self._build_materialized_dataset(repo_root)
                record = self._record_for_root(repo_root)
                dataset_yaml = Path(record.dataset_yaml_path)
                dump_yaml(
                    dataset_yaml,
                    {
                        "path": str(repo_root / "materialized"),
                        "train": "images/train",
                        "val": "images/val",
                        "names": {0: "card", 1: "chip"},
                    },
                )
                run_config = self._run_config(repo_root)
                result = run_export(
                    run_config=run_config,
                    dataset_record=record,
                    dataset_yaml=dataset_yaml,
                    weights="artifacts/runs/demo/weights/best.pt",
                    dry_run=True,
                )
                self.assertIn("fraction=1.0", result["commands"][0] if len(result["commands"]) == 1 else result["commands"][-1])
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_build_artifact_metadata_reports_file_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "model.engine"
            source.write_bytes(b"1234567890")
            metadata = build_artifact_metadata(
                run_id="demo-run",
                weights_path="weights/best.pt",
                job={"format": "engine", "int8": True},
                engine="python",
                required=True,
                source_path=source,
                archived_path=source,
            )
            self.assertEqual(metadata["precision"], "int8")
            self.assertGreater(metadata["size_mb"], 0)

    def test_compare_against_baseline_rejects_large_int8_drop_and_warns_on_recall(self) -> None:
        settings = assessment_settings({})
        gate = compare_against_baseline(
            baseline_metrics={"map50_95": 0.80, "map50": 0.90, "recall": 0.88},
            candidate_metrics={"map50_95": 0.10, "map50": 0.89, "recall": 0.70},
            precision="int8",
            settings=settings,
        )
        self.assertFalse(gate["accepted"])
        self.assertIn("map50_95 dropped", gate["reasons"][0])
        self.assertTrue(any("recall dropped" in warning for warning in gate["warnings"]))

    def test_load_export_manifest_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["YOLOCTL_ROOT"] = tmpdir
            try:
                with self.assertRaises(YoloCtlError):
                    load_export_manifest("missing-run")
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_assess_export_artifacts_fails_when_required_artifact_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "configs" / "runs").mkdir(parents=True)
            (repo_root / "artifacts" / "manifests" / "demo-run").mkdir(parents=True)
            artifact_path = repo_root / "artifacts" / "exports" / "demo-run" / "engine-int8" / "demo.engine"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(b"engine")
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                run_config = self._run_config(repo_root)
                record = DatasetVersionRecord(
                    dataset_key="demo",
                    version_id="dsv-demo-v1",
                    version_label="v1",
                    task="detect",
                    source_type="materialized_local",
                    sync_state="local_only",
                    class_names=["card"],
                    splits={"train": "images/train", "val": "images/val"},
                    materialized_path=str(repo_root / "materialized"),
                )
                dataset_yaml = repo_root / "configs" / "datasets" / "demo.yaml"
                dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
                dump_yaml(dataset_yaml, {"path": str(repo_root / "materialized"), "train": "images/train", "val": "images/val", "names": {0: "card"}})
                with (repo_root / "artifacts" / "manifests" / "demo-run" / "export.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "result": {
                                "artifacts": [
                                    {
                                        "path": str(artifact_path),
                                        "precision": "int8",
                                        "variant": "engine-int8",
                                        "required": True,
                                    }
                                ]
                            }
                        },
                        handle,
                    )
                validation_results = [
                    {"metrics": {"metrics/mAP50-95(B)": 0.8, "metrics/mAP50(B)": 0.9, "metrics/recall(B)": 0.88}, "speed": {"inference": 12.0}},
                    {"metrics": {"metrics/mAP50-95(B)": 0.1, "metrics/mAP50(B)": 0.85, "metrics/recall(B)": 0.80}, "speed": {"inference": 5.0}},
                ]
                with patch("yoloctl.cli.run_model_validation", side_effect=validation_results), patch(
                    "yoloctl.cli.run_benchmark",
                    return_value={"results": {"fps": 60.0, "latency_ms": 5.0}},
                ):
                    result = _assess_export_artifacts(
                        run_config=run_config,
                        record=record,
                        dataset_yaml=dataset_yaml,
                        weights=str(repo_root / "weights" / "best.pt"),
                        dry_run=False,
                    )
                self.assertTrue(result["summary"]["failed"])
                self.assertEqual(result["summary"]["required_failures"], 1)
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_assess_export_artifacts_raises_when_artifact_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "configs" / "runs").mkdir(parents=True)
            (repo_root / "artifacts" / "manifests" / "demo-run").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                run_config = self._run_config(repo_root)
                record = DatasetVersionRecord(
                    dataset_key="demo",
                    version_id="dsv-demo-v1",
                    version_label="v1",
                    task="detect",
                    source_type="materialized_local",
                    sync_state="local_only",
                    class_names=["card"],
                    splits={"train": "images/train", "val": "images/val"},
                    materialized_path=str(repo_root / "materialized"),
                )
                dataset_yaml = repo_root / "configs" / "datasets" / "demo.yaml"
                dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
                dump_yaml(dataset_yaml, {"path": str(repo_root / "materialized"), "train": "images/train", "val": "images/val", "names": {0: "card"}})
                with (repo_root / "artifacts" / "manifests" / "demo-run" / "export.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "result": {
                                "artifacts": [
                                    {
                                        "path": str(repo_root / "missing.engine"),
                                        "precision": "int8",
                                        "variant": "engine-int8",
                                        "required": True,
                                    }
                                ]
                            }
                        },
                        handle,
                    )
                with patch(
                    "yoloctl.cli.run_model_validation",
                    return_value={"metrics": {"metrics/mAP50-95(B)": 0.8}, "speed": {"inference": 12.0}},
                ):
                    with self.assertRaises(YoloCtlError):
                        _assess_export_artifacts(
                            run_config=run_config,
                            record=record,
                            dataset_yaml=dataset_yaml,
                            weights=str(repo_root / "weights" / "best.pt"),
                            dry_run=False,
                        )
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)


if __name__ == "__main__":
    unittest.main()

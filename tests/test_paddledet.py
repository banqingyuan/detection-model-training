from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from yoloctl.cli import (
    handle_cost_estimate,
    handle_export_run,
    handle_quant_qat_run,
    handle_train_resume,
    handle_vast_advise,
)
from yoloctl.config import RunConfig
from yoloctl.datasets import DatasetVersionRecord, save_dataset_record
from yoloctl.exceptions import YoloCtlError
from yoloctl.paddledet_dataset import prepare_paddledet_dataset
from yoloctl.paddledet_runner import run_paddledet_train, run_paddledet_validation
from yoloctl.yamlio import dump_yaml


class PaddleDetTests(unittest.TestCase):
    def _write_image(self, path: Path, size: tuple[int, int] = (100, 50)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size=size, color=(255, 255, 255)).save(path)

    def _setup_repo(self, root: Path) -> tuple[Path, Path, Path]:
        repo_root = root / "repo"
        repo_root.mkdir()
        os.environ["YOLOCTL_ROOT"] = str(repo_root)
        dataset_root = repo_root / "materialized"
        for split in ("train", "val"):
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        self._write_image(dataset_root / "images" / "train" / "sample-a.jpg", size=(100, 50))
        self._write_image(dataset_root / "images" / "train" / "sample-b.jpg", size=(80, 80))
        self._write_image(dataset_root / "images" / "val" / "sample-val.jpg", size=(64, 32))
        (dataset_root / "labels" / "train" / "sample-a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        (dataset_root / "labels" / "train" / "sample-b.txt").write_text("", encoding="utf-8")
        (dataset_root / "labels" / "val" / "sample-val.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")

        dataset_yaml = repo_root / "configs" / "datasets" / "demo.yaml"
        dump_yaml(
            dataset_yaml,
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "card", 1: "chip"},
            },
        )

        record = DatasetVersionRecord(
            dataset_key="demo",
            version_id="dsv-demo-v1",
            version_label="v1",
            task="detect",
            source_type="materialized_local",
            sync_state="local_only",
            class_names=["card", "chip"],
            splits={"train": "images/train", "val": "images/val"},
            split_counts={"train": {"images": 2, "labels": 2}, "val": {"images": 1, "labels": 1}},
            materialized_path=str(dataset_root),
            dataset_yaml_path=str(dataset_yaml),
        )
        save_dataset_record(record)

        paddledet_root = repo_root / "external" / "PaddleDetection"
        (paddledet_root / "tools").mkdir(parents=True, exist_ok=True)
        (paddledet_root / "tools" / "train.py").write_text("print('train')\n", encoding="utf-8")
        (paddledet_root / "tools" / "eval.py").write_text("print('eval')\n", encoding="utf-8")
        paddledet_config = paddledet_root / "configs" / "picodet" / "picodet_s_demo.yml"
        dump_yaml(
            paddledet_config,
            {
                "metric": "COCO",
                "TrainDataset": {"name": "COCODataSet"},
                "EvalDataset": {"name": "COCODataSet"},
            },
        )
        return repo_root, dataset_yaml, paddledet_config

    def _write_paddledet_run_config(
        self,
        repo_root: Path,
        paddledet_config: Path,
        *,
        eval_weights: str | None = "weights/picodet_best.pdparams",
    ) -> Path:
        config_path = repo_root / "configs" / "runs" / "picodet-demo.yaml"
        paddledet_payload: dict[str, object] = {
            "root": str(paddledet_config.parents[2]),
            "config": str(paddledet_config.relative_to(paddledet_config.parents[2])),
            "pretrain_weights": "https://example.com/picodet_pretrained.pdparams",
            "device": "0,1",
            "eval": True,
            "overrides": {"num_classes": 2},
        }
        if eval_weights:
            paddledet_payload["eval_weights"] = eval_weights
        dump_yaml(
            config_path,
            {
                "run_id": "picodet-demo",
                "backend": "paddledet",
                "task": "detect",
                "dataset_key": "demo",
                "dataset_version_id": "dsv-demo-v1",
                "train": {"device": "0,1", "project": "artifacts/runs", "name": "picodet-demo"},
                "export": {},
                "tracking": {"vast_profile": "configs/vast/profiles/default-docker-detect.yaml"},
                "paddledet": paddledet_payload,
            },
        )
        return config_path

    def tearDown(self) -> None:
        os.environ.pop("YOLOCTL_ROOT", None)

    def test_run_config_parses_paddledet_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root, _, paddledet_config = self._setup_repo(Path(tmpdir))
            config_path = self._write_paddledet_run_config(repo_root, paddledet_config)
            run_config = RunConfig.from_file(config_path)
            self.assertTrue(run_config.is_paddledet)
            self.assertIsNone(run_config.model)
            self.assertEqual(run_config.paddledet.eval_weights, "weights/picodet_best.pdparams")

    def test_run_config_requires_paddledet_root_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir) / "repo"
            repo_root.mkdir()
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            config_path = repo_root / "configs" / "runs" / "broken.yaml"
            dump_yaml(
                config_path,
                {
                    "run_id": "broken",
                    "backend": "paddledet",
                    "task": "detect",
                    "dataset_key": "demo",
                    "train": {},
                    "export": {},
                    "tracking": {},
                    "paddledet": {},
                },
            )
            with self.assertRaises(YoloCtlError):
                RunConfig.from_file(config_path)

    def test_prepare_paddledet_dataset_writes_coco_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root, dataset_yaml, _ = self._setup_repo(Path(tmpdir))
            from yoloctl.datasets import get_dataset_record

            summary = prepare_paddledet_dataset(
                record=get_dataset_record("demo", version_id="dsv-demo-v1"),
                dataset_yaml=dataset_yaml,
                run_id="picodet-demo",
                require_val=True,
                dry_run=False,
            )
            self.assertEqual(summary["train_images"], 2)
            self.assertEqual(summary["val_images"], 1)
            train_json = Path(summary["train_annotation_path"])
            payload = json.loads(train_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["images"]), 2)
            self.assertEqual(len(payload["annotations"]), 1)
            self.assertEqual(payload["annotations"][0]["bbox"], [30.0, 15.0, 40.0, 20.0])
            self.assertEqual(payload["categories"][1]["name"], "chip")
            self.assertTrue((repo_root / "artifacts" / "paddledet_datasets").exists())

    def test_prepare_paddledet_dataset_rejects_invalid_class_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, dataset_yaml, _ = self._setup_repo(Path(tmpdir))
            bad_label = Path(os.environ["YOLOCTL_ROOT"]) / "materialized" / "labels" / "train" / "sample-a.txt"
            bad_label.write_text("5 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            from yoloctl.datasets import get_dataset_record

            with self.assertRaises(YoloCtlError):
                prepare_paddledet_dataset(
                    record=get_dataset_record("demo", version_id="dsv-demo-v1"),
                    dataset_yaml=dataset_yaml,
                    run_id="picodet-demo",
                    require_val=True,
                    dry_run=True,
                )

    def test_run_paddledet_train_dry_run_builds_distributed_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root, dataset_yaml, paddledet_config = self._setup_repo(Path(tmpdir))
            config_path = self._write_paddledet_run_config(repo_root, paddledet_config)
            run_config = RunConfig.from_file(config_path)
            from yoloctl.datasets import get_dataset_record

            result = run_paddledet_train(
                run_config=run_config,
                dataset_record=get_dataset_record("demo", version_id="dsv-demo-v1"),
                dataset_yaml=dataset_yaml,
                weights="custom-pretrain.pdparams",
                dry_run=True,
            )
            self.assertEqual(result["backend"], "paddledet")
            self.assertIn("paddle.distributed.launch", result["command"])
            self.assertIn("pretrain_weights=custom-pretrain.pdparams", result["command"])
            self.assertIn("TrainDataset.anno_path=", result["command"])
            self.assertEqual(result["config_overrides"]["num_classes"], 2)

    def test_run_paddledet_validation_dry_run_uses_eval_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root, dataset_yaml, paddledet_config = self._setup_repo(Path(tmpdir))
            config_path = self._write_paddledet_run_config(repo_root, paddledet_config)
            run_config = RunConfig.from_file(config_path)
            from yoloctl.datasets import get_dataset_record

            result = run_paddledet_validation(
                run_config=run_config,
                dataset_record=get_dataset_record("demo", version_id="dsv-demo-v1"),
                dataset_yaml=dataset_yaml,
                weights=None,
                dry_run=True,
            )
            self.assertIn("tools/eval.py", result["command"])
            self.assertEqual(result["weights"], "weights/picodet_best.pdparams")

    def test_run_paddledet_validation_requires_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root, dataset_yaml, paddledet_config = self._setup_repo(Path(tmpdir))
            config_path = self._write_paddledet_run_config(repo_root, paddledet_config, eval_weights=None)
            run_config = RunConfig.from_file(config_path)
            from yoloctl.datasets import get_dataset_record

            with self.assertRaises(YoloCtlError):
                run_paddledet_validation(
                    run_config=run_config,
                    dataset_record=get_dataset_record("demo", version_id="dsv-demo-v1"),
                    dataset_yaml=dataset_yaml,
                    weights=None,
                    dry_run=True,
                )

    def test_unsupported_commands_raise_for_paddledet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root, _, paddledet_config = self._setup_repo(Path(tmpdir))
            config_path = self._write_paddledet_run_config(repo_root, paddledet_config)
            qat_config = repo_root / "configs" / "quant" / "picodet-qat.yaml"
            dump_yaml(qat_config, {"base_run_config": str(config_path)})

            with self.assertRaises(YoloCtlError):
                handle_export_run(SimpleNamespace(config=str(config_path), weights=None, engine="auto", dry_run=True))
            with self.assertRaises(YoloCtlError):
                handle_train_resume(SimpleNamespace(config=str(config_path), checkpoint=None, engine="auto", dry_run=True))
            with self.assertRaises(YoloCtlError):
                handle_cost_estimate(SimpleNamespace(config=str(config_path), profile=None, limit=5, dry_run=True))
            with self.assertRaises(YoloCtlError):
                handle_vast_advise(SimpleNamespace(config=str(config_path), profile=None, limit=5, dry_run=True))
            with self.assertRaises(YoloCtlError):
                handle_quant_qat_run(SimpleNamespace(config=str(qat_config), weights=None, dry_run=True))

    def test_handle_train_run_and_val_dry_run_emit_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root, _, paddledet_config = self._setup_repo(Path(tmpdir))
            config_path = self._write_paddledet_run_config(repo_root, paddledet_config)
            from yoloctl.cli import handle_train_run, handle_train_val

            output = io.StringIO()
            with redirect_stdout(output):
                handle_train_run(SimpleNamespace(config=str(config_path), weights=None, engine="auto", dry_run=True))
            train_manifest = json.loads(output.getvalue())
            self.assertEqual(train_manifest["result"]["backend"], "paddledet")

            output = io.StringIO()
            with redirect_stdout(output):
                handle_train_val(SimpleNamespace(config=str(config_path), weights=None, engine="auto", dry_run=True))
            val_manifest = json.loads(output.getvalue())
            self.assertEqual(val_manifest["result"]["backend"], "paddledet")


if __name__ == "__main__":
    unittest.main()

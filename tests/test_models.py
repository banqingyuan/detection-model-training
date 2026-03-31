from __future__ import annotations

import unittest
from pathlib import Path

from yoloctl.exceptions import YoloCtlError
from yoloctl.config import RunConfig
from yoloctl.models import validate_model
from yoloctl.ultralytics_runner import export_jobs
from yoloctl.yamlio import dump_yaml


class ModelTests(unittest.TestCase):
    def test_yolo11_weights_name(self) -> None:
        spec = validate_model("yolo11", "n")
        self.assertEqual(spec.weights_name, "yolo11n.pt")

    def test_yolo26_weights_name(self) -> None:
        spec = validate_model("yolo26", "x")
        self.assertEqual(spec.weights_name, "yolo26x.pt")

    def test_invalid_family_raises(self) -> None:
        with self.assertRaises(YoloCtlError):
            validate_model("yolo99", "n")

    def test_export_jobs_respect_format_precision_support(self) -> None:
        config_path = Path("artifacts/tmp-test-run.yaml")
        dump_yaml(
            config_path,
            {
                "run_id": "demo",
                "task": "detect",
                "dataset_key": "demo",
                "dataset_version_id": "dsv-20260325T000000Z-v1",
                "dataset_version_label": "v1",
                "model": {"family": "yolo26", "size": "n"},
                "train": {"imgsz": 640, "device": "0"},
                "export": {
                    "formats": ["torchscript", "onnx", "engine"],
                    "precision_targets": ["fp16", "int8"],
                    "end2end_modes": [True],
                    "calibration": {"fraction": 0.25},
                },
                "tracking": {},
            },
        )
        run_config = RunConfig.from_file(config_path)
        jobs = export_jobs(run_config, Path("configs/datasets/example-detect.yaml"))
        by_format = {job["format"]: job for job in jobs}
        self.assertTrue(by_format["torchscript"]["half"])
        self.assertFalse(by_format["torchscript"]["int8"])
        self.assertTrue(by_format["onnx"]["half"])
        self.assertFalse(by_format["onnx"]["int8"])
        self.assertFalse(by_format["engine"]["half"])
        self.assertTrue(by_format["engine"]["int8"])
        config_path.unlink()


if __name__ == "__main__":
    unittest.main()

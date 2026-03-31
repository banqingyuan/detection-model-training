from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from yoloctl.advisory import estimate_run_workload, recommend_offers
from yoloctl.config import RunConfig, VastProfile
from yoloctl.datasets import DatasetVersionRecord
from yoloctl.yamlio import dump_yaml


class AdvisoryTests(unittest.TestCase):
    def _build_run_config(self, root: Path, family: str = "yolo11", size: str = "n") -> RunConfig:
        config_path = root / "run.yaml"
        dump_yaml(
            config_path,
            {
                "run_id": "demo-run",
                "task": "detect",
                "dataset_key": "demo",
                "dataset_version_id": "dsv-demo-v1",
                "model": {"family": family, "size": size},
                "train": {
                    "epochs": 40,
                    "imgsz": 640,
                    "batch": 16,
                    "project": "artifacts/runs",
                    "name": "demo-run",
                },
                "export": {
                    "include_best_pt": True,
                    "formats": ["torchscript", "onnx", "engine"],
                    "precision_targets": ["fp16"],
                },
                "tracking": {"vast_profile": "configs/vast/profiles/default-docker-detect.yaml"},
            },
        )
        return RunConfig.from_file(config_path)

    def _build_record(self) -> DatasetVersionRecord:
        return DatasetVersionRecord(
            dataset_key="demo",
            version_id="dsv-demo-v1",
            version_label="v1",
            task="detect",
            source_type="local_archive",
            sync_state="cloud_synced",
            class_names=["card", "chip"],
            splits={"train": "images/train", "val": "images/val"},
            split_counts={
                "train": {"images": 12000, "labels": 12000},
                "val": {"images": 2000, "labels": 2000},
            },
            source_archive_size=2 * 1024 * 1024 * 1024,
            cloud_uri="oss://bucket/demo/source.zip",
            cloud_manifest_uri="oss://bucket/demo/manifest.yaml",
            materialized_path="artifacts/datasets/demo/v1",
            dataset_yaml_path="configs/datasets/demo.yaml",
        )

    def test_estimate_run_workload_reports_vram_and_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._build_run_config(Path(tmpdir))
            record = self._build_record()
            profile = VastProfile(name="demo", search={"storage": 200}, launch={"disk": 200})
            estimate = estimate_run_workload(run_config, record, profile=profile)
            self.assertGreater(estimate["estimated_required_vram_gb"], 0)
            self.assertGreater(estimate["estimated_required_disk_gb"], 0)
            self.assertEqual(estimate["train_images"], 12000)

    def test_recommend_offers_prefers_lower_total_cost_not_low_hourly_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = self._build_run_config(Path(tmpdir), family="yolo26", size="m")
            record = self._build_record()
            profile = VastProfile(name="demo", search={"storage": 200}, launch={"disk": 200})
            offers = [
                {
                    "id": 1,
                    "gpu_name": "RTX 4090",
                    "gpu_ram": 24564,
                    "disk_space": 600,
                    "dlperf": 55.0,
                    "dph_total": 0.21,
                    "inet_down_cost": 0.003,
                    "inet_up_cost": 0.004,
                    "reliability": 0.992,
                    "geolocation": "Cheap but slow",
                    "avail_vol_dph": 0.002,
                },
                {
                    "id": 2,
                    "gpu_name": "RTX 4090",
                    "gpu_ram": 24564,
                    "disk_space": 600,
                    "dlperf": 125.0,
                    "dph_total": 0.33,
                    "inet_down_cost": 0.003,
                    "inet_up_cost": 0.004,
                    "reliability": 0.998,
                    "geolocation": "Faster and cheaper overall",
                    "avail_vol_dph": 0.002,
                },
                {
                    "id": 3,
                    "gpu_name": "RTX 4090",
                    "gpu_ram": 6144,
                    "disk_space": 600,
                    "dlperf": 200.0,
                    "dph_total": 0.40,
                    "inet_down_cost": 0.003,
                    "inet_up_cost": 0.004,
                    "reliability": 0.999,
                    "geolocation": "Too little VRAM",
                },
            ]
            result = recommend_offers(run_config, record, profile, offers)
            self.assertEqual(result["recommendations"][0]["offer_id"], 2)
            self.assertEqual(result["recommendations"][0]["rank"], 1)
            self.assertEqual(result["rejected"][0]["offer_id"], 3)


if __name__ == "__main__":
    unittest.main()

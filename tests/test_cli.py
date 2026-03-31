from __future__ import annotations

import unittest

from yoloctl.cli import build_parser


class CliTests(unittest.TestCase):
    def test_dataset_register_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "dataset",
                "register",
                "--dataset-key",
                "demo",
                "--version-label",
                "v1",
                "--source-archive",
                "/tmp/demo.zip",
                "--class-name",
                "card",
                "--materialized-path",
                "/workspace/datasets/demo/v1",
            ]
        )
        self.assertEqual(args.dataset_key, "demo")
        self.assertEqual(args.version_label, "v1")

    def test_dataset_sync_push_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "dataset",
                "sync",
                "push",
                "--dataset-key",
                "demo",
                "--version-id",
                "dsv-20260325T000000Z-v1",
                "--dry-run",
            ]
        )
        self.assertEqual(args.dataset_key, "demo")
        self.assertTrue(args.dry_run)

    def test_dataset_import_release_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "dataset",
                "import-release",
                "--manifest-uri",
                "/tmp/release.json",
                "--dataset-key",
                "demo",
            ]
        )
        self.assertEqual(args.manifest_uri, "/tmp/release.json")
        self.assertEqual(args.dataset_key, "demo")

    def test_dataset_merge_draft_alias_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "dataset",
                "merge-draft",
                "--from-release",
                "dsv-20260325T000000Z-release",
                "--version-label",
                "merge-v2",
            ]
        )
        self.assertEqual(args.from_release, "dsv-20260325T000000Z-release")
        self.assertEqual(args.version_label, "merge-v2")

    def test_dataset_review_open_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "dataset",
                "review",
                "open",
                "--dataset-key",
                "demo",
                "--version-id",
                "dsv-demo-v1",
                "--weights",
                "weights/best.pt",
                "--issues-report",
                "artifacts/reports/demo.json",
                "--split",
                "train,val",
            ]
        )
        self.assertEqual(args.dataset_key, "demo")
        self.assertEqual(args.version_id, "dsv-demo-v1")
        self.assertEqual(args.weights, "weights/best.pt")
        self.assertEqual(args.issues_report, "artifacts/reports/demo.json")

    def test_dataset_review_finalize_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "dataset",
                "review",
                "finalize",
                "--dataset-key",
                "demo",
                "--version-id",
                "dsv-demo-v1",
                "--session-id",
                "rsv-20260326T000000Z",
                "--new-version-label",
                "v2-reviewed",
            ]
        )
        self.assertEqual(args.session_id, "rsv-20260326T000000Z")
        self.assertEqual(args.new_version_label, "v2-reviewed")

    def test_train_run_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["train", "run", "--config", "configs/runs/yolo11n-sample.yaml", "--dry-run"])
        self.assertEqual(args.engine, "auto")
        self.assertTrue(args.dry_run)

    def test_vast_advise_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["vast", "advise", "--config", "configs/runs/yolo11n-sample.yaml"])
        self.assertEqual(args.config, "configs/runs/yolo11n-sample.yaml")
        self.assertEqual(args.limit, 10)

    def test_export_assess_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["export", "assess", "--config", "configs/runs/yolo26n-sample.yaml", "--dry-run"])
        self.assertEqual(args.config, "configs/runs/yolo26n-sample.yaml")
        self.assertTrue(args.dry_run)

    def test_quant_qat_run_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["quant", "qat", "run", "--config", "configs/quant/yolo26n-int8-qat.yaml"])
        self.assertEqual(args.config, "configs/quant/yolo26n-int8-qat.yaml")
        self.assertIsNone(args.weights)

    def test_cost_estimate_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["cost", "estimate", "--config", "configs/runs/yolo26n-sample.yaml", "--dry-run"])
        self.assertEqual(args.config, "configs/runs/yolo26n-sample.yaml")
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()

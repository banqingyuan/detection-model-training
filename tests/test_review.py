from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from yoloctl.datasets import create_dataset_record, get_dataset_record, save_dataset_record
from yoloctl.review import (
    PREDICTION_CACHE_VERSION,
    compare_label_objects_to_predictions,
    create_or_resume_review_session,
    finalize_review_session,
    get_item_payload,
    load_review_items,
)
from yoloctl.review_app import create_review_app


class ReviewTests(unittest.TestCase):
    def test_compare_label_objects_enriches_fp_with_overlapping_gt_context(self) -> None:
        gt_objects = [
            {
                "id": "gt-0",
                "class_id": 0,
                "class_name": "RK",
                "x1": 100.0,
                "y1": 100.0,
                "x2": 140.0,
                "y2": 160.0,
                "source": "gt",
            }
        ]
        prediction_objects = [
            {
                "id": "pred-0",
                "class_id": 0,
                "class_name": "RK",
                "x1": 101.0,
                "y1": 101.0,
                "x2": 141.0,
                "y2": 161.0,
                "source": "pred",
                "confidence": 0.95,
            },
            {
                "id": "pred-1",
                "class_id": 1,
                "class_name": "BK",
                "x1": 101.0,
                "y1": 101.0,
                "x2": 141.0,
                "y2": 161.0,
                "source": "pred",
                "confidence": 0.67,
            },
        ]

        summary, issue_items = compare_label_objects_to_predictions(
            gt_objects,
            prediction_objects,
            label_role="gt",
            gt_objects=gt_objects,
            draft_objects=[],
            image_size=(640, 360),
        )

        self.assertEqual(summary["fp"], 1)
        self.assertEqual(len(issue_items), 1)
        issue = issue_items[0]
        self.assertEqual(issue["type"], "fp")
        self.assertEqual(issue["pred_label"], "BK")
        self.assertEqual(issue["gt_label"], "RK")
        self.assertEqual(issue["gt_object_id"], "gt-0")
        self.assertEqual(issue["pred_object_id"], "pred-1")
        self.assertGreater(issue["iou"], 0.9)

    def _build_dataset(self, repo_root: Path) -> tuple[Path, Path]:
        materialized = repo_root / "artifacts" / "datasets" / "demo" / "v1"
        for split in ("train", "val"):
            (materialized / "images" / split).mkdir(parents=True, exist_ok=True)
            (materialized / "labels" / split).mkdir(parents=True, exist_ok=True)

        image_train = materialized / "images" / "train" / "sample-train.jpg"
        image_val = materialized / "images" / "val" / "sample-val.jpg"
        Image.new("RGB", (640, 360), color=(240, 240, 240)).save(image_train)
        Image.new("RGB", (640, 360), color=(220, 220, 220)).save(image_val)
        (materialized / "labels" / "train" / "sample-train.txt").write_text(
            "0 0.500000 0.500000 0.200000 0.300000\n",
            encoding="utf-8",
        )
        (materialized / "labels" / "val" / "sample-val.txt").write_text(
            "1 0.250000 0.250000 0.100000 0.100000\n",
            encoding="utf-8",
        )
        return materialized, image_train

    def _register_dataset(self, repo_root: Path, materialized: Path):
        record = create_dataset_record(
            dataset_key="demo",
            version_label="v1",
            version_id="dsv-demo-v1",
            class_names=["card", "chip"],
            splits={"train": "images/train", "val": "images/val"},
            materialized_path=str(materialized),
            source_type="materialized_local",
        )
        save_dataset_record(record)
        return record

    def test_create_or_resume_review_session_builds_items_from_issue_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                materialized, image_train = self._build_dataset(repo_root)
                record = self._register_dataset(repo_root, materialized)
                issues_path = repo_root / "issues.json"
                issues_path.write_text(
                    json.dumps(
                        [
                            {
                                "image_path": str(image_train),
                                "fp": 1,
                                "fn": 2,
                                "confusions": 1,
                                "fp_details": [{"pred": "chip"}],
                                "fn_details": [{"gt": "card"}],
                                "confusion_details": [{"gt": "card", "pred": "chip"}],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )

                session_dir, session = create_or_resume_review_session(
                    record,
                    selected_splits=["train"],
                    issues_report=str(issues_path),
                )
                self.assertEqual(session.selected_splits, ["train"])
                image_id = self._first_image_id(session_dir)
                payload = get_item_payload(session_dir, image_id=image_id)
                self.assertEqual(payload["item"]["issue_summary"]["severity"], 5)
                self.assertEqual(payload["item"]["pred_class_names"], ["chip"])
                resumed_dir, resumed = create_or_resume_review_session(
                    record,
                    selected_splits=["train"],
                    issues_report=str(issues_path),
                )
                self.assertEqual(resumed.session_id, session.session_id)
                self.assertEqual(resumed_dir, session_dir)
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_review_session_save_and_finalize_creates_new_dataset_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                materialized, _ = self._build_dataset(repo_root)
                record = self._register_dataset(repo_root, materialized)
                session_dir, _ = create_or_resume_review_session(record, selected_splits=["train"])
                item_payload = get_item_payload(session_dir, image_id=self._first_image_id(session_dir))
                item = item_payload["item"]
                item_payload["draft_objects"][0]["class_id"] = 1
                item_payload["draft_objects"][0]["class_name"] = "chip"

                from yoloctl.review import save_item_draft

                save_item_draft(
                    session_dir=session_dir,
                    image_id=item["image_id"],
                    draft_objects=item_payload["draft_objects"],
                    status="fixed",
                    note="class corrected",
                )
                finalized = finalize_review_session(session_dir, new_version_label="v2-reviewed", merge_note="fix bad labels")
                new_record = get_dataset_record("demo", version_id=finalized.version_id)
                self.assertEqual(new_record.parents, [record.version_id])
                self.assertEqual(new_record.metadata["review_session_id"], Path(session_dir).name)
                base_label = (materialized / "labels" / "train" / "sample-train.txt").read_text(encoding="utf-8")
                new_label = (_resolve_record_materialized(new_record) / "labels" / "train" / "sample-train.txt").read_text(
                    encoding="utf-8"
                )
                self.assertTrue(base_label.startswith("0 "))
                self.assertTrue(new_label.startswith("1 "))
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_review_api_lists_items_and_saves_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                materialized, _ = self._build_dataset(repo_root)
                record = self._register_dataset(repo_root, materialized)
                session_dir, _ = create_or_resume_review_session(record, selected_splits=["train"])
                image_id = self._first_image_id(session_dir)
                app = create_review_app(session_dir, start_indexer=False)
                client = TestClient(app)
                items = client.get("/api/items").json()["items"]
                self.assertEqual(len(items), 1)
                payload = client.get(f"/api/items/{image_id}").json()
                payload["draft_objects"][0]["x1"] += 10
                payload["draft_objects"][0]["x2"] += 10
                saved = client.post(
                    f"/api/items/{image_id}/save",
                    json={
                        "draft_objects": payload["draft_objects"],
                        "status": "fixed",
                        "note": "shifted box",
                    },
                ).json()
                self.assertEqual(saved["item"]["status"], "fixed")
                self.assertTrue(Path(saved["item"]["draft_label_path"]).exists())
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_review_api_can_apply_and_replace_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                materialized, _ = self._build_dataset(repo_root)
                record = self._register_dataset(repo_root, materialized)
                session_dir, _ = create_or_resume_review_session(
                    record,
                    selected_splits=["train"],
                    weights="weights/demo.pt",
                )
                image_id = self._first_image_id(session_dir)
                fake_predictions = [
                    {
                        "id": "pred-0",
                        "class_id": 1,
                        "class_name": "chip",
                        "x1": 10.0,
                        "y1": 10.0,
                        "x2": 50.0,
                        "y2": 50.0,
                        "source": "pred",
                        "confidence": 0.91,
                    }
                ]
                app = create_review_app(session_dir, start_indexer=False)
                client = TestClient(app)
                with patch("yoloctl.review.predict_item_objects", return_value=fake_predictions):
                    detail = client.post(f"/api/items/{image_id}/predict").json()
                    self.assertEqual(len(detail["prediction_objects"]), 1)
                    self.assertIn("baseline_issue_items", detail)
                    self.assertIn("draft_issue_items", detail)
                    self.assertIn("focus_box", detail["baseline_issue_items"][0])
                    target_draft_id = detail["draft_objects"][0]["id"]
                    adopted = client.post(
                        f"/api/items/{image_id}/apply-prediction",
                        json={"prediction_id": "pred-0", "mode": "append"},
                    ).json()
                    self.assertEqual(adopted["draft_objects"][-1]["class_name"], "chip")
                    replaced = client.post(
                        f"/api/items/{image_id}/apply-prediction",
                        json={
                            "prediction_id": "pred-0",
                            "mode": "replace_selected",
                            "target_draft_id": target_draft_id,
                        },
                    ).json()
                    self.assertEqual(replaced["draft_objects"][0]["class_name"], "chip")
                    class_only = client.post(
                        f"/api/items/{image_id}/apply-prediction",
                        json={
                            "prediction_id": "pred-0",
                            "mode": "class_only",
                            "target_draft_id": target_draft_id,
                        },
                    ).json()
                    self.assertEqual(class_only["draft_objects"][0]["class_name"], "chip")
                    box_only = client.post(
                        f"/api/items/{image_id}/apply-prediction",
                        json={
                            "prediction_id": "pred-0",
                            "mode": "box_only",
                            "target_draft_id": target_draft_id,
                        },
                    ).json()
                    self.assertEqual(box_only["draft_objects"][0]["x1"], 10.0)
                    replaced = client.post(f"/api/items/{image_id}/replace-with-predictions").json()
                    self.assertEqual(len(replaced["draft_objects"]), 1)
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_review_payload_keeps_baseline_diagnostics_while_draft_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                materialized, _ = self._build_dataset(repo_root)
                record = self._register_dataset(repo_root, materialized)
                session_dir, _ = create_or_resume_review_session(
                    record,
                    selected_splits=["train"],
                    weights="weights/demo.pt",
                )
                image_id = self._first_image_id(session_dir)
                fake_predictions = [
                    {
                        "id": "pred-0",
                        "class_id": 1,
                        "class_name": "chip",
                        "x1": 256.0,
                        "y1": 126.0,
                        "x2": 384.0,
                        "y2": 234.0,
                        "source": "pred",
                        "confidence": 0.91,
                    }
                ]
                with patch("yoloctl.review.predict_item_objects", return_value=fake_predictions):
                    payload = get_item_payload(session_dir, image_id=image_id, force_predict=True)
                    self.assertEqual(payload["baseline_issue_summary"]["cls"], 1)
                    self.assertEqual(payload["draft_issue_summary"]["cls"], 1)
                    payload["draft_objects"][0]["class_id"] = 1
                    payload["draft_objects"][0]["class_name"] = "chip"
                    from yoloctl.review import save_item_draft

                    saved = save_item_draft(
                        session_dir=session_dir,
                        image_id=image_id,
                        draft_objects=payload["draft_objects"],
                        status="fixed",
                    )
                self.assertEqual(saved["baseline_issue_summary"]["cls"], 1)
                self.assertEqual(saved["draft_issue_summary"]["severity"], 0)
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_review_payload_backfills_baseline_cache_for_old_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                materialized, _ = self._build_dataset(repo_root)
                record = self._register_dataset(repo_root, materialized)
                session_dir, _ = create_or_resume_review_session(
                    record,
                    selected_splits=["train"],
                    weights="weights/demo.pt",
                )
                image_id = self._first_image_id(session_dir)
                cache_path = session_dir / "cache" / "predictions" / f"{image_id}.json"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(
                        {
                            "objects": [
                                {
                                    "id": "pred-0",
                                    "class_id": 1,
                                    "class_name": "chip",
                                    "x1": 256.0,
                                    "y1": 126.0,
                                    "x2": 384.0,
                                    "y2": 234.0,
                                    "source": "pred",
                                    "confidence": 0.91,
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                payload = get_item_payload(session_dir, image_id=image_id)
                self.assertEqual(payload["baseline_issue_summary"]["cls"], 1)
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                self.assertEqual(cached["version"], PREDICTION_CACHE_VERSION)
                self.assertIn("baseline_issue_items", cached)
                self.assertIn("baseline_issue_summary", cached)
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def test_load_review_items_ignores_torn_trailing_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "artifacts" / "registry" / "datasets").mkdir(parents=True)
            (repo_root / "configs" / "datasets").mkdir(parents=True)
            os.environ["YOLOCTL_ROOT"] = str(repo_root)
            try:
                materialized, _ = self._build_dataset(repo_root)
                record = self._register_dataset(repo_root, materialized)
                session_dir, _ = create_or_resume_review_session(record, selected_splits=["train"])
                items_path = session_dir / "items.jsonl"
                items_path.write_text(
                    items_path.read_text(encoding="utf-8") + 'at": "2026-03-26T09:43:49.193601+00:00"}\n',
                    encoding="utf-8",
                )
                items = load_review_items(session_dir)
                self.assertEqual(len(items), 1)
            finally:
                os.environ.pop("YOLOCTL_ROOT", None)

    def _first_image_id(self, session_dir: Path) -> str:
        from yoloctl.review import load_review_items

        return load_review_items(session_dir)[0].image_id


def _resolve_record_materialized(record) -> Path:
    materialized = Path(record.materialized_path)
    if materialized.is_absolute():
        return materialized
    return Path(os.environ["YOLOCTL_ROOT"]) / materialized


if __name__ == "__main__":
    unittest.main()

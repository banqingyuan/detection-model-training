from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yoloctl.config import VastProfile
from yoloctl.vast import build_search_query, build_search_command, invoice_snapshot, show_instances
from yoloctl.yamlio import dump_yaml


class VastTests(unittest.TestCase):
    def test_profile_query_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.yaml"
            dump_yaml(
                profile_path,
                {
                    "name": "demo",
                    "search": {
                        "query_clauses": ["reliability > 0.98", "rentable=True"],
                        "order": "score-",
                        "limit": 10,
                    },
                    "launch": {"gpu_name": "RTX_4090", "num_gpus": "1", "image": "demo:image"},
                },
            )
            profile = VastProfile.from_file(profile_path)
            query = build_search_query(profile)
            command = build_search_command(profile)
            self.assertIn("gpu_name=RTX_4090", query)
            self.assertIn("num_gpus=1", query)
            self.assertEqual(command[0:3], ["vastai", "search", "offers"])

    def test_show_instances_sdk_uses_non_quiet_mode(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = []

            def show_instances(self, quiet: bool = False):
                self.calls.append(quiet)
                return [{"id": 123}]

        client = FakeClient()
        with patch("yoloctl.vast._load_vast_client", return_value=client):
            result = show_instances(use_sdk=True)
        self.assertEqual(client.calls, [False])
        self.assertEqual(result["response"], [{"id": 123}])

    def test_invoice_snapshot_sdk_uses_non_quiet_mode(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = []

            def show_invoices(self, **kwargs):
                self.calls.append(kwargs)
                return [{"id": 456}]

        client = FakeClient()
        with patch("yoloctl.vast._load_vast_client", return_value=client):
            result = invoice_snapshot(use_sdk=True, start_date="2026-03-01", end_date="2026-03-25")
        self.assertEqual(client.calls[0]["quiet"], False)
        self.assertEqual(client.calls[0]["start_date"], "2026-03-01")
        self.assertEqual(client.calls[0]["end_date"], "2026-03-25")
        self.assertEqual(result["response"], [{"id": 456}])


if __name__ == "__main__":
    unittest.main()

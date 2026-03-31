from __future__ import annotations

import unittest

from yoloctl.exceptions import YoloCtlError
from yoloctl.oss import build_oss_sync_command, format_command, parse_oss_uri


class OssTests(unittest.TestCase):
    def test_build_sync_command(self) -> None:
        command = build_oss_sync_command("oss://bucket/demo/", "/workspace/demo", delete=True)
        joined = format_command(command)
        self.assertIn("ossutil sync", joined)
        self.assertIn("--delete", joined)
        self.assertIn("/workspace/demo", joined)

    def test_parse_oss_uri(self) -> None:
        bucket, key = parse_oss_uri("oss://demo-bucket/datasets/demo/source.zip")
        self.assertEqual(bucket, "demo-bucket")
        self.assertEqual(key, "datasets/demo/source.zip")

    def test_parse_oss_uri_requires_object_key(self) -> None:
        with self.assertRaises(YoloCtlError):
            parse_oss_uri("oss://demo-bucket")


if __name__ == "__main__":
    unittest.main()

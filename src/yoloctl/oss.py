from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .exceptions import YoloCtlError


@dataclass
class OssCommandResult:
    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


def build_oss_sync_command(
    source_uri: str,
    destination: str,
    delete: bool = False,
    update: bool = True,
    jobs: int = 4,
    parallel: int = 4,
) -> list[str]:
    if not source_uri.startswith("oss://"):
        raise YoloCtlError("OSS sync source must start with oss://")
    command = [
        "ossutil",
        "sync",
        source_uri,
        destination,
        "-j",
        str(jobs),
        "--parallel",
        str(parallel),
    ]
    if update:
        command.append("-u")
    if delete:
        command.append("--delete")
    return command


def build_oss_cp_command(
    source: str | Path,
    destination: str | Path,
    update: bool = True,
) -> list[str]:
    command = ["ossutil", "cp", str(source), str(destination)]
    if update:
        command.append("-u")
    return command


def build_oss_stat_command(uri: str) -> list[str]:
    if not uri.startswith("oss://"):
        raise YoloCtlError("OSS stat target must start with oss://")
    return ["ossutil", "stat", uri]


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def parse_oss_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("oss://"):
        raise YoloCtlError("OSS target must start with oss://")
    parsed = urlparse(uri)
    bucket = parsed.netloc.strip()
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise YoloCtlError(f"OSS URI must include a bucket and object key: {uri}")
    return bucket, key


def _ossutil_available() -> bool:
    return shutil.which("ossutil") is not None


def _prefer_sdk_backend() -> bool:
    backend = os.getenv("YOLOCTL_OSS_BACKEND", "").strip().lower()
    if backend == "ossutil":
        return False
    if backend == "sdk":
        return True
    return not _ossutil_available()


def _resolve_oss_endpoint() -> str:
    endpoint = (
        os.getenv("YOLOCTL_OSS_ENDPOINT")
        or os.getenv("OSS_ENDPOINT")
        or os.getenv("OSS_COLLECTION_ENDPOINT")
    )
    if not endpoint:
        raise YoloCtlError(
            "OSS endpoint is required for SDK uploads. Set YOLOCTL_OSS_ENDPOINT, OSS_ENDPOINT, or OSS_COLLECTION_ENDPOINT."
        )
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    return endpoint


def _resolve_oss_region(endpoint: str) -> str:
    explicit = (
        os.getenv("YOLOCTL_OSS_REGION")
        or os.getenv("OSS_REGION")
        or os.getenv("OSS_COLLECTION_REGION")
    )
    if explicit:
        return explicit
    host = endpoint.split("://", 1)[-1]
    match = re.search(r"oss-([a-z0-9-]+)(?:-internal)?\.aliyuncs\.com$", host)
    if not match:
        raise YoloCtlError(
            "Unable to infer OSS region from endpoint. Set YOLOCTL_OSS_REGION or OSS_REGION explicitly."
        )
    return match.group(1)


def _sdk_bucket(uri: str) -> tuple[Any, str]:
    try:
        import oss2
        from oss2.credentials import EnvironmentVariableCredentialsProvider
    except ImportError as exc:
        raise YoloCtlError(
            "OSS SDK fallback requires the 'oss2' package. Install project dependencies before syncing."
        ) from exc

    missing = [
        name
        for name in ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET")
        if not os.getenv(name)
    ]
    if missing:
        raise YoloCtlError(f"Missing required OSS environment variables: {', '.join(missing)}")

    bucket_name, object_key = parse_oss_uri(uri)
    endpoint = _resolve_oss_endpoint()
    region = _resolve_oss_region(endpoint)
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    return bucket, object_key


def run_command(
    command: list[str],
    dry_run: bool = False,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        return None
    if shutil.which(command[0]) is None:
        raise YoloCtlError(f"Required command not found in PATH: {command[0]}")
    return subprocess.run(command, cwd=cwd, check=check, text=True, capture_output=True)


def upload_oss_object(source: Path, destination_uri: str) -> subprocess.CompletedProcess[str] | OssCommandResult:
    if _prefer_sdk_backend():
        if not source.exists():
            raise YoloCtlError(f"Source file does not exist: {source}")
        bucket, object_key = _sdk_bucket(destination_uri)
        try:
            bucket.put_object_from_file(object_key, str(source))
        except Exception as exc:
            raise YoloCtlError(f"Failed to upload '{source}' to '{destination_uri}': {exc}") from exc
        return OssCommandResult(
            args=["oss-sdk", "put_object_from_file", str(source), destination_uri],
            returncode=0,
            stdout=f"Uploaded {source.name} to {destination_uri}",
        )
    return run_command(build_oss_cp_command(source, destination_uri), check=True)  # type: ignore[return-value]


def download_oss_object(source_uri: str, destination: Path) -> subprocess.CompletedProcess[str] | OssCommandResult:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _prefer_sdk_backend():
        bucket, object_key = _sdk_bucket(source_uri)
        try:
            bucket.get_object_to_file(object_key, str(destination))
        except Exception as exc:
            raise YoloCtlError(f"Failed to download '{source_uri}' to '{destination}': {exc}") from exc
        return OssCommandResult(
            args=["oss-sdk", "get_object_to_file", source_uri, str(destination)],
            returncode=0,
            stdout=f"Downloaded {source_uri} to {destination}",
        )
    return run_command(build_oss_cp_command(source_uri, destination), check=True)  # type: ignore[return-value]


def stat_oss_object(uri: str) -> subprocess.CompletedProcess[str] | OssCommandResult:
    if _prefer_sdk_backend():
        bucket, object_key = _sdk_bucket(uri)
        try:
            exists = bucket.object_exists(object_key)
        except Exception as exc:
            return OssCommandResult(
                args=["oss-sdk", "object_exists", uri],
                returncode=1,
                stderr=str(exc),
            )
        return OssCommandResult(
            args=["oss-sdk", "object_exists", uri],
            returncode=0 if exists else 1,
            stdout="Object exists." if exists else "",
            stderr="" if exists else f"Object not found: {uri}",
        )
    return run_command(build_oss_stat_command(uri), check=False)  # type: ignore[return-value]

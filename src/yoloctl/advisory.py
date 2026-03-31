from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import RunConfig, VastProfile
from .datasets import DatasetVersionRecord
from .exceptions import YoloCtlError
from .paths import project_root
from .ultralytics_runner import export_jobs

BASE_IMAGES_PER_HOUR_AT_DLPERF_100 = 180000.0
BASE_BATCH_SIZE = 16.0
BASE_IMGSZ = 640.0
STARTUP_BUFFER_HOURS = 0.12
GIB = 1024.0 ** 3

MODEL_COMPLEXITY = {
    "yolo11": {"n": 1.00, "s": 1.75, "m": 3.05, "l": 4.90, "x": 7.20},
    "yolo26": {"n": 1.15, "s": 2.00, "m": 3.45, "l": 5.45, "x": 8.10},
}

MODEL_VRAM_BASE_GB = {
    "yolo11": {"n": 3.5, "s": 5.0, "m": 7.5, "l": 10.5, "x": 14.0},
    "yolo26": {"n": 4.0, "s": 5.8, "m": 8.6, "l": 12.0, "x": 15.5},
}

MODEL_VRAM_PER_IMAGE_GB = {
    "yolo11": {"n": 0.18, "s": 0.22, "m": 0.30, "l": 0.40, "x": 0.55},
    "yolo26": {"n": 0.20, "s": 0.24, "m": 0.33, "l": 0.44, "x": 0.60},
}

WEIGHTS_SIZE_GB = {
    "n": 0.015,
    "s": 0.028,
    "m": 0.055,
    "l": 0.095,
    "x": 0.145,
}

EXPORT_BASE_HOURS_AT_DLPERF_100 = {
    "torchscript": 0.03,
    "onnx": 0.05,
    "engine": 0.12,
    "openvino": 0.08,
}

EXPORT_ARTIFACT_MULTIPLIER = {
    "torchscript": 1.2,
    "onnx": 1.35,
    "engine": 2.4,
    "openvino": 1.7,
}


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _dataset_archive_gib(record: DatasetVersionRecord) -> float:
    if record.source_archive_size:
        return record.source_archive_size / GIB
    if record.source_archive_path:
        archive_path = Path(record.source_archive_path)
        if not archive_path.is_absolute():
            archive_path = project_root() / archive_path
        if archive_path.exists():
            return archive_path.stat().st_size / GIB
    return 0.0


def _train_image_count(record: DatasetVersionRecord) -> int:
    return int(record.split_counts.get("train", {}).get("images", 0))


def _split_image_count(record: DatasetVersionRecord, split_name: str) -> int:
    return int(record.split_counts.get(split_name, {}).get("images", 0))


def _launch_disk_gb(profile: VastProfile | None) -> float:
    if profile is None:
        return 0.0
    value = profile.launch.get("disk") or profile.search.get("storage") or 0
    return float(value)


def _search_storage_gb(profile: VastProfile | None) -> float:
    if profile is None:
        return 0.0
    return float(profile.search.get("storage") or profile.launch.get("disk") or 0)


def _complexity(run_config: RunConfig) -> float:
    return MODEL_COMPLEXITY[run_config.model.family][run_config.model.size]


def _batch_size(run_config: RunConfig) -> float:
    batch = run_config.train.get("batch", BASE_BATCH_SIZE)
    return max(1.0, float(batch))


def _imgsz(run_config: RunConfig) -> float:
    imgsz = run_config.train.get("imgsz", BASE_IMGSZ)
    return max(64.0, float(imgsz))


def _export_artifacts_gib(run_config: RunConfig) -> float:
    weights_size = WEIGHTS_SIZE_GB[run_config.model.size]
    total = weights_size if run_config.export.get("include_best_pt", True) else 0.0
    jobs = export_jobs(run_config, Path("placeholder-dataset.yaml"))
    for job in jobs:
        fmt = str(job.get("format"))
        multiplier = EXPORT_ARTIFACT_MULTIPLIER.get(fmt, 1.0)
        size = weights_size * multiplier
        if job.get("int8"):
            size *= 1.15
        total += size
    return total


def _export_hours_at_reference_dlperf(run_config: RunConfig) -> float:
    jobs = export_jobs(run_config, Path("placeholder-dataset.yaml"))
    total = 0.0
    for job in jobs:
        fmt = str(job.get("format"))
        job_hours = EXPORT_BASE_HOURS_AT_DLPERF_100.get(fmt, 0.05)
        if job.get("half"):
            job_hours += 0.01
        if job.get("int8"):
            job_hours += 0.08 + 0.04 * float(job.get("fraction", 1.0))
        if job.get("end2end") is not None:
            job_hours += 0.01
        total += job_hours
    return total * _complexity(run_config)


def estimate_run_workload(
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    profile: VastProfile | None = None,
) -> dict[str, Any]:
    train_images = _train_image_count(dataset_record)
    if train_images <= 0:
        raise YoloCtlError(
            f"Dataset version '{dataset_record.version_id}' has no recorded train image count. "
            "Run dataset prepare/validate before estimating cost."
        )

    epochs = max(1, int(run_config.train.get("epochs", 100)))
    imgsz = _imgsz(run_config)
    batch = _batch_size(run_config)
    complexity = _complexity(run_config)
    resolution_factor = (imgsz / BASE_IMGSZ) ** 2
    batch_factor = (batch / BASE_BATCH_SIZE) ** 0.70
    normalized_train_image_passes = train_images * epochs * complexity * resolution_factor / max(batch_factor, 0.5)

    archive_gib = _dataset_archive_gib(dataset_record)
    artifact_gib = _export_artifacts_gib(run_config)
    estimated_download_gib = archive_gib
    estimated_upload_gib = artifact_gib

    required_vram_gb = (
        MODEL_VRAM_BASE_GB[run_config.model.family][run_config.model.size]
        + batch
        * MODEL_VRAM_PER_IMAGE_GB[run_config.model.family][run_config.model.size]
        * resolution_factor
    )

    inflated_dataset_gib = archive_gib * 2.5 if archive_gib else 0.0
    required_disk_gb = max(
        _launch_disk_gb(profile),
        20.0 + inflated_dataset_gib + artifact_gib,
    )
    profile_disk_gb = _launch_disk_gb(profile)
    profile_search_storage_gb = _search_storage_gb(profile)

    warnings: list[str] = []
    if profile and profile_disk_gb and profile_disk_gb < required_disk_gb:
        warnings.append(
            f"Current profile launch.disk={_round(profile_disk_gb, 1)} GB is below the estimated need of {_round(required_disk_gb, 1)} GB."
        )
    if profile and profile_search_storage_gb and profile_search_storage_gb < required_disk_gb:
        warnings.append(
            f"Current profile search.storage={_round(profile_search_storage_gb, 1)} GB may filter out machines that fit the estimated disk need of {_round(required_disk_gb, 1)} GB."
        )

    return {
        "run_id": run_config.run_id,
        "dataset_key": dataset_record.dataset_key,
        "dataset_version_id": dataset_record.version_id,
        "model_family": run_config.model.family,
        "model_size": run_config.model.size,
        "epochs": epochs,
        "imgsz": int(imgsz),
        "batch": int(batch),
        "train_images": train_images,
        "val_images": _split_image_count(dataset_record, "val"),
        "test_images": _split_image_count(dataset_record, "test"),
        "dataset_archive_gib": _round(archive_gib),
        "estimated_download_gib": _round(estimated_download_gib),
        "estimated_upload_gib": _round(estimated_upload_gib),
        "estimated_artifacts_gib": _round(artifact_gib),
        "estimated_required_disk_gb": _round(math.ceil(required_disk_gb), 1),
        "estimated_required_vram_gb": _round(required_vram_gb),
        "resolution_factor": _round(resolution_factor),
        "complexity_factor": _round(complexity),
        "normalized_train_image_passes": _round(normalized_train_image_passes),
        "export_hours_at_dlperf_100": _round(_export_hours_at_reference_dlperf(run_config)),
        "startup_buffer_hours": STARTUP_BUFFER_HOURS,
        "warnings": warnings,
        "heuristic_note": (
            "This estimate is heuristic. It combines dataset size, train image count, epochs, imgsz, batch, "
            "and model size with Vast offer dlperf/hourly pricing to compare machines by estimated total cost."
        ),
    }


def _offer_gpu_ram_gb(offer: dict[str, Any]) -> float:
    raw = offer.get("gpu_ram") or offer.get("gpu_total_ram") or 0
    return float(raw) / 1024.0


def _candidate_cache_hint(download_cost: float, volume_hourly: float, total_hours: float) -> str:
    if volume_hourly <= 0 or total_hours <= 0:
        return "No volume pricing available on this offer; default to OSS as the canonical dataset source."
    volume_single_run_cost = volume_hourly * total_hours
    if download_cost <= 0:
        return "Dataset download cost is negligible; volume caching is unlikely to improve single-run economics."
    if volume_single_run_cost > download_cost:
        return "For a single run, OSS download is cheaper than paying for a same-host volume cache."
    break_even_reuses = max(2, math.ceil(volume_single_run_cost / download_cost) + 1)
    return (
        f"Volume caching may pay off if you expect roughly {break_even_reuses}+ repeated runs on the same host."
    )


def evaluate_offer_cost(
    offer: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    dlperf = float(offer.get("dlperf") or 0.0)
    if dlperf <= 0:
        raise YoloCtlError(f"Offer {offer.get('id')} has no dlperf and cannot be ranked")

    training_hours = workload["normalized_train_image_passes"] / (
        BASE_IMAGES_PER_HOUR_AT_DLPERF_100 * (dlperf / 100.0)
    )
    export_hours = workload["export_hours_at_dlperf_100"] / max(dlperf / 100.0, 0.05)
    total_hours = float(workload["startup_buffer_hours"]) + training_hours + export_hours

    hourly_cost = float(offer.get("dph_total") or 0.0)
    download_cost = float(workload["estimated_download_gib"]) * float(offer.get("inet_down_cost") or 0.0)
    upload_cost = float(workload["estimated_upload_gib"]) * float(offer.get("inet_up_cost") or 0.0)
    network_cost = download_cost + upload_cost
    total_cost = hourly_cost * total_hours + network_cost

    volume_hourly = float(offer.get("avail_vol_dph") or 0.0)
    gpu_ram_gb = _offer_gpu_ram_gb(offer)
    disk_space_gb = float(offer.get("disk_space") or 0.0)
    reliability = float(offer.get("reliability") or 0.0)

    return {
        "offer_id": int(offer["id"]),
        "location": offer.get("geolocation"),
        "gpu_name": offer.get("gpu_name"),
        "gpu_ram_gb": _round(gpu_ram_gb),
        "disk_space_gb": _round(disk_space_gb),
        "dlperf": _round(dlperf),
        "reliability": _round(reliability, 4),
        "hourly_cost_usd": _round(hourly_cost, 4),
        "estimated_train_hours": _round(training_hours),
        "estimated_export_hours": _round(export_hours),
        "estimated_total_hours": _round(total_hours),
        "estimated_network_cost_usd": _round(network_cost, 4),
        "estimated_total_cost_usd": _round(total_cost, 4),
        "cache_hint": _candidate_cache_hint(download_cost, volume_hourly, total_hours),
        "raw_offer": offer,
    }


def recommend_offers(
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    profile: VastProfile,
    offers: list[dict[str, Any]],
) -> dict[str, Any]:
    workload = estimate_run_workload(run_config, dataset_record, profile=profile)
    required_vram = float(workload["estimated_required_vram_gb"])
    required_disk = float(workload["estimated_required_disk_gb"])

    recommendations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for offer in offers:
        gpu_ram_gb = _offer_gpu_ram_gb(offer)
        disk_space_gb = float(offer.get("disk_space") or 0.0)
        if gpu_ram_gb < required_vram:
            rejected.append(
                {
                    "offer_id": int(offer["id"]),
                    "reason": (
                        f"Estimated VRAM need is {_round(required_vram)} GB but offer only exposes {_round(gpu_ram_gb)} GB."
                    ),
                }
            )
            continue
        if disk_space_gb < required_disk:
            rejected.append(
                {
                    "offer_id": int(offer["id"]),
                    "reason": (
                        f"Estimated disk need is {_round(required_disk)} GB but offer only has {_round(disk_space_gb)} GB."
                    ),
                }
            )
            continue
        recommendations.append(evaluate_offer_cost(offer, workload))

    recommendations.sort(
        key=lambda item: (
            item["estimated_total_cost_usd"],
            item["estimated_total_hours"],
            -item["reliability"],
        )
    )

    if recommendations:
        best_cost = recommendations[0]["estimated_total_cost_usd"]
        fastest_hours = min(item["estimated_total_hours"] for item in recommendations)
        for rank, recommendation in enumerate(recommendations, start=1):
            notes = []
            if rank == 1:
                notes.append("Lowest estimated total cost for the current workload.")
            if recommendation["estimated_total_hours"] == fastest_hours:
                notes.append("Fastest estimated completion among the candidate offers.")
            if recommendation["reliability"] >= 0.998:
                notes.append("High host reliability based on Vast reliability score.")
            recommendation["rank"] = rank
            recommendation["relative_cost_vs_best"] = _round(
                recommendation["estimated_total_cost_usd"] / max(best_cost, 0.0001),
                3,
            )
            recommendation["fit_margin_vram_gb"] = _round(
                recommendation["gpu_ram_gb"] - required_vram,
                2,
            )
            recommendation["fit_margin_disk_gb"] = _round(
                recommendation["disk_space_gb"] - required_disk,
                2,
            )
            recommendation["selection_notes"] = notes

    return {
        "profile": str(profile.source_path) if profile.source_path else profile.name,
        "workload": workload,
        "recommendations": recommendations,
        "rejected": rejected,
    }


def summarize_market_estimate(
    run_config: RunConfig,
    dataset_record: DatasetVersionRecord,
    profile: VastProfile | None,
    offers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    workload = estimate_run_workload(run_config, dataset_record, profile=profile)
    payload = {"workload": workload}
    if profile:
        payload["profile"] = str(profile.source_path) if profile.source_path else profile.name
    if offers is not None and profile is not None:
        recommendations = recommend_offers(run_config, dataset_record, profile, offers)
        payload["market"] = {
            "candidate_count": len(recommendations["recommendations"]),
            "rejected_count": len(recommendations["rejected"]),
            "best_offer": recommendations["recommendations"][0] if recommendations["recommendations"] else None,
            "top_offers": recommendations["recommendations"][:3],
        }
    return payload

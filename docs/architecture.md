# Architecture

## Scope

- Training target: Ubuntu on Docker-based Vast.ai instances
- Supported tasks: detect only
- Supported backends:
  - `ultralytics` for official `YOLO11` and `YOLO26`
  - `paddledet` for `PP-PicoDet`
- Primary dataset source of truth: Alibaba Cloud OSS
- Optional cache layer: Vast volumes on the same host only

## Modules

- `yoloctl.dataset`
  - Dataset index in `artifacts/registry/datasets/index.yaml`
  - Immutable version manifests in `artifacts/registry/datasets/<dataset_key>/<version_id>.yaml`
  - Ultralytics detect YAML materialization in `configs/datasets/`
  - OSS zip upload and manifest verification through `ossutil` when available, with official Alibaba Cloud Python SDK fallback when it is not
- `yoloctl.vast`
  - Vast profile loading from `configs/vast/profiles/`
  - Search, launch, lifecycle, invoice, and volume wrappers
  - SDK-first execution with CLI preview and fallback
  - Cost-aware recommendation mode that ranks offers by estimated total run cost instead of hourly price alone
- `yoloctl.train` / `yoloctl.export`
  - Run config loading from `configs/runs/`
  - Backend-aware train/val dispatch
  - Ultralytics command generation plus Python API execution when available, CLI fallback otherwise
  - PaddleDetection subprocess execution for PicoDet with config overrides and derived COCO annotations
  - Precision is chosen per format so unsupported `INT8` combinations are not generated for TorchScript or ONNX
  - `export assess` is a separate acceptance gate that validates exported artifacts against the baseline checkpoint
- `yoloctl.quant`
  - Deterministic calibration subset generation under `artifacts/calibration/`
  - Experimental Python-only QAT lane that produces exportable checkpoints for follow-up export and assessment
- `yoloctl.cost`
  - Invoice snapshots
  - Snapshot summarization
  - Guarded instance and volume cleanup
- `yoloctl.manifests`
  - Stage manifests under `artifacts/manifests/<run_id>/`

## Key Runtime Contracts

- Dataset governance is backed by an index plus immutable version manifests, not a single mutable list file.
- Run configs select a dataset key and immutable dataset version ID, then describe train/export settings.
- PicoDet run configs point to an external PaddleDetection checkout and a native PicoDet `.yml`; `yoloctl` does not vendor PaddleDetection.
- Dataset sync is zip-first: source archives and version manifests are the canonical cloud objects in OSS.
- PicoDet training derives COCO annotations from the managed YOLO-format dataset into `artifacts/paddledet_datasets/` and passes those paths through Paddle config overrides.
- Merge support in v1 is metadata-only through parent version IDs and lineage manifests.
- Offer recommendation is heuristic: it combines dataset volume, train image count, model size, epochs, imgsz, batch, export plan, and live Vast offer metrics like `dlperf`, `dph_total`, network cost, disk, and VRAM.
- Training and export commands always produce a manifest payload so later automation can attach metrics, instance IDs, cost snapshots, and artifact URIs.
- INT8 export defaults to a deterministic calibration subset derived from the materialized dataset; legacy `fraction` fallback is opt-in.
- Export acceptance gates compare each artifact to the baseline checkpoint using `map50_95` as the blocking metric and report `map50`/`recall` as warnings.
- YOLO26 export must support both `end2end=True` and `end2end=False` paths.
- PicoDet currently supports only `train run` and `train val`; resume/export/benchmark/QAT/cost advice remain Ultralytics-only.
- Pruning is intentionally outside the default train/export loop.

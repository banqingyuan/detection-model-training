# YOLO Cloud Training

`YOLO Cloud Training` is a Vast.ai-first training scaffold for object detection workflows with multiple training backends:

- Ultralytics `YOLO11` / `YOLO26`
- PaddleDetection `PP-PicoDet`

It is designed around these boundaries:

- Ubuntu + Docker-based Vast.ai instances only
- Object detection only in v1
- Alibaba Cloud OSS as the source of truth for datasets and exported artifacts
- Vast volumes as optional same-host cache, not primary storage
- Unified `train run` / `train val` entrypoints across supported backends
- Ultralytics-only `export`, `benchmark`, `quant qat`, `cost estimate`, and `vast advise`

## What This Repository Includes

- `yoloctl dataset ...` for productized dataset governance, immutable version manifests, YAML generation, OSS sync, and lineage
- `yoloctl dataset review ...` for local annotation walkthrough, issue-driven triage, manual box/class edits, and review-session finalization
- `yoloctl vast ...` for offer search, instance lifecycle, and volume operations
- `yoloctl vast advise ...` for heuristic cost-aware offer ranking based on dataset size, model size, epochs, imgsz, batch, and live Vast pricing
- `yoloctl train ...` for Ultralytics YOLO and PaddleDetection PicoDet training / validation
- `yoloctl export ...` for multi-format export, assessment, and benchmark command generation
- `yoloctl quant qat ...` for an experimental QAT fine-tuning lane that feeds back into `export run -> export assess`
- `yoloctl cost ...` for workload estimation, invoice snapshots, reporting, and guarded cleanup actions
- Config templates for datasets, Vast profiles, and train/export runs
- Manifest generation for reproducibility and cost tracing

## Quick Start

### 1. Bootstrap on Ubuntu

```bash
./scripts/bootstrap_ubuntu.sh
source .venv/bin/activate
pip install -e .
```

### 2. Set Credentials

```bash
export VAST_API_KEY="your-vast-api-key"
export OSS_ACCESS_KEY_ID="your-oss-access-key-id"
export OSS_ACCESS_KEY_SECRET="your-oss-access-key-secret"
export OSS_ENDPOINT="https://oss-cn-hangzhou.aliyuncs.com"
export OSS_REGION="cn-hangzhou"
```

### 3. Register a Dataset Version

```bash
yoloctl dataset register \
  --dataset-key custom-detect \
  --version-label v0 \
  --source-archive /workspace/datasets/custom-detect-v0.zip \
  --detect-yaml /workspace/datasets/custom-detect-v0/detect.yaml \
  --materialized-path /workspace/datasets/custom-detect-v0
```

To push a registered zip-first version to OSS later:

```bash
yoloctl dataset sync push \
  --dataset-key custom-detect \
  --version-id <generated-version-id> \
  --cloud-prefix oss://your-bucket/datasets
```

`yoloctl dataset sync push` prefers `ossutil` when it is available. If `ossutil` is not installed, it falls back to the official Alibaba Cloud Python SDK (`oss2`) using the same `OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET`, `OSS_ENDPOINT`, and `OSS_REGION` environment variables.

To import an incremental release manifest published by the image pipeline backend:

```bash
yoloctl dataset import-release \
  --manifest-uri oss://your-bucket/training_releases/card-detect-prod/release-001/manifest.yaml
```

To turn an imported merge-candidate release into a metadata-only merge draft:

```bash
yoloctl dataset merge-draft \
  --from-release <imported-release-version-id> \
  --version-label merge-card-detect-v2
```

The repository already includes a bootstrapped local dataset record:

- dataset key: `self-build-v3`
- version label: `v3`
- version id: `dsv-20260325T045541Z-v3`
- extracted path: `artifacts/datasets/self-build-v3/v3`
- classes: 28
- splits: `train=6870`, `val=1963`, `test=982`

### 4. Preview Vast Search

```bash
yoloctl vast search --profile configs/vast/profiles/default-docker-detect.yaml --dry-run
```

### 5. Compare Offers by Estimated Total Cost

```bash
yoloctl vast advise --config configs/runs/yolo11n-sample.yaml --limit 5
```

This command is heuristic rather than authoritative. It estimates total cost from:

- dataset archive size and train image count
- `epochs`, `imgsz`, `batch`
- model family and size
- live Vast offer `dlperf`, `dph_total`, network cost, disk, and VRAM

### 6. Preview or Estimate Cost

```bash
yoloctl cost estimate --config configs/runs/yolo11n-sample.yaml --dry-run
```

### 7. Preview Training

```bash
yoloctl train run --config configs/runs/yolo11n-sample.yaml --dry-run
```

### 8. Preview Export

```bash
yoloctl export run --config configs/runs/yolo26n-sample.yaml --dry-run
```

### 9. Assess Exported Artifacts Against the Baseline

```bash
yoloctl export assess --config configs/runs/yolo26n-sample.yaml --dry-run
```

This command reads the latest `artifacts/manifests/<run_id>/export.json`, validates the baseline checkpoint and each exported artifact, compares metric deltas, and fails if any required artifact exceeds the configured accuracy gate.

### 10. Preview the Experimental QAT Lane

```bash
yoloctl quant qat run --config configs/quant/yolo26n-int8-qat.yaml --dry-run
```

### 11. Preview PicoDet Training

Install the PicoDet extras first:

```bash
pip install -e .[pico]
```

Then prepare a local PaddleDetection checkout and point the run config at it:

```bash
yoloctl train run --config configs/runs/picodet-s-sample.yaml --dry-run
```

Validate a PicoDet checkpoint:

```bash
yoloctl train val --config configs/runs/picodet-s-sample.yaml --dry-run
```

PicoDet uses the existing managed YOLO-format dataset as source input. `yoloctl` derives COCO annotations under `artifacts/paddledet_datasets/<dataset_key>/<version_id>/<run_id>/` and passes those paths to PaddleDetection via config overrides.

PicoDet prerequisites:

- A separate PaddleDetection working copy, referenced by `paddledet.root`
- A PicoDet config file inside that working copy, referenced by `paddledet.config`
- A compatible `paddlepaddle` installation in the training environment
- Any model-specific overrides you still need, such as `num_classes`, via `paddledet.overrides`

PicoDet does not yet support `train resume`, `export`, `benchmark`, `quant qat`, `cost estimate`, or `vast advise`.

### 12. Review and Fix Local Labels in a Browser

Launch the local review tool against a materialized dataset version:

```bash
yoloctl dataset review open \
  --dataset-key self-build-v3 \
  --version-id dsv-20260325T045541Z-v3
```

To seed the queue from an existing issue report and enable model predictions:

```bash
yoloctl dataset review open \
  --dataset-key self-build-v3 \
  --version-id dsv-20260325T045541Z-v3 \
  --issues-report artifacts/reports/eval_errors/yolo11n-280e-1280-datav3-test-conf025/all_errors.json \
  --weights /path/to/model.pt
```

To list saved review sessions:

```bash
yoloctl dataset review sessions \
  --dataset-key self-build-v3 \
  --version-id dsv-20260325T045541Z-v3
```

To convert a review session draft into a new dataset version:

```bash
yoloctl dataset review finalize \
  --dataset-key self-build-v3 \
  --version-id dsv-20260325T045541Z-v3 \
  --session-id <session-id> \
  --new-version-label v3-reviewed
```

## Repository Layout

```text
configs/
  datasets/
  runs/
  vast/profiles/
docker/
docs/
scripts/
src/yoloctl/
tests/
artifacts/
  manifests/
  registry/datasets/
  reports/
```

## Official References

- Ultralytics Train: <https://docs.ultralytics.com/modes/train/>
- Ultralytics Val: <https://docs.ultralytics.com/modes/val/>
- Ultralytics Export: <https://docs.ultralytics.com/modes/export/>
- Ultralytics Benchmark: <https://docs.ultralytics.com/modes/benchmark/>
- Ultralytics YOLO11: <https://docs.ultralytics.com/models/yolo11/>
- Ultralytics YOLO26: <https://docs.ultralytics.com/models/yolo26/>
- Ultralytics Detect Datasets: <https://docs.ultralytics.com/datasets/detect/>
- PaddleDetection PicoDet configs: <https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/picodet>
- Vast CLI Quickstart: <https://docs.vast.ai/cli/get-started>
- Vast Python SDK: <https://docs.vast.ai/sdk/python/quickstart>
- Vast Storage Types: <https://docs.vast.ai/documentation/instances/storage/types>
- Vast Volumes: <https://docs.vast.ai/documentation/instances/storage/volumes>
- Vast Cloud Sync: <https://docs.vast.ai/documentation/instances/storage/cloud-sync>
- Vast Billing: <https://docs.vast.ai/documentation/reference/billing>
- Alibaba Cloud OSS `ossutil sync`: <https://www.alibabacloud.com/help/en/oss/developer-reference/sync-synchronize-an-oss-file-to-a-local-device>
- Alibaba Cloud OSS Python SDK initialization: <https://www.alibabacloud.com/help/en/oss/initialization-2>
- Alibaba Cloud OSS Python SDK object existence check: <https://www.alibabacloud.com/help/en/oss/developer-reference/determine-whether-an-object-exists>

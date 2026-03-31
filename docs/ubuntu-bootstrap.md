# Ubuntu Bootstrap

## 1. Prepare the VM or Docker Instance

```bash
git clone <your-repo-url> /workspace/project
cd /workspace/project
./scripts/bootstrap_ubuntu.sh
source .venv/bin/activate
```

## 2. Install Supporting Tools

- Vast CLI or SDK credentials
- Alibaba Cloud `ossutil` or the Python SDK fallback credentials (`OSS_ACCESS_KEY_ID`, `OSS_ACCESS_KEY_SECRET`, `OSS_ENDPOINT`, `OSS_REGION`)
- `nvidia-smi` accessible inside the instance

## 3. Verify the Environment

```bash
./scripts/check_training_env.sh
```

## 4. Register a Dataset

```bash
yoloctl dataset register \
  --dataset-key custom-detect \
  --version-label v0 \
  --source-archive /workspace/datasets/custom-detect-v0.zip \
  --detect-yaml /workspace/datasets/custom-detect-v0/detect.yaml \
  --materialized-path /workspace/datasets/custom-detect-v0
```

If you are using the bootstrapped local dataset from this repo, validate it directly instead:

```bash
yoloctl dataset validate \
  --dataset-key self-build-v3 \
  --version-id dsv-20260325T045541Z-v3
```

## 5. Sync and Train

```bash
yoloctl dataset status --dataset-key self-build-v3
yoloctl dataset sync push \
  --dataset-key self-build-v3 \
  --version-id dsv-20260325T045541Z-v3 \
  --cloud-prefix oss://your-bucket/datasets
yoloctl train run --config configs/runs/yolo11n-sample.yaml
```

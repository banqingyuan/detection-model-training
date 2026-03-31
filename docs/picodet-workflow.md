# PicoDet Workflow

## Prerequisites

- Install the optional PicoDet extras:
  - `pip install -e .[pico]`
- Install a compatible `paddlepaddle` in the same environment
- Prepare a separate PaddleDetection checkout
- Choose a PicoDet config inside that checkout

This repository does not vendor PaddleDetection. `yoloctl` references it by path through the run config.

## Run Config

Use `configs/runs/picodet-s-sample.yaml` as a starting point.

Required fields:

- `backend: paddledet`
- `paddledet.root`
- `paddledet.config`

Useful optional fields:

- `paddledet.pretrain_weights`
- `paddledet.eval_weights`
- `paddledet.device`
- `paddledet.eval`
- `paddledet.overrides`

`paddledet.overrides` is the place to set model-specific values such as `num_classes` without editing the upstream config in place.

## Dataset Adaptation

Managed datasets remain in the current YOLO detect format. For PicoDet runs, `yoloctl` derives COCO annotations under:

`artifacts/paddledet_datasets/<dataset_key>/<version_id>/<run_id>/`

The generated files include:

- `annotations/instances_train.json`
- `annotations/instances_val.json`

Images stay in the original materialized dataset directory. `yoloctl` passes the image and annotation paths to PaddleDetection with runtime config overrides.

## Commands

Preview a training command:

```bash
yoloctl train run --config configs/runs/picodet-s-sample.yaml --dry-run
```

Preview a validation command:

```bash
yoloctl train val --config configs/runs/picodet-s-sample.yaml --dry-run
```

Run training:

```bash
yoloctl train run --config configs/runs/picodet-s-sample.yaml
```

Run validation:

```bash
yoloctl train val --config configs/runs/picodet-s-sample.yaml --weights /path/to/model_final.pdparams
```

## Current Limits

The PicoDet backend currently supports only:

- `yoloctl train run`
- `yoloctl train val`

The following remain Ultralytics-only:

- `train resume`
- `export run`
- `export benchmark`
- `export assess`
- `quant qat run`
- `cost estimate`
- `vast advise`

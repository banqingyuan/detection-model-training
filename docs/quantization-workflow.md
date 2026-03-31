# Quantization Workflow

## PTQ Acceptance Flow

1. Train a baseline checkpoint with `yoloctl train run`.
2. Export deployment artifacts with `yoloctl export run`.
3. Validate the baseline checkpoint and each exported artifact with `yoloctl export assess`.

`export assess` compares every artifact against the baseline checkpoint, blocks on `map50_95`, records `map50` and `recall` as warnings, and writes both a stage manifest and an assessment report.

## Deterministic Calibration

- INT8 export uses a deterministic calibration subset under `artifacts/calibration/<run_id>/`.
- The subset is derived from the configured split, seeded for reproducibility, and defaults to `class_balanced` sampling.
- If no materialized dataset is available, the legacy `fraction` flow is only used when `allow_fraction_fallback=true`.

## Experimental QAT Lane

- QAT is isolated from the main train/export flow.
- Use `yoloctl quant qat run --config configs/quant/yolo26n-int8-qat.yaml`.
- The command is Python-only, experimental, and intended to produce a checkpoint that still goes through `export run -> export assess`.

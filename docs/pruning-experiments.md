# Pruning Experiment Lane

## Current Project Policy

Pruning is intentionally isolated from the default training loop.

This repository does **not** promise that a structurally pruned model can be pushed back through the native Ultralytics `YOLO()` train/val/export lifecycle unchanged. That boundary follows the current official Ultralytics guidance from the community discussion linked in the main plan.

## What v1 Supports

- keeping pruning settings in separate experiment configs
- attaching pruning metadata to manifests later
- documenting the exact checkpoint and export lineage used in a pruning experiment

## What v1 Does Not Automate

- structural pruning inside the main `yoloctl train run` command
- automatic retraining of pruned checkpoints through the standard Ultralytics pipeline
- pruning-aware benchmark comparisons as part of the required acceptance flow


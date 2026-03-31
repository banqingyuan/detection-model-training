# Vast and OSS Strategy

## Why OSS Is Primary

The project treats Alibaba Cloud OSS as the primary store for datasets and exported artifacts because it is cross-instance, durable, and easier to version as the long-term source of truth.

## Why Vast Volumes Are Optional

Based on Vast official storage documentation:

- volumes are local to a single physical machine
- volumes can only be reattached on that same host
- volumes are billed separately while they exist
- volumes are useful for same-host persistence, not cross-host portability

That makes volumes a good cache for repeated training on a stable host, but a poor primary store for multi-host, budget-aware training pipelines.

## Practical Policy

- Keep canonical datasets and exported artifacts in OSS
- Sync to the instance with `ossutil sync`
- Optionally copy hot datasets into a same-host Vast volume for repeated runs
- Delete unused volumes explicitly to stop charges
- Snapshot invoices regularly because stopped instances and volumes can still generate storage charges

## Cloud Sync Note

Vast documents native Cloud Sync for Amazon S3, Google Drive, Backblaze, and Dropbox. Since OSS is not explicitly documented there, this repository uses OSS-native tooling inside the instance instead of depending on Vast Cloud Sync.


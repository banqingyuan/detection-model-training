#!/usr/bin/env bash
set -euo pipefail

echo "Python: $(python3 --version 2>/dev/null || echo missing)"
echo "Git: $(git --version 2>/dev/null || echo missing)"
echo "NVIDIA SMI:"
nvidia-smi || true
echo "Vast CLI:"
vastai --help >/dev/null 2>&1 && echo "present" || echo "missing"
echo "OSS Util:"
ossutil help >/dev/null 2>&1 && echo "present" || echo "missing"


#!/usr/bin/env bash
# Tears down the execution model C stack for the weather subject.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="${AGENTVERIFY_BENCH_VENV:-$HOME/.venvs/agentverify-bench}"
export PATH="$VENV/bin:$PATH"

npx --yes aws-cdk@2 destroy \
  --app "python app.py" \
  --force \
  "$@"

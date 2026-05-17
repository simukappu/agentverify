#!/usr/bin/env bash
# Deploys the execution model C stack for the LangGraph subject via AWS CDK.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="${AGENTVERIFY_BENCH_VENV:-$HOME/.venvs/agentverify-bench}"
export PATH="$VENV/bin:$PATH"

npx --yes aws-cdk@2 deploy \
  --app "python app.py" \
  --require-approval never \
  --outputs-file outputs.json \
  "$@"

echo
echo "Stack outputs written to: $SCRIPT_DIR/outputs.json"

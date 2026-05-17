#!/usr/bin/env bash
# Deploys the execution model C stack for the weather subject via AWS CDK.
#
# Required:
#   AWS credentials in the environment (the work AWS account).
#   ~/.venvs/agentverify-bench with aws-cdk-lib + aws-cdk.aws-bedrock-agentcore-alpha.
#
# Cost note: a single deploy creates one Lambda + one AgentCore Evaluations Evaluator resource. AgentCore Evaluations Evaluator carries no idle charge; Lambda pricing applies only on invocation. Tag-based cleanup: ``project=agentverify-execution-model-bench``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="${AGENTVERIFY_BENCH_VENV:-$HOME/.venvs/agentverify-bench}"
export PATH="$VENV/bin:$PATH"

# CDK CLI is invoked via npx so we don't require a global install.
npx --yes aws-cdk@2 deploy \
  --app "python app.py" \
  --require-approval never \
  --outputs-file outputs.json \
  "$@"

echo
echo "Stack outputs written to: $SCRIPT_DIR/outputs.json"

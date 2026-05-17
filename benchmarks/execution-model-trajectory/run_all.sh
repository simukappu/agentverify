#!/usr/bin/env bash
# Driver for the execution model comparison benchmark.
#
# Runs all 12 cells (3 execution models × 2 subjects × 2 scenarios) and writes a timestamped results JSON + Markdown into ./results/.
#
# Prerequisites (see README.md "How to run" for details):
#
#   1. Bench venv populated:
#      uv venv ~/.venvs/agentverify-bench --python 3.14 --seed
#      ~/.venvs/agentverify-bench/bin/pip install -e ".[bedrock,dev]"
#      ~/.venvs/agentverify-bench/bin/pip install -e "examples/strands-weather-forecaster"
#      ~/.venvs/agentverify-bench/bin/pip install -e "examples/langgraph-multi-agent-supervisor"
#      ~/.venvs/agentverify-bench/bin/pip install deepeval strands-agents-tools
#      ~/.venvs/agentverify-bench/bin/pip install aws-cdk-lib aws-cdk.aws-bedrock-agentcore-alpha boto3
#
#   2. AWS credentials (Bedrock invoke + AgentCore Evaluations + Lambda) and a real OPENAI_API_KEY in the environment.
#
#   3. CDK bootstrap done in the target account x region:
#      cd strands-weather-forecaster/agentcore_evaluator_cdk
#      npx --yes aws-cdk@2 bootstrap --app "python app.py"
#
#   4. Both AgentCore Evaluations stacks deployed (uses the per-subject deploy.sh).
#
# Tear down the AgentCore Evaluations stacks afterwards with each subject's destroy.sh; the script does not auto-destroy.
#
# Flags:
#   --preliminary           Mark the run as preliminary (development laptop).
#   --scenario {dev,ci,all} Which scenario(s) to run (default: all).
#   any other arguments     Forwarded to collect_results.py (e.g. --cold-start-seconds, --runs).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="${AGENTVERIFY_BENCH_VENV:-$HOME/.venvs/agentverify-bench}"
export PATH="$VENV/bin:$PATH"
export AWS_REGION="${AWS_REGION:-us-east-1}"

# Source ~/.env_apikeys when present so OPENAI_API_KEY (and any other LLM-provider keys) propagate to the collector subprocess. The file is expected to use ``export VAR="value"`` lines.
if [[ -f "$HOME/.env_apikeys" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.env_apikeys"
  set +a
fi

REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

DATESTAMP="$(date -u +%Y-%m-%dT%H%M%S)"

# Parse our own --preliminary and --scenario; forward everything else to the Python collector.
PREFIX="results"
COLLECTOR_FLAGS=()
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preliminary)
      PREFIX="preliminary"
      COLLECTOR_FLAGS+=("--preliminary")
      shift
      ;;
    --scenario)
      COLLECTOR_FLAGS+=("--scenario" "$2")
      shift 2
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

RESULTS_JSON="$RESULTS_DIR/$PREFIX-$DATESTAMP.json"
RESULTS_MD="$RESULTS_DIR/$PREFIX-$DATESTAMP.md"

echo "Driver: agentverify execution model benchmark"
echo "  venv:     $VENV"
echo "  region:   $AWS_REGION"
echo "  output:   $RESULTS_JSON"
echo

cd "$REPO_ROOT"
COLLECTOR_ARGS=("--output-json" "$RESULTS_JSON" "--output-md" "$RESULTS_MD")
if [[ ${#COLLECTOR_FLAGS[@]} -gt 0 ]]; then
  COLLECTOR_ARGS+=("${COLLECTOR_FLAGS[@]}")
fi
if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
  COLLECTOR_ARGS+=("${PASSTHROUGH[@]}")
fi
"$VENV/bin/python" "$SCRIPT_DIR/collect_results.py" "${COLLECTOR_ARGS[@]}"

echo
echo "Done."
echo "  JSON: $RESULTS_JSON"
echo "  MD:   $RESULTS_MD"

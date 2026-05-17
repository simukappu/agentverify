# Execution Model C — AgentCore Evaluations Custom Code-Based Evaluator (CDK, LangGraph subject)

This directory holds the AWS CDK Python application that deploys execution model C for the LangGraph multi-agent supervisor subject. It is a sibling stack to `../../strands-weather-forecaster/agentcore_evaluator_cdk/` with a different function name (`agentverify-execution-model-bench-langgraph`) and evaluator name (`agentverify_bench_langgraph`).

The driver test that calls this evaluator lives one directory up at `../agentcore_test.py`.

## One-time setup: CDK bootstrap

If you have already deployed the strands-weather-forecaster stack, the CDK bootstrap is already in place for this account x region. If you are deploying this stack first (or in a new account x region), run:

```bash
export AWS_REGION=us-east-1
PATH="$HOME/.venvs/agentverify-bench/bin:$PATH" \
  npx --yes aws-cdk@2 bootstrap --app "python app.py"
```

## Deploy / Destroy

```bash
./deploy.sh    # creates / updates the stack, writes outputs.json
./destroy.sh   # tears the stack down
```

`deploy.sh` writes `outputs.json` with the evaluator id, evaluator ARN, and Lambda ARN. The driver test reads this file automatically.

All resources are tagged `project=agentverify-execution-model-bench`, `subject=langgraph-multi-agent-supervisor`.

# Execution Model C — AgentCore Evaluations Custom Code-Based Evaluator (CDK)

This directory holds the AWS CDK Python application that deploys execution model C for the strands-weather-forecaster subject:

- A Python Lambda function that implements the trajectory assertion (`lambda_src/lambda_function.py`).
- A CloudWatch Log group with a one-week retention.
- An AgentCore Evaluations Custom code-based evaluator wired to the Lambda via the alpha L2 construct `aws_cdk.aws_bedrock_agentcore_alpha.Evaluator`.

The driver test that calls this evaluator lives one directory up at `../agentcore_test.py`.

## One-time setup: CDK bootstrap

CDK requires a one-time bootstrap per AWS account x region that creates the staging S3 bucket and IAM roles. This is independent of the benchmark itself and is **not** included in the deploy.sh setup LOC count.

```bash
export AWS_REGION=us-east-1
PATH="$HOME/.venvs/agentverify-bench/bin:$PATH" \
  npx --yes aws-cdk@2 bootstrap --app "python app.py"
```

The bootstrap stack (`CDKToolkit`) and its assets persist after the benchmark. To remove them entirely (after all CDK work in this account is done):

```bash
aws cloudformation delete-stack --stack-name CDKToolkit --region us-east-1
```

## Deploy / Destroy

```bash
./deploy.sh    # creates / updates the stack, writes outputs.json
./destroy.sh   # tears the stack down
```

`deploy.sh` writes `outputs.json` with the evaluator id, evaluator ARN, and Lambda ARN. The driver test reads this file automatically.

## AWS resources created

| Resource | Logical ID | Purpose |
|---|---|---|
| `AWS::Lambda::Function` | `EvaluatorFunction` | Runs the trajectory assertion |
| `AWS::Logs::LogGroup` | `EvaluatorFunctionLogs` | Captures Lambda execution logs (one-week retention) |
| `AWS::BedrockAgentCore::Evaluator` | `TrajectoryEvaluator` | The AgentCore Evaluations custom evaluator entry point |
| `AWS::IAM::Role` (Lambda execution role) | `EvaluatorFunctionServiceRole` | Lambda execution + log writes |

All resources are tagged `project=agentverify-execution-model-bench`, `subject=strands-weather-forecaster`.

## Cost notes

- Lambda: free-tier covers the benchmark's invocation count. Per-call charge at full rate is on the order of `$0.0000002 + memory_GB * $0.0000166667 / s` plus request count.
- AgentCore Evaluations Evaluator: no idle charge documented for the Custom code-based evaluator. The on-demand `Evaluate` Data Plane call that drives this benchmark incurs the cost of the Lambda execution plus any AgentCore Evaluations-side processing.
- CloudWatch Logs: a few cents per GB ingested. The benchmark logs the evaluation event for debugging, which is bounded by the trajectory size (under 5 KB per call).

Tear down with `destroy.sh` after measurements complete. The CDK bootstrap stack remains until manually removed.

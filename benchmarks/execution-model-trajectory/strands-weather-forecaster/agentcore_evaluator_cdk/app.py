"""CDK app entrypoint for the execution model C stack (Strands subject).

Run via the helper scripts ``deploy.sh`` and ``destroy.sh`` rather than invoking ``cdk`` directly, so the bench venv and node-version env var are picked up consistently.
"""

import aws_cdk as cdk
from aws_cdk import aws_iam as iam

from stack import ExecutionModelBenchStack

# Deploy with the caller's own credentials (a narrowly scoped role), not the CDK
# bootstrap roles, whose CloudFormation execution role has AdministratorAccess.
# Assets go to a dedicated bucket, and every role the stack creates carries a
# permissions boundary. See ../../README.md, step 4.
SYNTHESIZER = cdk.CliCredentialsStackSynthesizer(
    file_assets_bucket_name="agentverify-bench-assets-${AWS::AccountId}-${AWS::Region}",
    bucket_prefix="strands-weather-forecaster/",
)
BOUNDARY_POLICY_NAME = "agentverify-bench-lambda-boundary"

app = cdk.App()
stack = ExecutionModelBenchStack(
    app,
    "AgentverifyExecutionModelBenchWeather",
    description="Execution model C benchmark — Strands weather forecaster code-based evaluator",
    tags={"project": "agentverify-execution-model-bench", "subject": "strands-weather-forecaster"},
    synthesizer=SYNTHESIZER,
)
iam.PermissionsBoundary.of(stack).apply(
    iam.ManagedPolicy.from_managed_policy_name(stack, "LambdaBoundary", BOUNDARY_POLICY_NAME)
)
app.synth()

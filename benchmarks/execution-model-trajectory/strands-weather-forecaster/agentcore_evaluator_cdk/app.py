"""CDK app entrypoint for the execution model C stack (Strands subject).

Run via the helper scripts ``deploy.sh`` and ``destroy.sh`` rather than invoking ``cdk`` directly, so the bench venv and node-version env var are picked up consistently.
"""

import aws_cdk as cdk

from stack import ExecutionModelBenchStack


app = cdk.App()
ExecutionModelBenchStack(
    app,
    "AgentverifyExecutionModelBenchWeather",
    description="Execution model C benchmark — Strands weather forecaster code-based evaluator",
    tags={"project": "agentverify-execution-model-bench", "subject": "strands-weather-forecaster"},
)
app.synth()

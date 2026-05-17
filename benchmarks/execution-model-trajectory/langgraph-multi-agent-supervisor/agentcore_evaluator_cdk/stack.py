"""CDK stack for execution model C — AgentCore Evaluations Custom code-based evaluator.

LangGraph subject. Mirrors the strands-weather-forecaster stack with a different function name, evaluator name, and Lambda source path.
"""

from pathlib import Path

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    aws_lambda as _lambda,
    aws_logs as logs,
)
from aws_cdk.aws_bedrock_agentcore_alpha import (
    EvaluationLevel,
    Evaluator,
    EvaluatorConfig,
)
from constructs import Construct


LAMBDA_SRC_DIR = Path(__file__).parent / "lambda_src"
FUNCTION_NAME = "agentverify-execution-model-bench-langgraph"


class ExecutionModelBenchStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        log_group = logs.LogGroup(
            self,
            "EvaluatorFunctionLogs",
            log_group_name=f"/aws/lambda/{FUNCTION_NAME}",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        evaluator_lambda = _lambda.Function(
            self,
            "EvaluatorFunction",
            function_name=FUNCTION_NAME,
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset(str(LAMBDA_SRC_DIR)),
            timeout=Duration.seconds(60),
            memory_size=256,
            log_group=log_group,
            description=(
                "Execution model C benchmark evaluator: enforces tool ordering, argument matching, and cross-step running-total flow on synthesised LangGraph supervisor trajectory spans."
            ),
        )

        evaluator = Evaluator(
            self,
            "TrajectoryEvaluator",
            evaluator_name="agentverify_bench_langgraph",
            level=EvaluationLevel.SESSION,
            description="agentverify execution-model benchmark — langgraph-multi-agent-supervisor subject",
            evaluator_config=EvaluatorConfig.code_based(
                lambda_function=evaluator_lambda,
            ),
        )

        cdk.CfnOutput(
            self,
            "EvaluatorId",
            value=evaluator.evaluator_id,
            description="Pass to agentcore_test.py via AGENTCORE_EVALUATOR_ID",
        )
        cdk.CfnOutput(self, "EvaluatorArn", value=evaluator.evaluator_arn)
        cdk.CfnOutput(self, "LambdaArn", value=evaluator_lambda.function_arn)

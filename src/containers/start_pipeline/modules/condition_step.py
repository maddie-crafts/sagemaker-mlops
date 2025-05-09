from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.parameters import ParameterFloat

def get_condition_step(
    eval_step,
    evaluation_report,
    register_step,
    training_step,
    region
):
    f1_threshold = ParameterFloat("F1Threshold", 0.5)
    precision_threshold = ParameterFloat("PrecisionThreshold", 0.5)
    recall_threshold = ParameterFloat("RecallThreshold", 0.5)

    fail_step = FailStep(
        name="FailIfBadMetrics",
        error_message=Join(on=" ", values=[
            "Model failed quality checks (F1, Precision, Recall below threshold):",
            f1_threshold,
            precision_threshold,
            recall_threshold
        ])
    )

    cleanup_lambda = Lambda(function_arn=f"arn:aws:lambda:{region}:ACCOUNT_ID:function:DeleteModelArtifacts")

    cleanup_step = LambdaStep(
        name="CleanupModelArtifacts",
        lambda_func=cleanup_lambda,
        inputs={"s3_model_artifact": training_step.properties.ModelArtifacts.S3ModelArtifacts}
    )

    condition_step = ConditionStep(
        name="CheckMetricsCondition",
        conditions=[
            ConditionGreaterThanOrEqualTo(JsonGet(step_name=eval_step.name, property_file=evaluation_report, json_path="binary_classification_metrics.f1.value"), f1_threshold),
            ConditionGreaterThanOrEqualTo(JsonGet(step_name=eval_step.name, property_file=evaluation_report, json_path="binary_classification_metrics.precision.value"), precision_threshold),
            ConditionGreaterThanOrEqualTo(JsonGet(step_name=eval_step.name, property_file=evaluation_report, json_path="binary_classification_metrics.recall.value"), recall_threshold),
        ],
        if_steps=[register_step],
        else_steps=[fail_step, cleanup_step]
    )

    return condition_step, [f1_threshold, precision_threshold, recall_threshold]
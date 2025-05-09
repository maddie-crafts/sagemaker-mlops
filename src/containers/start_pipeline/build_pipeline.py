from utils.session_utils import create_boto_and_sagemaker_sessions, get_role
import os
from modules.processing_step import get_preprocessing_step
from modules.training_step import get_training_step
from modules.evaluation_step import get_evaluation_step
from modules.register_step import get_register_step
from modules.condition_step import get_condition_step
from sagemaker.workflow.pipeline import Pipeline

region = os.environ['AWS_REGION']
bucket = os.environ['BUCKET_NAME']
secret_name = os.environ['SECRET_NAME']
span_theme = os.environ['SPAN_THEME']

base_uri = f"s3://{bucket}/train/{span_theme.replace(' ', '-').replace(',', '')}"
base_job_prefix = f"spancat-{span_theme.replace(' ', '-').replace(',', '')}"
model_package_group_name = base_job_prefix

boto_session, sess = create_boto_and_sagemaker_sessions(region)
role = get_role(boto_session)

# Steps
processing_step, span_theme_param, scored_date_param, proc_inst_type, proc_inst_count = get_preprocessing_step(
    role, region, sess, base_uri, base_job_prefix, secret_name
)

training_step, train_inst_type, train_inst_count = get_training_step(
    role, sess, base_uri, base_job_prefix, span_theme_param, processing_step
)

eval_step, evaluation_report = get_evaluation_step(
    role, region, sess, base_uri, base_job_prefix, proc_inst_type, proc_inst_count, training_step, processing_step
)

register_step = get_register_step(
    role, sess, model_package_group_name, training_step, eval_step, evaluation_report
)

condition_step, metric_thresholds = get_condition_step(
    eval_step, evaluation_report, register_step, training_step, region
)

# Pipeline
pipeline = Pipeline(
    name=f"spancat-process-train-eval-{span_theme.replace(' ', '-').replace(',', '')}",
    parameters=[
        proc_inst_count,
        proc_inst_type,
        train_inst_type,
        train_inst_count,
        *metric_thresholds,
        scored_date_param,
        span_theme_param
    ],
    steps=[processing_step, training_step, eval_step, condition_step]
)

pipeline.upsert(role_arn=role)
pipeline.start(parameters={
    "ProcessingInstanceCount": 1,
    "ProcessingInstanceType": "ml.m5.xlarge",
    "TrainingInstanceType": "ml.g5.2xlarge",
    "TrainingInstanceCount": 1,
    "span_theme": span_theme,
    "ScoredDateGreaterThan": "2025-04-01",
})

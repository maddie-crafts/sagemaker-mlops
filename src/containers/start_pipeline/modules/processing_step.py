from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterInteger
)
import os

def get_preprocessing_step(
    role,
    region,
    sess,
    base_uri,
    base_job_prefix,
    span_default="Organization",
    annotation_after_default="2025-04-01",
    secret_name="database-secrets-in-secret-manager"
):
    # Pipeline parameters
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=1
    )

    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value="ml.m5.xlarge"
    )

    scored_date_domain_greater_than = ParameterString(
        name="ScoredDateGreaterThan",
        default_value=annotation_after_default
    )

    span_theme = ParameterString(
        name="span_theme",
        default_value=span_default
    )

    # Cache config
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    sklearn_processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/spancat-preprocess",
        sagemaker_session=sess,
        env={'AWS_DEFAULT_REGION': region}
    )

    processing_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=Join(
                    on="/",
                    values=[
                        base_uri,
                        base_job_prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        'train',
                    ]
                )
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=Join(
                    on="/",
                    values=[
                        base_uri,
                        base_job_prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        'validation',
                    ]
                )
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=Join(
                    on="/",
                    values=[
                        base_uri,
                        base_job_prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        'test',
                    ],
                ),
            ),
        ],
        code=os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/get_data.py")),
        arguments=[
            "--span_theme", span_theme,
            "--scored-date-greater-than", scored_date_domain_greater_than,
            "--secret_name", secret_name
        ]
    )

    processing_step = ProcessingStep(
        name='PrepareData',
        step_args=processing_args,
        cache_config=cache_config
    )

    return processing_step, span_theme, scored_date_domain_greater_than, processing_instance_type, processing_instance_count
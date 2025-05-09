from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.steps import CacheConfig
import os

def get_evaluation_step(
    role,
    region,
    sess,
    base_uri,
    base_job_prefix,
    processing_instance_type,
    processing_instance_count,
    training_step,
    processing_step,
):
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    evaluation_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=sess,
        env={"AWS_DEFAULT_REGION": region,
             "BUCKET_NAME": base_uri.split('/')[2],
             "KEY_PREFIX": '/'.join(base_uri.split('/')[3:] + [base_job_prefix])
             }
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )

    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        code=os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/evaluation.py")),
        inputs=[
            ProcessingInput(source=training_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=processing_step.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=Join(on="/", values=[
                    base_uri,
                    base_job_prefix,
                    ExecutionVariables.PIPELINE_EXECUTION_ID,
                    "evaluation"
                ])
            )
        ],
        property_files=[evaluation_report],
        cache_config=cache_config
    )

    return eval_step, evaluation_report
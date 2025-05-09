from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.steps import CacheConfig
import os

def get_training_step(
    role,
    sess,
    base_uri,
    span_theme_param,
    processing_step,
):
    # Parameters
    train_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value="ml.g5.2xlarge"
    )
    train_instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1
    )

    # Cache
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    estimator = PyTorch(
        entry_point="spacy_training.py",
        source_dir="src",
        instance_type=train_instance_type,
        instance_count=train_instance_count,
        role=role,
        framework_version="2.0.1",
        py_version="py310",
        hyperparameters={
            "span_theme": span_theme_param
        },
        output_path=base_uri,
        sagemaker_session=sess,
        metric_definitions=[
            {"Name": "precision", "Regex": "precision': {'value': (.*?)}"},
            {"Name": "recall", "Regex": "recall': {'value': (.*?)}"},
            {"Name": "f1score", "Regex": "f1': {'value': (.*?)}"},
            {"Name": "auc", "Regex": "auc': {'value': (.*?)}"}
        ],
        environment={"HUGGINGFACE_HUB_CACHE": "/tmp/.cache"},
    )

    train_args = estimator.fit(
        inputs={
            'train': TrainingInput(processing_step.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri),
            'validation': TrainingInput(processing_step.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri),
            'test': TrainingInput(processing_step.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri)
        },
        wait=False
    )

    training_step = TrainingStep(
        name="TrainModel",
        step_args=train_args,
        cache_config=cache_config
    )

    return training_step, train_instance_type, train_instance_count
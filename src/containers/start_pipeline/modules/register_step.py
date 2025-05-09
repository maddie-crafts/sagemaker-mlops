from sagemaker.model import Model
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.functions import Join

def get_register_step(
    role,
    sess,
    model_package_group_name,
    training_step,
    eval_step
):
    model = Model(
        image_uri=training_step.properties.AlgorithmSpecification.TrainingImage,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sess,
        role=role
    )

    register_step = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.g4dn.xlarge"],
            transform_instances=["ml.g4dn.xlarge"],
            model_package_group_name=model_package_group_name,
            model_metrics=ModelMetrics(
                model_statistics=MetricsSource(
                    s3_uri=Join(on="/", values=[
                        eval_step.outputs[0].destination,
                        "evaluation.json"
                    ]),
                    content_type="application/json"
                )
            ),
            approval_status="PendingManualApproval"
        )
    )

    return register_step
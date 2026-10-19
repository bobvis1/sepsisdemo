"""SageMaker Pipeline: features -> train (logreg + XGBoost) -> evaluate -> conditional register.

    BuildFeatures -> TrainLogreg  -> EvaluateLogreg  \
                  -> TrainXGBoost -> EvaluateXGBoost -> CheckXGBoostBeatsBaseline
                                                          -> (if true) RegisterXGBoost

The condition compares hospital A validation AUPRC from each evaluation.json. The model is
registered with status PendingManualApproval and evaluation.json attached as its model
statistics. Caching is off: all code ships inside the image, which a step cache key cannot see.
Build with a PipelineSession; nothing runs until the pipeline is started.
"""

from __future__ import annotations

import subprocess

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.model_metrics import MetricsSource, ModelMetrics
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)
from sagemaker.core.workflow.conditions import ConditionGreaterThan
from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.core.workflow.functions import Join, JsonGet
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train.configs import InputData

from sepsis.config import Infra, Settings, boto_session
from sepsis.sm_common import default_image, hp_to_args, make_processor, make_trainer

PIPELINE_NAME = "sepsis-pipeline"
IN, OUT = "/opt/ml/processing/input", "/opt/ml/processing/output"
AUPRC_PATH = "hospital_A_validation.metrics.auprc.value"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class TrainingStepWithTags(TrainingStep):
    """ModelTrainer (sagemaker 3.25.0) emits tags as {'key','value'}, but a pipeline definition
    needs {'Key','Value'}. The request is built lazily, so normalise it here."""

    @property
    def arguments(self):
        request = super().arguments
        request["Tags"] = [
            {"Key": t.get("key", t.get("Key")), "Value": t.get("value", t.get("Value"))}
            for t in request.get("Tags", [])
        ]
        return request


def _output_uri(infra: Infra, *parts: str) -> Join:
    """s3://<bucket>/pipeline/<execution id>/<parts...>, resolved at run time."""
    return Join(
        on="/",
        values=[f"s3://{infra.bucket}/pipeline", ExecutionVariables.PIPELINE_EXECUTION_ID, *parts],
    )


def _s3_input(name: str, uri, local_path: str) -> ProcessingInput:
    return ProcessingInput(
        input_name=name,
        s3_input=ProcessingS3Input(
            s3_uri=uri, local_path=local_path, s3_data_type="S3Prefix", s3_input_mode="File"
        ),
    )


def _s3_output(name: str, uri, local_path: str) -> ProcessingOutput:
    return ProcessingOutput(
        output_name=name,
        s3_output=ProcessingS3Output(s3_uri=uri, local_path=local_path, s3_upload_mode="EndOfJob"),
    )


def build_pipeline(
    s: Settings,
    infra: Infra,
    *,
    image_uri: str | None = None,
    xgb_hp: list[str] | None = None,
    logreg_hp: list[str] | None = None,
    n_boot: int = 1000,
    spot: bool = True,
) -> Pipeline:
    image = image_uri or default_image(infra)
    session = PipelineSession(boto_session=boto_session(), default_bucket=infra.bucket)

    input_data = ParameterString("InputDataUrl", default_value=f"s3://{infra.bucket}/data/parquet")
    proc_type = ParameterString("ProcessingInstanceType", default_value="ml.t3.large")
    train_type = ParameterString("TrainingInstanceType", default_value="ml.m5.large")
    approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")
    boot = ParameterString("BootstrapResamples", default_value=str(n_boot))  # CLI args are strings

    # 1. Features (same code and image as the local run; output verified identical in Phase 3)
    features = make_processor(
        session, s, infra, module="sepsis.process", name="sepsis-features",
        instance_type=proc_type, image_uri=image,
    )  # fmt: skip
    features_step = ProcessingStep(
        name="BuildFeatures",
        step_args=features.run(
            inputs=[_s3_input("parquet", input_data, IN)],
            outputs=[_s3_output("features", _output_uri(infra, "features"), OUT)],
        ),
    )
    features_uri = features_step.properties.ProcessingOutputConfig.Outputs[
        "features"
    ].S3Output.S3Uri

    steps, train_steps, eval_steps, prop_files = [features_step], {}, {}, {}
    for model, hp in (("logreg", logreg_hp or []), ("xgboost", xgb_hp or [])):
        trainer = make_trainer(
            session, s, infra, model=model, cli_args=hp_to_args(hp),
            instance_type=train_type, image_uri=image, spot=spot,
        )  # fmt: skip
        train_steps[model] = TrainingStepWithTags(
            name=f"Train{model.capitalize()}",
            step_args=trainer.train(
                input_data_config=[
                    InputData(channel_name="train", data_source=Join("/", [features_uri, "train"])),
                    InputData(
                        channel_name="validation",
                        data_source=Join("/", [features_uri, "validation"]),
                    ),
                ]
            ),
        )

        evaluator = make_processor(
            session, s, infra, module="sepsis.evaluate", name=f"sepsis-evaluate-{model}",
            instance_type=proc_type, image_uri=image,
        )  # fmt: skip
        prop_files[model] = PropertyFile(
            name=f"{model}-evaluation", output_name="evaluation", path="evaluation.json"
        )
        eval_steps[model] = ProcessingStep(
            name=f"Evaluate{model.capitalize()}",
            step_args=evaluator.run(
                inputs=[
                    _s3_input(
                        "model",
                        train_steps[model].properties.ModelArtifacts.S3ModelArtifacts,
                        "/opt/ml/processing/model",
                    ),
                    _s3_input("features", features_uri, "/opt/ml/processing/features"),
                ],
                outputs=[
                    _s3_output(
                        "evaluation",
                        _output_uri(infra, "evaluation", model),
                        "/opt/ml/processing/evaluation",
                    )
                ],
                arguments=["--model-name", model, "--n-boot", boot],
            ),
            property_files=[prop_files[model]],
        )
        steps += [train_steps[model], eval_steps[model]]

    # 3. Register the XGBoost model only if it beats the baseline on hospital A validation AUPRC
    def auprc(model: str) -> JsonGet:
        return JsonGet(
            step_name=eval_steps[model].name, property_file=prop_files[model], json_path=AUPRC_PATH
        )

    evaluation_json = Join(
        "/",
        [eval_steps["xgboost"].properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
         "evaluation.json"],
    )  # fmt: skip
    builder = ModelBuilder(
        s3_model_data_url=train_steps["xgboost"].properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=image,
        sagemaker_session=session,
        role_arn=infra.role_arn,
    )
    register_step = ModelStep(
        name="RegisterXGBoost",
        step_args=builder.register(
            model_package_group_name=infra.model_package_group_name,
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            model_metrics=ModelMetrics(
                model_statistics=MetricsSource(
                    content_type="application/json", s3_uri=evaluation_json
                )
            ),
            approval_status=approval,
            customer_metadata_properties={"git_sha": git_sha()},
        ),
    )
    condition = ConditionStep(
        name="CheckXGBoostBeatsBaseline",
        conditions=[ConditionGreaterThan(left=auprc("xgboost"), right=auprc("logreg"))],
        if_steps=[register_step],
        else_steps=[],
    )
    steps.append(condition)

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data, proc_type, train_type, approval, boot],
        steps=steps,
        sagemaker_session=session,
    )


def plain_session(s: Settings, infra: Infra) -> Session:
    return Session(boto_session=boto_session(), default_bucket=infra.bucket)

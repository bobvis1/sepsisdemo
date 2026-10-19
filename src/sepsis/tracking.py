"""Log SageMaker runs to the managed MLflow tracking server (optional; needs the `tracking` extra).

Training and processing jobs are network-isolated and cannot reach MLflow, so the launchers log
from the client after a job finishes, reading each job's params.json / metrics.json and the
pipeline's evaluation.json from S3.

    from sepsis.tracking import log_pipeline_execution, log_tuning_job
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path

from sepsis.config import Infra, boto_session

EVAL_METRICS = ("auroc", "auprc", "utility", "false_alarms_per_patient_day", "median_lead_time_h")


def _mlflow(infra: Infra):
    if not infra.mlflow_tracking_server_arn:
        raise SystemExit("No MLflow tracking server in config/outputs.json (enable_mlflow=false?)")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(infra.mlflow_tracking_server_arn)
    return mlflow


def _read_json_from_tar(s3, uri: str, names: tuple[str, ...]) -> dict[str, dict]:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.tar.gz"
        s3.download_file(bucket, key, str(path))
        with tarfile.open(path) as tf:
            return {n: json.load(tf.extractfile(n)) for n in names}


def _eval_metrics(evaluation: dict) -> dict[str, float]:
    """Flatten evaluation.json into hospital_A.auprc, hospital_B.auprc_ci_low, ... metrics."""
    out = {"threshold": evaluation["threshold"]}
    for tag, h in (("hospital_A", "hospital_A_validation"), ("hospital_B", "hospital_B")):
        out[f"{tag}.alert_rate"] = evaluation[h]["alert_rate"]
        for k in EVAL_METRICS:
            m = evaluation[h]["metrics"][k]
            out[f"{tag}.{k}"] = m["value"]
            out[f"{tag}.{k}_ci_low"], out[f"{tag}.{k}_ci_high"] = m["ci_low"], m["ci_high"]
    return out


def log_pipeline_execution(infra: Infra, execution_id: str, *, image_tag: str = "") -> list[str]:
    """One MLflow run per trained model in a pipeline execution. Returns the run ids."""
    boto = boto_session()
    sm, s3 = boto.client("sagemaker"), boto.client("s3")
    mlflow = _mlflow(infra)
    mlflow.set_experiment("sepsis-pipeline")
    arn = next(
        e["PipelineExecutionArn"]
        for e in sm.list_pipeline_executions(PipelineName="sepsis-pipeline")[
            "PipelineExecutionSummaries"
        ]
        if e["PipelineExecutionArn"].endswith(execution_id)
    )
    steps = {
        st["StepName"]: st
        for st in sm.list_pipeline_execution_steps(PipelineExecutionArn=arn)[
            "PipelineExecutionSteps"
        ]
    }
    run_ids = []
    for model in ("logreg", "xgboost"):
        train = steps[f"Train{model.capitalize()}"]
        job = sm.describe_training_job(
            TrainingJobName=train["Metadata"]["TrainingJob"]["Arn"].rsplit("/", 1)[-1]
        )
        out = _read_json_from_tar(
            s3, job["ModelArtifacts"]["S3ModelArtifacts"], ("params.json", "metrics.json")
        )
        evaluation = json.loads(
            s3.get_object(
                Bucket=infra.bucket,
                Key=f"pipeline/{execution_id}/evaluation/{model}/evaluation.json",
            )["Body"].read()
        )
        with mlflow.start_run(run_name=f"{model}-{execution_id}") as run:
            mlflow.set_tags(
                {
                    "pipeline_execution": execution_id,
                    "training_job": job["TrainingJobName"],
                    "image": job["AlgorithmSpecification"]["TrainingImage"].rsplit("/", 1)[-1],
                    "spot": str(job.get("EnableManagedSpotTraining", False)),
                    "hospital_B_role": "external test, evaluation only",
                }
            )
            mlflow.log_params(out["params.json"])
            mlflow.log_metrics({k.replace(":", "."): v for k, v in out["metrics.json"].items()})
            mlflow.log_metrics(_eval_metrics(evaluation))
            mlflow.log_metric("billable_seconds", job.get("BillableTimeInSeconds", 0))
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "evaluation.json"
                p.write_text(json.dumps(evaluation, indent=2))
                mlflow.log_artifact(str(p))
            run_ids.append(run.info.run_id)
    return run_ids


def log_tuning_job(infra: Infra, tuning_job: str) -> str:
    """A parent run for the tuning job with one nested run per completed trial."""
    boto = boto_session()
    sm = boto.client("sagemaker")
    mlflow = _mlflow(infra)
    mlflow.set_experiment("sepsis-tuning")
    desc = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job)
    objective = desc["HyperParameterTuningJobConfig"]["HyperParameterTuningJobObjective"]
    trials, token = [], None
    while True:
        kw = {"HyperParameterTuningJobName": tuning_job, "MaxResults": 100}
        resp = sm.list_training_jobs_for_hyper_parameter_tuning_job(
            **kw, **({"NextToken": token} if token else {})
        )
        trials += resp["TrainingJobSummaries"]
        token = resp.get("NextToken")
        if not token:
            break
    done = [t for t in trials if t["TrainingJobStatus"] == "Completed"]
    with mlflow.start_run(run_name=tuning_job) as parent:
        mlflow.set_tags(
            {
                "tuning_job": tuning_job,
                "strategy": desc["HyperParameterTuningJobConfig"]["Strategy"],
            }
        )
        mlflow.log_params({"objective": objective["MetricName"], "n_trials": len(done)})
        best = desc.get("BestTrainingJob")
        if best:
            mlflow.log_metric(
                "best_" + objective["MetricName"].replace(":", "."),
                best["FinalHyperParameterTuningJobObjectiveMetric"]["Value"],
            )
            mlflow.set_tag("best_training_job", best["TrainingJobName"])
        for t in sorted(done, key=lambda t: t["CreationTime"]):
            with mlflow.start_run(run_name=t["TrainingJobName"], nested=True):
                mlflow.log_params(t["TunedHyperParameters"])
                mlflow.log_metric(
                    objective["MetricName"].replace(":", "."),
                    t["FinalHyperParameterTuningJobObjectiveMetric"]["Value"],
                )
    return parent.info.run_id

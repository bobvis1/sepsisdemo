import json

from sepsis.train import sm_hyperparameters_argv


def test_sm_hyperparameters_become_cli_args(tmp_path):
    f = tmp_path / "hyperparameters.json"
    f.write_text(
        json.dumps(
            {
                "model": '"xgboost"',
                "max_depth": "5",
                "eta": "0.05",
                "_tuning_objective_metric": "validation:auprc",
                "sagemaker_program": "x.py",
            }
        )
    )
    assert sm_hyperparameters_argv(f) == [
        "--model", "xgboost", "--max-depth", "5", "--eta", "0.05",
    ]  # fmt: skip

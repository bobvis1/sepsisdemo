"""Load user settings and Terraform outputs.

`config/settings.yaml` (git-ignored) holds the user's choices; `config/outputs.json`
(git-ignored, written by `scripts/infra.sh outputs`) holds resource identifiers created by
Terraform. Env vars SEPSIS_AWS_PROFILE / SEPSIS_REGION override the settings file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "config" / "settings.yaml"
OUTPUTS_PATH = ROOT / "config" / "outputs.json"


@dataclass(frozen=True)
class Settings:
    aws_profile: str
    region: str
    owner: str
    name_prefix: str


@dataclass(frozen=True)
class Infra:
    bucket: str
    kms_key_arn: str
    role_arn: str
    subnet_ids: list[str]
    security_group_id: str
    ecr_repository_url: str


def load_settings() -> Settings:
    raw = yaml.safe_load(SETTINGS_PATH.read_text())
    return Settings(
        aws_profile=os.environ.get("SEPSIS_AWS_PROFILE", raw["aws_profile"]),
        region=os.environ.get("SEPSIS_REGION", raw["region"]),
        owner=raw["owner"],
        name_prefix=raw.get("name_prefix", "sepsis"),
    )


def load_infra() -> Infra:
    raw = json.loads(OUTPUTS_PATH.read_text())
    v = {k: x["value"] for k, x in raw.items()}
    return Infra(
        bucket=v["bucket"],
        kms_key_arn=v["kms_key_arn"],
        role_arn=v["role_arn"],
        subnet_ids=v["subnet_ids"],
        security_group_id=v["security_group_id"],
        ecr_repository_url=v["ecr_repository_url"],
    )


def boto_session():
    import boto3

    s = load_settings()
    return boto3.Session(profile_name=s.aws_profile, region_name=s.region)

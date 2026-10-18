"""Print the AWS identity used by this project (sanity check for profile/SSO login)."""

from sepsis.config import boto_session, load_settings

s = load_settings()
ident = boto_session().client("sts").get_caller_identity()
print(f"profile={s.aws_profile} region={s.region} arn={ident['Arn']}")

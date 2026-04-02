import os as _os

if _os.environ.get("USE_JUDGE_MODEL", "0") == "1":
    from .deploy_policy_with_judge import *
else:
    from .deploy_policy import *

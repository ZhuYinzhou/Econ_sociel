# Environment registry.
#
# IMPORTANT:
# Offline stages (S1/S2/S3a/S3b) typically use `huggingface_dataset_env`.
# If we eagerly import every env module here, a syntax/dep error in an *unused*
# env (e.g. `hisim_social_env`) can still break offline training at import time.

from .huggingface_dataset_env import HuggingFaceDatasetEnv

try:
    from .hisim_social_env import HiSimSocialEnv  # optional (only needed for Stage4 online RL)
except Exception:
    HiSimSocialEnv = None  # type: ignore

REGISTRY = {"huggingface_dataset_env": HuggingFaceDatasetEnv}
if HiSimSocialEnv is not None:
    REGISTRY["hisim_social_env"] = HiSimSocialEnv
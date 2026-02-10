from .huggingface_dataset_env import HuggingFaceDatasetEnv

try:
    from .hisim_social_env import HiSimSocialEnv  # optional (only needed for Stage4 online RL)
except Exception:
    HiSimSocialEnv = None  # type: ignore

REGISTRY = {"huggingface_dataset_env": HuggingFaceDatasetEnv}
if HiSimSocialEnv is not None:
    REGISTRY["hisim_social_env"] = HiSimSocialEnv
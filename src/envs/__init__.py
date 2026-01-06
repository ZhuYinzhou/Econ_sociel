# This file can be used to register custom environments if needed in the future.
# For now, we are loading datasets directly from Hugging Face.

from .huggingface_dataset_env import HuggingFaceDatasetEnv
from .hisim_social_env import HiSimSocialEnv

REGISTRY = {
    "huggingface_dataset_env": HuggingFaceDatasetEnv,
    "hisim_social_env": HiSimSocialEnv,
} 
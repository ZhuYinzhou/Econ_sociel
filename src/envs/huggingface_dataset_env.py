import gym
from gym import spaces
import numpy as np
import os
from datasets import load_dataset, Dataset, IterableDataset, DatasetDict
try:
    # Optional: concatenate_datasets is available in HuggingFace datasets
    from datasets import concatenate_datasets  # type: ignore
except Exception:
    concatenate_datasets = None  # type: ignore
try:
    from datasets import load_from_disk  # type: ignore
except Exception:
    load_from_disk = None  # type: ignore
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import re
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

class HuggingFaceDatasetEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, **kwargs):
        super().__init__()
        
        # hf_dataset_path can be:
        # - str: a HF hub dataset name or a local save_to_disk directory
        # - list[str]: multiple datasets to be merged (concatenated) AFTER loading the same split
        self.dataset_path = kwargs.get("hf_dataset_path", "gsm8k")
        self.dataset_config_name = kwargs.get("hf_dataset_config_name", None)
        self.dataset_split = kwargs.get("dataset_split", "train")
        self.is_streaming = kwargs.get("dataset_streaming", False)
        self.use_random_sampling = kwargs.get("use_random_sampling", True)  # 添加随机采样选项
        self.use_dataset_episode = kwargs.get("use_dataset_episode", False)  # 数据集级别episode选项
        # Optional discrete-action size for training (e.g., stance classification K=3)
        # Note: action_space remains Text; this n_actions is for runner/buffer/action_selector shapes.
        self.n_actions = int(kwargs.get("n_actions", 1))

        # Optional filter for datasets that contain both core/non-core samples
        # - None: use all samples
        # - True: only samples with sample["is_core_user"] == True
        # - False: only samples with sample["is_core_user"] == False
        self.filter_is_core_user = kwargs.get("filter_is_core_user", None)
        if isinstance(self.filter_is_core_user, str):
            s = self.filter_is_core_user.strip().lower()
            if s in ("core", "true", "1", "yes"):
                self.filter_is_core_user = True
            elif s in ("noncore", "non-core", "false", "0", "no"):
                self.filter_is_core_user = False
            else:
                self.filter_is_core_user = None
        
        self.question_field = kwargs.get("question_field_name", "question")
        self.answer_field = kwargs.get("answer_field_name", "answer")

        # ===== Optional: oversample a specific stance-id label on train split =====
        # Motivation: stage-1/2 offline stance classification can be highly imbalanced (e.g., label1 < 1%).
        # We only oversample TRAIN split; validation/test must keep the original distribution.
        self.oversample_enabled = bool(kwargs.get("oversample_enabled", False))
        self.oversample_only_train = bool(kwargs.get("oversample_only_train", True))
        self.oversample_label_id = int(kwargs.get("oversample_label_id", 1))
        self.oversample_target_ratio = float(kwargs.get("oversample_target_ratio", 0.05))  # e.g., 0.05~0.10
        self.oversample_seed = int(kwargs.get("oversample_seed", 42))
        # Safety cap to avoid exploding dataset size (multiplier for the minority class)
        self.oversample_max_multiplier = int(kwargs.get("oversample_max_multiplier", 30))
        if self.oversample_max_multiplier < 1:
            self.oversample_max_multiplier = 1
        if self.oversample_target_ratio < 0.0:
            self.oversample_target_ratio = 0.0
        if self.oversample_target_ratio > 0.99:
            self.oversample_target_ratio = 0.99
        
        # For reward calculation, if needed directly in env
        self.reward_args = kwargs.get("reward_config", {})
        # Whether to emit belief/z supervision fields from samples (for z_transition training)
        self.emit_belief_fields = bool(kwargs.get("emit_belief_fields", True))
        # Verbosity control: this env can print very long observations every step.
        # Disable for training runs to avoid terminal spam.
        self.verbose_step_logging = bool(kwargs.get("verbose_step_logging", True))
        # Optional: population belief dim for offline transition supervision.
        # - K=3: categorical distribution over stances
        # - K=1: scalar z in [-1,1]
        self.population_belief_dim = int(kwargs.get("population_belief_dim", 3))
        self.population_belief_dim = max(1, self.population_belief_dim)

        try:
            # Support both:
            # - HuggingFace hub datasets via load_dataset(...)
            # - local datasets saved by DatasetDict.save_to_disk(...) via load_from_disk(...)
            # Also support merging multiple datasets (list of paths/names) by concatenation.

            def _load_one(path_or_name: str):
                use_local = bool(isinstance(path_or_name, str) and os.path.isdir(path_or_name))
                if use_local:
                    if self.is_streaming:
                        logger.warning("dataset_streaming=True is not supported for load_from_disk datasets. Forcing streaming=False.")
                        self.is_streaming = False
                    if load_from_disk is None:
                        raise RuntimeError("datasets.load_from_disk is unavailable. Please ensure `datasets` is installed correctly.")
                    loaded = load_from_disk(path_or_name)
                    if isinstance(loaded, DatasetDict):
                        if self.dataset_split not in loaded:
                            raise KeyError(
                                f"Split '{self.dataset_split}' not found in dataset at {path_or_name}. Available: {list(loaded.keys())}"
                            )
                        ds = loaded[self.dataset_split]
                    else:
                        ds = loaded
                    logger.info(f"Loaded dataset from disk: {path_or_name}, split: {self.dataset_split}, streaming={self.is_streaming}")
                    return ds
                else:
                    ds = load_dataset(
                        path_or_name,
                        name=self.dataset_config_name,
                        split=self.dataset_split,
                        streaming=self.is_streaming
                    )
                    return ds

            # Normalize dataset_path into a list or a single string
            ds_sources: List[str] = []
            if isinstance(self.dataset_path, (list, tuple)):
                ds_sources = [str(x) for x in self.dataset_path if str(x).strip() != ""]
            elif isinstance(self.dataset_path, str) and "," in self.dataset_path:
                # allow comma-separated for convenience in CLI/env
                ds_sources = [s.strip() for s in self.dataset_path.split(",") if s.strip()]
            else:
                ds_sources = [str(self.dataset_path)]

            if self.is_streaming and len(ds_sources) > 1:
                raise RuntimeError("Merging multiple datasets is not supported in streaming mode. Set dataset_streaming: false.")

            if len(ds_sources) == 1:
                self.dataset = _load_one(ds_sources[0])
            else:
                if concatenate_datasets is None:
                    raise RuntimeError("datasets.concatenate_datasets is unavailable; cannot merge multiple datasets.")
                parts = []
                for src in ds_sources:
                    parts.append(_load_one(src))
                self.dataset = concatenate_datasets(parts)
                logger.info(
                    f"Merged {len(parts)} datasets by concatenation for split='{self.dataset_split}': "
                    + ", ".join(ds_sources)
                    + f" | merged_num_samples={len(self.dataset)}"
                )
            if self.is_streaming:
                self.dataset_iterator = iter(self.dataset)
                # For IterableDataset, we can't easily get the length.
                # We might need a max_episodes arg from config for termination in streaming mode.
                logger.info(f"Loaded IterableDataset: {self.dataset_path}, split: {self.dataset_split}")
            else:
                self.dataset_list = list(self.dataset) # Convert to list for easier iteration and shuffling if needed
                # Apply optional filter on samples
                if self.filter_is_core_user is not None:
                    want = bool(self.filter_is_core_user)
                    before = len(self.dataset_list)
                    self.dataset_list = [s for s in self.dataset_list if bool(s.get("is_core_user", False)) == want]
                    after = len(self.dataset_list)
                    logger.info(f"Filtered dataset by is_core_user={want}: {before} -> {after}")

                # Optional: oversample minority label on TRAIN split only
                try:
                    do_oversample = bool(self.oversample_enabled)
                    if self.oversample_only_train and str(self.dataset_split).lower() != "train":
                        do_oversample = False
                    if do_oversample:
                        na = int(getattr(self, "n_actions", 1))
                        if na <= 1:
                            logger.warning("[HFEnv] oversample_enabled=True but n_actions<=1; skipping oversampling.")
                        else:
                            # Compute label ids from boxed answer: \boxed{<id>}
                            boxed_re = getattr(self, "_boxed_int_re", re.compile(r"\\boxed\{\s*([-+]?\d+)\s*\}"))
                            y: List[int] = []
                            idx_by_label: Dict[int, List[int]] = {i: [] for i in range(na)}
                            missing = 0
                            for i, s in enumerate(self.dataset_list):
                                a = s.get(self.answer_field, "")
                                if not isinstance(a, str):
                                    missing += 1
                                    y.append(-1)
                                    continue
                                m = boxed_re.search(a)
                                if not m:
                                    missing += 1
                                    y.append(-1)
                                    continue
                                try:
                                    sid = int(m.group(1))
                                except Exception:
                                    missing += 1
                                    y.append(-1)
                                    continue
                                y.append(sid)
                                if 0 <= sid < na:
                                    idx_by_label[sid].append(i)

                            counts = [len(idx_by_label[i]) for i in range(na)]
                            total = int(sum(counts))
                            lid = int(self.oversample_label_id)
                            if not (0 <= lid < na):
                                logger.warning(f"[HFEnv] oversample_label_id={lid} out of range for n_actions={na}; skipping oversampling.")
                            elif total <= 0:
                                logger.warning("[HFEnv] No valid boxed labels found; skipping oversampling.")
                            elif len(idx_by_label.get(lid, [])) <= 0:
                                logger.warning(f"[HFEnv] No samples for label_id={lid}; skipping oversampling.")
                            else:
                                cur = float(counts[lid]) / float(total) if total > 0 else 0.0
                                tgt = float(self.oversample_target_ratio)
                                if tgt <= 0.0 or cur >= tgt:
                                    logger.info(
                                        f"[HFEnv] Oversampling skipped (current_ratio={cur:.4f} >= target_ratio={tgt:.4f}). "
                                        f"counts={counts}, missing_boxed={missing}"
                                    )
                                else:
                                    # x >= (tgt*total - count_lid) / (1 - tgt)
                                    need = int(np.ceil((tgt * float(total) - float(counts[lid])) / max(1e-8, (1.0 - tgt))))
                                    # Cap by max_multiplier
                                    max_extra = int(self.oversample_max_multiplier * counts[lid] - counts[lid])
                                    if max_extra < 0:
                                        max_extra = 0
                                    extra = int(min(need, max_extra))
                                    if extra <= 0:
                                        logger.info(
                                            f"[HFEnv] Oversampling computed extra<=0 (need={need}, cap_extra={max_extra}); skipping."
                                        )
                                    else:
                                        rng = random.Random(int(self.oversample_seed))
                                        src_indices = list(idx_by_label[lid])
                                        add_samples = [self.dataset_list[rng.choice(src_indices)] for _ in range(extra)]
                                        before_n = len(self.dataset_list)
                                        self.dataset_list.extend(add_samples)
                                        after_n = len(self.dataset_list)

                                        # Recompute counts for logging
                                        # (Only for valid boxed ids; duplicates are included)
                                        new_counts = counts[:]
                                        new_counts[lid] = new_counts[lid] + extra
                                        new_total = int(sum(new_counts))
                                        new_ratio = float(new_counts[lid]) / float(new_total) if new_total > 0 else 0.0
                                        logger.info(
                                            f"[HFEnv] Oversampled label_id={lid} on split='{self.dataset_split}': "
                                            f"target_ratio={tgt:.3f}, current_ratio={cur:.3f} -> new_ratio={new_ratio:.3f}; "
                                            f"added={extra}, size={before_n}->{after_n}, counts={counts}->{new_counts}, missing_boxed={missing}"
                                        )
                except Exception as e:
                    logger.warning(f"[HFEnv] Oversampling failed (skipped): {e}")
                self.dataset_iterator = None # Will be created in reset
                self.current_data_idx = -1
                self.num_samples = len(self.dataset_list)
                logger.info(f"Loaded Dataset: {self.dataset_path}, split: {self.dataset_split}, num_samples: {self.num_samples}")
                
                # 如果使用随机采样，打乱数据集
                if self.use_random_sampling and not self.use_dataset_episode:
                    random.shuffle(self.dataset_list)
                    logger.info("Dataset shuffled for random sampling")

        except Exception as e:
            logger.error(f"Failed to load dataset '{self.dataset_path}' (config: {self.dataset_config_name}, split: {self.dataset_split}): {e}")
            raise

        self.max_question_length = kwargs.get("max_question_length", 1024)
        self.max_answer_length = kwargs.get("max_answer_length", 1024) # For action space

        # ===== Runtime sample validation (helps rule out data issues that can cause NaNs) =====
        # Some degenerate samples (e.g., empty question/answer) can lead to all-padding inputs and
        # downstream NaNs in attention. We count and optionally skip such samples.
        self._invalid_sample_skipped = 0
        self._invalid_sample_last_reason = ""
        self._invalid_sample_max_resample = int(kwargs.get("invalid_sample_max_resample", 2000))
        self._invalid_sample_max_resample = max(1, self._invalid_sample_max_resample)
        self._boxed_int_re = re.compile(r"\\boxed\{\s*([-+]?\d+)\s*\}")

        # ===== Optional: log stance label distribution (helps diagnose acc plateaus) =====
        # Many stage-1/2 runs are pure stance-id classification (K=3). If the dataset is imbalanced
        # (e.g., one label takes ~0.76), accuracy can appear "stuck" even when learning progresses slowly.
        self.log_label_distribution = bool(kwargs.get("log_label_distribution", True))
        try:
            if (
                self.log_label_distribution
                and (not bool(getattr(self, "is_streaming", False)))
                and hasattr(self, "dataset_list")
                and isinstance(getattr(self, "dataset_list", None), list)
            ):
                na = int(getattr(self, "n_actions", 1))
                if na and na > 1 and len(self.dataset_list) > 0:
                    counts = [0 for _ in range(na)]
                    missing = 0
                    out_of_range = 0
                    non_string = 0
                    for s in self.dataset_list:
                        if not isinstance(s, dict):
                            continue
                        a = s.get(self.answer_field, "")
                        if not isinstance(a, str):
                            non_string += 1
                            continue
                        m = self._boxed_int_re.search(a)
                        if not m:
                            missing += 1
                            continue
                        try:
                            sid = int(m.group(1))
                        except Exception:
                            missing += 1
                            continue
                        if 0 <= sid < na:
                            counts[sid] += 1
                        else:
                            out_of_range += 1

                    total = int(sum(counts))
                    if total > 0:
                        props = [c / float(total) for c in counts]
                        logger.info(
                            f"[HFEnv] Label distribution (boxed id) over loaded split='{self.dataset_split}', "
                            f"is_core_user={self.filter_is_core_user}, n_actions={na}: "
                            + ", ".join([f"{i}:{counts[i]}({props[i]:.3f})" for i in range(na)])
                            + f" | missing_boxed={missing}, out_of_range={out_of_range}, non_string={non_string}"
                        )
        except Exception as e:
            logger.warning(f"[HFEnv] Failed to log label distribution: {e}")

        # Define action/observation spaces - what the agent "outputs" / "sees"
        # Using gym.spaces.Text requires gym version 0.26+
        self.action_space = spaces.Text(max_length=self.max_answer_length)
        self.observation_space = spaces.Text(max_length=self.max_question_length)

        # Current sample from the dataset
        self.current_sample: Optional[Dict] = None
        self.current_question: Optional[str] = None
        self.current_ground_truth_answer: Optional[str] = None
        self.episode_count = 0  # 添加episode计数器，用于追踪问题

        # 数据集级别episode的状态追踪
        if self.use_dataset_episode:
            self.step_count = 0  # 当前episode内的步数
            self.episode_limit = self.num_samples if not self.is_streaming else 1000  # 使用数据集大小作为episode限制
            self.current_episode_samples = []  # 当前episode处理的所有样本
            self.episode_results = []  # 当前episode的所有结果
        else:
            # Episode specifics (each question is an episode)
            self.episode_length = 0  # Steps within current episode (always 1)
            self.episode_limit = 1

    def _validate_sample(self, sample: Any) -> Tuple[bool, str]:
        """
        Return (is_valid, reason_if_invalid).
        - Requires non-empty question/answer strings
        - If n_actions > 1 (classification), requires answer contains \\boxed{<id>} within [0, n_actions-1]
        """
        if not isinstance(sample, dict):
            return False, f"sample_not_dict:{type(sample)}"
        q = sample.get(self.question_field, "")
        a = sample.get(self.answer_field, "")
        if not isinstance(q, str) or not isinstance(a, str):
            return False, f"non_string_fields:q={type(q)},a={type(a)}"
        if q.strip() == "":
            return False, "empty_question"
        if a.strip() == "":
            return False, "empty_answer"
        # Only enforce boxed-id format for classification-like settings
        try:
            na = int(getattr(self, "n_actions", 1))
        except Exception:
            na = 1
        if na > 1:
            m = self._boxed_int_re.search(a)
            if not m:
                return False, "answer_missing_boxed_id"
            try:
                sid = int(m.group(1))
            except Exception:
                return False, "boxed_id_parse_error"
            if sid < 0 or sid >= na:
                return False, f"boxed_id_out_of_range:{sid}"
        return True, ""

    def _get_next_sample(self) -> Optional[Dict]:
        """Get next sample, skipping invalid ones (up to a max resample limit)."""

        def _sample_once() -> Optional[Dict]:
            # Original sampling logic
            if self.is_streaming:
                try:
                    # In streaming mode, loop until we find a matching sample if filtering is enabled
                    while True:
                        sample = next(self.dataset_iterator)
                        if self.filter_is_core_user is None:
                            return sample
                        want = bool(self.filter_is_core_user)
                        if bool(sample.get("is_core_user", False)) == want:
                            return sample
                except StopIteration:
                    logger.info("Streaming dataset iterator exhausted.")
                    return None
            else:
                if self.use_dataset_episode:
                    # 数据集级别episode：顺序遍历所有样本
                    self.current_data_idx += 1
                    if self.current_data_idx < self.num_samples:
                        return self.dataset_list[self.current_data_idx]
                    else:
                        logger.info("Dataset-level episode completed: all samples processed.")
                        return None
                elif self.use_random_sampling:
                    # 随机采样：每次随机选择一个样本
                    if self.num_samples > 0:
                        random_idx = random.randint(0, self.num_samples - 1)
                        sample = self.dataset_list[random_idx]
                        logger.debug(f"Random sampling: selected index {random_idx}")
                        return sample
                    else:
                        logger.info("Dataset is empty.")
                        return None
                else:
                    # 顺序采样：原有逻辑
                    self.current_data_idx += 1
                    if self.current_data_idx < self.num_samples:
                        return self.dataset_list[self.current_data_idx]
                    else:
                        logger.info("Non-streaming dataset iterator exhausted.")
                        return None

        attempts = 0
        while attempts < self._invalid_sample_max_resample:
            attempts += 1
            sample = _sample_once()
            if sample is None:
                return None
            ok, reason = self._validate_sample(sample)
            if ok:
                return sample
            self._invalid_sample_skipped += 1
            self._invalid_sample_last_reason = str(reason)
            if self._invalid_sample_skipped <= 3 or (self._invalid_sample_skipped % 200 == 0):
                logger.warning(f"Skipped invalid sample (count={self._invalid_sample_skipped}): {reason}")
        logger.error(
            f"Too many invalid samples encountered (attempts={attempts}, skipped={self._invalid_sample_skipped}). "
            f"Last reason: {self._invalid_sample_last_reason}"
        )
        return None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed) # Gym 0.26+
        
        # 设置随机种子以确保可重现性
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if self.use_dataset_episode:
            # 数据集级别episode：重置状态，从第一个样本开始
            self.current_data_idx = -1  # 会被_get_next_sample递增到0
            self.step_count = 0
            self.current_episode_samples = []
            self.episode_results = []
            self.episode_count += 1
            logger.info(f"Starting dataset-level Episode {self.episode_count}: will process {self.num_samples} samples")
        else:
            # 原有逻辑：每个问题一个episode
            if not self.is_streaming and (self.dataset_iterator is None or self.current_data_idx >= self.num_samples -1):
                # Re-initialize iterator for non-streaming dataset if exhausted or first time
                if not self.use_random_sampling:
                    # 只有在顺序采样时才重置索引
                    self.current_data_idx = -1 # Will be incremented by _get_next_sample

        self.current_sample = self._get_next_sample()
        
        if self.current_sample is None:
            # Handle dataset exhaustion, e.g., by raising an error or returning a special state
            # For now, let's raise an error to make it explicit during development.
            # In a long run, might want to loop the dataset or have a max_episodes from config.
            raise StopIteration("Dataset exhausted. Implement looping or max_episode limit if needed.")

        self.current_question = str(self.current_sample.get(self.question_field, ""))
        self.current_ground_truth_answer = str(self.current_sample.get(self.answer_field, ""))
        
        if not self.use_dataset_episode:
            self.episode_length = 0
            self.episode_count += 1
        
        # 记录问题变化以便调试
        if self.use_dataset_episode:
            question_preview = self.current_question[:100] + "..." if len(self.current_question) > 100 else self.current_question
            if bool(getattr(self, "verbose_step_logging", True)):
                logger.info(f"Episode {self.episode_count}, Step {self.step_count + 1}/{self.num_samples}: {question_preview}")
        else:
            question_preview = self.current_question[:100] + "..." if len(self.current_question) > 100 else self.current_question
            if bool(getattr(self, "verbose_step_logging", True)):
                logger.info(f"Episode {self.episode_count}: New question - {question_preview}")
        
        # Observation is the question text
        # Preprocess if necessary (e.g., tokenization if obs_space was Box)
        # For now, passing raw text, MAC needs to handle it.
        observation = self.current_question 
        
        info = {"sample": self.current_sample} # Pass the whole sample for potential use in reward or logging
        # Expose validation/skip counters to help diagnose data issues
        info["invalid_sample_skipped"] = int(getattr(self, "_invalid_sample_skipped", 0))
        info["invalid_sample_last_reason"] = str(getattr(self, "_invalid_sample_last_reason", ""))
        return observation, info

    def get_belief_tensor(self, belief_inputs: Optional[Dict[str, Any]], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        A minimal tensorizer to match EpisodeRunner._get_post_transition_data() expectations.
        This enables offline z_transition supervision by providing population_z (K=3 categorical or K=1 scalar).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bi = belief_inputs if isinstance(belief_inputs, dict) else {}
        pop_dim = int(getattr(self, "population_belief_dim", 3))
        pop_dim = max(1, pop_dim)
        pz_raw = bi.get("population_z")
        if pz_raw is None:
            pz_raw = bi.get("z_t")

        # population_z
        if pop_dim == 1:
            # scalar in [-1,1]
            try:
                if isinstance(pz_raw, torch.Tensor):
                    z = float(pz_raw.detach().flatten()[0].item())
                elif isinstance(pz_raw, (list, tuple)) and len(pz_raw) > 0:
                    z = float(pz_raw[0])
                else:
                    z = float(pz_raw) if pz_raw is not None else 0.0
            except Exception:
                z = 0.0
            z = float(max(-1.0, min(1.0, z)))
            pz = [z]
        else:
            # categorical simplex (default K=3)
            pz = pz_raw if pz_raw is not None else [1.0 / float(pop_dim) for _ in range(pop_dim)]
            try:
                pz = [float(x) for x in list(pz)[:pop_dim]]
            except Exception:
                pz = [1.0 / float(pop_dim) for _ in range(pop_dim)]
        s = float(sum(max(0.0, x) for x in pz))
        if s <= 0:
            pz = [1.0 / float(pop_dim) for _ in range(pop_dim)]
        else:
            pz = [max(0.0, x) / s for x in pz]

        # neighbor stance counts (optional; keep zeros)
        nb = bi.get("neighbor_stance_counts") or [0, 0, 0]
        try:
            nb = [int(x) for x in list(nb)[:3]]
        except Exception:
            nb = [0, 0, 0]

        is_core = bi.get("is_core_user", False)
        try:
            is_core = bool(is_core)
        except Exception:
            is_core = False

        return {
            "population_z": torch.tensor(pz, dtype=torch.float32, device=device),
            "neighbor_stance_counts": torch.tensor(nb, dtype=torch.float32, device=device),
            "is_core_user": torch.tensor([1 if is_core else 0], dtype=torch.int64, device=device),
        }

    def step(self, action: Any, extra_info: Optional[Dict[str, Any]] = None) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: The primary action (final answer string from coordinator/agent)
            extra_info: Additional information for reward calculation including:
                - 'agent_responses': List[str] - Individual agent responses
                - 'commitment_text': str - Coordinator's commitment text
                - 'agent_log_probs': Optional[List[float]] - Token log probabilities for AL reward
                - 'prompt_embeddings': Optional[torch.Tensor] - Agent prompt embeddings
                - 'belief_states': Optional[torch.Tensor] - Agent belief states
        """
        if self.current_sample is None:
            raise RuntimeError("step() called before reset() or after dataset exhaustion.")

        if self.use_dataset_episode:
            self.step_count += 1
        else:
            self.episode_length += 1
        
        # Extract primary action and additional info
        if isinstance(action, dict):
            llm_answer_str = str(action.get("answer", ""))
            if extra_info is None:
                extra_info = action  # Use action dict as extra_info if not provided separately
        else:
            llm_answer_str = str(action)
            
        if extra_info is None:
            extra_info = {}

        # === Verbose step logging (VERY noisy for long social-media observations) ===
        if self.verbose_step_logging:
            logger.info("=" * 80)
            logger.info(f"🔍 QUESTION: {self.current_question}")
            logger.info("=" * 80)
        
        # Strategy will be logged by MAC
        
        # Executor responses will be logged by MAC
        
        # Coordinator commitment will be logged by MAC
        
        if self.verbose_step_logging:
            logger.info(f"📖 GROUND TRUTH: {self.current_ground_truth_answer}")
            logger.info("=" * 80)

        # --- Reward Calculation ---
        # 权重（如果某项权重为 0，则跳过该项的昂贵计算，尤其是 TF-IDF 相似度）
        al_weight = float(getattr(self.reward_args, "al_weight", 0.3))
        ts_weight = float(getattr(self.reward_args, "ts_weight", 0.5))
        cc_weight = float(getattr(self.reward_args, "cc_weight", 0.2))

        # Task-Specific (TS) reward: 基于正确性（对 stance_id/boxed-id 监督最常用）
        is_correct = self._evaluate_answer(llm_answer_str, self.current_ground_truth_answer)
        reward_ts = 1.0 if is_correct else 0.0

        # Action Likelihood (AL) reward: 仅当权重>0 时计算
        reward_al = 0.0
        if al_weight > 0:
            reward_al = self._calculate_action_likelihood_reward(extra_info)

        # Collaborative Contribution (CC) reward: 仅当权重>0 时计算
        reward_cc = 0.0
        if cc_weight > 0:
            reward_cc = self._calculate_collaborative_contribution_reward(
                llm_answer_str, extra_info, is_correct
            )

        total_reward = al_weight * float(reward_al) + ts_weight * float(reward_ts) + cc_weight * float(reward_cc)
        
        # 显示奖励信息（可关闭）
        if self.verbose_step_logging:
            logger.info(f"🎯 REWARD BREAKDOWN:")
            logger.info(f"   TS (Task-Specific): {reward_ts:.3f} * {ts_weight:.1f} = {reward_ts * ts_weight:.3f}")
            logger.info(f"   AL (Action Likelihood): {reward_al:.3f} * {al_weight:.1f} = {reward_al * al_weight:.3f}")
            logger.info(f"   CC (Collaborative): {reward_cc:.3f} * {cc_weight:.1f} = {reward_cc * cc_weight:.3f}")
            logger.info(f"   TOTAL REWARD: {total_reward:.3f}")
            logger.info("=" * 80)
        
        # --- 数据集级别episode的特殊处理 ---
        if self.use_dataset_episode:
            # 记录当前样本结果
            step_result = {
                "question": self.current_question,
                "ground_truth": self.current_ground_truth_answer,
                "llm_answer": llm_answer_str,
                "is_correct": is_correct,
                "reward_ts": reward_ts,
                "reward_al": reward_al,
                "reward_cc": reward_cc,
                "total_reward": total_reward
            }
            self.current_episode_samples.append(self.current_sample)
            self.episode_results.append(step_result)
            
            # 检查是否完成整个数据集
            terminated = (self.step_count >= self.num_samples)
            
            if not terminated:
                # 获取下一个样本
                self.current_sample = self._get_next_sample()
                if self.current_sample is None:
                    terminated = True
                else:
                    self.current_question = str(self.current_sample.get(self.question_field, ""))
                    self.current_ground_truth_answer = str(self.current_sample.get(self.answer_field, ""))
            
            if terminated:
                # Episode结束，计算整体统计
                total_correct = sum(1 for r in self.episode_results if r["is_correct"])
                accuracy = total_correct / len(self.episode_results) if self.episode_results else 0.0
                avg_reward = sum(r["total_reward"] for r in self.episode_results) / len(self.episode_results) if self.episode_results else 0.0
                
                logger.info(f"📊 DATASET-LEVEL EPISODE {self.episode_count} COMPLETED:")
                logger.info(f"   Total samples: {len(self.episode_results)}")
                logger.info(f"   Correct answers: {total_correct}")
                logger.info(f"   Accuracy: {accuracy:.3f}")
                logger.info(f"   Average reward: {avg_reward:.3f}")
                logger.info("=" * 80)
                
                next_observation = ""  # Episode结束，无下一个观察
            else:
                next_observation = self.current_question  # 下一个问题作为下一个观察
        else:
            # 原有逻辑：每个问题一个episode
            terminated = True
            next_observation = ""  # Placeholder

        truncated = False # Not typically used if episode length is fixed at 1 or based on dataset size
        
        info = {
            "is_correct": is_correct,
            "reward_ts": reward_ts,
            "reward_al": reward_al,
            "reward_cc": reward_cc,
            "llm_answer": llm_answer_str,
            "ground_truth_answer": self.current_ground_truth_answer
        }

        # Pass-through optional fields from dataset samples.
        # - structured conditioning fields: used by encoder-only z_transition training
        # - target_distribution_prob: used by Stage1/2 soft-label supervised training
        try:
            if isinstance(self.current_sample, dict):
                for k in (
                    "core_stance_id_t",
                    "core_action_type_id_t",
                    "has_user_history",
                    "has_neighbors",
                    "neighbor_action_type_counts_t",
                    "neighbor_stance_counts_t",
                    "target_distribution_prob",
                ):
                    if k in self.current_sample:
                        info[k] = self.current_sample.get(k)
        except Exception:
            pass

        # Optional: emit belief/z supervision fields for offline transition training
        if self.emit_belief_fields and isinstance(self.current_sample, dict):
            # stage index
            st = self.current_sample.get("stage_t", self.current_sample.get("t", 0))
            try:
                st = int(st)
            except Exception:
                st = 0
            info["t"] = int(st)

            pop_dim = int(getattr(self, "population_belief_dim", 3))
            pop_dim = max(1, pop_dim)

            # z supervision fields
            z_t = self.current_sample.get("z_t")
            z_target = self.current_sample.get("z_target")
            z_mask = self.current_sample.get("z_mask", 0.0)
            if pop_dim == 1:
                # scalar regression target
                try:
                    if isinstance(z_target, torch.Tensor):
                        zt = float(z_target.detach().flatten()[0].item())
                    elif isinstance(z_target, (list, tuple)) and len(z_target) > 0:
                        zt = float(z_target[0])
                    else:
                        zt = float(z_target) if z_target is not None else 0.0
                except Exception:
                    zt = 0.0
                info["z_target"] = [float(max(-1.0, min(1.0, zt)))]
            else:
                if isinstance(z_target, (list, tuple)) and len(z_target) >= pop_dim:
                    info["z_target"] = [float(x) for x in list(z_target)[:pop_dim]]
            if z_mask is not None:
                try:
                    info["z_mask"] = float(z_mask)
                except Exception:
                    info["z_mask"] = 0.0

            # Provide belief_inputs_{pre,post} so EpisodeRunner can call get_belief_tensor() and fill batch z_t / belief_pre_population_z
            if pop_dim == 1:
                # z_t / z_target can be scalar in dataset; keep as scalar and let get_belief_tensor() handle it.
                info["belief_inputs_pre"] = {
                    "population_z": z_t,
                    "is_core_user": bool(self.current_sample.get("is_core_user", False)),
                    "neighbor_stance_counts": [0, 0, 0],
                }
                info["belief_inputs_post"] = {
                    "population_z": z_target,
                    "is_core_user": bool(self.current_sample.get("is_core_user", False)),
                    "neighbor_stance_counts": [0, 0, 0],
                }
            else:
                if isinstance(z_t, (list, tuple)) and len(z_t) >= pop_dim:
                    info["belief_inputs_pre"] = {
                        "population_z": [float(x) for x in list(z_t)[:pop_dim]],
                        "is_core_user": bool(self.current_sample.get("is_core_user", False)),
                        "neighbor_stance_counts": [0, 0, 0],
                    }
                if isinstance(z_target, (list, tuple)) and len(z_target) >= pop_dim:
                    info["belief_inputs_post"] = {
                        "population_z": [float(x) for x in list(z_target)[:pop_dim]],
                    "is_core_user": bool(self.current_sample.get("is_core_user", False)),
                    "neighbor_stance_counts": [0, 0, 0],
                }
        
        # 为数据集级别episode添加额外信息
        if self.use_dataset_episode:
            info.update({
                "step_count": self.step_count,
                "total_steps": self.num_samples,
                "progress": self.step_count / self.num_samples
            })
            
            if terminated:
                # 添加整体统计信息
                total_correct = sum(1 for r in self.episode_results if r["is_correct"])
                info.update({
                    "episode_accuracy": total_correct / len(self.episode_results) if self.episode_results else 0.0,
                    "episode_avg_reward": sum(r["total_reward"] for r in self.episode_results) / len(self.episode_results) if self.episode_results else 0.0,
                    "total_samples_processed": len(self.episode_results)
                })
        
        return next_observation, total_reward, terminated, truncated, info

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """Extracts content from \\boxed{} with improved fallback mechanisms."""
        if not isinstance(text, str): # Ensure text is a string
            return None
        
        # Primary: Look for \\boxed{content}
        match = re.search(r"\\boxed\{([\s\S]*?)\}", text)
        if match:
            content = match.group(1).strip()
            return content if content else None
        
        # Fallback 1: Look for boxed{content} without backslash
        match = re.search(r"boxed\{([\s\S]*?)\}", text)
        if match:
            content = match.group(1).strip()
            logger.info(f"Found 'boxed{{}}' without backslash: {content}")
            return content if content else None
        
        # Fallback 2: Look for the last number in the text (often the final answer)
        numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        last_number_candidate = numbers[-1] if numbers else None
        
        # Fallback 3: Look for "answer is" patterns, but prefer last number if found
        patterns = [
            r"(?:answer is|answer:|final answer is|final answer:|the answer is)\s*([+-]?\d+(?:\.\d+)?)",
            r"(?:therefore|thus|so)\s*[^0-9]*([+-]?\d+(?:\.\d+)?)",
            r"(?:equals|=)\s*([+-]?\d+(?:\.\d+)?)",
        ]
        
        pattern_matches = []
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                pattern_matches.append(match.group(1))
        
        # Strategy: If we have both pattern matches and a last number, 
        # prefer the last number if it appears in the later part of the text
        if last_number_candidate and pattern_matches:
            # Check if the last number appears after the pattern matches
            last_number_pos = text.rfind(last_number_candidate)
            pattern_positions = []
            for pattern_match in pattern_matches:
                pos = text.rfind(pattern_match)
                if pos != -1:
                    pattern_positions.append(pos)
            
            # If last number appears after pattern matches, prefer it
            if pattern_positions and last_number_pos > max(pattern_positions):
                logger.info(f"Using last number as it appears after pattern matches: {last_number_candidate}")
                return last_number_candidate
            elif pattern_matches:
                logger.info(f"Using pattern match: {pattern_matches[0]}")
                return pattern_matches[0]
        
        # If only pattern matches exist
        if pattern_matches:
            logger.info(f"Using pattern match: {pattern_matches[0]}")
            return pattern_matches[0]
        
        # If only last number exists
        if last_number_candidate:
            logger.info(f"Using last number in text as fallback: {last_number_candidate}")
            return last_number_candidate
        
        logger.warning(f"No answer found in text: {text[:100]}...")
        return None

    def _normalize_number_string(self, s: Optional[str]) -> Optional[str]:
        """Normalizes a string potentially representing a number."""
        if s is None:
            return None
        # Remove commas used as thousand separators
        s_no_commas = s.replace(",", "")
        # Remove trailing ".0" or ".00" etc. to treat 123.0 as 123 for int comparison
        # but keep 123.5 as 123.5
        if '.' in s_no_commas:
            parts = s_no_commas.split('.')
            if len(parts) == 2 and all(c == '0' for c in parts[1]):
                return parts[0] # Return only integer part if fractional part is all zeros
        return s_no_commas

    def _evaluate_answer(self, llm_answer: str, ground_truth_answer: str) -> bool:
        logger.debug(f"Evaluating LLM Answer: '{llm_answer}' vs Ground Truth: '{ground_truth_answer}'")

        llm_boxed_content = self._extract_boxed_content(llm_answer)
        gt_boxed_content = self._extract_boxed_content(ground_truth_answer)

        logger.debug(f"Boxed Content - LLM: '{llm_boxed_content}', GT: '{gt_boxed_content}'")

        # 如果两个都没有boxed内容，尝试直接从文本中提取数字
        if llm_boxed_content is None and gt_boxed_content is None:
            logger.info("Both answers lack \\boxed{} format, attempting direct text comparison")
            # 尝试从文本中提取最后的数字
            llm_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", llm_answer)
            gt_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", ground_truth_answer)
            
            if llm_numbers and gt_numbers:
                llm_boxed_content = llm_numbers[-1]
                gt_boxed_content = gt_numbers[-1]
                logger.info(f"Extracted numbers - LLM: '{llm_boxed_content}', GT: '{gt_boxed_content}'")
            else:
                logger.info(f"Evaluation failed: Unable to extract numerical answers. LLM: '{llm_answer[:100]}...', GT: '{ground_truth_answer[:100]}...'")
                return False

        # 如果只有一个有boxed内容，这通常表示格式问题
        if llm_boxed_content is None or gt_boxed_content is None:
            logger.info(f"Evaluation failed: Inconsistent answer formats. LLM boxed: '{llm_boxed_content}', GT boxed: '{gt_boxed_content}'")
            logger.info(f"Full answers - LLM: '{llm_answer[:150]}...', GT: '{ground_truth_answer[:150]}...'")
            return False

        # Normalize the string content from \boxed{} before attempting float conversion or string comparison
        norm_llm_content = self._normalize_number_string(llm_boxed_content)
        norm_gt_content = self._normalize_number_string(gt_boxed_content)
        
        logger.debug(f"Normalized Boxed Content - LLM: '{norm_llm_content}', GT: '{norm_gt_content}'")

        if norm_llm_content is None or norm_gt_content is None: # Should not happen if _extract_boxed_content returned non-None
             return False

        try:
            # Attempt to convert both to floats for numerical comparison
            llm_val = float(norm_llm_content)
            gt_val = float(norm_gt_content)

            # Check for near-equality
            if abs(llm_val - gt_val) < 1e-5:
                logger.info(f"✅ Correct answer: {llm_val} == {gt_val}")
                return True
            else:
                logger.info(f"❌ Numeric mismatch: LLM val {llm_val} vs GT val {gt_val}")
                return False
        except ValueError:
            # If conversion to float fails, fall back to string comparison
            logger.debug(f"ValueError converting to float. Comparing normalized strings: '{norm_llm_content}' vs '{norm_gt_content}'")
            if norm_llm_content == norm_gt_content:
                logger.info(f"✅ Correct answer (string match): '{norm_llm_content}'")
                return True
            else:
                # Last resort: compare the original (just stripped) boxed content
                if llm_boxed_content.strip() == gt_boxed_content.strip():
                    logger.info(f"✅ Correct answer (original content match): '{llm_boxed_content.strip()}'")
                    return True
                logger.info(f"❌ String mismatch after float conversion failed. LLM: '{norm_llm_content}', GT: '{norm_gt_content}'")
                return False

    def _calculate_action_likelihood_reward(self, extra_info: Dict[str, Any]) -> float:
        """
        计算动作似然性奖励 r^AL
        基于以下因素：
        1. Agent响应的一致性和质量
        2. 响应与commitment的相似度
        3. 响应的正确性（通过数字匹配检查）
        4. 如果API失败或响应无效，返回0.0
        """
        try:
            # 检查是否有有效的agent响应
            agent_responses = extra_info.get('agent_responses', [])
            commitment_text = extra_info.get('commitment_text', '')
            
            # 如果没有有效响应或响应包含错误信息，返回0.0
            if not agent_responses or not commitment_text:
                return 0.0
            
            # 检查响应是否包含API错误信息
            error_indicators = ['Error: Could not generate response', 'API Error', 'HTTP error', 'Failed to generate']
            for response in agent_responses:
                if any(error in str(response) for error in error_indicators):
                    return 0.0
            
            if any(error in str(commitment_text) for error in error_indicators):
                return 0.0
            
            # 过滤掉包含错误的响应
            valid_responses = [resp for resp in agent_responses 
                             if not any(error in str(resp) for error in error_indicators)]
            
            if not valid_responses:
                return 0.0
            
            # 方法1: 检查响应中数字答案的一致性
            response_numbers = []
            commitment_numbers = []
            
            # 从每个响应中提取数字
            for response in valid_responses:
                numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", response)
                if numbers:
                    response_numbers.append(numbers[-1])  # 使用最后一个数字作为答案
            
            # 从commitment中提取数字
            commit_numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", commitment_text)
            if commit_numbers:
                commitment_numbers = commit_numbers[-1]  # 使用最后一个数字作为答案
            
            # 计算数字一致性奖励
            numerical_consistency = 0.0
            if response_numbers and commitment_numbers:
                # 检查所有响应的数字是否与commitment一致
                consistent_count = sum(1 for num in response_numbers if num == commitment_numbers)
                numerical_consistency = consistent_count / len(response_numbers)
            
            # 方法2: 基于响应与commitment的文本相似度
            text_similarity = 0.0
            if valid_responses and commitment_text:
                similarities = []
                for response in valid_responses:
                    # 检查响应长度，太短的响应（如错误消息）给予低分
                    if len(response.strip()) < 10:
                        similarities.append(0.0)
                    else:
                        sim = self._calculate_text_similarity(response, commitment_text)
                        similarities.append(sim)
                
                if similarities:
                    text_similarity = np.mean(similarities)
            
            # 方法3: 响应长度和质量评估
            quality_score = 0.0
            if valid_responses:
                quality_scores = []
                for response in valid_responses:
                    response_length = len(response.strip())
                    # 合理长度的响应得分更高
                    if 20 <= response_length <= 500:  # 合理长度范围
                        length_score = 1.0
                    elif 10 <= response_length <= 1000:  # 可接受范围
                        length_score = 0.7
                    else:  # 太短或太长
                        length_score = 0.3
                    
                    # 检查是否包含推理过程
                    has_reasoning = any(keyword in response.lower() 
                                      for keyword in ['first', 'then', 'therefore', 'because', 'so'])
                    reasoning_score = 0.3 if has_reasoning else 0.0
                    
                    quality_scores.append(length_score + reasoning_score)
                
                quality_score = np.mean(quality_scores)
            
            # 组合所有因素，数字一致性权重最高
            if numerical_consistency > 0:
                # 如果有数字一致性，主要基于此计算
                al_reward = 0.6 * numerical_consistency + 0.25 * text_similarity + 0.15 * quality_score
            else:
                # 如果没有数字一致性，主要基于文本相似度
                al_reward = 0.7 * text_similarity + 0.3 * quality_score
            
            return min(1.0, max(0.0, al_reward))
            
        except Exception as e:
            logger.warning(f"Error calculating AL reward: {e}")
            return 0.0

    def _calculate_collaborative_contribution_reward(self, final_answer: str, 
                                                   extra_info: Dict[str, Any], 
                                                   is_correct: bool) -> float:
        """
        计算协作贡献奖励 r^CC
        基于以下启发式规则：
        1. 如果最终答案正确且智能体响应多样化，给予高奖励
        2. 如果智能体响应与commitment一致，给予中等奖励
        3. 考虑响应的独特性和互补性
        4. 如果API失败或响应无效，返回0.0
        """
        try:
            agent_responses = extra_info.get('agent_responses', [])
            commitment_text = extra_info.get('commitment_text', '')
            
            # 检查是否有API错误
            error_indicators = ['Error: Could not generate response', 'API Error', 'HTTP error', 'Failed to generate']
            
            # 如果最终答案包含错误信息，返回0.0
            if any(error in str(final_answer) for error in error_indicators):
                return 0.0
            
            # 如果commitment包含错误信息，返回0.0
            if any(error in str(commitment_text) for error in error_indicators):
                return 0.0
            
            # 过滤掉包含错误的响应
            valid_responses = []
            for response in agent_responses:
                if not any(error in str(response) for error in error_indicators):
                    valid_responses.append(response)
            
            # 如果没有有效响应，返回0.0
            if not valid_responses:
                return 0.0
            
            # 基础奖励：正确性贡献（只有正确时才有基础奖励）
            base_reward = 0.3 if is_correct else 0.0
            
            # 多样性奖励：智能体响应的多样性（只考虑有效响应）
            diversity_reward = 0.0
            if len(valid_responses) > 1:
                unique_responses = len(set([resp.strip().lower() for resp in valid_responses]))
                diversity_ratio = unique_responses / len(valid_responses)
                diversity_reward = 0.3 * diversity_ratio
            
            # 一致性奖励：与commitment的一致性（只考虑有效响应）
            consistency_reward = 0.0
            if valid_responses and commitment_text:
                consistencies = []
                for response in valid_responses:
                    # 检查响应长度
                    if len(response.strip()) < 10:
                        consistencies.append(0.0)
                    else:
                        sim = self._calculate_text_similarity(response, commitment_text)
                        consistencies.append(sim)
                
                if consistencies:
                    avg_consistency = np.mean(consistencies)
                    consistency_reward = 0.2 * avg_consistency
            
            # 质量奖励：如果最终答案包含推理过程且不包含错误
            quality_reward = 0.0
            if final_answer and len(final_answer.strip()) > 10:
                # 简单启发式：长度合理且包含数学术语的答案质量更高
                answer_length = len(final_answer.split())
                has_reasoning = any(keyword in final_answer.lower() 
                                  for keyword in ['because', 'since', 'therefore', 'thus', 'so'])
                
                if 10 <= answer_length <= 200 and has_reasoning:
                    quality_reward = 0.2
                elif 5 <= answer_length <= 50:
                    quality_reward = 0.1
            
            total_cc_reward = base_reward + diversity_reward + consistency_reward + quality_reward
            return min(1.0, max(0.0, total_cc_reward))
            
        except Exception as e:
            logger.warning(f"Error calculating CC reward: {e}")
            return 0.0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本之间的相似度
        使用TF-IDF向量和余弦相似度
        """
        try:
            if not text1.strip() or not text2.strip():
                return 0.0
            
            # 使用TF-IDF向量化
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            texts = [text1, text2]
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # 计算余弦相似度
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(max(0.0, min(1.0, similarity)))
            
        except Exception as e:
            logger.warning(f"Error calculating text similarity: {e}")
            return 0.0

    def get_env_info(self) -> Dict[str, Any]:
        # This info is used by the runner to setup the batch scheme.
        return {
            "episode_limit": self.episode_limit,
            "n_actions": int(self.n_actions),  # For discrete-action training (shapes only)
            "obs_shape": (self.max_question_length,), # For scheme vshape
            "state_shape": (1,), # Placeholder for scheme vshape
            # Any other info needed by the runner or learner
        }

    def render(self, mode='human'):
        if mode == 'human':
            if self.current_sample:
                print("-" * 30)
                print(f"Current Question: {self.current_question}")
                print(f"Ground Truth Answer: {self.current_ground_truth_answer}")
                print("-" * 30)
            else:
                print("No current sample to render. Call reset() first.")

    def close(self):
        # Clean up resources if any (e.g., closing file handles if dataset was local)
        logger.info("Closing HuggingFaceDatasetEnv.")
        pass

# Example Usage (for testing purposes):
if __name__ == '__main__':
    env_args_gsm8k = {
        "hf_dataset_path": "gsm8k",
        "hf_dataset_config_name": "main",
        "dataset_split": "test",
        "question_field_name": "question",
        "answer_field_name": "answer",
        "max_question_length": 1024,
        "max_answer_length": 200,
        "dataset_streaming": False, # For testing, non-streaming is easier
        "use_random_sampling": False, # Disable random sampling for deterministic testing
        "use_dataset_episode": False # Disable dataset-level episode for deterministic testing
    }
    
    # env = HuggingFaceDatasetEnv(**env_args_gsm8k)
    # obs, info = env.reset()
    # print("Observation (Question):", obs)
    # print("Ground Truth (from info['sample']):", info['sample'][env.answer_field])
    
    # # Simulate a step
    # dummy_action = "The answer is \\boxed{10}." 
    # next_obs, reward, terminated, truncated, step_info = env.step(dummy_action)
    # print(f"LLM's Action: {dummy_action}")
    # print(f"Next Obs: {next_obs}")
    # print(f"Reward: {reward}")
    # print(f"Terminated: {terminated}")
    # print(f"Step Info: {step_info}")
    # env.render()

    # env.reset() # Try another one
    # env.render()

    # Test MATH dataset
    env_args_math = {
        "hf_dataset_path": "competition_math",
        "dataset_split": "test", # Using test split which is smaller
        "question_field_name": "problem",
        "answer_field_name": "solution",
        "max_question_length": 2048,
        "max_answer_length": 2048,
        "dataset_streaming": False,
        "use_random_sampling": False, # Disable random sampling for deterministic testing
        "use_dataset_episode": False # Disable dataset-level episode for deterministic testing
    }
    math_env = HuggingFaceDatasetEnv(**env_args_math)
    obs, info = math_env.reset()
    math_env.render()
    # Simulate a step for MATH
    # For MATH, answers are more complex, often with LaTeX. Evaluation is harder.
    # Example: ground truth might be "\\boxed{-\\frac{1}{2}}"
    # dummy_math_action = "The final answer is \\boxed{-\\frac{1}{2}}"
    dummy_math_action = info['sample'][math_env.answer_field] # Give correct answer
    next_obs, reward, terminated, truncated, step_info = math_env.step(dummy_math_action)
    print(f"LLM's Action: {dummy_math_action}")
    print(f"Reward: {reward}") # Should be 1.0
    print(f"Step Info: {step_info}")
    math_env.render()

    # Test an incorrect answer for MATH
    obs, info = math_env.reset()
    math_env.render()
    dummy_math_action_wrong = "The final answer is \\boxed{42}"
    next_obs, reward, terminated, truncated, step_info = math_env.step(dummy_math_action_wrong)
    print(f"LLM's Action (Wrong): {dummy_math_action_wrong}")
    print(f"Reward: {reward}") # Should be 0.0
    print(f"Step Info: {step_info}")
    math_env.render() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import copy
import torch
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional
from collections import deque
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore

# Import necessary components
from utils.logging import get_logger
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY

# Optional distributed support (torchrun/DDP)
try:  # pragma: no cover
    import torch.distributed as dist  # type: ignore
    from torch.nn.parallel import DistributedDataParallel as DDP  # type: ignore
    _HAS_DIST = True
except Exception:  # pragma: no cover
    dist = None  # type: ignore
    DDP = None  # type: ignore
    _HAS_DIST = False

# YAML loader: prefer PyYAML; fallback to ruamel.yaml if PyYAML isn't available.
try:
    import yaml  # type: ignore
    _HAS_PYYAML = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    _HAS_PYYAML = False
    try:
        from ruamel.yaml import YAML  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("No YAML parser available. Please install PyYAML (pyyaml) or ruamel.yaml.") from e

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ECON Framework Training Script')
    
    # Basic parameters
    parser.add_argument('--config', type=str, default='src/config/config.yaml', help='Configuration file path')
    parser.add_argument('--executor_model', type=str, help='Executor LLM model name')
    parser.add_argument('--coordinator_model', type=str, help='Coordinator LLM model name')
    parser.add_argument('--n_agents', type=int, help='Number of agents')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--log_dir', type=str, help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint root directory (overrides config.logging.checkpoint_path)')
    parser.add_argument('--final_save_dir', type=str, help='Final model output directory (overrides default <checkpoint_dir>/final)')
    parser.add_argument('--api_key', type=str, help='Together API key')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--env', type=str, help='Environment name')
    parser.add_argument('--load_model_path', type=str, help='Optional checkpoint directory to load (expects files saved by learner.save_models)')

    # Distributed training (torchrun)
    parser.add_argument('--distributed', action='store_true', help='Enable DistributedDataParallel (use with torchrun).')
    parser.add_argument('--ddp_backend', type=str, default='nccl', help='DDP backend (nccl/gloo).')
    parser.add_argument('--ddp_find_unused_params', action='store_true', help='DDP find_unused_parameters=True (safer, slower).')
    parser.add_argument('--ddp_reduce_batch_by_world_size', action='store_true', help='In DDP, divide config.train.batch_size by world_size per rank (keeps global effective batch).')
    
    # wandb related parameters
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb for experiment logging')
    parser.add_argument('--wandb_project', type=str, default='ECON-Framework', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, help='wandb username or organization name')
    parser.add_argument('--wandb_tags', type=str, help='wandb tags, separated by commas')
    
    return parser.parse_args()

def load_config(config_path: str) -> SimpleNamespace:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        if _HAS_PYYAML and yaml is not None:
            config_dict = yaml.safe_load(f)
        else:
            y = YAML(typ="safe")
            config_dict = y.load(f)
    
    # Recursively convert to SimpleNamespace for easy access
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d
    
    config = dict_to_namespace(config_dict)
    return config

def update_config_with_args(config: SimpleNamespace, args: Any) -> SimpleNamespace:
    """Update configuration with command line arguments"""
    # Only update non-None parameters
    if args.executor_model:
        if hasattr(config, 'llm'):
            config.llm.executor_model = args.executor_model
    
    if args.coordinator_model:
        if hasattr(config, 'llm'):
            config.llm.coordinator_model = args.coordinator_model
    
    if args.n_agents:
        config.n_agents = args.n_agents
    
    if args.experiment_name:
        if hasattr(config, 'logging'):
            config.logging.experiment_name = args.experiment_name
    
    if args.log_dir:
        if hasattr(config, 'logging'):
            config.logging.log_path = args.log_dir

    if getattr(args, "checkpoint_dir", None):
        if not hasattr(config, 'logging'):
            config.logging = SimpleNamespace()
        config.logging.checkpoint_path = str(args.checkpoint_dir)

    if getattr(args, "final_save_dir", None):
        # Dedicated final output dir; used by run_training() at the very end.
        config.final_save_dir = str(args.final_save_dir)
    
    if args.api_key:
        # Set API key in both places for compatibility
        config.together_api_key = args.api_key  # For direct access as args.together_api_key
        if hasattr(config, 'llm'):
            config.llm.together_api_key = args.api_key
    
    if args.seed:
        config.system.seed = args.seed
    
    if args.env:
        config.env = args.env

    if getattr(args, "load_model_path", None):
        config.load_model_path = str(args.load_model_path)

    # Distributed options (torchrun/DDP)
    if getattr(args, "distributed", None) is not None:
        config.distributed = bool(args.distributed)
    if getattr(args, "ddp_backend", None):
        config.ddp_backend = str(args.ddp_backend)
    if getattr(args, "ddp_find_unused_params", None) is not None:
        config.ddp_find_unused_params = bool(args.ddp_find_unused_params)
    if getattr(args, "ddp_reduce_batch_by_world_size", None) is not None:
        config.ddp_reduce_batch_by_world_size = bool(args.ddp_reduce_batch_by_world_size)
    
    # Add wandb related configuration
    if not hasattr(config, 'wandb'):
        config.wandb = SimpleNamespace()
    
    config.wandb.use_wandb = args.use_wandb
    if args.wandb_project:
        config.wandb.project = args.wandb_project
    if args.wandb_entity:
        config.wandb.entity = args.wandb_entity
    if args.wandb_tags:
        config.wandb.tags = args.wandb_tags.split(',')
    
    return config

def setup_experiment(config: SimpleNamespace):
    """Setup experiment environment and components"""
    # --- detect distributed env early (so we can silence TB on non-zero ranks) ---
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    dist_enabled = bool(getattr(config, "distributed", False)) or (world_size > 1)
    if dist_enabled and (not _HAS_DIST):
        raise RuntimeError("distributed=True but torch.distributed is not available in this environment.")

    # Respect config logging settings so TensorBoard points to the expected directory.
    log_dir = "logs"
    exp_name = None
    use_tb = True
    try:
        if hasattr(config, "logging"):
            log_dir = str(getattr(config.logging, "log_path", log_dir) or log_dir)
            exp_name = getattr(config.logging, "experiment_name", exp_name)
            use_tb = bool(getattr(config.logging, "use_tensorboard", use_tb))
    except Exception:
        pass
    # In DDP: only rank0 writes TensorBoard/metrics.jsonl to avoid duplicate event streams.
    if dist_enabled and rank != 0:
        use_tb = False
    logger = get_logger(log_dir=log_dir, experiment_name=exp_name, use_tensorboard=use_tb)
    logger.info("Setting up experiment environment...")
    
    # Set random seed
    seed = config.system.seed if hasattr(config, 'system') and hasattr(config.system, 'seed') else 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set device (DDP uses LOCAL_RANK)
    use_cuda = hasattr(config, 'system') and hasattr(config.system, 'use_cuda') and config.system.use_cuda and torch.cuda.is_available()
    device_num = config.system.device_num if hasattr(config, 'system') and hasattr(config.system, 'device_num') else 0
    if dist_enabled:
        device_num = int(local_rank)
        try:
            if hasattr(config, "system"):
                config.system.device_num = int(device_num)
        except Exception:
            pass
    device = torch.device(f"cuda:{device_num}" if use_cuda else "cpu")
    
    # Add device to config object for runner access
    config.device = device
    
    if use_cuda:
        torch.cuda.set_device(device_num)

    # Initialize process group (after device is set)
    if dist_enabled:
        try:
            backend = str(getattr(config, "ddp_backend", "nccl") or "nccl")
            if dist is not None and (not dist.is_initialized()):
                dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
            # expose for later logic (saving/batch scaling)
            config.distributed_rank = int(rank)
            config.distributed_local_rank = int(local_rank)
            config.distributed_world_size = int(world_size)
        except Exception as e:
            raise RuntimeError(f"Failed to init distributed process group: {e}")
    
    # Initialize Runner
    runner = r_REGISTRY[config.runner](args=config, logger=logger)
    
    # Setup schemes and groups
    # Prefer runner-provided scheme/groups (supports tokenized obs, extra fields, etc.)
    try:
        if hasattr(runner, "_build_scheme") and callable(getattr(runner, "_build_scheme")):
            scheme = runner._build_scheme()
        else:
            scheme = None
        if hasattr(runner, "_build_groups") and callable(getattr(runner, "_build_groups")):
            groups = runner._build_groups()
        else:
            groups = None
    except Exception as e:
        logger.warning(f"Failed to build scheme/groups from runner, fallback to default: {e}")
        scheme, groups = None, None

    if scheme is None:
        scheme = {
            "state": {"vshape": runner.env_info["state_shape"]},
            "obs": {
                "vshape": runner.env_info["obs_shape"],
                "group": "agents",
            },
            "actions": {
                "vshape": (1,),
                "group": "agents",
                "dtype": torch.long,
            },
            "avail_actions": {
                "vshape": (runner.env_info["n_actions"],),
                "group": "agents",
                "dtype": torch.int,
            },
            
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": torch.uint8},
            "belief_states": {"vshape": (config.belief_dim,), "group": "agents"},
        }

    if groups is None:
        groups = {"agents": config.n_agents}
    
    # Initialize MAC
    mac = mac_REGISTRY[config.mac](scheme, groups, config)
    
    # Setup runner
    runner.setup(scheme, groups, None, mac)
    
    # Initialize learner
    learner = le_REGISTRY[config.learner](mac=mac, scheme=scheme, logger=logger, args=config)
    # Ensure model components are moved to the intended device.
    # Some runners/envs place tensors on CUDA when enabled; without this, we can hit CPU/GPU mismatch.
    if use_cuda:
        try:
            learner.cuda()
            logger.info("Moved learner/MAC components to CUDA.")
        except Exception as e:
            logger.warning(f"Failed to move learner to CUDA: {e}")

    # Wrap trainable modules with DDP (must happen AFTER cuda() and AFTER optional load_models()).
    # Note: BasicMAC has DDP-safe wrappers for non-forward methods (generate_answer/save_models/etc).
    if dist_enabled and use_cuda:
        try:
            # In several ECON stages we intentionally train only a subset of heads/parameters
            # (e.g., Stage1/2 stance_head only; Stage3b action_type_head (+ optional z-fusion)),
            # so "unused parameters" in a given forward/backward is expected.
            # DDP requires find_unused_parameters=True in that case.
            find_unused = bool(getattr(config, "ddp_find_unused_params", False))
            if not find_unused:
                if bool(getattr(config, "train_belief_supervised", False)) or bool(getattr(config, "train_action_imitation", False)):
                    find_unused = True
                    if rank == 0:
                        logger.info("[DDP] Auto-enabled find_unused_parameters=True for partial-head supervised training (Stage1/2/3b).")

            def _has_trainable_params(m: Any) -> bool:
                try:
                    return any(bool(p.requires_grad) for p in m.parameters())
                except Exception:
                    return False

            # Wrap agent (only if it has trainable params)
            if getattr(mac, "agent", None) is not None and DDP is not None:
                if _has_trainable_params(mac.agent):
                    mac.agent = DDP(mac.agent, device_ids=[device_num], output_device=device_num, find_unused_parameters=find_unused)
                else:
                    if rank == 0:
                        logger.info("[DDP] Skipping DDP wrap for mac.agent: no trainable params (requires_grad=False).")

            # Wrap belief_encoder ONLY if it has trainable params.
            # In Stage1/2 you freeze BeliefEncoder, so DDP should be skipped to avoid PyTorch error.
            if getattr(mac, "belief_encoder", None) is not None and DDP is not None:
                if _has_trainable_params(mac.belief_encoder):
                    mac.belief_encoder = DDP(
                        mac.belief_encoder,
                        device_ids=[device_num],
                        output_device=device_num,
                        find_unused_parameters=find_unused,
                    )
                else:
                    if rank == 0:
                        logger.info("[DDP] Skipping DDP wrap for mac.belief_encoder: frozen in this stage (0 trainable params).")

            # Expose rank/world_size on runner for convenience
            try:
                runner.distributed_rank = int(rank)
                runner.distributed_world_size = int(world_size)
            except Exception:
                pass
            if rank == 0:
                logger.info(f"Enabled DDP: world_size={world_size}, backend={getattr(config,'ddp_backend','nccl')}, find_unused={find_unused}")
        except Exception as e:
            raise RuntimeError(f"Failed to wrap modules with DDP: {e}")

    # Optional: resume/load checkpoint
    load_path = str(getattr(config, "load_model_path", "") or "").strip()
    if load_path:
        try:
            if os.path.isdir(load_path):
                learner.load_models(load_path)
                logger.info(f"Loaded checkpoint from: {load_path}")
            else:
                logger.warning(f"load_model_path is set but not a directory: {load_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {load_path}: {e}")
    
    return runner, mac, learner, logger, device

def run_training(config: SimpleNamespace, runner, learner, logger, device):
    """Execute training loop"""
    logger.info("Starting training...")
    
    begin_time = time.time()
    
    # Training configuration
    t_max = getattr(config, 't_max', 2000000)
    test_interval = getattr(config, 'test_interval', 50000)
    log_interval = getattr(config, 'logging', SimpleNamespace()).log_interval if hasattr(config, 'logging') else 2000
    save_model_interval = getattr(config, 'logging', SimpleNamespace()).save_model_interval if hasattr(config, 'logging') else 10000
    
    # Early stopping related variables
    last_commitment = None
    last_total_loss = None
    patience_counter = 0

    # ===== Sliding window moving average (for noisy per-episode stats) =====
    # Default window=200; you can override via config.logging.moving_avg_window
    try:
        _logging_cfg = getattr(config, "logging", SimpleNamespace())
        moving_avg_window = int(getattr(_logging_cfg, "moving_avg_window", 200))
    except Exception:
        moving_avg_window = 200
    moving_avg_window = max(1, moving_avg_window)
    ma_buffers: Dict[str, deque] = {
        "loss_total": deque(maxlen=moving_avg_window),
        "belief_sup_acc": deque(maxlen=moving_avg_window),
        "reward_mean": deque(maxlen=moving_avg_window),
        # supervised diagnostics
        "belief_sup_soft_available_frac": deque(maxlen=moving_avg_window),
        "belief_sup_soft_used_frac": deque(maxlen=moving_avg_window),
        "belief_sup_soft_p1_mean": deque(maxlen=moving_avg_window),
        "belief_sup_grad_norm": deque(maxlen=moving_avg_window),
        "belief_sup_entropy": deque(maxlen=moving_avg_window),
        "belief_sup_maxprob": deque(maxlen=moving_avg_window),
        "belief_sup_logit_abs_mean": deque(maxlen=moving_avg_window),
        "belief_sup_logit_std": deque(maxlen=moving_avg_window),
        "belief_sup_p0_mean": deque(maxlen=moving_avg_window),
        "belief_sup_p1_mean": deque(maxlen=moving_avg_window),
        "belief_sup_p0_gt05_frac": deque(maxlen=moving_avg_window),
        "belief_sup_hard_pred0_frac": deque(maxlen=moving_avg_window),
        "belief_sup_delta01_mean": deque(maxlen=moving_avg_window),
        "belief_sup_marginal_gap0": deque(maxlen=moving_avg_window),
        "belief_sup_marginal_gap1": deque(maxlen=moving_avg_window),
        "belief_sup_z_gate": deque(maxlen=moving_avg_window),
        "belief_sup_pred0_frac": deque(maxlen=moving_avg_window),
        "belief_sup_pred1_frac": deque(maxlen=moving_avg_window),
        "belief_sup_pred2_frac": deque(maxlen=moving_avg_window),
        "belief_sup_gt0_frac": deque(maxlen=moving_avg_window),
        "belief_sup_gt1_frac": deque(maxlen=moving_avg_window),
        "belief_sup_gt2_frac": deque(maxlen=moving_avg_window),
    }

    ma_skipped: Dict[str, int] = {
        "loss_total": 0,
        "belief_sup_acc": 0,
        "reward_mean": 0,
        "belief_sup_soft_available_frac": 0,
        "belief_sup_soft_used_frac": 0,
        "belief_sup_soft_p1_mean": 0,
        "belief_sup_grad_norm": 0,
        "belief_sup_entropy": 0,
        "belief_sup_maxprob": 0,
        "belief_sup_logit_abs_mean": 0,
        "belief_sup_logit_std": 0,
        "belief_sup_p0_mean": 0,
        "belief_sup_p1_mean": 0,
        "belief_sup_p0_gt05_frac": 0,
        "belief_sup_hard_pred0_frac": 0,
        "belief_sup_delta01_mean": 0,
        "belief_sup_marginal_gap0": 0,
        "belief_sup_marginal_gap1": 0,
        "belief_sup_z_gate": 0,
        "belief_sup_pred0_frac": 0,
        "belief_sup_pred1_frac": 0,
        "belief_sup_pred2_frac": 0,
        "belief_sup_gt0_frac": 0,
        "belief_sup_gt1_frac": 0,
        "belief_sup_gt2_frac": 0,
    }

    def _ma_update(name: str, value: float) -> float:
        """Update moving average buffer with finite values only; return current mean (or NaN if empty)."""
        buf = ma_buffers.get(name)
        if buf is None:
            buf = deque(maxlen=moving_avg_window)
            ma_buffers[name] = buf
        v = float(value)
        # Do NOT let a single NaN/Inf poison the whole moving average curve
        if not np.isfinite(v):
            ma_skipped[name] = int(ma_skipped.get(name, 0)) + 1
            return float(np.mean(buf)) if len(buf) > 0 else float("nan")
        buf.append(v)
        return float(np.mean(buf)) if len(buf) > 0 else v
    
    # Get early stopping thresholds from configuration
    early_stopping = getattr(config, 'early_stopping', SimpleNamespace())
    commitment_threshold = getattr(early_stopping, 'commitment_threshold', 0.01)
    reward_threshold = getattr(early_stopping, 'reward_threshold', 0.7)
    loss_threshold = getattr(early_stopping, 'loss_threshold', 0.0001)
    patience = getattr(early_stopping, 'patience', 5)

    # Training loop
    episode = 0
    t_env = 0

    # Distributed info (rank0 handles side effects like saving/checkpointing/eval)
    dist_rank = int(getattr(config, "distributed_rank", 0) or 0)
    dist_world = int(getattr(config, "distributed_world_size", 1) or 1)
    is_rank0 = (dist_rank == 0)

    # ===== Stage1/2: supervised mini-batch replay (critical for stability) =====
    # The runner returns an EpisodeBatch with batch_size_run (often 1). If we train on it directly,
    # supervised learning degenerates into single-sample SGD (effective_count ~= n_agents), which is
    # extremely unstable and prone to fast collapse for imbalanced labels.
    #
    # Here we keep a small replay of recent EpisodeBatch samples (size=train.buffer_size) and,
    # every train.update_interval episodes, sample train.batch_size of them to form a true mini-batch.
    supervised_mode = bool(getattr(config, "train_belief_supervised", False))
    # store tuples: (gt_label:int|None, EpisodeBatch)
    supervised_replay = []
    try:
        _train_cfg = getattr(config, "train", SimpleNamespace())
        supervised_batch_size = int(getattr(_train_cfg, "batch_size", 16))
        supervised_buffer_size = int(getattr(_train_cfg, "buffer_size", 128))
        supervised_update_interval = int(getattr(_train_cfg, "update_interval", 1))
        # optional: enforce at least K samples of label-1 in each supervised batch (if available in replay)
        supervised_min_label1 = int(getattr(_train_cfg, "supervised_min_label1_per_batch", 0))
    except Exception:
        supervised_batch_size = 16
        supervised_buffer_size = 128
        supervised_update_interval = 1
        supervised_min_label1 = 0
    supervised_batch_size = max(1, supervised_batch_size)
    supervised_buffer_size = max(supervised_batch_size, supervised_buffer_size)
    supervised_update_interval = max(1, supervised_update_interval)
    supervised_min_label1 = max(0, supervised_min_label1)

    # In DDP: reduce per-rank batch_size so global effective batch stays the same (and per-GPU memory drops).
    # Example: global batch_size=64, world_size=4 -> per-rank=16.
    try:
        reduce_bs_flag = getattr(config, "ddp_reduce_batch_by_world_size", None)
        reduce_bs = True if reduce_bs_flag is None else bool(reduce_bs_flag)
        if dist_world > 1 and reduce_bs:
            per_rank = max(1, int(supervised_batch_size) // int(dist_world))
            if per_rank < int(supervised_batch_size):
                if is_rank0:
                    logger.info(f"[DDP] train.batch_size(global)={supervised_batch_size} -> per-rank={per_rank} (world_size={dist_world})")
                supervised_batch_size = int(per_rank)
                supervised_buffer_size = max(supervised_batch_size, supervised_buffer_size)
    except Exception:
        pass

    def _concat_episode_batches(batches):
        if not batches:
            return None
        b0 = batches[0]
        from components.episode_buffer import EpisodeBatch
        out = EpisodeBatch(
            scheme=b0.scheme,
            groups=b0.groups,
            batch_size=len(batches),
            max_seq_length=b0.max_seq_length,
            device=b0.device,
        )
        for k in out.data.keys():
            try:
                out.data[k] = torch.cat([b.data[k] for b in batches], dim=0)
            except Exception:
                # keep default zeros for this key if concat fails
                pass
        return out

    # ===== B: curriculum schedule (optional) =====
    # config.curriculum example:
    # curriculum:
    #   enabled: true
    #   t_env_steps: [20000, 60000]   # thresholds (env steps) to advance
    #   n_stages: [7, 10, 13]         # must be len(t_env_steps)+1
    cur = getattr(config, "curriculum", SimpleNamespace())
    cur_enabled = bool(getattr(cur, "enabled", False))
    cur_t_env_steps = list(getattr(cur, "t_env_steps", [])) if cur_enabled else []
    cur_n_stages = list(getattr(cur, "n_stages", [])) if cur_enabled else []
    cur_idx = 0
    if cur_enabled and cur_n_stages:
        # initialize to the first stage count (should be <= original env_args.n_stages)
        try:
            ns0 = int(cur_n_stages[0])
            runner.set_env_n_stages(ns0)
            # keep config in sync for logging
            if hasattr(config, "env_args"):
                config.env_args.n_stages = ns0
        except Exception as e:
            logger.warning(f"Failed to initialize curriculum: {e}")
    
    while t_env < t_max:
        # curriculum advance
        if cur_enabled and cur_n_stages and cur_t_env_steps and (cur_idx + 1) < len(cur_n_stages):
            try:
                next_threshold = int(cur_t_env_steps[cur_idx])
            except Exception:
                next_threshold = None
            if next_threshold is not None and t_env >= next_threshold:
                cur_idx += 1
                try:
                    ns = int(cur_n_stages[cur_idx])
                    runner.set_env_n_stages(ns)
                    if hasattr(config, "env_args"):
                        config.env_args.n_stages = ns
                    logger.info(f"[Curriculum] Switched to n_stages={ns} at t_env={t_env}")
                except Exception as e:
                    logger.warning(f"Failed to advance curriculum at t_env={t_env}: {e}")

        # Run episode
        episode_batch = runner.run(test_mode=False)
        
        # Train learner
        if episode_batch is not None:
            if supervised_mode:
                # IMPORTANT (OOM fix):
                # EpisodeRunner builds EpisodeBatch tensors on args.device (CUDA in our runs).
                # If we keep a replay of EpisodeBatch objects on GPU, GPU memory will grow roughly
                # linearly with replay size until OOM (and DDP duplicates this on each rank).
                # So we store replay batches on CPU, and move only the sampled mini-batch back to CUDA.
                try:
                    if hasattr(episode_batch, "to"):
                        episode_batch = episode_batch.to(torch.device("cpu"))
                        # Optional: pin CPU memory to speed up non_blocking CPU->GPU transfer of sampled mini-batches.
                        # This improves throughput after moving replay to CPU.
                        try:
                            pin_flag = getattr(config, "supervised_replay_pin_memory", None)
                            pin_replay = True if pin_flag is None else bool(pin_flag)
                            if pin_replay and hasattr(episode_batch, "pin_memory"):
                                episode_batch = episode_batch.pin_memory()
                        except Exception:
                            pass
                except Exception:
                    pass
                # Push into supervised replay (also extract gt_action label for stratified sampling)
                gt_label = None
                try:
                    gt_t = episode_batch.data.get("gt_action", None)
                    # gt_action shape: (bs, T, 1); runner batch_size_run is usually 1
                    if isinstance(gt_t, torch.Tensor) and gt_t.numel() > 0:
                        gt_label = int(gt_t.view(-1)[0].item())
                except Exception:
                    gt_label = None
                supervised_replay.append((gt_label, episode_batch))
                if len(supervised_replay) > supervised_buffer_size:
                    supervised_replay = supervised_replay[-supervised_buffer_size:]

                # Only train when we have enough and on update_interval
                if len(supervised_replay) >= supervised_batch_size and (episode % supervised_update_interval == 0):
                    import random
                    # sample without replacement for a more diverse mini-batch
                    if supervised_min_label1 > 0:
                        # True "at least K": first ensure K label-1 samples, then fill remaining from the
                        # *entire remaining pool* (which may include more label-1).
                        n = len(supervised_replay)
                        one_idx = [i for i, (lab, _b) in enumerate(supervised_replay) if lab == 1]
                        k = min(int(supervised_min_label1), len(one_idx), supervised_batch_size)
                        picked_idx = set()
                        if k > 0:
                            picked_idx.update(random.sample(one_idx, k))
                        rem = supervised_batch_size - len(picked_idx)
                        if rem > 0:
                            pool_idx = [i for i in range(n) if i not in picked_idx]
                            if len(pool_idx) >= rem:
                                picked_idx.update(random.sample(pool_idx, rem))
                            else:
                                # fallback: if replay is tiny, sample with replacement from all
                                picked_idx.update(random.choices(list(range(n)), k=rem))
                        picked_list = [supervised_replay[i][1] for i in list(picked_idx)[:supervised_batch_size]]
                        sampled_batches = picked_list
                        label1_in_batch = float(k)
                    else:
                        sampled_batches = [b for (_lab, b) in random.sample(supervised_replay, supervised_batch_size)]
                        label1_in_batch = float("nan")
                    mb = _concat_episode_batches(sampled_batches)
                    # Move sampled mini-batch to the intended training device (per-rank CUDA in DDP).
                    try:
                        if mb is not None and hasattr(mb, "to"):
                            mb = mb.to(device)
                    except Exception:
                        pass
                    train_stats = learner.train(mb, t_env, episode)
                    # annotate for visibility
                    if isinstance(train_stats, dict):
                        train_stats["supervised_replay_size"] = float(len(supervised_replay))
                        train_stats["supervised_batch_size"] = float(supervised_batch_size)
                        train_stats["supervised_min_label1_per_batch"] = float(supervised_min_label1)
                        train_stats["supervised_label1_picked"] = float(label1_in_batch) if label1_in_batch == label1_in_batch else float("nan")
                else:
                    train_stats = {
                        "status": "pending_supervised_minibatch",
                        "supervised_replay_size": float(len(supervised_replay)),
                        "supervised_batch_size": float(supervised_batch_size),
                        "supervised_min_label1_per_batch": float(supervised_min_label1),
                    }
            else:
                train_stats = learner.train(episode_batch, t_env, episode)

            # === High-signal fixed metrics (core reward + secondary belief loss) ===
            # These are written to metrics.jsonl and TensorBoard regardless of console verbosity.
            try:
                if isinstance(train_stats, dict):
                    # core losses
                    for k in ("loss_total", "loss_belief", "loss_encoder", "loss_mixer"):
                        if k in train_stats:
                            logger.log_stat(f"train/{k}", float(train_stats[k]), t_env)
                    if "reward_mean" in train_stats:
                        logger.log_stat("train/reward_mean", float(train_stats["reward_mean"]), t_env)
                    # Stage1/2 supervised accuracy (if present)
                    if "belief_sup_acc" in train_stats:
                        logger.log_stat("train/belief_sup_acc", float(train_stats["belief_sup_acc"]), t_env)
                        # Stage3b-friendly alias (masked supervision)
                        logger.log_stat("train/action_sup_acc_masked", float(train_stats["belief_sup_acc"]), t_env)
                    # Stage3b masked CE loss alias (same as loss_total in supervised mode)
                    if bool(getattr(config, "train_action_imitation", False)):
                        # Prefer loss_belief (explicit supervised CE) to avoid confusion across modes.
                        if "loss_belief" in train_stats:
                            logger.log_stat("train/action_sup_loss_masked", float(train_stats["loss_belief"]), t_env)
                        elif "loss_total" in train_stats:
                            logger.log_stat("train/action_sup_loss_masked", float(train_stats["loss_total"]), t_env)
                        # Sanity: log difference if both exist (should be ~0 in S3b supervised mode)
                        if ("loss_belief" in train_stats) and ("loss_total" in train_stats):
                            try:
                                logger.log_stat(
                                    "train/action_sup_loss_minus_loss_belief",
                                    float(train_stats["loss_total"]) - float(train_stats["loss_belief"]),
                                    t_env,
                                )
                            except Exception:
                                pass
                    # Stage1/2 supervised diagnostics (if present)
                    for k in (
                        "belief_sup_soft_available_frac",
                        "belief_sup_soft_used_frac",
                        "belief_sup_soft_p1_mean",
                        "belief_sup_grad_norm",
                        "belief_sup_effective_count",
                        "belief_sup_entropy",
                        "belief_sup_maxprob",
                        "belief_sup_logit_abs_mean",
                        "belief_sup_logit_std",
                        "belief_sup_p0_mean",
                        "belief_sup_p1_mean",
                        "belief_sup_p0_gt05_frac",
                        "belief_sup_hard_pred0_frac",
                        "belief_sup_delta01_mean",
                        "belief_sup_marginal_gap0",
                        "belief_sup_marginal_gap1",
                        "belief_sup_z_gate",
                        "belief_sup_ce_w0",
                        "belief_sup_ce_w1",
                        "belief_sup_ce_w2",
                        "belief_sup_ce_w3",
                        "belief_sup_ce_w4",
                        "belief_sup_pred0_frac",
                        "belief_sup_pred1_frac",
                        "belief_sup_pred2_frac",
                        "belief_sup_gt0_frac",
                        "belief_sup_gt1_frac",
                        "belief_sup_gt2_frac",
                        # per-class recall/precision (helps debug collapse / class-0 missing)
                        "belief_sup_recall0",
                        "belief_sup_recall1",
                        "belief_sup_recall2",
                        "belief_sup_precision0",
                        "belief_sup_precision1",
                        "belief_sup_precision2",
                        # counts/flags (helps interpret recall/precision; avoids NaN confusion)
                        "belief_sup_gt0_count",
                        "belief_sup_gt1_count",
                        "belief_sup_gt2_count",
                        "belief_sup_pred0_count",
                        "belief_sup_pred1_count",
                        "belief_sup_pred2_count",
                        "belief_sup_correct0_count",
                        "belief_sup_correct1_count",
                        "belief_sup_correct2_count",
                        "belief_sup_has_gt0",
                        "belief_sup_has_gt1",
                        "belief_sup_has_gt2",
                        # Stage3b masked-supervision diagnostics
                        "belief_sup_possible_count",
                        "belief_sup_supervised_count",
                        "belief_sup_coverage",
                        "belief_sup_skipped_ratio",
                    ):
                        if k in train_stats:
                            logger.log_stat(f"train/{k}", float(train_stats[k]), t_env)
                    # Stage3b: expose coverage/skipped under action_sup namespace too
                    if "belief_sup_coverage" in train_stats:
                        logger.log_stat("train/action_sup_coverage", float(train_stats["belief_sup_coverage"]), t_env)
                    if "belief_sup_skipped_ratio" in train_stats:
                        logger.log_stat("train/action_sup_skipped_ratio", float(train_stats["belief_sup_skipped_ratio"]), t_env)

                    # sliding moving averages (smoothed curves)
                    if "loss_total" in train_stats:
                        ma_loss = _ma_update("loss_total", float(train_stats["loss_total"]))
                        logger.log_stat(f"train/loss_total_ma{moving_avg_window}", ma_loss, t_env)
                    if "belief_sup_acc" in train_stats:
                        ma_acc = _ma_update("belief_sup_acc", float(train_stats["belief_sup_acc"]))
                        logger.log_stat(f"train/belief_sup_acc_ma{moving_avg_window}", ma_acc, t_env)
                        logger.log_stat(f"train/action_sup_acc_masked_ma{moving_avg_window}", ma_acc, t_env)
                    # Optional: moving averages for supervised diagnostics (helps when noisy)
                    for k in (
                        "belief_sup_soft_available_frac",
                        "belief_sup_soft_used_frac",
                        "belief_sup_soft_p1_mean",
                        "belief_sup_grad_norm",
                        "belief_sup_effective_count",
                        "belief_sup_entropy",
                        "belief_sup_maxprob",
                        "belief_sup_logit_abs_mean",
                        "belief_sup_logit_std",
                        "belief_sup_p0_mean",
                        "belief_sup_p1_mean",
                        "belief_sup_p0_gt05_frac",
                        "belief_sup_hard_pred0_frac",
                        "belief_sup_delta01_mean",
                        "belief_sup_marginal_gap0",
                        "belief_sup_marginal_gap1",
                        "belief_sup_z_gate",
                        "belief_sup_ce_w0",
                        "belief_sup_ce_w1",
                        "belief_sup_ce_w2",
                        "belief_sup_ce_w3",
                        "belief_sup_ce_w4",
                        "belief_sup_pred0_frac",
                        "belief_sup_pred1_frac",
                        "belief_sup_pred2_frac",
                        "belief_sup_gt0_frac",
                        "belief_sup_gt1_frac",
                        "belief_sup_gt2_frac",
                        "belief_sup_recall0",
                        "belief_sup_recall1",
                        "belief_sup_recall2",
                        "belief_sup_precision0",
                        "belief_sup_precision1",
                        "belief_sup_precision2",
                        "belief_sup_gt0_count",
                        "belief_sup_gt1_count",
                        "belief_sup_gt2_count",
                        "belief_sup_pred0_count",
                        "belief_sup_pred1_count",
                        "belief_sup_pred2_count",
                        "belief_sup_correct0_count",
                        "belief_sup_correct1_count",
                        "belief_sup_correct2_count",
                        "belief_sup_has_gt0",
                        "belief_sup_has_gt1",
                        "belief_sup_has_gt2",
                    ):
                        if k in train_stats:
                            ma_k = _ma_update(k, float(train_stats[k]))
                            logger.log_stat(f"train/{k}_ma{moving_avg_window}", ma_k, t_env)
                    if "reward_mean" in train_stats:
                        ma_r = _ma_update("reward_mean", float(train_stats["reward_mean"]))
                        logger.log_stat(f"train/reward_mean_ma{moving_avg_window}", ma_r, t_env)
                    # Optional: surface how often we skipped non-finite values for MA (debugging stability)
                    try:
                        if int(ma_skipped.get("loss_total", 0)) > 0:
                            logger.log_stat("train/loss_total_ma_skipped_nonfinite", float(ma_skipped["loss_total"]), t_env)
                    except Exception:
                        pass

                    # z(t)->z(t+1) transition supervision (secondary users belief)
                    if "loss_z_transition" in train_stats:
                        logger.log_stat("train/loss_z_transition", float(train_stats["loss_z_transition"]), t_env)
                    # Stage3a diagnostics: ||z_pred - z_t|| and ||z_target - z_t||
                    if "z_pred_minus_z_t_l2" in train_stats:
                        logger.log_stat("train/z_pred_minus_z_t_l2", float(train_stats["z_pred_minus_z_t_l2"]), t_env)
                    if "z_target_minus_z_t_l2" in train_stats:
                        logger.log_stat("train/z_target_minus_z_t_l2", float(train_stats["z_target_minus_z_t_l2"]), t_env)
                    # Stage3a diagnostics: distribution + per-stage buckets + head norms
                    for k in (
                        "z_pred_entropy",
                        "z_pred_maxprob",
                        "z_pred_p0_mean",
                        "z_pred_p1_mean",
                        "z_pred_p2_mean",
                        "population_update_head_weight_l2",
                        "population_update_head_grad_l2",
                    ):
                        if k in train_stats:
                            logger.log_stat(f"train/{k}", float(train_stats[k]), t_env)
                    # per-stage buckets (avoid hardcoding stage count; just scan keys)
                    for k, v in train_stats.items():
                        if not isinstance(k, str):
                            continue
                        if k.startswith("z_pred_delta_l2_stage") or k.startswith("z_target_delta_l2_stage"):
                            logger.log_stat(f"train/{k}", float(v), t_env)
            except Exception as e:
                logger.warning(f"Failed to log fixed train metrics: {e}")
            
            # Log training statistics
            if episode % log_interval == 0:
                logger.info(f"Episode {episode}, t_env: {t_env}")
                for key, value in train_stats.items():
                    logger.info(f"  {key}: {value}")
                
                # Log to wandb if enabled
                if hasattr(config, 'wandb') and config.wandb.use_wandb:
                    log_to_wandb(train_stats, episode, 'train/')
            
            # Save model periodically (rank0 only in DDP)
            if is_rank0 and episode % save_model_interval == 0 and episode > 0:
                save_path = Path(config.logging.checkpoint_path) / f"episode_{episode}"
                save_path.mkdir(parents=True, exist_ok=True)
                learner.save_models(str(save_path))
                logger.info(f"Model saved at episode {episode}")
            
            # Test periodically (rank0 only in DDP)
            if is_rank0 and episode % test_interval == 0 and episode > 0:
                test_stats = run_test(runner, logger, config)
                logger.info(f"Test results at episode {episode}:")
                for key, value in test_stats.items():
                    logger.info(f"  {key}: {value}")
                
                # Log to wandb if enabled
                if hasattr(config, 'wandb') and config.wandb.use_wandb:
                    log_to_wandb(test_stats, episode, 'test/')

                # Also write the key test metrics to TensorBoard/metrics.jsonl
                try:
                    for k in (
                        "test_return_mean",
                        "core_action_type_acc",
                        "core_stance_acc",
                        "core_text_sim",
                        "secondary_z_kl",
                        # Stage3a encoder-only eval keys
                        "test_loss_z_transition",
                        "test_kl_target_zt",
                        "test_kl_target_zpred",
                        "test_z_pred_minus_z_t_l2",
                        "test_z_target_minus_z_t_l2",
                        "test_z_pred_entropy",
                        "test_z_pred_maxprob",
                        "test_z_pred_p0_mean",
                        "test_z_pred_p1_mean",
                        "test_z_pred_p2_mean",
                        "test_kl_target_zpred_nostage",
                        "test_kl_target_zpred_nogr",
                        "secondary_z_eval_steps",
                        # Stage3b masked eval (HF datasets)
                        "hf_eval_acc_masked",
                        "hf_eval_total_masked",
                        "hf_eval_skipped_unsup",
                        "hf_eval_coverage",
                        "hf_eval_skipped_ratio",
                        # Stage3b collapse / marginal sanity (K=5 action imitation)
                        "action_pred_entropy",
                        "action_pred_mode_frac",
                        "action_pred_kl_gt",
                        "action_unsup_pred_frac",
                        "action_unsup_gt_frac",
                        # per-class stance metrics on eval split
                        "stance_gt0_frac",
                        "stance_gt1_frac",
                        "stance_gt2_frac",
                        "stance_pred0_frac",
                        "stance_pred1_frac",
                        "stance_pred2_frac",
                        "stance_recall0",
                        "stance_recall1",
                        "stance_recall2",
                        "stance_precision0",
                        "stance_precision1",
                        "stance_precision2",
                        "stance_gt0_count",
                        "stance_gt1_count",
                        "stance_gt2_count",
                        "stance_pred0_count",
                        "stance_pred1_count",
                        "stance_pred2_count",
                        "stance_correct0_count",
                        "stance_correct1_count",
                        "stance_correct2_count",
                        "stance_has_gt0",
                        "stance_has_gt1",
                        "stance_has_gt2",
                        "stance_has_pred0",
                        "stance_has_pred1",
                        "stance_has_pred2",
                    ):
                        if k in test_stats:
                            logger.log_stat(f"test/{k}", float(test_stats[k]), t_env)
                    # Also log per-action marginals if present (Stage3b K=5)
                    try:
                        for k, v in test_stats.items():
                            if not isinstance(k, str):
                                continue
                            if k.startswith("action_gt") or k.startswith("action_pred") or k.startswith("action_recall") or k.startswith("action_precision"):
                                if isinstance(v, (int, float)) and (v == v):
                                    logger.log_stat(f"test/{k}", float(v), t_env)
                    except Exception:
                        pass
                    # Stage3a per-stage buckets (dynamic keys)
                    try:
                        for k, v in test_stats.items():
                            if not isinstance(k, str):
                                continue
                            if k.startswith("test_z_pred_delta_l2_stage") or k.startswith("test_z_target_delta_l2_stage") or k.startswith("test_stage_mask_sum_stage"):
                                logger.log_stat(f"test/{k}", float(v), t_env)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"Failed to log test metrics: {e}")
        
        episode += 1
        # For multi-step environments, count the actual number of env steps executed
        try:
            steps = int(getattr(runner, "t", 1))
            if steps <= 0:
                steps = int(getattr(config, "episode_length", 1))
        except Exception:
            steps = int(getattr(config, "episode_length", 1))
        t_env += steps
    
    # Final save
    final_dir = str(getattr(config, "final_save_dir", "") or "").strip()
    if final_dir:
        save_path = Path(final_dir)
    else:
        save_path = Path(config.logging.checkpoint_path) / "final"
    # Final save (rank0 only in DDP)
    if is_rank0:
        save_path.mkdir(parents=True, exist_ok=True)
        learner.save_models(str(save_path))
        logger.info("Training completed. Final model saved.")
    
    total_time = time.time() - begin_time
    logger.info(f"Total training time: {total_time:.2f} seconds")

def _safe_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """
    KL(p || q) with safety normalization and epsilon.
    p, q: 1D arrays
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    sp = float(p.sum())
    sq = float(q.sum())
    if sp <= 0:
        p = np.full_like(p, 1.0 / max(1, p.size))
    else:
        p = p / sp
    if sq <= 0:
        q = np.full_like(q, 1.0 / max(1, q.size))
    else:
        q = q / sq
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def run_test(runner, logger, config: SimpleNamespace):
    """Run test episodes (task-specific evaluation for social-media simulation)."""
    logger.info("Running test episodes...")
    
    test_episodes = int(getattr(config, "test_nepisode", 10))
    test_episodes = max(1, test_episodes)

    # Optional: evaluate on a different dataset split than training (useful for HF dataset env).
    # Example in YAML:
    #   eval_dataset_split: "test"
    eval_split = str(getattr(config, "eval_dataset_split", "") or "").strip() or None
    eval_runner = runner
    if eval_split and hasattr(config, "env") and str(getattr(config, "env", "")).strip() == "huggingface_dataset_env":
        try:
            # cache per-split runner to avoid reloading dataset every eval
            cache = getattr(runner, "_eval_runner_cache", None)
            if cache is None:
                cache = {}
                setattr(runner, "_eval_runner_cache", cache)
            if eval_split in cache:
                eval_runner = cache[eval_split]
            else:
                cfg_eval = copy.deepcopy(config)
                if hasattr(cfg_eval, "env_args") and hasattr(cfg_eval.env_args, "dataset_split"):
                    cfg_eval.env_args.dataset_split = eval_split
                # Optional: control evaluation sampling behavior independently from training.
                # This helps reduce eval noise on imbalanced splits.
                try:
                    eval_use_rs = getattr(cfg_eval, "eval_use_random_sampling", None)
                    if eval_use_rs is not None and hasattr(cfg_eval, "env_args"):
                        cfg_eval.env_args.use_random_sampling = bool(eval_use_rs)
                except Exception:
                    pass
                # Build a new runner/env for evaluation, but reuse the same MAC (model weights)
                eval_runner = r_REGISTRY[cfg_eval.runner](cfg_eval, logger)
                # reuse same scheme/groups/mac
                eval_runner.setup(getattr(runner, "scheme", None), getattr(runner, "groups", None), getattr(runner, "preprocess", None), getattr(runner, "mac", None))
                cache[eval_split] = eval_runner
            logger.info(f"[Eval] Using eval_dataset_split='{eval_split}' (train split='{getattr(getattr(config, 'env_args', SimpleNamespace()), 'dataset_split', None)}').")
        except Exception as e:
            logger.warning(f"Failed to create eval runner for split='{eval_split}', fallback to training runner: {e}")
            eval_runner = runner

    # ==========================
    # Stage3a encoder-only eval
    # ==========================
    # For z_transition dataset env, env_info doesn't naturally include z_pred.
    # We therefore compute metrics directly from EpisodeBatch + MAC/BeliefEncoder.
    try:
        if bool(getattr(config, "train_encoder_only", False)) and float(getattr(config, "z_transition_loss_weight", 0.0) or 0.0) > 0:
            import torch
            import torch.nn.functional as F

            test_episodes_eff = test_episodes
            # weighted sums (by mask count) for stable aggregation
            sum_loss = 0.0
            sum_kl_tgt_zt = 0.0
            sum_kl_tgt_zp = 0.0
            sum_pred_ent = 0.0
            sum_pred_max = 0.0
            sum_p0 = 0.0
            sum_p1 = 0.0
            sum_p2 = 0.0
            sum_dz_pred = 0.0
            sum_dz_tgt = 0.0
            sum_mask = 0.0

            # per-stage buckets
            stage_sum_dz_pred: Dict[int, float] = {}
            stage_sum_dz_tgt: Dict[int, float] = {}
            stage_sum_mask: Dict[int, float] = {}

            # ablations
            sum_kl_tgt_zp_nostage = 0.0
            sum_kl_tgt_zp_nogr = 0.0

            # helpers
            def _renorm(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
                p = torch.clamp(p, min=0.0)
                return p / torch.clamp(p.sum(dim=-1, keepdim=True), min=eps)

            def _kl_tgt_pred(tgt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
                # returns (N,)
                t = _renorm(tgt, eps=eps)
                q = _renorm(pred, eps=eps)
                return torch.sum(t * (torch.log(t + eps) - torch.log(q + eps)), dim=-1)

            for _ in range(test_episodes_eff):
                episode_batch = eval_runner.run(test_mode=True)
                if episode_batch is None:
                    continue

                mac = getattr(eval_runner, "mac", None)
                be = getattr(mac, "belief_encoder", None) if mac is not None else None
                if be is None or (not hasattr(be, "predict_next_population_belief")):
                    logger.warning("Stage3a eval: BeliefEncoder missing; skipping encoder-only eval.")
                    break

                # tensors: (bs, seq, K)
                z_t = episode_batch["z_t"][:, :-1]
                z_target = episode_batch["z_target"][:, :-1]
                z_mask = episode_batch["z_mask"][:, :-1]  # (bs, seq, 1)
                stage_t = episode_batch["stage_t"][:, :-1] if "stage_t" in episode_batch.scheme else None  # (bs, seq, 1)
                gr = episode_batch["group_representation"][:, :-1] if "group_representation" in episode_batch.scheme else None

                # flatten
                bs0, sl0, k0 = z_t.shape
                N = bs0 * sl0
                zt_f = z_t.reshape(N, k0).to(getattr(config, "device", None) or torch.device("cpu"))
                ztar_f = z_target.reshape(N, k0).to(zt_f.device)
                zm_f = z_mask.reshape(N).to(zt_f.device, dtype=torch.float32).clamp(min=0.0, max=1.0)
                denom = float(torch.clamp(zm_f.sum(), min=0.0).item())
                if denom <= 0.0:
                    continue

                gr_f = gr.reshape(N, -1).to(zt_f.device) if gr is not None else None
                st_f = stage_t.reshape(N, -1).to(zt_f.device) if stage_t is not None else None

                # forward
                with torch.no_grad():
                    lt = str(getattr(config, "z_transition_loss_type", "kl") or "kl").strip().lower()
                    if lt.startswith("dirichlet"):
                        if not hasattr(be, "predict_next_population_belief_alpha"):
                            raise RuntimeError("Stage3a eval: dirichlet requested but BeliefEncoder lacks predict_next_population_belief_alpha().")
                        if not hasattr(be, "compute_population_belief_loss_dirichlet_kl"):
                            raise RuntimeError("Stage3a eval: dirichlet requested but BeliefEncoder lacks compute_population_belief_loss_dirichlet_kl().")
                        alpha_pred = be.predict_next_population_belief_alpha(zt_f, group_repr=gr_f, stage_t=st_f)
                        zpred_f = be.population_belief_mean_from_alpha(alpha_pred)
                        # Prefer per-sample inferred alpha0_target if present in batch (EpisodeRunner passthrough)
                        a0_tgt = None
                        try:
                            if "z_alpha0_target" in episode_batch.scheme:
                                a0_tgt = episode_batch["z_alpha0_target"][:, :-1].reshape(N).to(zt_f.device)
                        except Exception:
                            a0_tgt = None
                        loss = be.compute_population_belief_loss_dirichlet_kl(
                            alpha_pred,
                            ztar_f,
                            zm_f,
                            alpha0_target=(a0_tgt if a0_tgt is not None else float(getattr(config, "dirichlet_alpha0_target", 10.0))),
                        )
                    else:
                        zpred_f = be.predict_next_population_belief(zt_f, group_repr=gr_f, stage_t=st_f, return_logits=False)
                        # loss
                        loss = be.compute_population_belief_loss(
                            zpred_f,
                            ztar_f,
                            zm_f,
                            loss_type=lt,
                        )

                    # KL baseline vs model
                    kl_tgt_zt = _kl_tgt_pred(ztar_f, zt_f)
                    kl_tgt_zp = _kl_tgt_pred(ztar_f, zpred_f)

                    # deltas
                    dz_pred = torch.norm((zpred_f - zt_f), p=2, dim=-1)
                    dz_tgt = torch.norm((ztar_f - zt_f), p=2, dim=-1)

                    # dist stats
                    if int(zpred_f.shape[-1]) == 3:
                        zp = _renorm(zpred_f)
                        ent = -torch.sum(zp * torch.log(zp + 1e-8), dim=-1)
                        mx = torch.max(zp, dim=-1)[0]
                        # masked weighted sums
                        sum_pred_ent += float((ent * zm_f).sum().item())
                        sum_pred_max += float((mx * zm_f).sum().item())
                        sum_p0 += float((zp[:, 0] * zm_f).sum().item())
                        sum_p1 += float((zp[:, 1] * zm_f).sum().item())
                        sum_p2 += float((zp[:, 2] * zm_f).sum().item())

                    # aggregate masked sums
                    sum_loss += float(loss.item()) * denom
                    sum_kl_tgt_zt += float((kl_tgt_zt * zm_f).sum().item())
                    sum_kl_tgt_zp += float((kl_tgt_zp * zm_f).sum().item())
                    sum_dz_pred += float((dz_pred * zm_f).sum().item())
                    sum_dz_tgt += float((dz_tgt * zm_f).sum().item())
                    sum_mask += denom

                    # stage buckets
                    if st_f is not None:
                        st1 = st_f.reshape(-1).to(dtype=torch.long)
                        for s in torch.unique(st1).tolist():
                            try:
                                si = int(s)
                            except Exception:
                                continue
                            sel = (st1 == si)
                            if not bool(sel.any().item()):
                                continue
                            m_s = zm_f[sel]
                            d_s = float(torch.clamp(m_s.sum(), min=0.0).item())
                            if d_s <= 0:
                                continue
                            stage_sum_mask[si] = float(stage_sum_mask.get(si, 0.0) + d_s)
                            stage_sum_dz_pred[si] = float(stage_sum_dz_pred.get(si, 0.0) + float((dz_pred[sel] * m_s).sum().item()))
                            stage_sum_dz_tgt[si] = float(stage_sum_dz_tgt.get(si, 0.0) + float((dz_tgt[sel] * m_s).sum().item()))

                    # stage ablation (strongly suggested): set all stage to 0
                    try:
                        if st_f is not None:
                            st0 = torch.zeros_like(st_f)
                            zpred_nostage = be.predict_next_population_belief(zt_f, group_repr=gr_f, stage_t=st0, return_logits=False)
                            kl_nostage = _kl_tgt_pred(ztar_f, zpred_nostage)
                            sum_kl_tgt_zp_nostage += float((kl_nostage * zm_f).sum().item())
                    except Exception:
                        pass

                    # group_repr sensitivity: zero-out group_repr
                    try:
                        if gr_f is not None:
                            gr0 = torch.zeros_like(gr_f)
                            zpred_nogr = be.predict_next_population_belief(zt_f, group_repr=gr0, stage_t=st_f, return_logits=False)
                            kl_nogr = _kl_tgt_pred(ztar_f, zpred_nogr)
                            sum_kl_tgt_zp_nogr += float((kl_nogr * zm_f).sum().item())
                    except Exception:
                        pass

            if sum_mask <= 0:
                return {"secondary_z_eval_steps": 0}

            out: Dict[str, Any] = {}
            out["test_loss_z_transition"] = float(sum_loss / sum_mask)
            out["test_kl_target_zt"] = float(sum_kl_tgt_zt / sum_mask)          # identity baseline
            out["test_kl_target_zpred"] = float(sum_kl_tgt_zp / sum_mask)       # model
            out["test_z_pred_minus_z_t_l2"] = float(sum_dz_pred / sum_mask)
            out["test_z_target_minus_z_t_l2"] = float(sum_dz_tgt / sum_mask)
            out["test_z_pred_entropy"] = float(sum_pred_ent / sum_mask) if sum_pred_ent > 0 else 0.0
            out["test_z_pred_maxprob"] = float(sum_pred_max / sum_mask) if sum_pred_max > 0 else 0.0
            out["test_z_pred_p0_mean"] = float(sum_p0 / sum_mask) if sum_p0 > 0 else 0.0
            out["test_z_pred_p1_mean"] = float(sum_p1 / sum_mask) if sum_p1 > 0 else 0.0
            out["test_z_pred_p2_mean"] = float(sum_p2 / sum_mask) if sum_p2 > 0 else 0.0
            # ablations
            if sum_kl_tgt_zp_nostage > 0:
                out["test_kl_target_zpred_nostage"] = float(sum_kl_tgt_zp_nostage / sum_mask)
            if sum_kl_tgt_zp_nogr > 0:
                out["test_kl_target_zpred_nogr"] = float(sum_kl_tgt_zp_nogr / sum_mask)

            # per-stage buckets (mean)
            for s, m in stage_sum_mask.items():
                if m <= 0:
                    continue
                out[f"test_z_pred_delta_l2_stage{s}"] = float(stage_sum_dz_pred.get(s, 0.0) / m)
                out[f"test_z_target_delta_l2_stage{s}"] = float(stage_sum_dz_tgt.get(s, 0.0) / m)
                out[f"test_stage_mask_sum_stage{s}"] = float(m)

            out["secondary_z_eval_steps"] = int(sum_mask)
            return out
    except Exception as e:
        logger.warning(f"Stage3a encoder-only eval failed; falling back to legacy run_test: {e}")

    returns: List[float] = []
    # core-user metrics
    core_action_type_acc: List[float] = []
    core_stance_acc: List[float] = []
    core_text_sim: List[float] = []
    core_valid_steps: List[int] = []
    # secondary-user belief metrics
    z_kl_list: List[float] = []
    z_eval_steps: int = 0

    # HF dataset eval (per-class stats on eval split).
    # - Stage1/2 stance classification: K=3
    # - Stage3b action imitation: K=5
    try:
        hf_k = int(getattr(getattr(config, "env_args", SimpleNamespace()), "n_actions", getattr(config, "n_actions", 3)))
    except Exception:
        hf_k = 3
    # Stage3b option: binary(0/1) imitation prior.
    # Keep model/env head as K=5 for Stage4, but report eval metrics with K=2 since we only care about labels {0,1}.
    try:
        if bool(getattr(config, "train_action_imitation", False)) and bool(getattr(config, "action_imitation_binary_01", False)):
            hf_k = 2
    except Exception:
        pass
    hf_k = max(1, hf_k)
    hf_gt_counts = [0 for _ in range(hf_k)]
    hf_pred_counts = [0 for _ in range(hf_k)]
    hf_correct_counts = [0 for _ in range(hf_k)]
    
    def _is_boxed_int(s: Any) -> bool:
        """Return True if string contains a \\boxed{<int>} (allows whitespace)."""
        try:
            import re
            if not isinstance(s, str):
                return False
            return re.search(r"\\boxed\{\s*[-+]?\d+\s*\}", s) is not None
        except Exception:
            return False

    def _parse_boxed_int(s: Any) -> Optional[int]:
        try:
            import re
            if not isinstance(s, str):
                return None
            m = re.search(r"\\boxed\{\s*([-+]?\d+)\s*\}", s)
            if not m:
                m = re.search(r"boxed\{\s*([-+]?\d+)\s*\}", s)
            if not m:
                return None
            return int(m.group(1))
        except Exception:
            return None

    # Aggregate masked-eval counters across ALL test episodes (stable + interpretable).
    hf_eval_total_all = 0
    hf_eval_correct_all = 0
    hf_eval_skipped_unsup_all = 0

    for _ in range(test_episodes):
        episode_batch = eval_runner.run(test_mode=True)
        if episode_batch is not None:
            # === return ===
            episode_return = float(episode_batch["reward"].sum().item())
            returns.append(episode_return)

            # === task-specific evaluation from env infos ===
            env_infos = getattr(eval_runner, "last_env_infos", None)
            if not isinstance(env_infos, list) or not env_infos:
                continue

            # Decide evaluation schema:
            # - Legacy social-sim envs may emit gt_t/gt_available/reward_action_type/reward_text.
            # - HuggingFaceDatasetEnv emits is_correct/reward_ts/reward_al/reward_cc (+ optional z_*).
            use_legacy_schema = False
            # Optional: Stage3b adjustment — only count post/retweet as supervised labels during evaluation.
            sup_ids = None
            try:
                if (not use_legacy_schema) and bool(getattr(config, "train_action_imitation", False)):
                    only_ids = getattr(config, "action_imitation_supervised_action_ids", None)
                    if isinstance(only_ids, (list, tuple)) and len(only_ids) > 0:
                        sup_ids = set(int(x) for x in only_ids)
            except Exception:
                sup_ids = None

            # Per-episode counters (will be accumulated into *_all)
            hf_eval_total = 0
            hf_eval_correct = 0
            hf_eval_skipped_unsup = 0

            for info in env_infos:
                if not isinstance(info, dict):
                    continue
                if ("gt_t" in info) or ("gt_available" in info) or ("reward_action_type" in info) or ("reward_text" in info):
                    use_legacy_schema = True
                    break

            # core: use per-step signals that env already provides
            # - reward_action_type: 1/0
            # - reward_ts: stance match 1/0
            # - reward_text: similarity 0..1
            # Only count steps that have next-step supervision (gt_t < n_stages).
            valid = 0
            sum_at = 0.0
            sum_st = 0.0
            sum_txt = 0.0

            for info in env_infos:
                if not isinstance(info, dict):
                    continue

                if use_legacy_schema:
                    try:
                        gt_t = int(info.get("gt_t", -1))
                    except Exception:
                        gt_t = -1
                    # A: only evaluate steps with usable supervision
                    try:
                        gt_av = int(info.get("gt_available", 1))
                    except Exception:
                        gt_av = 1
                    try:
                        # Use runtime env n_stages if available (curriculum may change it)
                        n_stages = int(getattr(getattr(runner, "env", None), "n_stages", getattr(getattr(config, "env_args", SimpleNamespace()), "n_stages", 13)))
                    except Exception:
                        n_stages = 13
                    if gt_av <= 0:
                        continue
                    if gt_t < 0 or gt_t >= n_stages:
                        continue

                    valid += 1
                    sum_at += float(info.get("reward_action_type", 0.0))
                    sum_st += float(info.get("reward_ts", 0.0))
                    sum_txt += float(info.get("reward_text", 0.0))
                else:
                    # HuggingFaceDatasetEnv schema: treat each step as valid.
                    # If sup_ids is set, only evaluate those labels; others are treated as latent (skipped).
                    gt = _parse_boxed_int(info.get("ground_truth_answer", ""))
                    if gt is None:
                        gt = _parse_boxed_int(info.get("ground_truth", ""))
                    if sup_ids is not None and gt is not None and int(gt) not in sup_ids:
                        hf_eval_skipped_unsup += 1
                        continue

                    valid += 1
                    # IMPORTANT: For HF datasets, evaluate boxed-id accuracy, not just "is boxed".
                    pr = _parse_boxed_int(info.get("llm_answer", ""))
                    sum_at += 1.0 if (gt is not None and pr is not None and int(gt) == int(pr)) else 0.0
                    # keep legacy "stance" slot as is_correct (HF env sets is_correct based on boxed-id equality)
                    sum_st += 1.0 if bool(info.get("is_correct", False)) else 0.0
                    # no explicit reward_text; use reward_al as a proxy (often 0 when al_weight=0)
                    try:
                        sum_txt += float(info.get("reward_al", 0.0))
                    except Exception:
                        sum_txt += 0.0

                    # Also compute per-class stats on eval split (boxed ids)
                    try:
                        if gt is not None and 0 <= int(gt) < hf_k:
                            hf_gt_counts[int(gt)] += 1
                        if pr is not None and 0 <= int(pr) < hf_k:
                            hf_pred_counts[int(pr)] += 1
                        if gt is not None and pr is not None and 0 <= int(gt) < hf_k and int(gt) == int(pr):
                            hf_correct_counts[int(gt)] += 1
                    except Exception:
                        pass

                    # additional masked-eval stats for Stage3b
                    try:
                        if gt is not None:
                            hf_eval_total += 1
                            if pr is not None and int(gt) == int(pr):
                                hf_eval_correct += 1
                    except Exception:
                        pass

                # secondary: stage boundary evaluation (z_mask == 1)
                try:
                    z_mask = float(info.get("z_mask", 0.0))
                except Exception:
                    z_mask = 0.0
                if z_mask > 0.5 and ("z_pred" in info) and ("z_target" in info):
                    z_eval_steps += 1
                    z_kl_list.append(_safe_kl(np.array(info["z_target"]), np.array(info["z_pred"])))

            if valid > 0:
                core_valid_steps.append(valid)
                core_action_type_acc.append(sum_at / valid)
                core_stance_acc.append(sum_st / valid)
                core_text_sim.append(sum_txt / valid)
            # accumulate masked-eval counters across episodes
            try:
                hf_eval_total_all += int(hf_eval_total)
                hf_eval_correct_all += int(hf_eval_correct)
                hf_eval_skipped_unsup_all += int(hf_eval_skipped_unsup)
            except Exception:
                pass
    
    # Calculate averages
    avg_return = float(np.mean(returns)) if returns else 0.0
    # Keep legacy "success" definition, but based on return.
    success_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in returns])) if returns else 0.0

    core_at = float(np.mean(core_action_type_acc)) if core_action_type_acc else 0.0
    core_st = float(np.mean(core_stance_acc)) if core_stance_acc else 0.0
    core_txt = float(np.mean(core_text_sim)) if core_text_sim else 0.0
    avg_core_steps = float(np.mean(core_valid_steps)) if core_valid_steps else 0.0

    z_kl = float(np.mean(z_kl_list)) if z_kl_list else 0.0
    
    # Per-class HF metrics on eval split (best-effort)
    hf_total_gt = float(sum(hf_gt_counts))
    hf_total_pred = float(sum(hf_pred_counts))
    hf_gt_frac = [float(c) / hf_total_gt if hf_total_gt > 0 else float("nan") for c in hf_gt_counts]
    hf_pred_frac = [float(c) / hf_total_pred if hf_total_pred > 0 else float("nan") for c in hf_pred_counts]
    hf_has_gt = [1.0 if hf_gt_counts[i] > 0 else 0.0 for i in range(hf_k)]
    hf_has_pred = [1.0 if hf_pred_counts[i] > 0 else 0.0 for i in range(hf_k)]
    hf_recall = [
        (float(hf_correct_counts[i]) / float(hf_gt_counts[i])) if hf_gt_counts[i] > 0 else 0.0
        for i in range(hf_k)
    ]
    hf_precision = [
        (float(hf_correct_counts[i]) / float(hf_pred_counts[i])) if hf_pred_counts[i] > 0 else 0.0
        for i in range(hf_k)
    ]

    out = {
        "test_return_mean": avg_return,
        "test_success_rate": success_rate,
        "test_episodes": len(returns),
        # Note: string metadata is printed in console logs; TB only logs numeric stats
        "eval_dataset_split": eval_split or str(getattr(getattr(config, "env_args", SimpleNamespace()), "dataset_split", "train")),
        # core user evaluation (next-step prediction)
        "core_action_type_acc": core_at,
        "core_stance_acc": core_st,
        "core_text_sim": core_txt,
        "core_eval_steps_mean": avg_core_steps,
        # secondary belief evaluation
        "secondary_z_kl": z_kl,
        "secondary_z_eval_steps": int(z_eval_steps),
    }
    # Stage3b masked-eval helpers (only meaningful for HF schema + action_imitation_supervised_action_ids)
    try:
        if hf_eval_total_all > 0:
            out["hf_eval_acc_masked"] = float(hf_eval_correct_all / max(1, hf_eval_total_all))
        out["hf_eval_total_masked"] = int(hf_eval_total_all)
        out["hf_eval_skipped_unsup"] = int(hf_eval_skipped_unsup_all)
        denom = float(hf_eval_total_all + hf_eval_skipped_unsup_all)
        out["hf_eval_coverage"] = float(hf_eval_total_all / denom) if denom > 0 else 0.0
        out["hf_eval_skipped_ratio"] = float(hf_eval_skipped_unsup_all / denom) if denom > 0 else 0.0
    except Exception:
        pass
    # Per-class keys:
    # - K=3: keep backward-compatible stance_* keys (Stage1/2).
    # - K!=3: emit action_* keys (Stage3b action imitation).
    if hf_k == 3:
        out.update(
            {
                "stance_gt0_frac": float(hf_gt_frac[0]) if len(hf_gt_frac) > 0 else float("nan"),
                "stance_gt1_frac": float(hf_gt_frac[1]) if len(hf_gt_frac) > 1 else float("nan"),
                "stance_gt2_frac": float(hf_gt_frac[2]) if len(hf_gt_frac) > 2 else float("nan"),
                "stance_pred0_frac": float(hf_pred_frac[0]) if len(hf_pred_frac) > 0 else float("nan"),
                "stance_pred1_frac": float(hf_pred_frac[1]) if len(hf_pred_frac) > 1 else float("nan"),
                "stance_pred2_frac": float(hf_pred_frac[2]) if len(hf_pred_frac) > 2 else float("nan"),
                "stance_recall0": float(hf_recall[0]) if len(hf_recall) > 0 else float("nan"),
                "stance_recall1": float(hf_recall[1]) if len(hf_recall) > 1 else float("nan"),
                "stance_recall2": float(hf_recall[2]) if len(hf_recall) > 2 else float("nan"),
                "stance_precision0": float(hf_precision[0]) if len(hf_precision) > 0 else float("nan"),
                "stance_precision1": float(hf_precision[1]) if len(hf_precision) > 1 else float("nan"),
                "stance_precision2": float(hf_precision[2]) if len(hf_precision) > 2 else float("nan"),
                "stance_gt0_count": float(hf_gt_counts[0]) if len(hf_gt_counts) > 0 else 0.0,
                "stance_gt1_count": float(hf_gt_counts[1]) if len(hf_gt_counts) > 1 else 0.0,
                "stance_gt2_count": float(hf_gt_counts[2]) if len(hf_gt_counts) > 2 else 0.0,
                "stance_pred0_count": float(hf_pred_counts[0]) if len(hf_pred_counts) > 0 else 0.0,
                "stance_pred1_count": float(hf_pred_counts[1]) if len(hf_pred_counts) > 1 else 0.0,
                "stance_pred2_count": float(hf_pred_counts[2]) if len(hf_pred_counts) > 2 else 0.0,
                "stance_correct0_count": float(hf_correct_counts[0]) if len(hf_correct_counts) > 0 else 0.0,
                "stance_correct1_count": float(hf_correct_counts[1]) if len(hf_correct_counts) > 1 else 0.0,
                "stance_correct2_count": float(hf_correct_counts[2]) if len(hf_correct_counts) > 2 else 0.0,
                "stance_has_gt0": float(hf_has_gt[0]) if len(hf_has_gt) > 0 else 0.0,
                "stance_has_gt1": float(hf_has_gt[1]) if len(hf_has_gt) > 1 else 0.0,
                "stance_has_gt2": float(hf_has_gt[2]) if len(hf_has_gt) > 2 else 0.0,
                "stance_has_pred0": float(hf_has_pred[0]) if len(hf_has_pred) > 0 else 0.0,
                "stance_has_pred1": float(hf_has_pred[1]) if len(hf_has_pred) > 1 else 0.0,
                "stance_has_pred2": float(hf_has_pred[2]) if len(hf_has_pred) > 2 else 0.0,
            }
        )
    else:
        for i in range(hf_k):
            out[f"action_gt{i}_frac"] = float(hf_gt_frac[i]) if i < len(hf_gt_frac) else float("nan")
            out[f"action_pred{i}_frac"] = float(hf_pred_frac[i]) if i < len(hf_pred_frac) else float("nan")
            out[f"action_recall{i}"] = float(hf_recall[i]) if i < len(hf_recall) else float("nan")
            out[f"action_precision{i}"] = float(hf_precision[i]) if i < len(hf_precision) else float("nan")
            out[f"action_gt{i}_count"] = float(hf_gt_counts[i]) if i < len(hf_gt_counts) else 0.0
            out[f"action_pred{i}_count"] = float(hf_pred_counts[i]) if i < len(hf_pred_counts) else 0.0
            out[f"action_correct{i}_count"] = float(hf_correct_counts[i]) if i < len(hf_correct_counts) else 0.0
            out[f"action_has_gt{i}"] = float(hf_has_gt[i]) if i < len(hf_has_gt) else 0.0
            out[f"action_has_pred{i}"] = float(hf_has_pred[i]) if i < len(hf_has_pred) else 0.0

        # === Stage3b sanity metrics: detect collapse on UNSUP actions (when partial supervision is enabled) ===
        try:
            import math

            eps = 1e-8
            # entropy of predicted marginal distribution over actions
            p = [float(x) for x in hf_pred_frac]
            if all((x == x) for x in p) and len(p) > 0:
                pp = [max(eps, x) for x in p]
                s = float(sum(pp))
                pp = [x / s for x in pp]
                out["action_pred_entropy"] = float(-sum(x * math.log(x + eps) for x in pp))
                out["action_pred_mode_frac"] = float(max(pp))

            # compare pred marginal to gt marginal (KL)
            q = [float(x) for x in hf_gt_frac]
            if all((x == x) for x in p) and all((x == x) for x in q) and len(p) == len(q) and len(p) > 0:
                pp = [max(eps, x) for x in p]
                qq = [max(eps, x) for x in q]
                sp = float(sum(pp))
                sq = float(sum(qq))
                pp = [x / sp for x in pp]
                qq = [x / sq for x in qq]
                out["action_pred_kl_gt"] = float(sum(pp[i] * (math.log(pp[i] + eps) - math.log(qq[i] + eps)) for i in range(len(pp))))

            # unsupervised mass (only meaningful when action_imitation_supervised_action_ids is set)
            sup_ids_cfg = None
            try:
                only_ids = getattr(config, "action_imitation_supervised_action_ids", None)
                if isinstance(only_ids, (list, tuple)) and len(only_ids) > 0:
                    sup_ids_cfg = set(int(x) for x in only_ids)
            except Exception:
                sup_ids_cfg = None
            if isinstance(sup_ids_cfg, set) and len(sup_ids_cfg) > 0:
                unsup = [i for i in range(hf_k) if i not in sup_ids_cfg]
                out["action_unsup_pred_frac"] = float(sum(float(hf_pred_frac[i]) for i in unsup if i < len(hf_pred_frac)))
                out["action_unsup_gt_frac"] = float(sum(float(hf_gt_frac[i]) for i in unsup if i < len(hf_gt_frac)))
        except Exception:
            pass

    return out

def setup_wandb(config: SimpleNamespace, logger):
    """Initialize wandb for experiment tracking"""
    if hasattr(config, 'wandb') and config.wandb.use_wandb:
        if wandb is None:
            logger.warning("wandb is enabled in config but package 'wandb' is not installed. Disabling wandb logging.")
            config.wandb.use_wandb = False
            return
        logger.info("Initializing wandb...")
        
        wandb.init(
            project=getattr(config.wandb, 'project', 'ECON-Framework'),
            entity=getattr(config.wandb, 'entity', None),
            tags=getattr(config.wandb, 'tags', None),
            config=dict(config.__dict__) if hasattr(config, '__dict__') else None,
            name=getattr(config.logging, 'experiment_name', 'econ_experiment')
        )
        
        logger.info("wandb initialized successfully")
        return True
    return False

def log_to_wandb(data: Dict, step: int, prefix: str = ''):
    """Log data to wandb"""
    if wandb is None:
        return
    if wandb.run is not None:
        wandb.log({f"{prefix}{k}": v for k, v in data.items()}, step=step)

def main():
    """Main training function"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Load configuration
        config = load_config(args.config)
        
        # Update configuration with command line arguments
        config = update_config_with_args(config, args)
        
        # Setup experiment
        runner, mac, learner, logger, device = setup_experiment(config)
        
        # Setup wandb if enabled
        setup_wandb(config, logger)
        
        # Run training
        run_training(config, runner, learner, logger, device)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up distributed process group (if any)
        try:
            if _HAS_DIST and dist is not None and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:
            pass
        # Clean up wandb
        if wandb is not None and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 
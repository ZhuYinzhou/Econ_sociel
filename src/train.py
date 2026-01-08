#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List
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
    logger = get_logger(log_dir=log_dir, experiment_name=exp_name, use_tensorboard=use_tb)
    logger.info("Setting up experiment environment...")
    
    # Set random seed
    seed = config.system.seed if hasattr(config, 'system') and hasattr(config.system, 'seed') else 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set device
    use_cuda = hasattr(config, 'system') and hasattr(config.system, 'use_cuda') and config.system.use_cuda and torch.cuda.is_available()
    device_num = config.system.device_num if hasattr(config, 'system') and hasattr(config.system, 'device_num') else 0
    device = torch.device(f"cuda:{device_num}" if use_cuda else "cpu")
    
    # Add device to config object for runner access
    config.device = device
    
    if use_cuda:
        torch.cuda.set_device(device_num)
    
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
    }

    def _ma_update(name: str, value: float) -> float:
        """Update moving average buffer and return current mean."""
        buf = ma_buffers.get(name)
        if buf is None:
            buf = deque(maxlen=moving_avg_window)
            ma_buffers[name] = buf
        buf.append(float(value))
        return float(np.mean(buf)) if len(buf) > 0 else float(value)
    
    # Get early stopping thresholds from configuration
    early_stopping = getattr(config, 'early_stopping', SimpleNamespace())
    commitment_threshold = getattr(early_stopping, 'commitment_threshold', 0.01)
    reward_threshold = getattr(early_stopping, 'reward_threshold', 0.7)
    loss_threshold = getattr(early_stopping, 'loss_threshold', 0.0001)
    patience = getattr(early_stopping, 'patience', 5)

    # Training loop
    episode = 0
    t_env = 0

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

                    # sliding moving averages (smoothed curves)
                    if "loss_total" in train_stats:
                        ma_loss = _ma_update("loss_total", float(train_stats["loss_total"]))
                        logger.log_stat(f"train/loss_total_ma{moving_avg_window}", ma_loss, t_env)
                    if "belief_sup_acc" in train_stats:
                        ma_acc = _ma_update("belief_sup_acc", float(train_stats["belief_sup_acc"]))
                        logger.log_stat(f"train/belief_sup_acc_ma{moving_avg_window}", ma_acc, t_env)
                    if "reward_mean" in train_stats:
                        ma_r = _ma_update("reward_mean", float(train_stats["reward_mean"]))
                        logger.log_stat(f"train/reward_mean_ma{moving_avg_window}", ma_r, t_env)

                    # z(t)->z(t+1) transition supervision (secondary users belief)
                    if "loss_z_transition" in train_stats:
                        logger.log_stat("train/loss_z_transition", float(train_stats["loss_z_transition"]), t_env)
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
            
            # Save model periodically
            if episode % save_model_interval == 0 and episode > 0:
                save_path = Path(config.logging.checkpoint_path) / f"episode_{episode}"
                save_path.mkdir(parents=True, exist_ok=True)
                learner.save_models(str(save_path))
                logger.info(f"Model saved at episode {episode}")
            
            # Test periodically
            if episode % test_interval == 0 and episode > 0:
                test_stats = run_test(runner, logger, config)
                logger.info(f"Test results at episode {episode}:")
                for key, value in test_stats.items():
                    logger.info(f"  {key}: {value}")
                
                # Log to wandb if enabled
                if hasattr(config, 'wandb') and config.wandb.use_wandb:
                    log_to_wandb(test_stats, episode, 'test/')

                # Also write the key test metrics to TensorBoard/metrics.jsonl
                try:
                    for k in ("test_return_mean", "core_action_type_acc", "core_stance_acc", "core_text_sim", "secondary_z_kl"):
                        if k in test_stats:
                            logger.log_stat(f"test/{k}", float(test_stats[k]), t_env)
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

    returns: List[float] = []
    # core-user metrics
    core_action_type_acc: List[float] = []
    core_stance_acc: List[float] = []
    core_text_sim: List[float] = []
    core_valid_steps: List[int] = []
    # secondary-user belief metrics
    z_kl_list: List[float] = []
    z_eval_steps: int = 0
    
    def _is_boxed_int(s: Any) -> bool:
        """Return True if string contains a \\boxed{<int>} (allows whitespace)."""
        try:
            import re
            if not isinstance(s, str):
                return False
            return re.search(r"\\boxed\{\s*[-+]?\d+\s*\}", s) is not None
        except Exception:
            return False

    for _ in range(test_episodes):
        episode_batch = runner.run(test_mode=True)
        if episode_batch is not None:
            # === return ===
            episode_return = float(episode_batch["reward"].sum().item())
            returns.append(episode_return)

            # === task-specific evaluation from env infos ===
            env_infos = getattr(runner, "last_env_infos", None)
            if not isinstance(env_infos, list) or not env_infos:
                continue

            # Decide evaluation schema:
            # - Legacy social-sim envs may emit gt_t/gt_available/reward_action_type/reward_text.
            # - HuggingFaceDatasetEnv emits is_correct/reward_ts/reward_al/reward_cc (+ optional z_*).
            use_legacy_schema = False
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
                    valid += 1
                    # "action type" here means output format is usable (\\boxed{<id>}).
                    sum_at += 1.0 if _is_boxed_int(info.get("llm_answer", "")) else 0.0
                    # stance correctness: prefer reward_ts; fallback to is_correct
                    try:
                        sum_st += float(info.get("reward_ts", 1.0 if info.get("is_correct", False) else 0.0))
                    except Exception:
                        sum_st += 0.0
                    # no explicit reward_text; use reward_al as a proxy (often 0 when al_weight=0)
                    try:
                        sum_txt += float(info.get("reward_al", 0.0))
                    except Exception:
                        sum_txt += 0.0

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
    
    # Calculate averages
    avg_return = float(np.mean(returns)) if returns else 0.0
    # Keep legacy "success" definition, but based on return.
    success_rate = float(np.mean([1.0 if r > 0 else 0.0 for r in returns])) if returns else 0.0

    core_at = float(np.mean(core_action_type_acc)) if core_action_type_acc else 0.0
    core_st = float(np.mean(core_stance_acc)) if core_stance_acc else 0.0
    core_txt = float(np.mean(core_text_sim)) if core_text_sim else 0.0
    avg_core_steps = float(np.mean(core_valid_steps)) if core_valid_steps else 0.0

    z_kl = float(np.mean(z_kl_list)) if z_kl_list else 0.0
    
    return {
        "test_return_mean": avg_return,
        "test_success_rate": success_rate,
        "test_episodes": len(returns),
        # core user evaluation (next-step prediction)
        "core_action_type_acc": core_at,
        "core_stance_acc": core_st,
        "core_text_sim": core_txt,
        "core_eval_steps_mean": avg_core_steps,
        # secondary belief evaluation
        "secondary_z_kl": z_kl,
        "secondary_z_eval_steps": int(z_eval_steps),
    }

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
        # Clean up wandb
        if wandb is not None and wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 
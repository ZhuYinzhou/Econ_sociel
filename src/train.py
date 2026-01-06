#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List
import wandb

# Import necessary components
from utils.logging import get_logger
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY

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
    parser.add_argument('--api_key', type=str, help='Together API key')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--env', type=str, help='Environment name')
    
    # wandb related parameters
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb for experiment logging')
    parser.add_argument('--wandb_project', type=str, default='ECON-Framework', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, help='wandb username or organization name')
    parser.add_argument('--wandb_tags', type=str, help='wandb tags, separated by commas')
    
    return parser.parse_args()

def load_config(config_path: str) -> SimpleNamespace:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
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
    
    if args.api_key:
        # Set API key in both places for compatibility
        config.together_api_key = args.api_key  # For direct access as args.together_api_key
        if hasattr(config, 'llm'):
            config.llm.together_api_key = args.api_key
    
    if args.seed:
        config.system.seed = args.seed
    
    if args.env:
        config.env = args.env
    
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
    logger = get_logger()
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
    
    # Get early stopping thresholds from configuration
    early_stopping = getattr(config, 'early_stopping', SimpleNamespace())
    commitment_threshold = getattr(early_stopping, 'commitment_threshold', 0.01)
    reward_threshold = getattr(early_stopping, 'reward_threshold', 0.7)
    loss_threshold = getattr(early_stopping, 'loss_threshold', 0.0001)
    patience = getattr(early_stopping, 'patience', 5)

    # Training loop
    episode = 0
    t_env = 0
    
    while t_env < t_max:
        # Run episode
        episode_batch = runner.run(test_mode=False)
        
        # Train learner
        if episode_batch is not None:
            train_stats = learner.train(episode_batch, t_env, episode)

            # === High-signal fixed metrics (core reward + secondary belief loss) ===
            # These are written to metrics.jsonl and TensorBoard regardless of console verbosity.
            try:
                if isinstance(train_stats, dict):
                    if "reward_mean" in train_stats:
                        logger.log_stat("train/reward_mean", float(train_stats["reward_mean"]), t_env)
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
                try:
                    gt_t = int(info.get("gt_t", -1))
                except Exception:
                    gt_t = -1
                try:
                    n_stages = int(getattr(getattr(config, "env_args", SimpleNamespace()), "n_stages", 13))
                except Exception:
                    n_stages = 13
                if gt_t < 0 or gt_t >= n_stages:
                    continue

                valid += 1
                sum_at += float(info.get("reward_action_type", 0.0))
                sum_st += float(info.get("reward_ts", 0.0))
                sum_txt += float(info.get("reward_text", 0.0))

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
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 
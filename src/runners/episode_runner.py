import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from utils.logging import Logger
from dataclasses import dataclass
from loguru import logger

@dataclass
class EpisodeMetrics:
    """Container for episode-specific metrics."""
    llm_responses: List[str] = None
    strategies: List[str] = None
    commitments: List[str] = None
    rewards: List[float] = None
    belief_states: List[torch.Tensor] = None
    rewards_al: List[float] = None
    rewards_ts: List[float] = None
    rewards_cc: List[float] = None
    
    def __post_init__(self):
        """Initialize empty lists."""
        self.llm_responses = []
        self.strategies = []
        self.commitments = []
        self.rewards = []
        self.belief_states = []
        self.rewards_al = []
        self.rewards_ts = []
        self.rewards_cc = []
    
    def add_step_data(self, extra_info: Dict[str, Any], 
                      reward: float, reward_al: float, reward_ts: float, reward_cc: float):
        """Add data from a single step."""
        if 'llm_responses' in extra_info:
            self.llm_responses.append(extra_info['llm_responses'])
        if 'strategy' in extra_info:
            self.strategies.append(extra_info['strategy'])
        if 'commitment' in extra_info:
            if isinstance(extra_info['commitment'], str):
                 self.commitments.append(extra_info['commitment'])
            elif isinstance(extra_info['commitment'], list):
                 self.commitments.extend(extra_info['commitment'])

        if 'belief_states' in extra_info:
            self.belief_states.append(extra_info['belief_states'])
        
        self.rewards.append(reward)
        self.rewards_al.append(reward_al)
        self.rewards_ts.append(reward_ts)
        self.rewards_cc.append(reward_cc)

class EpisodeRunner:
    """
    Episode runner for LLM-based MARL training.
    
    Handles episode execution, data collection, and coordination between
    environment interactions, LLM responses, and data storage.
    """
    def __init__(self, args: Any, logger: Logger):
        """
        Initialize episode runner.
        
        Args:
            args: Configuration arguments
            logger: Logger instance
        """
        self.args = args
        self.logger = logger
        
        # Environment and batch information
        self.env = None
        self.env_info = None
        self.batch = None
        
        # Training state
        self.t = 0  # Current timestep within episode
        self.t_env = 0  # Total timesteps across all episodes
        self.t_episodes = 0  # 添加episode计数器
        
        # Testing state
        self.test_returns = []
        self.train_returns = []
        self.last_test_t = 0
        self.last_save_t = 0
        
        # MAC and processing components
        self.mac = None
        self.batch_handler = None
        
        # Statistics tracking
        self.train_stats = {}
        self.test_stats = {}
        # store raw env step infos for the most recent episode (useful for task-specific evaluation)
        self.last_env_infos: List[Dict[str, Any]] = []

        # ===== Debug: forced-vs-parsed alignment (Stage4 social credit assignment) =====
        # We print ONCE (per process) when debug is enabled, to verify that env-consumed JSON
        # matches the policy-forced action_type/stance_id.
        self._forced_align_printed = False
        self._forced_align_action_n = 0
        self._forced_align_action_ok = 0
        self._forced_align_stance_n = 0
        self._forced_align_stance_ok = 0
        
        # Episode management
        self.episode_limit = 1  # Single step per episode for LLM environments
        self.n_agents = args.n_agents
        self.batch_size = self.args.batch_size_run
        # NOTE:
        # 当前 runner/env 逻辑是“单环境实例”采样；当 batch_size_run > 1 时，
        # EpisodeBatch 的 batch 维会被同一条轨迹广播填充（用于形状兼容/调试/并行化前过渡）。
        # 如果你需要真正的并行采样，需要引入 vectorized env 或多 env 实例。
        if self.batch_size != 1:
            self.logger.warning(
                f"EpisodeRunner batch_size_run={self.batch_size} (>1). "
                "Current implementation will broadcast a single env trajectory across batch dimension."
            )
        
        # Initialize environment using the registry and env_args from config
        self.env = self._init_environment()
        self.env_info = self.env.get_env_info()
        self.episode_limit = self.env_info["episode_limit"]
        self.obs_shape = self.env_info["obs_shape"]
        self.t = 0 # Step within the current episode
        
        # Initialize batch handling
        # max_seq_length for EpisodeBatch will be self.episode_limit + 1.
        # If episode_limit is 1 (one data sample = one step), max_seq_length is 2.
        self.new_batch = self._init_batch_handler()
        self.batch = self.new_batch()

    def setup(self, scheme: Dict, groups: Dict, preprocess: Any, mac: Any):
        """Setup with MAC."""
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.mac = mac  # 添加MAC
    
    def _init_environment(self):
        """Initialize and return the environment from the registry."""
        try:
            env_key = self.args.env
            # Prepare environment arguments including reward configuration
            # Convert SimpleNamespace to dict
            if hasattr(self.args.env_args, '__dict__'):
                env_kwargs = vars(self.args.env_args)
            else:
                env_kwargs = dict(self.args.env_args)
            
            # Add reward configuration if it exists
            if hasattr(self.args, 'reward'):
                env_kwargs['reward_config'] = self.args.reward
            
            return env_REGISTRY[env_key](**env_kwargs)
        except KeyError:
            self.logger.error(f"Environment '{self.args.env}' not found in registry. Available: {list(env_REGISTRY.keys())}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize environment '{self.args.env}': {e}")
            raise
    
    def _init_batch_handler(self):
        """Initialize and return the batch handler."""
        return partial(
            EpisodeBatch,
            scheme=self._build_scheme(),
            groups=self._build_groups(),
            batch_size=self.batch_size,
            max_seq_length=self.episode_limit + 1,
            device=self.args.device
        )

    # ===== B: curriculum support (update n_stages / episode_limit at runtime) =====
    def set_env_n_stages(self, n_stages: int) -> None:
        """
        Update env.n_stages (if supported) and rebuild episode_limit + EpisodeBatch factory.
        This enables curriculum training without restarting the whole process.
        """
        try:
            ns = int(n_stages)
        except Exception:
            return
        if ns <= 0:
            return
        if not hasattr(self.env, "n_stages"):
            self.logger.warning("Runner curriculum requested but env has no attribute n_stages.")
            return
        # update env
        try:
            self.env.n_stages = int(ns)
            # keep env episode_limit consistent
            if hasattr(self.env, "core_users"):
                self.env.episode_limit = int(ns) * len(getattr(self.env, "core_users"))
        except Exception as e:
            self.logger.warning(f"Failed to set env.n_stages={ns}: {e}")
            return

        # refresh env_info / episode_limit and rebuild batch factory
        try:
            self.env_info = self.env.get_env_info()
            self.episode_limit = int(self.env_info.get("episode_limit", self.episode_limit))
            self.new_batch = self._init_batch_handler()
            self.reset_runner_state()
            self.logger.info(f"[Curriculum] Updated env n_stages={ns}, episode_limit={self.episode_limit}")
        except Exception as e:
            self.logger.warning(f"Failed to refresh runner after setting n_stages={ns}: {e}")

    def run(self, test_mode: bool = False) -> EpisodeBatch:
        """
        Run a complete episode (processing one data sample from the dataset).
        
        Args:
            test_mode: Whether in testing mode
            
        Returns:
            Collected episode data for the processed sample.
        """
        try:
            # Reset environment and MAC hidden state
            # For HuggingFaceDatasetEnv, reset() loads the next data sample (e.g., a question)
            # and sets self.env.current_question and self.env.current_sample.
            # The observation returned by self.env.reset() is self.env.current_question.
            current_obs, env_step_info = self.env.reset() # env_step_info contains the full sample
            self.reset_runner_state() # Resets self.batch and self.t
            
            if current_obs is None: # Should be handled by env.reset() raising StopIteration
                self.logger.warning("Environment reset returned None observation. Stopping run.")
                return self.batch # Return empty or partially filled batch

            episode_return = 0
            if hasattr(self.mac, 'init_hidden'):
                self.mac.init_hidden(batch_size=self.batch_size)
            
            metrics = EpisodeMetrics()
            
            terminated = False
            _next_obs = current_obs
            self.last_env_infos = []

            # 多步 episode：循环直到 env 返回 terminated 或达到 episode_limit
            while (not terminated) and (self.t < self.episode_limit):
                pre_transition_data = self._get_pre_transition_data(_next_obs)
                self.batch.update(pre_transition_data, ts=self.t)

                discrete_actions, mac_extra_info = self._get_actions(test_mode, raw_observation_text=_next_obs)

                # Determine action for env.step()
                # Default: use coordinator commitment_text (string).
                # For offline classification-style training (HF dataset with \\boxed{id}),
                # you can set args.env_action_source="discrete_action_boxed" to use the chosen discrete action.
                action_source = str(getattr(self.args, "env_action_source", "commitment")).strip().lower()
                action_for_env_step = ""
                if action_source in ("discrete_action_boxed", "boxed", "discrete"):
                    try:
                        # discrete_actions is typically shape (bs, n_agents) or (n_agents,)
                        a = discrete_actions
                        if isinstance(a, torch.Tensor):
                            # take batch 0 if exists
                            if a.ndim >= 2:
                                a0 = a[0]
                            else:
                                a0 = a
                            # take agent 0 as the env action (you can change to majority vote later)
                            aid = int(a0[0].item()) if a0.numel() > 0 else 0
                        else:
                            aid = int(a)
                    except Exception:
                        aid = 0
                    action_for_env_step = f"\\boxed{{{aid}}}"
                elif action_source in ("llm_response_0", "executor0", "executor_0", "response0"):
                    # Use the first executor response directly (useful for hisim_social_env where env expects JSON action).
                    rs = mac_extra_info.get("llm_responses") or []
                    action_for_env_step = rs[0] if isinstance(rs, list) and len(rs) > 0 else ""
                else:
                    action_for_env_step = mac_extra_info.get("commitment_text", "")
                    if not action_for_env_step and mac_extra_info.get("llm_responses"):
                        action_for_env_step = mac_extra_info["llm_responses"][0] if mac_extra_info["llm_responses"] else ""

                step_extra_info = {
                    "agent_responses": mac_extra_info.get("llm_responses", []),
                    "commitment_text": mac_extra_info.get("commitment_text", ""),
                    "agent_log_probs": mac_extra_info.get("agent_log_probs"),
                    "prompt_embeddings": mac_extra_info.get("prompt_embeddings"),
                    "belief_states": mac_extra_info.get("belief_states"),
                    # optional: secondary user belief for env-side simulation
                    "secondary_z_next": mac_extra_info.get("secondary_z_next"),
                    "secondary_action_probs": mac_extra_info.get("secondary_action_probs"),
                }

                _next_obs, reward_total_float, terminated, _truncated, env_step_info = self.env.step(
                    action_for_env_step, extra_info=step_extra_info
                )
                # cache for evaluation
                if isinstance(env_step_info, dict):
                    self.last_env_infos.append(env_step_info)

                # ---- Debug: forced vs parsed alignment (print once) ----
                try:
                    dbg = bool(getattr(getattr(self.args, "system", None), "debug", False))
                    env_name = str(getattr(self.args, "env", "") or "").strip().lower()
                    if dbg and (not self._forced_align_printed) and env_name == "hisim_social_env":
                        f_at = None
                        f_sid = None
                        try:
                            fat_list = mac_extra_info.get("forced_action_types")
                            fsid_list = mac_extra_info.get("forced_stance_ids")
                            if isinstance(fat_list, list) and len(fat_list) > 0:
                                f_at = str(fat_list[0])
                            if isinstance(fsid_list, list) and len(fsid_list) > 0:
                                f_sid = fsid_list[0]
                        except Exception:
                            f_at = None
                            f_sid = None

                        parsed_at = env_step_info.get("action_type") if isinstance(env_step_info, dict) else None
                        parsed_sid = env_step_info.get("pred_stance_id") if isinstance(env_step_info, dict) else None

                        if f_at is not None and parsed_at is not None:
                            self._forced_align_action_n += 1
                            if str(f_at) == str(parsed_at):
                                self._forced_align_action_ok += 1

                        stance_actions = {"post", "retweet", "reply"}
                        if f_at is not None and str(f_at) in stance_actions and (f_sid is not None) and (parsed_sid is not None):
                            self._forced_align_stance_n += 1
                            try:
                                if int(f_sid) == int(parsed_sid):
                                    self._forced_align_stance_ok += 1
                            except Exception:
                                pass

                        # Print once after enough samples (or near episode end).
                        thr = int(getattr(self.args, "forced_align_log_after", 50))
                        thr = max(1, thr)
                        if (self._forced_align_action_n >= thr) or bool(terminated):
                            ar = self._forced_align_action_ok / float(max(1, self._forced_align_action_n))
                            sr = self._forced_align_stance_ok / float(max(1, self._forced_align_stance_n)) if self._forced_align_stance_n > 0 else float("nan")
                            sr_str = f"{sr:.3f}" if (sr == sr) else "nan"
                            self.logger.info(
                                f"[Debug][forced-align] action_type align: {self._forced_align_action_ok}/{self._forced_align_action_n}={ar:.3f} | "
                                f"stance_id align (stance-actions only): {self._forced_align_stance_ok}/{self._forced_align_stance_n}={sr_str}"
                            )
                            self._forced_align_printed = True
                except Exception:
                    pass

                reward_ts = env_step_info.get("reward_ts", 0.0)
                reward_al = env_step_info.get("reward_al", 0.0)
                reward_cc = env_step_info.get("reward_cc", 0.0)

                rewards_al_list = [reward_al] * self.n_agents
                rewards_ts_list = [reward_ts] * self.n_agents
                rewards_cc_list = [reward_cc] * self.n_agents

                episode_return += reward_total_float
                metrics.add_step_data(mac_extra_info, reward_total_float, reward_al, reward_ts, reward_cc)

                actions_for_batch_storage = discrete_actions[0] if discrete_actions.ndim > 1 else discrete_actions
                current_commitment_embedding = mac_extra_info.get("commitment_embedding")
                current_q_values = mac_extra_info.get("q_values")
                current_agent_prompt_embeddings = mac_extra_info.get("prompt_embeddings")
                current_group_representation = mac_extra_info.get("group_repr")
                current_belief_states = mac_extra_info.get("belief_states")

                post_data = self._get_post_transition_data(
                    actions_for_batch_storage,
                    reward_total_float,
                    terminated,
                    env_step_info,
                    rewards_al_list,
                    rewards_ts_list,
                    rewards_cc_list,
                    current_commitment_embedding,
                    current_q_values,
                    current_agent_prompt_embeddings,
                    current_group_representation,
                    current_belief_states,
                )
                self.batch.update(post_data, ts=self.t)
                self.t += 1
                # Maintain a true global environment-step counter used by action selection schedules.
                # Previously this runner never incremented t_env, causing epsilon schedules to stay at start.
                try:
                    self.t_env += 1
                except Exception:
                    self.t_env = int(getattr(self, "t_env", 0)) + 1
                # Best-effort epsilon schedule update (multinomial selector supports epsilon_decay).
                try:
                    sel = getattr(getattr(self.mac, "action_selector", None), "epsilon_decay", None)
                    if callable(sel):
                        sel(int(self.t_env))
                except Exception:
                    pass

            if not test_mode:
                self._handle_episode_end(metrics, episode_return, test_mode)
                self.t_episodes += 1

            if self.episode_limit > 0:
                self._add_final_data(_next_obs if not test_mode else current_obs)

            if not test_mode:
                self._add_llm_data_to_batch(metrics)

            return self.batch
            
        except StopIteration: # Raised by self.env.reset() if dataset is exhausted
            self.logger.info(f"Dataset exhausted after {self.t_env} samples.")
            # Potentially return the last partially filled batch or a special signal
            return self.batch # Or None, or raise further to signal completion
        except Exception as e:
            logger.error(f"Error during episode execution: {str(e)}")
            logger.exception("Exception details:")
            raise

    def _get_pre_transition_data(self, current_observation_text: str) -> Dict:
        """Get pre-transition data (current observation)."""
        # Preprocess (tokenize) the observation text using the MAC's preprocessor
        # Ensure obs_tensor is on the correct device (preprocess_observation should handle this)
        # The shape of obs_tensor should be (max_token_length,)
        obs_tensor = self.mac.preprocess_observation(current_observation_text) 

        # Other fields for scheme if needed (often placeholders for text envs)
        default_state_vshape = self.env_info.get("state_shape", (1,))
        default_avail_actions_vshape = (self.env_info.get("n_actions", 1),)

        return {
            # obs_tensor will be grouped by agent and batched by EpisodeBuffer.
            # For bs=1, EpisodeBuffer.update expects data for "obs" to be a list of tensors,
            # one for each agent, or a single tensor if "group" is not "agents".
            # If scheme["obs"] has "group": "agents", then obs_tensor should be provided for each agent.
            # Since the observation is global, we replicate it for each agent.
            "obs": [obs_tensor for _ in range(self.n_agents)],
            "state": [torch.zeros(*default_state_vshape, device=self.args.device)], 
            "avail_actions": [torch.ones(*default_avail_actions_vshape, device=self.args.device, dtype=torch.int64) for _ in range(self.n_agents)]
        }

    def _get_actions(self, test_mode: bool, raw_observation_text: Optional[str] = None) -> Tuple[torch.Tensor, Dict]:
        """Get actions and extra info from MAC."""
        # self.batch here contains the pre_transition_data at self.t (which is 0)
        # self.mac.select_actions expects the whole batch and current timestep t_ep.
        return self.mac.select_actions(
            self.batch, # Pass the current episode batch (contains tokenized obs at ts=0)
            t_ep=self.t,  # Current step in the episode (0)
            t_env=self.t_env, # Global step counter
            raw_observation_text=raw_observation_text, # Pass raw text for LLM prompts
            test_mode=test_mode
        )

    def _get_post_transition_data(self, discrete_actions_for_agents: torch.Tensor, 
                                reward_total: float,  # 修改为单个值
                                terminated: bool, env_info: Dict,
                                rewards_al: List[float], 
                                rewards_ts: List[float], 
                                rewards_cc: List[float],
                                commitment_embedding: Optional[torch.Tensor],
                                q_values_per_agent: Optional[torch.Tensor], # New: (n_agents, 1)
                                prompt_embeddings_per_agent: Optional[torch.Tensor], # New: (n_agents, 2)
                                group_representation: Optional[torch.Tensor],
                                belief_states: Optional[torch.Tensor]
                                ) -> Dict:
        """Get post-transition data."""
        
        # actions should be a tensor of shape (self.n_agents, expected_action_vshape_in_scheme)
        # scheme[actions][vshape] is (1,) for discrete actions.
        # So, actions should be (self.n_agents, 1)
        # Ensure `actions` (processed_actions from `run`) has this shape.
        # If `actions` from MAC is (n_agents, ), we might need to .view(-1, 1) if scheme expects (1,)

        # 对于全局奖励，创建标量张量
        final_reward_scalar = torch.tensor([reward_total], dtype=torch.float32, device=self.args.device)  # (1,)
        
        # 对于per-agent奖励，创建张量
        rewards_al_tensor = torch.tensor(rewards_al, dtype=torch.float32, device=self.args.device).view(self.n_agents, 1)
        rewards_ts_tensor = torch.tensor(rewards_ts, dtype=torch.float32, device=self.args.device).view(self.n_agents, 1)
        rewards_cc_tensor = torch.tensor(rewards_cc, dtype=torch.float32, device=self.args.device).view(self.n_agents, 1)

        # discrete_actions_for_agents should be a tensor of shape (self.n_agents, scheme_action_vshape)
        # If scheme_action_vshape is (1,), then (self.n_agents, 1)
        # If discrete_actions_for_agents is (self.n_agents, ), then .view(self.n_agents, 1)
        if discrete_actions_for_agents.ndim == 1:
             actions_for_batch = discrete_actions_for_agents.view(self.n_agents, 1)
        else:
             actions_for_batch = discrete_actions_for_agents
        actions_for_batch = actions_for_batch.to(device=self.args.device, dtype=torch.long)

        post_data_dict = {
            "actions": actions_for_batch, 
            "reward": final_reward_scalar, 
            "terminated": torch.tensor([terminated], dtype=torch.uint8, device=self.args.device),
            "reward_al": rewards_al_tensor,
            "reward_ts": rewards_ts_tensor,
            "reward_cc": rewards_cc_tensor,
            "filled": torch.tensor([1], dtype=torch.long, device=self.args.device)
        }

        # === offline supervised ground-truth label (HF dataset) ===
        # Prefer HuggingFaceDatasetEnv: env_info["ground_truth_answer"] like "\\boxed{2}"
        try:
            import re

            def _parse_boxed_int(s: Any) -> Optional[int]:
                if not isinstance(s, str):
                    return None
                m = re.search(r"\\boxed\{\s*([-+]?\d+)\s*\}", s)
                if not m:
                    # tolerate "boxed{2}" without backslash
                    m = re.search(r"boxed\{\s*([-+]?\d+)\s*\}", s)
                return int(m.group(1)) if m else None

            gt = None
            if isinstance(env_info, dict):
                gt = _parse_boxed_int(env_info.get("ground_truth_answer"))
                if gt is None:
                    gt = _parse_boxed_int(env_info.get("ground_truth"))
                # social env style: directly provides gt stance id
                if gt is None:
                    v = env_info.get("gt_stance_id")
                    if isinstance(v, int):
                        gt = int(v)
            if gt is not None:
                post_data_dict["gt_action"] = torch.tensor([gt], dtype=torch.int64, device=self.args.device)
        except Exception as e:
            self.logger.debug(f"Failed to parse gt_action from env_info: {e}")

        # === offline supervised soft label distribution (HF dataset) ===
        # Prefer env_info["target_distribution_prob"] which is a dict like {"0":0.2,"1":0.1,"2":0.7}
        try:
            if isinstance(env_info, dict) and "target_distribution_prob" in env_info:
                na = int(self.env_info.get("n_actions", 1))
                na = max(1, na)
                dist = env_info.get("target_distribution_prob")
                arr = [0.0 for _ in range(na)]
                if isinstance(dist, dict):
                    for k, v in dist.items():
                        try:
                            idx = int(k)
                            if 0 <= idx < na:
                                arr[idx] = float(v)
                        except Exception:
                            continue
                elif isinstance(dist, (list, tuple)):
                    for i, x in enumerate(list(dist)[:na]):
                        try:
                            arr[i] = float(x)
                        except Exception:
                            arr[i] = 0.0
                post_data_dict["gt_action_dist"] = torch.tensor(arr, dtype=torch.float32, device=self.args.device)
        except Exception as e:
            self.logger.debug(f"Failed to parse gt_action_dist from env_info: {e}")

        # === belief inputs (explicit, tensorized) ===
        # env_info may contain: belief_inputs_pre / belief_inputs_post (dict)
        try:
            if isinstance(env_info, dict):
                # stage index (global)
                if "t" in env_info:
                    # vshape=(1,) -> tensor shape (1,), will broadcast to (bs,1) in EpisodeBatch.update
                    post_data_dict["stage_t"] = torch.tensor([int(env_info.get("t", 0))], dtype=torch.int64, device=self.args.device)

                # tensorize via env helper if available
                get_bt = getattr(self.env, "get_belief_tensor", None)
                if callable(get_bt):
                    bi_pre = env_info.get("belief_inputs_pre")
                    bt_pre = get_bt(bi_pre, device=self.args.device) if bi_pre is not None else None
                    if isinstance(bt_pre, dict):
                        # canonical tensors
                        if "population_z" in bt_pre:
                            # vshape=(population_belief_dim,) -> tensor shape (K,), will broadcast to (bs,K)
                            post_data_dict["belief_pre_population_z"] = bt_pre["population_z"].to(self.args.device)
                            # for population_update_head training
                            post_data_dict["z_t"] = bt_pre["population_z"].to(self.args.device)
                        if "neighbor_stance_counts" in bt_pre:
                            post_data_dict["belief_pre_neighbor_counts"] = bt_pre["neighbor_stance_counts"].to(self.args.device)
                        if "is_core_user" in bt_pre:
                            # vshape=(1,) -> tensor shape (1,), will broadcast to (bs,1)
                            post_data_dict["belief_pre_is_core_user"] = bt_pre["is_core_user"].to(self.args.device)

                    bi_post = env_info.get("belief_inputs_post")
                    bt_post = get_bt(bi_post, device=self.args.device) if bi_post is not None else None
                    if isinstance(bt_post, dict):
                        if "population_z" in bt_post:
                            post_data_dict["belief_post_population_z"] = bt_post["population_z"].to(self.args.device)
                        if "neighbor_stance_counts" in bt_post:
                            post_data_dict["belief_post_neighbor_counts"] = bt_post["neighbor_stance_counts"].to(self.args.device)
                        if "is_core_user" in bt_post:
                            post_data_dict["belief_post_is_core_user"] = bt_post["is_core_user"].to(self.args.device)
        except Exception as e:
            self.logger.warning(f"Failed to add belief tensor fields to batch: {e}")

        # === optional latent-z supervision fields (global, from env_info) ===
        # env_info may contain: z_pred/z_target (len=population_belief_dim), z_mask (float)
        try:
            if isinstance(env_info, dict) and ("z_pred" in env_info or "z_target" in env_info or "z_mask" in env_info):
                z_pred = env_info.get("z_pred")
                z_target = env_info.get("z_target")
                z_mask = env_info.get("z_mask", 0.0)
                if isinstance(z_pred, list):
                    post_data_dict["z_pred"] = torch.tensor(z_pred, dtype=torch.float32, device=self.args.device)
                if isinstance(z_target, list):
                    post_data_dict["z_target"] = torch.tensor(z_target, dtype=torch.float32, device=self.args.device)
                post_data_dict["z_mask"] = torch.tensor([float(z_mask)], dtype=torch.float32, device=self.args.device)
        except Exception as e:
            self.logger.warning(f"Failed to add z supervision fields to batch: {e}")

        # === optional: structured conditioning fields for z_transition (global, from env_info) ===
        # These are passthrough fields from HuggingFaceDatasetEnv samples.
        try:
            if isinstance(env_info, dict):
                if "core_stance_id_t" in env_info:
                    post_data_dict["core_stance_id_t"] = torch.tensor([int(env_info.get("core_stance_id_t", -1))], dtype=torch.int64, device=self.args.device)
                if "core_action_type_id_t" in env_info:
                    post_data_dict["core_action_type_id_t"] = torch.tensor([int(env_info.get("core_action_type_id_t", -1))], dtype=torch.int64, device=self.args.device)
                if "has_user_history" in env_info:
                    post_data_dict["has_user_history"] = torch.tensor([int(env_info.get("has_user_history", 0))], dtype=torch.int64, device=self.args.device)
                if "has_neighbors" in env_info:
                    post_data_dict["has_neighbors"] = torch.tensor([int(env_info.get("has_neighbors", 0))], dtype=torch.int64, device=self.args.device)
                if "neighbor_action_type_counts_t" in env_info:
                    v = env_info.get("neighbor_action_type_counts_t")
                    if isinstance(v, dict):
                        order = ["post", "retweet", "reply", "like", "do_nothing"]
                        arr = [float(v.get(k, 0.0)) for k in order]
                    elif isinstance(v, (list, tuple)):
                        arr = [float(x) for x in list(v)[:5]]
                        if len(arr) < 5:
                            arr = arr + [0.0] * (5 - len(arr))
                    else:
                        arr = [0.0, 0.0, 0.0, 0.0, 0.0]
                    post_data_dict["neighbor_action_type_counts_t"] = torch.tensor(arr, dtype=torch.float32, device=self.args.device)
                if "neighbor_stance_counts_t" in env_info:
                    v = env_info.get("neighbor_stance_counts_t")
                    if isinstance(v, dict):
                        # stance order fixed to [Neutral,Oppose,Support] -> [0,1,2]
                        order = ["Neutral", "Oppose", "Support"]
                        arr = [float(v.get(k, 0.0)) for k in order]
                    elif isinstance(v, (list, tuple)):
                        arr = [float(x) for x in list(v)[:3]]
                        if len(arr) < 3:
                            arr = arr + [0.0] * (3 - len(arr))
                    else:
                        arr = [0.0, 0.0, 0.0]
                    post_data_dict["neighbor_stance_counts_t"] = torch.tensor(arr, dtype=torch.float32, device=self.args.device)
        except Exception as e:
            self.logger.debug(f"Failed to add structured conditioning fields to batch: {e}")

        if commitment_embedding is not None:
            if commitment_embedding.ndim == 1: 
                processed_commitment_embedding = commitment_embedding.unsqueeze(0).to(self.args.device)
            elif commitment_embedding.ndim == 2 and commitment_embedding.shape[0] == 1: # Already (1, embed_dim)
                processed_commitment_embedding = commitment_embedding.to(self.args.device)
            else:
                self.logger.warning(f"Unexpected commitment_embedding shape from MAC: {commitment_embedding.shape}. Expected (embed_dim,) or (1, embed_dim). Not adding to batch.")
                processed_commitment_embedding = None 
            
            if processed_commitment_embedding is not None:
                 post_data_dict["commitment_embedding"] = processed_commitment_embedding
        
        if q_values_per_agent is not None:  # expected (n_agents,1); accept common variants and normalize
            try:
                qv = q_values_per_agent
                if isinstance(qv, torch.Tensor):
                    # Common variants observed in this codebase:
                    # - (bs, n_agents) coming from BasicMAC ("q_values" is often (bs,n_agents))
                    # - (n_agents,) scalar per agent
                    # - (n_agents,1)
                    # - (1,n_agents,1)
                    if qv.ndim == 1 and qv.shape[0] == self.n_agents:
                        qv = qv.view(self.n_agents, 1)
                    elif qv.ndim == 2:
                        if qv.shape == (self.n_agents, 1):
                            pass
                        elif qv.shape == (1, self.n_agents):
                            qv = qv.view(self.n_agents, 1)
                        elif qv.shape[0] == 1 and qv.shape[1] == self.n_agents:
                            qv = qv.view(self.n_agents, 1)
                        else:
                            # if it's (bs, n_agents) with bs==1, squeeze to (n_agents,1)
                            if qv.shape[0] == 1 and qv.shape[1] == self.n_agents:
                                qv = qv.squeeze(0).view(self.n_agents, 1)
                    elif qv.ndim == 3 and qv.shape == (1, self.n_agents, 1):
                        qv = qv.squeeze(0)

                    if isinstance(qv, torch.Tensor) and qv.shape == (self.n_agents, 1):
                        post_data_dict["q_values"] = qv.to(self.args.device)
                    else:
                        self.logger.warning(
                            f"Unexpected q_values_per_agent shape: {getattr(q_values_per_agent, 'shape', None)} "
                            f"(normalized={getattr(qv, 'shape', None)}). Expected ({self.n_agents}, 1). Not adding to batch."
                        )
                else:
                    self.logger.warning(f"Unexpected q_values_per_agent type: {type(q_values_per_agent)}. Not adding to batch.")
            except Exception as e:
                self.logger.warning(f"Failed to normalize q_values_per_agent: {e}")

        if prompt_embeddings_per_agent is not None: # Expected shape (n_agents, 2) or (1, n_agents, 2)
            if prompt_embeddings_per_agent.shape == (self.n_agents, 2):
                post_data_dict["prompt_embeddings"] = prompt_embeddings_per_agent.to(self.args.device) # Shape: (n_agents, 2)
            elif prompt_embeddings_per_agent.shape == (1, self.n_agents, 2):
                post_data_dict["prompt_embeddings"] = prompt_embeddings_per_agent.squeeze(0).to(self.args.device) # Remove batch dim: (n_agents, 2)
            else:
                self.logger.warning(f"Unexpected prompt_embeddings_per_agent shape: {prompt_embeddings_per_agent.shape}. Expected ({self.n_agents}, 2) or (1, {self.n_agents}, 2). Not adding to batch.")

        if group_representation is not None:
            if group_representation.ndim == 1: 
                processed_group_representation = group_representation.to(self.args.device) # Shape: (embed_dim,)
            elif group_representation.ndim == 2 and group_representation.shape[0] == 1: # Shape: (1, embed_dim)
                processed_group_representation = group_representation.squeeze(0).to(self.args.device) # Remove batch dim: (embed_dim,)
            else:
                self.logger.warning(f"Unexpected group_representation shape from MAC: {group_representation.shape}. Expected (embed_dim,) or (1, embed_dim). Not adding to batch.")
                processed_group_representation = None 
            
            if processed_group_representation is not None:
                 post_data_dict["group_representation"] = processed_group_representation
        
        if belief_states is not None:
            expected_belief_dim = getattr(self.args, 'belief_dim', 64) # Get belief_dim from args, with a fallback
            if belief_states.shape == (self.n_agents, expected_belief_dim):
                post_data_dict["belief_states"] = belief_states.to(self.args.device) # Shape: (n_agents, belief_dim)
            elif belief_states.shape == (1, self.n_agents, expected_belief_dim):
                post_data_dict["belief_states"] = belief_states.squeeze(0).to(self.args.device) # Remove batch dim: (n_agents, belief_dim)
            else:
                self.logger.warning(f"Unexpected belief_states shape: {belief_states.shape}. Expected ({self.n_agents}, {expected_belief_dim}) or (1, {self.n_agents}, {expected_belief_dim}). Not adding to batch.")

        return post_data_dict

    def _handle_episode_end(self, metrics: EpisodeMetrics, 
                          episode_return: float, test_mode: bool):
        """Handle end of episode processing."""
        self._save_episode_metrics(metrics, test_mode)
        
        if test_mode:
            self.test_returns.append(episode_return)
            self.logger.log_stat("test_return", episode_return, self.t_env)
        else:
            self.train_returns.append(episode_return)
            self.logger.log_stat("train_return", episode_return, self.t_env)

    def _add_final_data(self, next_observation_text: str):
        """Add final (next) observations to batch at self.t (which is 1)."""
        # Preprocess the next (dummy) observation text
        next_obs_tensor = self.mac.preprocess_observation(next_observation_text)

        default_state_vshape = self.env_info.get("state_shape", (1,))
        default_avail_actions_vshape = (self.env_info.get("n_actions", 1),)

        last_data = {
            "obs": [next_obs_tensor for _ in range(self.n_agents)],
            "state": [torch.zeros(*default_state_vshape, device=self.args.device)], 
            "avail_actions": [torch.ones(*default_avail_actions_vshape, device=self.args.device, dtype=torch.int64) for _ in range(self.n_agents)],
            "filled": torch.tensor([0], dtype=torch.long, device=self.args.device)  # 最终状态标记为无效
        }
        self.batch.update(last_data, ts=self.t) # self.t is 1 here

    def reset_runner_state(self):
        """Reset runner's per-episode state (batch and timestep t)."""
        self.batch = self.new_batch() # Get a fresh batch from the handler
        self.t = 0 # Reset episode timestep

    def _build_scheme(self) -> Dict:
        """
        Build data scheme for episode batch.
        
        Returns:
            Data scheme dictionary
        """
        commitment_dim = getattr(self.args, 'commitment_embedding_dim', 768)
        belief_dim = getattr(self.args, 'belief_dim')
        pop_dim = int(getattr(self.args, "population_belief_dim", 3))
        pop_dim = max(1, pop_dim)
        # Max question length here refers to max token length after tokenization
        # It should come from env_args, which HuggingFaceDatasetEnv also uses.
        max_token_len = getattr(self.args.env_args, "max_question_length", 512)

        scheme = {
            "state": {"vshape": self.env_info["state_shape"]}, # Usually (1,) for these envs
            # obs is now token IDs, per agent
            "obs": {"vshape": (max_token_len,), "group": "agents", "dtype": torch.long},
            "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long}, # Symbolic actions
            "avail_actions": {
                "vshape": (self.env_info["n_actions"],), # n_actions usually 1
                "group": "agents",
                "dtype": torch.int64, # Changed from torch.int for consistency
            },
            "reward": {"vshape": (1,)}, # Global reward, will be unsqueezed by buffer if group="agents"
            "terminated": {"vshape": (1,), "dtype": torch.uint8},
            "filled": {"vshape": (1,), "dtype": torch.long},  # 添加filled字段，标记有效的时间步
            
            # Fields per agent (these are fine)
            "q_values": {"vshape": (1,), "group": "agents", "dtype": torch.float32}, 
            "prompt_embeddings": {"vshape": (2,), "group": "agents", "dtype": torch.float32}, 
            "belief_states": {"vshape": (belief_dim,), "group": "agents"},
            "reward_al": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
            "reward_ts": {"vshape": (1,), "group": "agents", "dtype": torch.float32},
            "reward_cc": {"vshape": (1,), "group": "agents", "dtype": torch.float32},

            # Global fields (these are fine)
            "commitment_embedding": {"vshape": (commitment_dim,), "dtype": torch.float32},
            "group_representation": {"vshape": (belief_dim,), "dtype": torch.float32}
            ,
            # === offline supervised label (global) ===
            # For HuggingFaceDatasetEnv stance-id training: ground-truth \\boxed{<id>} parsed to int.
            "gt_action": {"vshape": (1,), "dtype": torch.int64},
            # Optional soft-label distribution over stance ids (aligned to env_info["n_actions"])
            "gt_action_dist": {"vshape": (self.env_info.get("n_actions", 1),), "dtype": torch.float32},
            # latent z supervision (global)
            "z_pred": {"vshape": (pop_dim,), "dtype": torch.float32},
            "z_target": {"vshape": (pop_dim,), "dtype": torch.float32},
            "z_mask": {"vshape": (1,), "dtype": torch.float32},

            # === belief inputs (explicit, tensorized; global) ===
            # stage index
            "stage_t": {"vshape": (1,), "dtype": torch.int64},
            # for population belief update head (z(t) -> z(t+1))
            "z_t": {"vshape": (pop_dim,), "dtype": torch.float32},
            # belief inputs (pre/post)
            "belief_pre_population_z": {"vshape": (pop_dim,), "dtype": torch.float32},
            "belief_pre_neighbor_counts": {"vshape": (3,), "dtype": torch.float32},
            "belief_pre_is_core_user": {"vshape": (1,), "dtype": torch.int64},
            "belief_post_population_z": {"vshape": (pop_dim,), "dtype": torch.float32},
            "belief_post_neighbor_counts": {"vshape": (3,), "dtype": torch.float32},
            "belief_post_is_core_user": {"vshape": (1,), "dtype": torch.int64},
            # === optional structured conditioning for z_transition (global) ===
            "core_stance_id_t": {"vshape": (1,), "dtype": torch.int64},
            "core_action_type_id_t": {"vshape": (1,), "dtype": torch.int64},
            "has_user_history": {"vshape": (1,), "dtype": torch.int64},
            "has_neighbors": {"vshape": (1,), "dtype": torch.int64},
            "neighbor_action_type_counts_t": {"vshape": (5,), "dtype": torch.float32},
            "neighbor_stance_counts_t": {"vshape": (3,), "dtype": torch.float32},
        }
        return scheme

    def _build_groups(self) -> Dict:
        """
        Build groups for episode batch.
        
        Returns:
            Group definitions
        """
        return {
            "agents": self.args.n_agents
        }

    def _save_episode_metrics(self, metrics: EpisodeMetrics, test_mode: bool):
        """
        Save episode metrics.
        
        Args:
            metrics: Collected metrics
            test_mode: Whether in testing mode
        """
        stats = self.test_stats if test_mode else self.train_stats
        
        # Calculate average reward
        if metrics.rewards:
            stats['mean_reward'] = np.mean(metrics.rewards)
        
        # Calculate LLM response diversity
        if metrics.llm_responses:
            unique_responses = len(set(map(str, metrics.llm_responses)))
            stats['response_diversity'] = unique_responses / len(metrics.llm_responses)
        
        # Log statistics
        prefix = 'test_' if test_mode else 'train_'
        for k, v in stats.items():
            self.logger.log_stat(f"{prefix}{k}", v, self.t_env)

    def _add_llm_data_to_batch(self, metrics: EpisodeMetrics):
        """
        Add LLM-related data to episode batch.
        
        Args:
            metrics: Collected LLM metrics
        """
        self.logger.info("_add_llm_data_to_batch called. Current logic mostly commented out or for text logging only.")
        # try:
            # Only stack if there are items and they are stackable (e.g. tensors)
            # llm_data_to_add = {}
            # if metrics.llm_responses and all(isinstance(x, torch.Tensor) for x in metrics.llm_responses):
            #     llm_data_to_add["llm_responses"] = torch.stack(metrics.llm_responses) 
            
            # if metrics.belief_states: # Already added per step
            #    pass 

            # if llm_data_to_add: 
            #    self.batch.update(llm_data_to_add) 
            #    self.logger.info("_add_llm_data_to_batch: Added to batch - " + str(list(llm_data_to_add.keys())))

        # except Exception as e:
        #     logger.error(f"Error in _add_llm_data_to_batch: {str(e)}. Data types might be incompatible.")
        #     raise

    def reset(self):
        """Reset the runner state."""
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def get_env_info(self) -> Dict:
        """Get environment information."""
        return self.env_info

    def save_replay(self):
        """Save replay buffer."""
        self.env.save_replay()

    def close_env(self):
        """Close environment."""
        self.env.close()

    def log_train_stats_t(self):
        """Log training statistics."""
        self.logger.print_recent_stats()
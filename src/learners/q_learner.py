import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from components.episode_buffer import EpisodeBatch
from modules.mixer.mix_llm import LLMQMixer
from modules.belief_encoder import BeliefEncoder
from typing import Dict, List, Tuple, Optional, Any
import os

"""
Q-Learning algorithm with multi-agent coordination.

Learner for the ECON framework.

Implements Q-learning with:
- Multi-agent coordination
- LLM-based belief networks
- Mixing networks for global Q-values
- Dynamic reward systems
- Two-stage belief coordination
- Bayesian Nash Equilibrium (BNE) updates
"""

class ECONLearner:
    """
    Learner for the ECON framework.
    Handles the optimization of individual BeliefNetworks, the BeliefEncoder,
    and the CentralizedMixingNetwork (LLMQMixer).
    """
    
    def __init__(self, mac: Any, scheme: Dict, logger: Any, args: Any):
        self.args = args
        self.logger = logger
        self.mac = mac
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        
        self.last_target_update_episode = 0
        self.log_stats_t = -getattr(args, "learner_log_interval", 100) - 1

        self.mixer: Optional[LLMQMixer] = None
        self.target_mixer: Optional[LLMQMixer] = None
        self.belief_encoder: Optional[BeliefEncoder] = None
        self.target_belief_encoder: Optional[BeliefEncoder] = None
        self.target_mac = None

        self.belief_net_params: List = []
        self.encoder_params: List = []
        self.mixer_params: List = []

        self.belief_optimizer: Optional[Adam] = None
        self.encoder_optimizer: Optional[Adam] = None
        self.mixer_optimizer: Optional[Adam] = None

        self.gamma = getattr(args, "gamma", 0.99)
        self.lambda_e = getattr(args, "lambda_e", 0.1)
        self.lambda_sd = getattr(args, "lambda_sd", 0.1)
        self.lambda_m = getattr(args, "lambda_m", 0.1)
        self.lambda_belief = getattr(args.loss, "belief_weight", 0.1) if hasattr(args, 'loss') else 0.1
        self.z_loss_weight = getattr(args, "z_loss_weight", 0.0)
        self.z_transition_loss_weight = getattr(args, "z_transition_loss_weight", 0.0)
        self.z_head: Optional[nn.Module] = None
        self.train_encoder_only: bool = bool(getattr(args, "train_encoder_only", False))
        self.train_belief_supervised: bool = bool(getattr(args, "train_belief_supervised", False))
        self.freeze_belief_encoder_in_supervised: bool = bool(
            getattr(args, "freeze_belief_encoder_in_supervised", False)
        )
        self.belief_supervised_optimizer: Optional[Adam] = None
        
        self.bne_max_iterations = getattr(args, "bne_max_iterations", 5)
        self.bne_convergence_threshold = getattr(args, "bne_convergence_threshold", 0.01)
        self.stage2_weight = getattr(args, "stage2_weight", 0.3)  # Stage 2在总损失中的权重
        
        self._initialize_networks_and_optimizers(args)

    def _unwrap_module(self, m: Any) -> Any:
        """Unwrap DDP-wrapped modules for attribute access (e.g., .module)."""
        return getattr(m, "module", m)

    def _get_agent_module(self) -> Any:
        """
        Get underlying agent module (handles DDP wrapping and BasicMAC helpers).
        This is required for accessing custom attributes like last_s3b_bias_alpha / s3b_bias_alpha_param.
        """
        try:
            am = getattr(self.mac, "agent_module", None)
            if am is not None:
                return am
        except Exception:
            pass
        return self._unwrap_module(getattr(self.mac, "agent", None))

    def _initialize_networks_and_optimizers(self, args: Any):
        if getattr(args, "use_mixer", True):
            self.mixer = LLMQMixer(args)
            self.target_mixer = LLMQMixer(args)
            self.mixer_params = list(self.mixer.parameters())
            self.logger.info(f"Mixer initialized with {len(self.mixer_params)} parameters.")
        else:
            self.mixer = None
            self.target_mixer = None
            self.mixer_params = []
            self.logger.info("Mixer is disabled.")

        if hasattr(self.mac, 'belief_encoder') and self.mac.belief_encoder is not None:
            self.belief_encoder = self.mac.belief_encoder
            self.logger.info("Using BeliefEncoder from MAC.")
        elif getattr(args, "use_belief_encoder", True):
            self.belief_encoder = BeliefEncoder(args)
            self.logger.info("Initialized standalone BeliefEncoder.")
        else:
            self.belief_encoder = None
            self.logger.info("BeliefEncoder is disabled.")
        
        if self.belief_encoder is not None:
            if self.train_belief_supervised and self.freeze_belief_encoder_in_supervised:
                try:
                    for p in self.belief_encoder.parameters():
                        p.requires_grad = False
                    self.logger.info("Froze BeliefEncoder parameters for supervised belief training (Stage1/2).")
                except Exception as e:
                    self.logger.warning(f"Failed to freeze BeliefEncoder params in supervised mode: {e}")

            if self.train_encoder_only and bool(getattr(args, "train_population_update_head_only", False)):
                try:
                    for p in self.belief_encoder.parameters():
                        p.requires_grad = False

                    puh = getattr(self.belief_encoder, "population_update_head", None)
                    if puh is None:
                        raise RuntimeError("train_population_update_head_only=True but belief_encoder.population_update_head is None")
                    for p in puh.parameters():
                        p.requires_grad = True

                    if bool(getattr(self.belief_encoder, "population_update_use_stage", False)):
                        se = getattr(self.belief_encoder, "stage_embed", None)
                        if se is None:
                            raise RuntimeError(
                                "population_update_use_stage=True but belief_encoder.stage_embed is None. "
                                "Please ensure BeliefEncoder initializes stage_embed when stage conditioning is enabled."
                            )
                        for p in se.parameters():
                            p.requires_grad = True

                    if bool(getattr(args, "train_brief_encoder_in_stage3a", False)):
                        br = getattr(self.belief_encoder, "brief_encoder", None)
                        if br is not None:
                            for p in br.parameters():
                                p.requires_grad = True
                            self.logger.info("Stage3a: Unfroze belief_encoder.brief_encoder parameters.")
                        else:
                            self.logger.warning("Stage3a: train_brief_encoder_in_stage3a=True but belief_encoder.brief_encoder is None.")

                    mix_logit = getattr(self.belief_encoder, "population_update_mix_logit", None)
                    if isinstance(mix_logit, torch.nn.Parameter):
                        mix_logit.requires_grad = True

                    self.logger.info("Stage3a: Froze encoder except population_update_head (+optional stage_embed/+brief_encoder/+mix gate).")
                except Exception as e:
                    self.logger.warning(f"Stage3a: Failed to apply train_population_update_head_only freezing: {e}")

                try:
                    agent = self._get_agent_module()
                    if agent is not None and hasattr(agent, "parameters"):
                        for p in agent.parameters():
                            p.requires_grad = False
                        self.logger.info("Stage3a: Froze mac.agent parameters (encoder-only head training).")
                except Exception as e:
                    self.logger.warning(f"Stage3a: Failed to freeze mac.agent params: {e}")

            if (not self.train_belief_supervised) and (not self.train_encoder_only) and bool(getattr(args, "freeze_belief_encoder_in_rl", False)):
                try:
                    for p in self.belief_encoder.parameters():
                        p.requires_grad = False
                    self.logger.info("Stage4: Froze BeliefEncoder parameters in RL mode.")
                except Exception as e:
                    self.logger.warning(f"Stage4: Failed to freeze BeliefEncoder in RL: {e}")

            self.encoder_params = [p for p in self.belief_encoder.parameters() if p.requires_grad]
            self.target_belief_encoder = copy.deepcopy(self.belief_encoder)
            self.logger.info(f"BeliefEncoder trainable params: {len(self.encoder_params)}")
        else:
            self.encoder_params = []
            self.target_belief_encoder = None

        if self.z_loss_weight and self.z_loss_weight > 0 and self.belief_encoder is not None:
            belief_dim = getattr(args, "belief_dim", 128)
            self.z_head = nn.Linear(belief_dim, 3).to(self.device)
            self.encoder_params.extend(list(self.z_head.parameters()))
            self.logger.info("Initialized z_head for edge-population latent z supervision.")
            
        self.target_mac = copy.deepcopy(self.mac)

        self.belief_net_params = []
        if hasattr(self.mac, 'agents') and (isinstance(self.mac.agents, list) or isinstance(self.mac.agents, nn.ModuleList)):
            for agent_module in self.mac.agents:
                if hasattr(agent_module, 'belief_network') and agent_module.belief_network is not None:
                    self.belief_net_params.extend(list(agent_module.belief_network.parameters()))
                else:
                    self.logger.warning("An agent module in mac.agents is missing 'belief_network' or it's None.")
        elif hasattr(self.mac, 'agent') and hasattr(self.mac.agent, 'belief_network') and self.mac.agent.belief_network is not None: 
            self.logger.info("Treating mac.agent as the single BeliefNetwork provider.")
            self.belief_net_params.extend(list(self.mac.agent.belief_network.parameters()))
        else:
            self.logger.error("ECONLearner: Could not find belief_network parameters in MAC structure. BeliefNetwork losses might not work.")

        if (not self.train_belief_supervised) and (not self.train_encoder_only):
            agent = self._get_agent_module()
            try:
                if bool(getattr(args, "freeze_belief_network_in_rl", False)) and agent is not None and hasattr(agent, "belief_network") and agent.belief_network is not None:
                    for p in agent.belief_network.parameters():
                        p.requires_grad = False
                    self.logger.info("Stage4: Froze agent.belief_network parameters in RL mode.")
            except Exception as e:
                self.logger.warning(f"Stage4: Failed to freeze belief_network in RL: {e}")
            try:
                if bool(getattr(args, "freeze_stance_head_in_rl", False)) and agent is not None and hasattr(agent, "stance_head") and agent.stance_head is not None:
                    for p in agent.stance_head.parameters():
                        p.requires_grad = False
                    self.logger.info("Stage4: Froze agent.stance_head parameters in RL mode.")
            except Exception as e:
                self.logger.warning(f"Stage4: Failed to freeze stance_head in RL: {e}")
            try:
                if bool(getattr(args, "freeze_action_type_head_in_rl", False)) and agent is not None and hasattr(agent, "action_type_head") and agent.action_type_head is not None:
                    for p in agent.action_type_head.parameters():
                        p.requires_grad = False
                    self.logger.info("Stage4: Froze agent.action_type_head parameters in RL mode.")
            except Exception as e:
                self.logger.warning(f"Stage4: Failed to freeze action_type_head in RL: {e}")

        try:
            train_heads_rl = bool(getattr(args, "train_policy_heads_in_rl", True))
            if train_heads_rl and (not self.train_belief_supervised) and (not self.train_encoder_only):
                agent = self._get_agent_module()
                if agent is not None:
                    if hasattr(agent, "action_type_head") and getattr(agent, "action_type_head") is not None:
                        self.belief_net_params.extend(list(agent.action_type_head.parameters()))
                    if bool(getattr(args, "train_stance_head_in_rl", True)):
                        if hasattr(agent, "stance_head") and getattr(agent, "stance_head") is not None:
                            self.belief_net_params.extend(list(agent.stance_head.parameters()))
                    try:
                        p = getattr(agent, "s3b_bias_alpha_param", None)
                        if isinstance(p, torch.nn.Parameter) and bool(getattr(p, "requires_grad", False)):
                            self.belief_net_params.append(p)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning(f"Failed to include policy head params for RL training: {e}")

        self.belief_optimizer = None
        try:
            belief_lr = float(getattr(args, "belief_net_lr", getattr(args, "lr", 0.0)))
        except Exception:
            belief_lr = float(getattr(args, "lr", 0.0) or 0.0)

        if self.belief_net_params and belief_lr > 0:
            trainable_belief_params = [p for p in self.belief_net_params if getattr(p, "requires_grad", False)]
            if trainable_belief_params:
                self.belief_optimizer = Adam(
                    params=trainable_belief_params,
                    lr=belief_lr,
                    weight_decay=getattr(args, "weight_decay", 0.0),
                )
            else:
                self.logger.info(
                    "Belief optimizer skipped: no trainable belief/policy parameters (all requires_grad=False)."
                )
        else:
            if belief_lr <= 0:
                self.logger.info(f"Belief optimizer skipped: belief_net_lr={belief_lr} <= 0.")
        
        self.encoder_optimizer = None
        try:
            encoder_lr = float(getattr(args, "encoder_lr", getattr(args, "lr", 0.0)))
        except Exception:
            encoder_lr = float(getattr(args, "lr", 0.0) or 0.0)
        if self.encoder_params and self.belief_encoder and encoder_lr > 0:
            trainable_encoder_params = [p for p in self.encoder_params if getattr(p, "requires_grad", False)]
            if trainable_encoder_params:
                self.encoder_optimizer = Adam(
                    params=trainable_encoder_params,
                    lr=encoder_lr,
                    weight_decay=getattr(args, "weight_decay", 0.0),
                )
            else:
                self.logger.warning("Encoder optimizer skipped: no trainable encoder parameters (all requires_grad=False).")
        else:
            if encoder_lr <= 0:
                self.logger.info(f"Encoder optimizer skipped: encoder_lr={encoder_lr} <= 0.")
        
        self.mixer_optimizer = None
        if self.mixer_params and self.mixer:
            if (not self.train_belief_supervised) and (not self.train_encoder_only) and bool(getattr(args, "freeze_mixer_in_rl", False)):
                try:
                    for p in self.mixer.parameters():
                        p.requires_grad = False
                    self.logger.info("Stage4: Froze mixer parameters in RL mode.")
                except Exception as e:
                    self.logger.warning(f"Stage4: Failed to freeze mixer in RL: {e}")
                self.mixer_params = [p for p in self.mixer.parameters() if p.requires_grad]

            try:
                mixer_lr = float(getattr(args, "mixer_lr", getattr(args, "lr", 0.0)))
            except Exception:
                mixer_lr = float(getattr(args, "lr", 0.0) or 0.0)
            if mixer_lr > 0:
                trainable_mixer_params = [p for p in self.mixer_params if getattr(p, "requires_grad", False)]
                if trainable_mixer_params:
                    self.mixer_optimizer = Adam(
                        params=trainable_mixer_params,
                        lr=mixer_lr,
                        weight_decay=getattr(args, "weight_decay", 0.0),
                    )
                else:
                    self.logger.info("Mixer optimizer skipped: no trainable mixer parameters (all requires_grad=False).")
            else:
                self.logger.info(f"Mixer optimizer skipped: mixer_lr={mixer_lr} <= 0.")

        if self.train_belief_supervised:
            try:
                agent = getattr(self.mac, "agent", None)
                if agent is not None and hasattr(agent, "parameters"):
                    if bool(getattr(args, "train_action_imitation", False)):
                        try:
                            for p in agent.parameters():
                                p.requires_grad = False

                            if hasattr(agent, "action_type_head") and getattr(agent, "action_type_head") is not None:
                                for p in agent.action_type_head.parameters():
                                    p.requires_grad = True
                            else:
                                raise RuntimeError("train_action_imitation=True but agent.action_type_head is missing/None")

                            try:
                                if bool(getattr(args, "use_population_belief_in_action_head", False)):
                                    if hasattr(agent, "population_belief_proj") and getattr(agent, "population_belief_proj") is not None:
                                        for p in agent.population_belief_proj.parameters():
                                            p.requires_grad = True
                                    if hasattr(agent, "population_belief_gate_logit") and getattr(agent, "population_belief_gate_logit") is not None:
                                        agent.population_belief_gate_logit.requires_grad = True
                                    self.logger.info("Stage3b: Unfroze population_belief_proj/gate for z_t-conditioned action imitation.")
                            except Exception:
                                pass

                            if hasattr(agent, "belief_network") and getattr(agent, "belief_network") is not None:
                                for p in agent.belief_network.parameters():
                                    p.requires_grad = False
                            if hasattr(agent, "stance_head") and getattr(agent, "stance_head") is not None:
                                for p in agent.stance_head.parameters():
                                    p.requires_grad = False

                            self.logger.info("Stage3b: Froze mac.agent except action_type_head (offline action imitation).")
                        except Exception as e:
                            self.logger.warning(f"Stage3b: Failed to apply train_action_imitation freezing: {e}")

                    self.belief_supervised_optimizer = Adam(
                        params=[p for p in agent.parameters() if getattr(p, "requires_grad", False)],
                        lr=belief_lr,
                        weight_decay=getattr(args, "weight_decay", 0.0),
                    )
                    self.logger.info("Initialized belief_supervised_optimizer for offline classification training.")
            except Exception as e:
                self.logger.warning(f"Failed to init belief_supervised_optimizer: {e}")

        if self.mixer is None:
            self.logger.warning("ECONLearner: Mixer is None. Global Q-value calculation and related losses will be skipped during training.")
        if self.belief_encoder is None:
            self.logger.warning("ECONLearner: BeliefEncoder is None. Group representation E and related losses will be skipped.")
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> Dict:
        """
        Train the ECON framework using the provided batch data with two-stage coordination.
        
        Args:
            batch: Episode batch data
            t_env: Current environment timestep
            episode_num: Current episode number
            
        Returns:
            Dictionary containing training statistics
        """
        rewards = batch["reward"][:, :-1].to(self.device)
        terminated = batch["terminated"][:, :-1].float().to(self.device)
        mask = batch["filled"][:, :-1].float().to(self.device)
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.train_belief_supervised:
            if "gt_action" not in batch.scheme:
                self.logger.warning("train_belief_supervised=True but gt_action not in batch.scheme; skipping.")
                return {"status": "skipped_no_gt_action"}
            if self.belief_supervised_optimizer is None:
                self.logger.warning("train_belief_supervised=True but belief_supervised_optimizer is None; skipping.")
                return {"status": "skipped_no_supervised_optimizer"}

            if hasattr(self.mac, 'init_hidden'):
                self.mac.init_hidden(batch.batch_size)

            try:
                belief_sup_micro_bs = int(getattr(self.args, "belief_supervised_micro_batch_size", 0))
            except Exception:
                belief_sup_micro_bs = 0
            belief_sup_micro_bs = max(0, belief_sup_micro_bs)

            ce_weight = None
            ce_weight_list = None
            try:
                w = getattr(self.args, "belief_supervised_class_weights", None)
                if isinstance(w, (list, tuple)) and len(w) > 0:
                    ce_weight_list = [float(x) for x in w]
                    ce_weight = torch.tensor([float(x) for x in w], device=self.device, dtype=torch.float32)
            except Exception:
                ce_weight = None
                ce_weight_list = None

            total_loss = torch.tensor(0.0, device=self.device)
            total_correct = 0.0
            total_correct_prob = 0.0
            total_count = 0.0
            possible_count = 0.0
            supervised_count = 0.0
            soft_available_steps = 0  # timesteps where gt_action_dist has any valid mass
            soft_used_steps = 0       # timesteps where we actually used soft CE (not fallback hard CE)
            soft_p1_sum = 0.0         # mean mass on class-1 across valid rows (helps detect Oppose signal)
            soft_p_count = 0
            dbg_entropy_sum = 0.0
            dbg_entropy_count = 0
            dbg_maxprob_sum = 0.0
            dbg_maxprob_count = 0
            dbg_logit_abs_sum = 0.0
            dbg_logit_std_sum = 0.0
            dbg_logit_count = 0
            dbg_p0_sum = 0.0
            dbg_p0_count = 0
            dbg_p1_sum = 0.0
            dbg_p1_count = 0
            dbg_p0_gt05_sum = 0.0
            dbg_p0_gt05_count = 0
            dbg_hard_pred0_sum = 0.0
            dbg_hard_pred0_count = 0
            dbg_delta01_sum = 0.0
            dbg_delta01_count = 0
            pred_counts = None  # type: ignore
            gt_counts = None  # type: ignore
            correct_counts = None  # type: ignore
            self.belief_supervised_optimizer.zero_grad()

            s3b_binary_01 = False
            try:
                if bool(getattr(self.args, "train_action_imitation", False)):
                    s3b_binary_01 = bool(getattr(self.args, "action_imitation_binary_01", False))
            except Exception:
                s3b_binary_01 = False
            try:
                s3b_preference_scorer = bool(getattr(self.args, "s3b_preference_scorer", False))
            except Exception:
                s3b_preference_scorer = False

            use_argmax_metrics = True
            try:
                use_argmax_metrics = bool(getattr(self.args, "belief_supervised_metrics_use_argmax", True))
            except Exception:
                use_argmax_metrics = True

            for t in range(batch.max_seq_length - 1):
                bs_total = int(batch.batch_size)
                if belief_sup_micro_bs <= 0 or belief_sup_micro_bs >= bs_total:
                    spans = [(0, bs_total)]
                else:
                    spans = [(i, min(i + belief_sup_micro_bs, bs_total)) for i in range(0, bs_total, belief_sup_micro_bs)]

                m_t_full = mask[:, t]
                if m_t_full.ndim > 1:
                    m_t_full = m_t_full.view(bs_total)
                sup_full = None
                try:
                    if bool(getattr(self.args, "train_action_imitation", False)):
                        only_ids = getattr(self.args, "action_imitation_supervised_action_ids", None)
                        if isinstance(only_ids, (list, tuple)) and len(only_ids) > 0:
                            y_full = batch["gt_action"][:, t].to(self.device)
                            if y_full.ndim > 1:
                                y_full = y_full.view(bs_total)
                            y_full = y_full.long()
                            sup_full = torch.zeros_like(y_full, dtype=torch.bool)
                            for _i in only_ids:
                                try:
                                    ii = int(_i)
                                except Exception:
                                    continue
                                sup_full |= (y_full == ii)
                except Exception:
                    sup_full = None

                if isinstance(sup_full, torch.Tensor):
                    total_w = torch.clamp((m_t_full.float() * sup_full.float()).sum() * float(self.args.n_agents), min=1.0).to(self.device)
                else:
                    total_w = torch.clamp(m_t_full.float().sum() * float(self.args.n_agents), min=1.0).to(self.device)
                try:
                    possible_count += float((m_t_full.float().sum() * float(self.args.n_agents)).item())
                except Exception:
                    pass

                loss_t_val = 0.0

                for (s0, s1) in spans:
                    b_slice = batch[slice(s0, s1)]
                    agent_logits, _info = self.mac.forward(b_slice, t, train_mode=True)
                    if not isinstance(agent_logits, torch.Tensor) or agent_logits.ndim != 3:
                        continue
                    if not torch.isfinite(agent_logits).all():
                        self.logger.warning(f"belief_supervised: non-finite agent_logits at t={t}; skipping slice {s0}:{s1}.")
                        continue
                    bs, na, nc = agent_logits.shape

                    y = b_slice["gt_action"][:, t].to(self.device)
                    if y.ndim > 1:
                        y = y.view(bs)
                    y = torch.clamp(y.long(), min=0, max=max(0, int(nc) - 1))

                    logits_flat = agent_logits.reshape(bs * na, nc)
                    nc_eff = int(nc)
                    ce_weight_eff = ce_weight
                    if bool(s3b_binary_01):
                        nc_eff = 2
                        logits_flat = logits_flat[:, :2]
                        y = torch.clamp(y, min=0, max=1)
                        try:
                            if isinstance(ce_weight, torch.Tensor) and ce_weight.numel() >= 2:
                                ce_weight_eff = ce_weight[:2]
                            else:
                                ce_weight_eff = None
                        except Exception:
                            ce_weight_eff = None
                    y_exp = y.unsqueeze(1).expand(bs, na).reshape(-1)

                    try:
                        with torch.no_grad():
                            p = F.softmax(logits_flat, dim=-1)
                            ent = (-p * torch.log(torch.clamp(p, min=1e-12))).sum(dim=-1)
                            mx = p.max(dim=-1)[0]
                            if torch.isfinite(ent).all():
                                dbg_entropy_sum += float(ent.mean().item())
                                dbg_entropy_count += 1
                            if torch.isfinite(mx).all():
                                dbg_maxprob_sum += float(mx.mean().item())
                                dbg_maxprob_count += 1
                            if torch.isfinite(logits_flat).all():
                                dbg_logit_abs_sum += float(logits_flat.abs().mean().item())
                                dbg_logit_std_sum += float(logits_flat.std(dim=-1).mean().item())
                                dbg_logit_count += 1
                    except Exception:
                        pass

                    m_t = m_t_full[s0:s1]
                    w = m_t.float().unsqueeze(1).expand(bs, na).reshape(-1)
                    if isinstance(sup_full, torch.Tensor):
                        sup_s = sup_full[s0:s1].to(self.device)
                        sup_w = sup_s.float().unsqueeze(1).expand(bs, na).reshape(-1)
                        w = w * sup_w

                    use_soft = bool(getattr(self.args, "belief_supervised_use_soft_labels", False)) and ("gt_action_dist" in batch.scheme)
                    try:
                        label_smoothing = float(getattr(self.args, "belief_supervised_label_smoothing", 0.0) or 0.0)
                    except Exception:
                        label_smoothing = 0.0
                    if use_soft:
                        try:
                            psl = b_slice["gt_action_dist"][:, t].to(self.device)
                            if psl.ndim == 1:
                                psl = psl.view(bs, -1)
                            if psl.shape[-1] != nc:
                                if psl.shape[-1] > nc:
                                    psl = psl[:, :nc]
                                else:
                                    pad = torch.zeros(bs, nc - psl.shape[-1], device=self.device, dtype=psl.dtype)
                                    psl = torch.cat([psl, pad], dim=-1)
                            if bool(s3b_binary_01):
                                psl = psl[:, :2]
                                if psl.shape[-1] != 2:
                                    pad = torch.zeros(bs, 2 - psl.shape[-1], device=self.device, dtype=psl.dtype)
                                    psl = torch.cat([psl, pad], dim=-1)
                            psl = torch.clamp(psl.float(), min=0.0)
                            ps = psl.sum(dim=-1, keepdim=True)
                            valid = (ps.squeeze(-1) > 0)
                            if (not bool(valid.any().item())) and label_smoothing > 0.0:
                                k = int(psl.shape[-1])
                                k = max(2, k)
                                y2 = y.clamp(min=0, max=k - 1).view(-1)
                                psl = torch.full((bs, k), fill_value=label_smoothing / float(k - 1), device=self.device, dtype=torch.float32)
                                psl.scatter_(1, y2.view(-1, 1), 1.0 - label_smoothing)
                                ps = psl.sum(dim=-1, keepdim=True)
                                valid = torch.ones_like(ps.squeeze(-1), dtype=torch.bool)
                            if bool(valid.any().item()):
                                soft_available_steps += 1
                                try:
                                    if nc > 1:
                                        soft_p1_sum += float(psl[valid, 1].mean().item())
                                        soft_p_count += 1
                                except Exception:
                                    pass
                                psl = torch.where(ps > 0, psl / ps, torch.full_like(psl, 1.0 / float(psl.shape[-1])))
                                try:
                                    mix = float(getattr(self.args, "belief_supervised_soft_label_mix", 1.0))
                                except Exception:
                                    mix = 1.0
                                mix = float(max(0.0, min(1.0, mix)))
                                if mix < 1.0:
                                    y_oh = torch.zeros_like(psl)
                                    y_oh.scatter_(1, y.view(-1, 1).clamp(min=0, max=psl.shape[-1]-1), 1.0)
                                    psl = mix * psl + (1.0 - mix) * y_oh
                                if bool(s3b_binary_01) and bool(s3b_preference_scorer):
                                    bias_logit = logits_flat[:, 1] - logits_flat[:, 0]
                                    p1_target = psl[:, 1].clamp(0.0, 1.0)
                                    p1_exp = p1_target.unsqueeze(1).expand(bs, na).reshape(-1)
                                    bce = F.binary_cross_entropy_with_logits(bias_logit, p1_exp, reduction="none")
                                    if torch.isfinite(bce).all():
                                        loss_sum = (bce * w).sum()
                                        (loss_sum / total_w).backward()
                                        loss_t_val += float((loss_sum.detach() / total_w).item())
                                        soft_used_steps += 1
                                        loss_used = True
                                    else:
                                        loss_used = False
                                else:
                                    p_exp = psl.unsqueeze(1).expand(bs, na, psl.shape[-1]).reshape(bs * na, psl.shape[-1])
                                    logp = F.log_softmax(logits_flat, dim=-1)
                                    if bool(s3b_binary_01) and logp.shape[-1] != p_exp.shape[-1]:
                                        logp = logp[:, : p_exp.shape[-1]]
                                    s_ce = -(p_exp * logp).sum(dim=-1)
                                    if torch.isfinite(s_ce).all():
                                        loss_sum = (s_ce * w).sum()
                                        (loss_sum / total_w).backward()
                                        loss_t_val += float((loss_sum.detach() / total_w).item())
                                        soft_used_steps += 1
                                        loss_used = True
                                    else:
                                        loss_used = False
                            else:
                                loss_used = False
                        except Exception:
                            loss_used = False
                    else:
                        loss_used = False

                    if not loss_used:
                        if bool(s3b_binary_01) and bool(s3b_preference_scorer):
                            bias_logit = logits_flat[:, 1] - logits_flat[:, 0]
                            y_bin = y_exp.float().clamp(0.0, 1.0)
                            ce = F.binary_cross_entropy_with_logits(bias_logit, y_bin, reduction="none")
                        else:
                            ce = F.cross_entropy(logits_flat, y_exp, reduction="none", weight=ce_weight_eff)
                        if not torch.isfinite(ce).all():
                            self.logger.warning(f"belief_supervised: non-finite CE at t={t}; skipping slice {s0}:{s1}.")
                            continue
                        loss_sum = (ce * w).sum()
                        (loss_sum / total_w).backward()
                        loss_t_val += float((loss_sum.detach() / total_w).item())

                    total_count += float(w.sum().item())
                    supervised_count += float(w.sum().item())
                    try:
                        with torch.no_grad():
                            m = (w > 0.5)
                            try:
                                if bool(m.any().item()):
                                    p_all = F.softmax(logits_flat, dim=-1)
                                    idx = torch.arange(p_all.shape[0], device=self.device)
                                    p_true = p_all[idx, y_exp]
                                    total_correct_prob += float((p_true * w).sum().item())
                            except Exception:
                                pass

                            if use_argmax_metrics:
                                pred = logits_flat.argmax(dim=-1)
                                total_correct += float(((pred == y_exp).float() * w).sum().item())
                            if pred_counts is None:
                                pred_counts = torch.zeros(nc_eff, device=self.device, dtype=torch.float32)
                            if gt_counts is None:
                                gt_counts = torch.zeros(nc_eff, device=self.device, dtype=torch.float32)
                            if correct_counts is None:
                                correct_counts = torch.zeros(nc_eff, device=self.device, dtype=torch.float32)
                            if bool(m.any().item()):
                                try:
                                    logits_m = logits_flat[m]
                                    if isinstance(logits_m, torch.Tensor) and logits_m.ndim == 2 and logits_m.shape[-1] >= 2:
                                        pm = F.softmax(logits_m, dim=-1)
                                        dbg_p0_sum += float(pm[:, 0].mean().item())
                                        dbg_p0_count += 1
                                        dbg_p1_sum += float(pm[:, 1].mean().item())
                                        dbg_p1_count += 1
                                        dbg_p0_gt05_sum += float((pm[:, 0] > 0.5).float().mean().item())
                                        dbg_p0_gt05_count += 1
                                        dbg_hard_pred0_sum += float((logits_m[:, 0] > logits_m[:, 1]).float().mean().item())
                                        dbg_hard_pred0_count += 1
                                        dbg_delta01_sum += float((logits_m[:, 0] - logits_m[:, 1]).mean().item())
                                        dbg_delta01_count += 1
                                except Exception:
                                    pass
                                gc = torch.bincount(y_exp[m], minlength=nc_eff).float()
                                gt_counts[: gc.numel()] += gc
                                if use_argmax_metrics:
                                    pc = torch.bincount(pred[m], minlength=nc_eff).float()
                                    pred_counts[: pc.numel()] += pc
                                    corr = (pred == y_exp) & m
                                    if bool(corr.any().item()):
                                        cc = torch.bincount(y_exp[corr], minlength=nc_eff).float()
                                        correct_counts[: cc.numel()] += cc
                    except Exception:
                        pass

                total_loss = total_loss + torch.tensor(float(loss_t_val), device=self.device, dtype=torch.float32)

            steps = max(1, int(batch.max_seq_length - 1))
            total_loss = total_loss / float(steps)

            if float(total_count) <= 0.0:
                self.logger.warning("belief_supervised: effective_count==0 after label-masking; skipping optimizer step.")
                try:
                    self.belief_supervised_optimizer.zero_grad(set_to_none=True)
                except Exception:
                    self.belief_supervised_optimizer.zero_grad()
                return {
                    "status": "belief_supervised_skipped_no_labeled",
                    "loss_total": 0.0,
                    "loss_belief": 0.0,
                    "loss_encoder": 0.0,
                    "loss_mixer": 0.0,
                    "belief_sup_acc": 0.0,
                    "belief_sup_effective_count": 0.0,
                    "reward_mean": float(rewards.mean().item()) if rewards.numel() > 0 else 0.0,
                }

            if not torch.isfinite(total_loss).all():
                self.logger.warning("belief_supervised: total_loss is NaN/Inf; skipping optimizer step for this batch.")
                try:
                    self.belief_supervised_optimizer.zero_grad(set_to_none=True)
                except Exception:
                    self.belief_supervised_optimizer.zero_grad()
                return {
                    "status": "belief_supervised_skipped_nan",
                    "loss_total": float("nan"),
                    "loss_belief": float("nan"),
                    "loss_encoder": 0.0,
                    "loss_mixer": 0.0,
                    "belief_sup_acc": float(acc) if "acc" in locals() else 0.0,
                    "reward_mean": float(rewards.mean().item()) if rewards.numel() > 0 else 0.0,
                }

            try:
                agent = getattr(self.mac, "agent", None)
                if agent is not None and hasattr(agent, "parameters"):
                    torch.nn.utils.clip_grad_norm_(list(agent.parameters()), 10.0)
            except Exception:
                pass
            grad_norm = 0.0
            try:
                agent = getattr(self.mac, "agent", None)
                if agent is not None and hasattr(agent, "parameters"):
                    s = 0.0
                    for p in agent.parameters():
                        if p is None or (not isinstance(p, torch.Tensor)) or (p.grad is None):
                            continue
                        g = p.grad
                        if not torch.isfinite(g).all():
                            continue
                        s += float(g.detach().float().pow(2).sum().item())
                    grad_norm = float(s ** 0.5)
            except Exception:
                grad_norm = 0.0
            self.belief_supervised_optimizer.step()

            if use_argmax_metrics:
                acc = (total_correct / max(1.0, total_count)) if total_count > 0 else 0.0
            else:
                acc = (total_correct_prob / max(1.0, total_count)) if total_count > 0 else 0.0
            try:
                nc_stats = int(pred_counts.numel()) if isinstance(pred_counts, torch.Tensor) else int(nc)
            except Exception:
                nc_stats = int(nc)
            pred_frac = [float("nan")] * int(nc_stats)
            gt_frac = [float("nan")] * int(nc_stats)
            recall = [float("nan")] * int(nc_stats)
            precision = [float("nan")] * int(nc_stats)
            pred_cnt = [0.0] * int(nc_stats)
            gt_cnt = [0.0] * int(nc_stats)
            correct_cnt = [0.0] * int(nc_stats)
            has_gt = [0.0] * int(nc_stats)
            try:
                if use_argmax_metrics:
                    if isinstance(pred_counts, torch.Tensor) and pred_counts.sum().item() > 0:
                        pf = (pred_counts / pred_counts.sum()).detach().cpu().tolist()
                        pred_frac = [float(x) for x in pf]
                else:
                    try:
                        if dbg_p0_count > 0 and dbg_p1_count > 0:
                            p0 = float(dbg_p0_sum / max(1, dbg_p0_count))
                            p1 = float(dbg_p1_sum / max(1, dbg_p1_count))
                            pred_frac = [p0, p1] + ([float("nan")] * max(0, int(nc_stats) - 2))
                    except Exception:
                        pass
                if isinstance(gt_counts, torch.Tensor) and gt_counts.sum().item() > 0:
                    gf = (gt_counts / gt_counts.sum()).detach().cpu().tolist()
                    gt_frac = [float(x) for x in gf]
                if (not use_argmax_metrics) and isinstance(gt_counts, torch.Tensor):
                    try:
                        eff_total = float(gt_counts.sum().item())
                        for i in range(int(nc_stats)):
                            if i < int(gt_counts.numel()):
                                gti = float(gt_counts[i].item())
                                gt_cnt[i] = gti
                                has_gt[i] = 1.0 if gti > 0 else 0.0
                            if i < len(pred_frac):
                                pi = float(pred_frac[i])
                                if pi == pi and eff_total > 0:
                                    pred_cnt[i] = float(pi * eff_total)
                    except Exception:
                        pass
                if use_argmax_metrics and isinstance(correct_counts, torch.Tensor):
                    if isinstance(gt_counts, torch.Tensor):
                        for i in range(int(nc_stats)):
                            gti = float(gt_counts[i].item())
                            ci = float(correct_counts[i].item())
                            gt_cnt[i] = gti
                            correct_cnt[i] = ci
                            has_gt[i] = 1.0 if gti > 0 else 0.0
                            recall[i] = (ci / gti) if gti > 0 else 0.0
                    if isinstance(pred_counts, torch.Tensor):
                        for i in range(int(nc_stats)):
                            pi = float(pred_counts[i].item())
                            ci = float(correct_counts[i].item())
                            pred_cnt[i] = pi
                            if correct_cnt[i] == 0.0:
                                correct_cnt[i] = ci
                            precision[i] = (ci / pi) if pi > 0 else 0.0
            except Exception:
                pass
            gap0 = float("nan")
            gap1 = float("nan")
            try:
                if len(pred_frac) > 0 and len(gt_frac) > 0:
                    if (pred_frac[0] == pred_frac[0]) and (gt_frac[0] == gt_frac[0]):
                        gap0 = float(pred_frac[0] - gt_frac[0])
                if len(pred_frac) > 1 and len(gt_frac) > 1:
                    if (pred_frac[1] == pred_frac[1]) and (gt_frac[1] == gt_frac[1]):
                        gap1 = float(pred_frac[1] - gt_frac[1])
            except Exception:
                gap0 = float("nan")
                gap1 = float("nan")
            z_gate = float("nan")
            try:
                agent = getattr(self.mac, "agent", None)
                g = getattr(agent, "population_belief_gate_logit", None) if agent is not None else None
                if isinstance(g, torch.Tensor):
                    z_gate = float(torch.sigmoid(g.detach()).float().item())
            except Exception:
                z_gate = float("nan")
            return {
                "status": "belief_supervised",
                "loss_total": float(total_loss.item()),
                "loss_belief": float(total_loss.item()),
                "loss_encoder": 0.0,
                "loss_mixer": 0.0,
                "belief_sup_acc": float(acc),
                "belief_sup_possible_count": float(possible_count),
                "belief_sup_supervised_count": float(supervised_count),
                "belief_sup_coverage": float(supervised_count / max(1.0, possible_count)) if possible_count > 0 else 0.0,
                "belief_sup_skipped_ratio": float(1.0 - (supervised_count / max(1.0, possible_count))) if possible_count > 0 else 0.0,
                "belief_sup_effective_count": float(total_count),
                "reward_mean": float(rewards.mean().item()) if rewards.numel() > 0 else 0.0,
                "belief_sup_grad_norm": float(grad_norm),
                "belief_sup_entropy": float(dbg_entropy_sum / max(1, dbg_entropy_count)) if dbg_entropy_count > 0 else float("nan"),
                "belief_sup_maxprob": float(dbg_maxprob_sum / max(1, dbg_maxprob_count)) if dbg_maxprob_count > 0 else float("nan"),
                "belief_sup_logit_abs_mean": float(dbg_logit_abs_sum / max(1, dbg_logit_count)) if dbg_logit_count > 0 else float("nan"),
                "belief_sup_logit_std": float(dbg_logit_std_sum / max(1, dbg_logit_count)) if dbg_logit_count > 0 else float("nan"),
                "belief_sup_p0_mean": float(dbg_p0_sum / max(1, dbg_p0_count)) if dbg_p0_count > 0 else float("nan"),
                "belief_sup_p1_mean": float(dbg_p1_sum / max(1, dbg_p1_count)) if dbg_p1_count > 0 else float("nan"),
                "belief_sup_p0_gt05_frac": float(dbg_p0_gt05_sum / max(1, dbg_p0_gt05_count)) if dbg_p0_gt05_count > 0 else float("nan"),
                "belief_sup_hard_pred0_frac": float(dbg_hard_pred0_sum / max(1, dbg_hard_pred0_count)) if dbg_hard_pred0_count > 0 else float("nan"),
                "belief_sup_delta01_mean": float(dbg_delta01_sum / max(1, dbg_delta01_count)) if dbg_delta01_count > 0 else float("nan"),
                "belief_sup_soft_available_frac": float(soft_available_steps / float(steps)) if steps > 0 else 0.0,
                "belief_sup_soft_used_frac": float(soft_used_steps / float(steps)) if steps > 0 else 0.0,
                "belief_sup_soft_p1_mean": float(soft_p1_sum / max(1, soft_p_count)) if soft_p_count > 0 else float("nan"),
                "belief_sup_pred0_frac": float(pred_frac[0]) if len(pred_frac) > 0 else float("nan"),
                "belief_sup_pred1_frac": float(pred_frac[1]) if len(pred_frac) > 1 else float("nan"),
                "belief_sup_pred2_frac": float(pred_frac[2]) if len(pred_frac) > 2 else float("nan"),
                "belief_sup_gt0_frac": float(gt_frac[0]) if len(gt_frac) > 0 else float("nan"),
                "belief_sup_gt1_frac": float(gt_frac[1]) if len(gt_frac) > 1 else float("nan"),
                "belief_sup_gt2_frac": float(gt_frac[2]) if len(gt_frac) > 2 else float("nan"),
                "belief_sup_marginal_gap0": float(gap0),
                "belief_sup_marginal_gap1": float(gap1),
                "belief_sup_z_gate": float(z_gate),
                "belief_sup_recall0": float(recall[0]) if len(recall) > 0 else float("nan"),
                "belief_sup_recall1": float(recall[1]) if len(recall) > 1 else float("nan"),
                "belief_sup_recall2": float(recall[2]) if len(recall) > 2 else float("nan"),
                "belief_sup_precision0": float(precision[0]) if len(precision) > 0 else float("nan"),
                "belief_sup_precision1": float(precision[1]) if len(precision) > 1 else float("nan"),
                "belief_sup_precision2": float(precision[2]) if len(precision) > 2 else float("nan"),
                "belief_sup_gt0_count": float(gt_cnt[0]) if len(gt_cnt) > 0 else 0.0,
                "belief_sup_gt1_count": float(gt_cnt[1]) if len(gt_cnt) > 1 else 0.0,
                "belief_sup_gt2_count": float(gt_cnt[2]) if len(gt_cnt) > 2 else 0.0,
                "belief_sup_pred0_count": float(pred_cnt[0]) if len(pred_cnt) > 0 else 0.0,
                "belief_sup_pred1_count": float(pred_cnt[1]) if len(pred_cnt) > 1 else 0.0,
                "belief_sup_pred2_count": float(pred_cnt[2]) if len(pred_cnt) > 2 else 0.0,
                "belief_sup_correct0_count": float(correct_cnt[0]) if len(correct_cnt) > 0 else 0.0,
                "belief_sup_correct1_count": float(correct_cnt[1]) if len(correct_cnt) > 1 else 0.0,
                "belief_sup_correct2_count": float(correct_cnt[2]) if len(correct_cnt) > 2 else 0.0,
                "belief_sup_has_gt0": float(has_gt[0]) if len(has_gt) > 0 else 0.0,
                "belief_sup_has_gt1": float(has_gt[1]) if len(has_gt) > 1 else 0.0,
                "belief_sup_has_gt2": float(has_gt[2]) if len(has_gt) > 2 else 0.0,
                "belief_sup_ce_w0": float(ce_weight_list[0]) if isinstance(ce_weight_list, list) and len(ce_weight_list) > 0 else float("nan"),
                "belief_sup_ce_w1": float(ce_weight_list[1]) if isinstance(ce_weight_list, list) and len(ce_weight_list) > 1 else float("nan"),
                "belief_sup_ce_w2": float(ce_weight_list[2]) if isinstance(ce_weight_list, list) and len(ce_weight_list) > 2 else float("nan"),
                "belief_sup_ce_w3": float(ce_weight_list[3]) if isinstance(ce_weight_list, list) and len(ce_weight_list) > 3 else float("nan"),
                "belief_sup_ce_w4": float(ce_weight_list[4]) if isinstance(ce_weight_list, list) and len(ce_weight_list) > 4 else float("nan"),
            }

        if self.train_encoder_only:
            if self.belief_encoder is None:
                self.logger.warning("train_encoder_only=True but belief_encoder is None; skipping.")
                return {"status": "skipped_encoder_none"}

            if hasattr(self.mac, 'init_hidden'):
                self.mac.init_hidden(batch.batch_size)

            group_repr_list = []
            for t in range(batch.max_seq_length - 1):
                _, mac_info_t = self.mac.forward(batch, t, train_mode=True)
                gr = mac_info_t.get("group_repr")
                if gr is None:
                    bs_t = mac_info_t.get("belief_states")
                    if bs_t is not None and callable(getattr(self.belief_encoder, "__call__", None)):
                        try:
                            gr = self.belief_encoder(bs_t)
                        except Exception:
                            gr = None
                if gr is None:
                    gr = torch.zeros(batch.batch_size, getattr(self.args, "belief_dim", 128), device=self.device)
                group_repr_list.append(gr)

            group_representation_seq = torch.stack(group_repr_list, dim=1)  # (bs, seq, belief_dim)

            encoder_loss = torch.tensor(0.0, device=self.device)
            z_loss = torch.tensor(0.0, device=self.device)
            z_tr_loss = torch.tensor(0.0, device=self.device)
            z_pred_minus_z_t_l2 = None
            z_target_minus_z_t_l2 = None
            z_pred_entropy = None
            z_pred_maxprob = None
            z_pred_p0 = None
            z_pred_p1 = None
            z_pred_p2 = None
            z_pred_delta_l2_by_stage = {}
            z_target_delta_l2_by_stage = {}
            pop_update_weight_l2 = None
            pop_update_grad_l2 = None

            if self.z_head is not None and "z_target" in batch.scheme and "z_mask" in batch.scheme:
                z_logits = self.z_head(group_representation_seq)  # (bs, seq, 3)
                z_logp = F.log_softmax(z_logits, dim=-1)
                z_target = batch["z_target"][:, :-1].to(self.device)  # (bs, seq, K)
                z_mask = batch["z_mask"][:, :-1].to(self.device)      # (bs, seq, 1)
                z_mask = z_mask * mask.unsqueeze(-1)
                z_target = torch.clamp(z_target, min=0.0)
                z_sum = z_target.sum(dim=-1, keepdim=True)
                if z_target.shape[-1] == 3:
                    z_target = torch.where(z_sum > 0, z_target / z_sum, torch.full_like(z_target, 1.0 / 3.0))
                kl = F.kl_div(z_logp, z_target, reduction="none").sum(dim=-1, keepdim=True)
                denom = torch.clamp(z_mask.sum(), min=1.0)
                z_loss = (kl * z_mask).sum() / denom
                encoder_loss = encoder_loss + self.z_loss_weight * z_loss

            try:
                if (
                    self.z_transition_loss_weight
                    and self.z_transition_loss_weight > 0
                    and hasattr(self.belief_encoder, "predict_next_population_belief")
                    and hasattr(self.belief_encoder, "compute_population_belief_loss")
                    and "z_target" in batch.scheme
                    and "z_mask" in batch.scheme
                    and ("z_t" in batch.scheme or "belief_pre_population_z" in batch.scheme)
                ):
                    z_t_seq = batch["z_t"][:, :-1].to(self.device) if "z_t" in batch.scheme else batch["belief_pre_population_z"][:, :-1].to(self.device)
                    z_target_seq = batch["z_target"][:, :-1].to(self.device)
                    z_mask_seq = batch["z_mask"][:, :-1].to(self.device) * mask.unsqueeze(-1)
                    stage_t_seq = batch["stage_t"][:, :-1].to(self.device) if "stage_t" in batch.scheme else None

                    bs, seq_len, k = z_t_seq.shape
                    z_t_flat = z_t_seq.reshape(bs * seq_len, k)
                    z_target_flat = z_target_seq.reshape(bs * seq_len, k)
                    z_mask_flat = z_mask_seq.reshape(bs * seq_len)
                    gr_flat = group_representation_seq.reshape(bs * seq_len, -1)
                    st_flat = stage_t_seq.reshape(bs * seq_len, -1) if stage_t_seq is not None else None

                    lt = str(getattr(self.args, "z_transition_loss_type", "kl") or "kl").strip().lower()
                    if lt.startswith("dirichlet"):
                        if not hasattr(self.belief_encoder, "predict_next_population_belief_alpha"):
                            raise RuntimeError("Dirichlet z_transition requested but BeliefEncoder lacks predict_next_population_belief_alpha().")
                        if not hasattr(self.belief_encoder, "compute_population_belief_loss_dirichlet_kl"):
                            raise RuntimeError("Dirichlet z_transition requested but BeliefEncoder lacks compute_population_belief_loss_dirichlet_kl().")
                        alpha_pred = self.belief_encoder.predict_next_population_belief_alpha(
                            z_t_flat,
                            group_repr=gr_flat,
                            stage_t=st_flat,
                        )
                        z_pred_flat = self.belief_encoder.population_belief_mean_from_alpha(alpha_pred)
                        alpha0_tgt_flat = None
                        try:
                            if "z_alpha0_target" in batch.scheme:
                                a0_seq = batch["z_alpha0_target"][:, :-1].to(self.device)  # (bs, seq, 1)
                                alpha0_tgt_flat = a0_seq.reshape(bs * seq_len)
                        except Exception:
                            alpha0_tgt_flat = None
                        if alpha0_tgt_flat is None:
                            alpha0_tgt_flat = float(getattr(self.args, "dirichlet_alpha0_target", 10.0))
                        z_tr_loss = self.belief_encoder.compute_population_belief_loss_dirichlet_kl(
                            alpha_pred,
                            z_target_flat,
                            z_mask_flat,
                            alpha0_target=alpha0_tgt_flat,
                        )
                        try:
                            zm = z_mask_flat.to(alpha_pred.device, dtype=alpha_pred.dtype).clamp(min=0.0, max=1.0)
                            denom = torch.clamp(zm.sum(), min=1.0)
                            a0 = alpha_pred.sum(dim=-1)
                            train_stats_alpha0 = (a0 * zm).sum() / denom
                            try:
                                k_dir = int(alpha_pred.shape[-1])
                                ap = torch.clamp(alpha_pred, min=float(getattr(self.belief_encoder, "dirichlet_alpha_min", 1e-6)))
                                ap0 = torch.clamp(ap.sum(dim=-1), min=1e-6)
                                logB = torch.sum(torch.lgamma(ap), dim=-1) - torch.lgamma(ap0)
                                ent = logB + (ap0 - float(k_dir)) * torch.digamma(ap0) - torch.sum((ap - 1.0) * torch.digamma(ap), dim=-1)
                                train_stats_dir_entropy = (ent * zm).sum() / denom
                                ap0u = ap0.unsqueeze(-1)
                                var = (ap * (ap0u - ap)) / (ap0u * ap0u * (ap0u + 1.0))
                                var_sum = var.sum(dim=-1)
                                train_stats_dir_varsum = (var_sum * zm).sum() / denom
                            except Exception:
                                train_stats_dir_entropy = None
                                train_stats_dir_varsum = None
                        except Exception:
                            train_stats_alpha0 = None
                            train_stats_dir_entropy = None
                            train_stats_dir_varsum = None
                    else:
                        z_pred_flat = self.belief_encoder.predict_next_population_belief(
                            z_t_flat,
                            group_repr=gr_flat,
                            stage_t=st_flat,
                            return_logits=False,
                        )
                        z_tr_loss = self.belief_encoder.compute_population_belief_loss(
                            z_pred_flat,
                            z_target_flat,
                            z_mask_flat,
                            loss_type=lt,
                        )
                        train_stats_alpha0 = None
                        train_stats_dir_entropy = None
                        train_stats_dir_varsum = None
                    encoder_loss = encoder_loss + self.z_transition_loss_weight * z_tr_loss

                    try:
                        zm = z_mask_flat.to(z_pred_flat.device, dtype=z_pred_flat.dtype).clamp(min=0.0, max=1.0)
                        denom = torch.clamp(zm.sum(), min=1.0)
                        dz_pred = torch.norm((z_pred_flat - z_t_flat), p=2, dim=-1)
                        dz_tgt = torch.norm((z_target_flat - z_t_flat), p=2, dim=-1)
                        z_pred_minus_z_t_l2 = (dz_pred * zm).sum() / denom
                        z_target_minus_z_t_l2 = (dz_tgt * zm).sum() / denom

                        if int(z_pred_flat.shape[-1]) == 3:
                            eps = 1e-8
                            zp = torch.clamp(z_pred_flat, min=0.0)
                            zp = zp / torch.clamp(zp.sum(dim=-1, keepdim=True), min=eps)
                            ent = -torch.sum(zp * torch.log(zp + eps), dim=-1)  # (N,)
                            mx = torch.max(zp, dim=-1)[0]  # (N,)
                            z_pred_entropy = (ent * zm).sum() / denom
                            z_pred_maxprob = (mx * zm).sum() / denom
                            z_pred_p0 = (zp[:, 0] * zm).sum() / denom
                            z_pred_p1 = (zp[:, 1] * zm).sum() / denom
                            z_pred_p2 = (zp[:, 2] * zm).sum() / denom

                        if st_flat is not None:
                            st1 = st_flat.reshape(-1).to(dtype=torch.long)
                            for s in torch.unique(st1).tolist():
                                try:
                                    s_int = int(s)
                                except Exception:
                                    continue
                                sel = (st1 == s_int)
                                if sel.any():
                                    denom_s = torch.clamp(zm[sel].sum(), min=1.0)
                                    z_pred_delta_l2_by_stage[s_int] = float((dz_pred[sel] * zm[sel]).sum().item() / denom_s.item())
                                    z_target_delta_l2_by_stage[s_int] = float((dz_tgt[sel] * zm[sel]).sum().item() / denom_s.item())
                    except Exception:
                        z_pred_minus_z_t_l2 = None
                        z_target_minus_z_t_l2 = None
                        z_pred_entropy = None
                        z_pred_maxprob = None
            except Exception as e:
                self.logger.warning(f"train_encoder_only: z_transition_loss skipped due to error: {e}")

            if self.encoder_optimizer:
                self.encoder_optimizer.zero_grad()
                encoder_loss.backward()
                try:
                    puh = getattr(self.belief_encoder, "population_update_head", None)
                    if puh is not None:
                        gn2 = 0.0
                        for p in puh.parameters():
                            if p.grad is None:
                                continue
                            g = p.grad.detach()
                            gn2 += float(torch.sum(g * g).item())
                        pop_update_grad_l2 = float(gn2 ** 0.5)
                except Exception:
                    pop_update_grad_l2 = None
                torch.nn.utils.clip_grad_norm_(self.encoder_params, 10.0)
                self.encoder_optimizer.step()

            try:
                puh = getattr(self.belief_encoder, "population_update_head", None)
                if puh is not None:
                    wn2 = 0.0
                    for p in puh.parameters():
                        w = p.detach()
                        wn2 += float(torch.sum(w * w).item())
                    pop_update_weight_l2 = float(wn2 ** 0.5)
            except Exception:
                pop_update_weight_l2 = None

            train_stats = {
                "status": "encoder_only",
                "loss_total": float(encoder_loss.item()),
                "loss_encoder": float(encoder_loss.item()),
            }
            if self.z_head is not None:
                train_stats["loss_z"] = float(z_loss.item())
            if self.z_transition_loss_weight and self.z_transition_loss_weight > 0:
                train_stats["loss_z_transition"] = float(z_tr_loss.item())
                try:
                    if "train_stats_alpha0" in locals() and train_stats_alpha0 is not None:
                        train_stats["z_pred_alpha0_mean"] = float(train_stats_alpha0.item())
                    if "train_stats_dir_entropy" in locals() and train_stats_dir_entropy is not None:
                        train_stats["z_pred_dirichlet_entropy"] = float(train_stats_dir_entropy.item())
                    if "train_stats_dir_varsum" in locals() and train_stats_dir_varsum is not None:
                        train_stats["z_pred_dirichlet_varsum"] = float(train_stats_dir_varsum.item())
                    if z_pred_minus_z_t_l2 is not None:
                        train_stats["z_pred_minus_z_t_l2"] = float(z_pred_minus_z_t_l2.item())
                    if z_target_minus_z_t_l2 is not None:
                        train_stats["z_target_minus_z_t_l2"] = float(z_target_minus_z_t_l2.item())
                    if z_pred_entropy is not None:
                        train_stats["z_pred_entropy"] = float(z_pred_entropy.item())
                    if z_pred_maxprob is not None:
                        train_stats["z_pred_maxprob"] = float(z_pred_maxprob.item())
                    if z_pred_p0 is not None:
                        train_stats["z_pred_p0_mean"] = float(z_pred_p0.item())
                    if z_pred_p1 is not None:
                        train_stats["z_pred_p1_mean"] = float(z_pred_p1.item())
                    if z_pred_p2 is not None:
                        train_stats["z_pred_p2_mean"] = float(z_pred_p2.item())
                    if pop_update_weight_l2 is not None:
                        train_stats["population_update_head_weight_l2"] = float(pop_update_weight_l2)
                    if pop_update_grad_l2 is not None:
                        train_stats["population_update_head_grad_l2"] = float(pop_update_grad_l2)
                    for s, v in (z_pred_delta_l2_by_stage or {}).items():
                        train_stats[f"z_pred_delta_l2_stage{s}"] = float(v)
                    for s, v in (z_target_delta_l2_by_stage or {}).items():
                        train_stats[f"z_target_delta_l2_stage{s}"] = float(v)
                except Exception:
                    pass
            return train_stats

        use_mixer = (self.mixer is not None) and (self.target_mixer is not None)
        if not use_mixer:
            if not bool(getattr(self, "_warned_no_mixer_fallback", False)):
                self.logger.warning(
                    "Mixer is None; falling back to VDN-style sum mixing (mixer loss disabled). "
                    "RL training will proceed without Coordinator-LLM / mixer objectives."
                )
                self._warned_no_mixer_fallback = True

        if hasattr(self.mac, 'init_hidden'):
            self.mac.init_hidden(batch.batch_size)
        if hasattr(self.target_mac, 'init_hidden'):
            self.target_mac.init_hidden(batch.batch_size)

        
        def _select_chosen_q(q_all: torch.Tensor, actions_t: torch.Tensor) -> torch.Tensor:
            """
            q_all: (bs, n_agents, n_actions)
            actions_t: (bs, n_agents, 1) or (bs, n_agents)
            returns: (bs, n_agents)
            """
            if actions_t.ndim == 3 and actions_t.shape[-1] == 1:
                a = actions_t.long()
            elif actions_t.ndim == 2:
                a = actions_t.long().unsqueeze(-1)
            else:
                a = actions_t.long().reshape(actions_t.shape[0], actions_t.shape[1], 1)
            return q_all.gather(-1, a).squeeze(-1)
        
        list_belief_states_stage1, list_prompt_embeddings_stage1, list_local_q_values_stage1, list_group_repr_stage1 = [], [], [], []
        list_belief_states_stage1_next, list_prompt_embeddings_stage1_next, list_local_q_values_stage1_next, list_group_repr_stage1_next = [], [], [], []
        
        list_commitment_features_t = [] 
        has_commitment_features_in_batch = "commitment_embedding" in batch.scheme
        
        self.logger.debug(f"Commitment embedding in batch scheme: {has_commitment_features_in_batch}")
        if has_commitment_features_in_batch:
            self.logger.debug(f"Commitment embedding scheme: {batch.scheme['commitment_embedding']}")

        self.logger.debug("Starting Stage 1: Individual belief formation")
        action_ent_sum = torch.tensor(0.0, device=self.device)
        action_ent_count = torch.tensor(0.0, device=self.device)
        action_mode_sum = torch.tensor(0.0, device=self.device)
        action_mode_count = torch.tensor(0.0, device=self.device)
        chosen_ent_sum = torch.tensor(0.0, device=self.device)
        chosen_ent_count = torch.tensor(0.0, device=self.device)
        chosen_mode_sum = torch.tensor(0.0, device=self.device)
        chosen_mode_count = torch.tensor(0.0, device=self.device)
        for t in range(batch.max_seq_length - 1):
            agent_outs_t, mac_info_t = self.mac.forward(batch, t, train_mode=True)
            list_belief_states_stage1.append(mac_info_t["belief_states"])
            list_prompt_embeddings_stage1.append(mac_info_t["prompt_embeddings"])
            try:
                if isinstance(agent_outs_t, torch.Tensor) and agent_outs_t.ndim == 3 and agent_outs_t.shape[-1] > 1:
                    logits = agent_outs_t
                    p = torch.softmax(logits, dim=-1)
                    ent = -(p * torch.log(torch.clamp(p, min=1e-8))).sum(dim=-1)  # (bs, n_agents)
                    action_ent_sum = action_ent_sum + ent.mean()
                    action_ent_count = action_ent_count + 1.0
                    a = logits.argmax(dim=-1)  # (bs, n_agents)
                    bs = int(a.shape[0])
                    if bs > 0:
                        mf = []
                        for bi in range(bs):
                            cnt = torch.bincount(a[bi].reshape(-1), minlength=int(logits.shape[-1])).float()
                            denom = float(max(1, int(a.shape[1])))
                            mf.append((cnt.max() / denom).clamp(0.0, 1.0))
                        action_mode_sum = action_mode_sum + torch.stack(mf).mean()
                        action_mode_count = action_mode_count + 1.0
            except Exception:
                pass
            try:
                if "actions" in batch.scheme:
                    a_t = batch["actions"][:, t].to(self.device)  # (bs, n_agents, 1) usually
                    if isinstance(a_t, torch.Tensor):
                        a = a_t.long()
                        if a.ndim == 3 and a.shape[-1] == 1:
                            a = a.squeeze(-1)
                        if a.ndim == 2:
                            bs = int(a.shape[0])
                            A = int(agent_outs_t.shape[-1]) if (isinstance(agent_outs_t, torch.Tensor) and agent_outs_t.ndim == 3 and agent_outs_t.shape[-1] > 1) else int(getattr(self.args, "n_actions", 1))
                            if bs > 0 and A > 1:
                                ents = []
                                mfs = []
                                for bi in range(bs):
                                    cnt = torch.bincount(a[bi].reshape(-1), minlength=A).float()
                                    p = cnt / torch.clamp(cnt.sum(), min=1.0)
                                    ent = -(p * torch.log(torch.clamp(p, min=1e-8))).sum()
                                    ents.append(ent)
                                    mfs.append((cnt.max() / float(max(1, int(a.shape[1])))).clamp(0.0, 1.0))
                                chosen_ent_sum = chosen_ent_sum + torch.stack(ents).mean()
                                chosen_ent_count = chosen_ent_count + 1.0
                                chosen_mode_sum = chosen_mode_sum + torch.stack(mfs).mean()
                                chosen_mode_count = chosen_mode_count + 1.0
            except Exception:
                pass
            try:
                if isinstance(agent_outs_t, torch.Tensor) and agent_outs_t.ndim == 3 and agent_outs_t.shape[-1] > 1:
                    a_t = batch["actions"][:, t].to(self.device)  # (bs, n_agents, 1)
                    list_local_q_values_stage1.append(_select_chosen_q(agent_outs_t, a_t))
                else:
                    list_local_q_values_stage1.append(mac_info_t["q_values"])
            except Exception:
                list_local_q_values_stage1.append(mac_info_t["q_values"])
            list_group_repr_stage1.append(mac_info_t["group_repr"])

            if has_commitment_features_in_batch:
                if t < batch.max_seq_length - 1:  # 确保时间步有效
                    try:
                        commitment_emb_t = batch["commitment_embedding"][:, t]
                        list_commitment_features_t.append(commitment_emb_t)
                        self.logger.debug(f"Added commitment_embedding at t={t}, shape: {commitment_emb_t.shape}")
                    except (KeyError, IndexError) as e:
                        self.logger.warning(f"Failed to get commitment_embedding at t={t}: {e}")
                        dummy_emb = torch.zeros(batch.batch_size, self.args.commitment_embedding_dim, device=self.device)
                        list_commitment_features_t.append(dummy_emb)
                        self.logger.debug(f"Created dummy commitment_embedding at t={t}")

            target_agent_outs_next, target_mac_info_t_next = self.target_mac.forward(batch, t + 1, train_mode=True)
            list_belief_states_stage1_next.append(target_mac_info_t_next["belief_states"])
            list_prompt_embeddings_stage1_next.append(target_mac_info_t_next["prompt_embeddings"])
            try:
                if isinstance(target_agent_outs_next, torch.Tensor) and target_agent_outs_next.ndim == 3 and target_agent_outs_next.shape[-1] > 1:
                    list_local_q_values_stage1_next.append(target_agent_outs_next.max(dim=-1)[0])
                else:
                    list_local_q_values_stage1_next.append(target_mac_info_t_next["q_values"])
            except Exception:
                list_local_q_values_stage1_next.append(target_mac_info_t_next["q_values"])
            list_group_repr_stage1_next.append(target_mac_info_t_next["group_repr"])

        belief_states_stage1_stacked = torch.stack(list_belief_states_stage1, dim=1)
        prompt_embeddings_stage1_stacked = torch.stack(list_prompt_embeddings_stage1, dim=1)
        local_q_values_stage1_stacked = torch.stack(list_local_q_values_stage1, dim=1)
        group_representation_stage1_stacked = torch.stack(list_group_repr_stage1, dim=1)

        belief_states_stage1_next_stacked = torch.stack(list_belief_states_stage1_next, dim=1)
        local_q_values_stage1_next_stacked = torch.stack(list_local_q_values_stage1_next, dim=1)

        
        self.logger.debug("Starting Stage 2: BNE coordination")
        belief_states_stage2, prompt_embeddings_stage2, local_q_values_stage2, group_representation_stage2 = self._perform_bne_coordination(
            belief_states_stage1_stacked,
            prompt_embeddings_stage1_stacked,
            local_q_values_stage1_stacked,
            group_representation_stage1_stacked,
            batch
        )

        commitment_features_t_stacked = None
        if has_commitment_features_in_batch and list_commitment_features_t:
            try:
                commitment_features_t_stacked = torch.stack(list_commitment_features_t, dim=1)
                self.logger.debug(f"Stacked commitment_features shape: {commitment_features_t_stacked.shape}")
            except Exception as e:
                self.logger.warning(f"Failed to stack commitment_features: {e}")
                commitment_features_t_stacked = torch.zeros(
                    batch.batch_size, batch.max_seq_length - 1, self.args.commitment_embedding_dim, 
                    device=self.device
                )
                self.logger.debug(f"Created dummy commitment_features_t_stacked shape: {commitment_features_t_stacked.shape}")
        elif has_commitment_features_in_batch:
            commitment_features_t_stacked = torch.zeros(
                batch.batch_size, batch.max_seq_length - 1, self.args.commitment_embedding_dim, 
                device=self.device
            )
            self.logger.debug(f"Created dummy commitment_features (empty list) shape: {commitment_features_t_stacked.shape}")

        bs_x_seq_len = batch.batch_size * (batch.max_seq_length - 1)


        prompt_embeddings_stage2_flat = prompt_embeddings_stage2.reshape(bs_x_seq_len, self.n_agents, -1)
        local_q_values_stage2_flat = local_q_values_stage2.reshape(bs_x_seq_len, self.n_agents)
        group_representation_stage2_flat = group_representation_stage2.reshape(bs_x_seq_len, -1)

        local_q_values_stage1_next_flat = local_q_values_stage1_next_stacked.reshape(bs_x_seq_len, self.n_agents)

        commitment_features_flat = None
        if commitment_features_t_stacked is not None:
            commitment_features_flat = commitment_features_t_stacked.reshape(bs_x_seq_len, -1)
            self.logger.debug(f"Flattened commitment_features shape: {commitment_features_flat.shape}")

        if use_mixer:
            mixer_results_stage2 = self.mixer(
                local_q_values=local_q_values_stage2_flat,
                prompt_embeddings=prompt_embeddings_stage2_flat,
                group_representation=group_representation_stage2_flat
            )
            q_total_stage2_flat = mixer_results_stage2["Q_tot"]

            target_group_repr_next = self.target_belief_encoder(
                belief_states_stage1_next_stacked.reshape(bs_x_seq_len, self.n_agents, -1)
            ).reshape(bs_x_seq_len, -1)
            target_prompt_embeddings_next_flat = torch.stack(list_prompt_embeddings_stage1_next, dim=1).reshape(bs_x_seq_len, self.n_agents, -1)
            
            target_mixer_results_next = self.target_mixer(
                local_q_values=local_q_values_stage1_next_flat,
                prompt_embeddings=target_prompt_embeddings_next_flat,
                group_representation=target_group_repr_next
            )
            q_total_target_next_flat = target_mixer_results_next["Q_tot"].detach()
        else:
            q_total_stage2_flat = local_q_values_stage2_flat.mean(dim=1)
            q_total_target_next_flat = local_q_values_stage1_next_flat.mean(dim=1).detach()

        rewards_flat = rewards.reshape(bs_x_seq_len)
        terminated_flat = terminated.reshape(bs_x_seq_len)
        mask_flat = mask.reshape(bs_x_seq_len)

        target_q_total_flat = rewards_flat + self.gamma * (1 - terminated_flat) * q_total_target_next_flat

        
        belief_loss, belief_loss_components = self._calculate_belief_network_loss(
            belief_states_stage1_stacked,
            belief_states_stage2,
            local_q_values_stage1_stacked,
            local_q_values_stage2,
            target_q_total_flat.reshape(batch.batch_size, batch.max_seq_length - 1),
            rewards.squeeze(-1),
            mask.squeeze(-1)
        )

        if use_mixer:
            F_i_for_LSD = mixer_results_stage2.get("F_i_for_LSD")
            
            self.logger.debug(f"F_i_for_LSD is None: {F_i_for_LSD is None}")
            self.logger.debug(f"commitment_features_flat is None: {commitment_features_flat is None}")
            self.logger.debug(f"lambda_sd: {self.lambda_sd}")
            
            total_mix_loss, loss_components = self.mixer.calculate_mix_loss(
                Q_tot=q_total_stage2_flat,
                local_q_values=local_q_values_stage2_flat,
                F_i_for_LSD=F_i_for_LSD,
                commitment_text_features=commitment_features_flat,
                target_Q_tot=target_q_total_flat,
                rewards_total=rewards_flat,
                gamma=self.gamma,
                lambda_sd=self.lambda_sd,
                lambda_m=self.lambda_m,
                terminated=terminated_flat,
                mask_flat=mask_flat
            )
        else:
            total_mix_loss = torch.tensor(0.0, device=self.device)
            loss_components = {"mixer_disabled": 1.0}

        
        encoder_loss = self._calculate_encoder_loss(
            belief_states_stage1_stacked,
            belief_states_stage2,
            group_representation_stage1_stacked,
            group_representation_stage2
        )

        z_loss = torch.tensor(0.0, device=self.device)
        if self.z_head is not None and "z_target" in batch.scheme and "z_mask" in batch.scheme:
            z_logits = self.z_head(group_representation_stage2)  # (bs, seq, 3)
            z_logp = F.log_softmax(z_logits, dim=-1)
            z_target = batch["z_target"][:, :-1].to(self.device)  # (bs, seq, 3)
            z_mask = batch["z_mask"][:, :-1].to(self.device)      # (bs, seq, 1)
            z_mask = z_mask * mask.unsqueeze(-1)
            z_target = torch.clamp(z_target, min=0.0)
            z_sum = z_target.sum(dim=-1, keepdim=True)
            z_target = torch.where(z_sum > 0, z_target / z_sum, torch.full_like(z_target, 1.0 / 3.0))

            kl = F.kl_div(z_logp, z_target, reduction="none").sum(dim=-1, keepdim=True)  # (bs, seq, 1)
            denom = torch.clamp(z_mask.sum(), min=1.0)
            z_loss = (kl * z_mask).sum() / denom
            encoder_loss = encoder_loss + self.z_loss_weight * z_loss

        z_tr_loss = torch.tensor(0.0, device=self.device)
        try:
            if (
                self.z_transition_loss_weight
                and self.z_transition_loss_weight > 0
                and self.belief_encoder is not None
                and hasattr(self.belief_encoder, "compute_loss")
                and "z_target" in batch.scheme
                and "z_mask" in batch.scheme
                and ("z_t" in batch.scheme or "belief_pre_population_z" in batch.scheme)
            ):
                z_t_seq = batch["z_t"][:, :-1].to(self.device) if "z_t" in batch.scheme else batch["belief_pre_population_z"][:, :-1].to(self.device)
                z_target_seq = batch["z_target"][:, :-1].to(self.device)
                z_mask_seq = batch["z_mask"][:, :-1].to(self.device) * mask.unsqueeze(-1)
                stage_t_seq = batch["stage_t"][:, :-1].to(self.device) if "stage_t" in batch.scheme else None

                gr_seq = group_representation_stage2

                bs, seq_len, k = z_t_seq.shape
                z_t_flat = z_t_seq.reshape(bs * seq_len, k)
                z_target_flat = z_target_seq.reshape(bs * seq_len, k)
                z_mask_flat = z_mask_seq.reshape(bs * seq_len)
                gr_flat = gr_seq.reshape(bs * seq_len, -1)
                st_flat = stage_t_seq.reshape(bs * seq_len, -1) if stage_t_seq is not None else None

                lt = str(getattr(self.args, "z_transition_loss_type", "kl") or "kl").strip().lower()
                if lt.startswith("dirichlet"):
                    if not hasattr(self.belief_encoder, "predict_next_population_belief_alpha"):
                        raise RuntimeError("Dirichlet z_transition requested but BeliefEncoder lacks predict_next_population_belief_alpha().")
                    if not hasattr(self.belief_encoder, "compute_population_belief_loss_dirichlet_kl"):
                        raise RuntimeError("Dirichlet z_transition requested but BeliefEncoder lacks compute_population_belief_loss_dirichlet_kl().")
                    alpha_pred = self.belief_encoder.predict_next_population_belief_alpha(
                        z_t_flat,
                        group_repr=gr_flat,
                        stage_t=st_flat,
                    )
                    z_tr_loss = self.belief_encoder.compute_population_belief_loss_dirichlet_kl(
                        alpha_pred,
                        z_target_flat,
                        z_mask_flat,
                        alpha0_target=(
                            (batch["z_alpha0_target"][:, :-1].to(self.device).reshape(bs * seq_len))
                            if ("z_alpha0_target" in batch.scheme)
                            else float(getattr(self.args, "dirichlet_alpha0_target", 10.0))
                        ),
                    )
                else:
                    z_tr_loss = self.belief_encoder.compute_loss(
                        z_t_flat,
                        z_target_flat,
                        z_mask_flat,
                        group_repr=gr_flat,
                        stage_t=st_flat,
                        loss_type=lt,
                    )
                encoder_loss = encoder_loss + self.z_transition_loss_weight * z_tr_loss
        except Exception as e:
            self.logger.warning(f"z_transition_loss skipped due to error: {e}")

        total_loss = belief_loss + encoder_loss + total_mix_loss

        if self.belief_optimizer:
            self.belief_optimizer.zero_grad()
        if self.encoder_optimizer:
            self.encoder_optimizer.zero_grad()
        if self.mixer_optimizer:
            self.mixer_optimizer.zero_grad()

        total_loss.backward()

        if getattr(self, "belief_net_params", None):
            torch.nn.utils.clip_grad_norm_(self.belief_net_params, 10.0)
        if getattr(self, "encoder_params", None):
            torch.nn.utils.clip_grad_norm_(self.encoder_params, 10.0)
        if getattr(self, "mixer_params", None):
            torch.nn.utils.clip_grad_norm_(self.mixer_params, 10.0)

        if self.belief_optimizer:
            self.belief_optimizer.step()
        if self.encoder_optimizer:
            self.encoder_optimizer.step()
        if self.mixer_optimizer:
            self.mixer_optimizer.step()

        if episode_num - self.last_target_update_episode >= getattr(self.args, "target_update_interval", 200):
            self._update_targets()
            self.last_target_update_episode = episode_num

        train_stats = {
            "loss_total": total_loss.item(),
            "loss_belief": belief_loss.item(),
            "loss_encoder": encoder_loss.item(),
            "loss_mixer": total_mix_loss.item(),
            "q_total_stage1_mean": torch.stack(list_local_q_values_stage1).mean().item(),
            "q_total_stage2_mean": local_q_values_stage2.mean().item(),
            "reward_mean": rewards_flat.mean().item(),
        }
        try:
            if isinstance(belief_loss_components, dict):
                for k, v in belief_loss_components.items():
                    if isinstance(v, torch.Tensor):
                        if v.numel() == 1 and torch.isfinite(v).all():
                            train_stats[str(k)] = float(v.detach().item())
                    elif isinstance(v, (int, float)):
                        vv = float(v)
                        if vv == vv:
                            train_stats[str(k)] = vv
        except Exception:
            pass
        try:
            if float(action_ent_count.item()) > 0:
                train_stats["action_pred_entropy"] = float((action_ent_sum / action_ent_count).item())
            if float(action_mode_count.item()) > 0:
                train_stats["action_pred_mode_frac"] = float((action_mode_sum / action_mode_count).item())
            if float(chosen_ent_count.item()) > 0:
                train_stats["action_chosen_entropy"] = float((chosen_ent_sum / chosen_ent_count).item())
            if float(chosen_mode_count.item()) > 0:
                train_stats["action_chosen_mode_frac"] = float((chosen_mode_sum / chosen_mode_count).item())
        except Exception:
            pass
        try:
            agent = self._get_agent_module()
            if agent is not None:
                a = getattr(agent, "last_s3b_bias_alpha", None)
                bm = getattr(agent, "last_s3b_bias_logit_mean", None)
                bs = getattr(agent, "last_s3b_bias_logit_std", None)
                af = getattr(agent, "last_s3b_bias_applied_frac", None)
                if a is not None:
                    train_stats["s3b_bias_alpha"] = float(a)
                if bm is not None:
                    train_stats["s3b_bias_logit_mean"] = float(bm)
                if bs is not None:
                    train_stats["s3b_bias_logit_std"] = float(bs)
                if af is not None:
                    train_stats["s3b_bias_applied_frac"] = float(af)
        except Exception:
            pass
        if self.z_head is not None:
            train_stats["loss_z"] = z_loss.item()
        if self.z_transition_loss_weight and self.z_transition_loss_weight > 0:
            train_stats["loss_z_transition"] = z_tr_loss.item()
        
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                train_stats[f"mixer_{key}"] = value.item()
            else:
                train_stats[f"mixer_{key}"] = value

        return train_stats

    def _perform_bne_coordination(self, belief_states_stage1: torch.Tensor, 
                                 prompt_embeddings_stage1: torch.Tensor,
                                 local_q_values_stage1: torch.Tensor,
                                 group_representation_stage1: torch.Tensor,
                                 batch: EpisodeBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行贝叶斯纳什均衡协调，实现Stage 2的belief更新
        
        Args:
            belief_states_stage1: Stage 1的belief states
            prompt_embeddings_stage1: Stage 1的prompt embeddings
            local_q_values_stage1: Stage 1的local Q values
            group_representation_stage1: Stage 1的group representation
            batch: Episode batch data
            
        Returns:
            Tuple of (belief_states_stage2, prompt_embeddings_stage2, local_q_values_stage2, group_representation_stage2)
        """
        batch_size, seq_len, n_agents, belief_dim = belief_states_stage1.shape
        
        belief_states_current = belief_states_stage1.clone()
        prompt_embeddings_current = prompt_embeddings_stage1.clone()
        local_q_values_current = local_q_values_stage1.clone()
        group_representation_current = group_representation_stage1.clone()
        
        for iteration in range(self.bne_max_iterations):
            belief_states_prev = belief_states_current.clone()

            new_beliefs_ts = []
            new_prompt_ts = []
            new_q_ts = []
            new_group_ts = []
            for t in range(seq_len):
                current_beliefs_t = belief_states_current[:, t]  # (batch, n_agents, belief_dim)
                current_group_repr_t = group_representation_current[:, t]  # (batch, group_dim)

                agent_interactions = self._calculate_agent_interactions(
                    current_beliefs_t, current_group_repr_t
                )

                updated_beliefs_t = self._update_beliefs_bne(
                    current_beliefs_t, agent_interactions, batch, t
                )

                updated_prompt_emb_t, updated_q_vals_t = self._recompute_agent_outputs(
                    updated_beliefs_t, batch, t
                )

                updated_group_repr_t = self.belief_encoder(updated_beliefs_t)

                new_beliefs_ts.append(updated_beliefs_t)
                new_prompt_ts.append(updated_prompt_emb_t)
                new_q_ts.append(updated_q_vals_t)
                new_group_ts.append(updated_group_repr_t)

            belief_states_current = torch.stack(new_beliefs_ts, dim=1)
            prompt_embeddings_current = torch.stack(new_prompt_ts, dim=1)
            local_q_values_current = torch.stack(new_q_ts, dim=1)
            group_representation_current = torch.stack(new_group_ts, dim=1)
            
            belief_change = torch.norm(belief_states_current - belief_states_prev).item()
            if belief_change < self.bne_convergence_threshold:
                self.logger.debug(f"BNE converged after {iteration + 1} iterations, change: {belief_change:.6f}")
                break
        
        return belief_states_current, prompt_embeddings_current, local_q_values_current, group_representation_current

    def _calculate_agent_interactions(self, beliefs: torch.Tensor, group_repr: torch.Tensor) -> torch.Tensor:
        """
        计算智能体之间的互动影响矩阵
        
        Args:
            beliefs: (batch, n_agents, belief_dim)
            group_repr: (batch, group_dim)
            
        Returns:
            interaction matrix: (batch, n_agents, n_agents)
        """
        batch_size, n_agents, belief_dim = beliefs.shape
        
        beliefs_normalized = F.normalize(beliefs, p=2, dim=-1)
        similarity_matrix = torch.bmm(beliefs_normalized, beliefs_normalized.transpose(-2, -1))
        
        group_influence = group_repr.unsqueeze(1).expand(-1, n_agents, -1)  # (batch, n_agents, group_dim)
        
        interaction_weights = torch.softmax(similarity_matrix, dim=-1)
        
        return interaction_weights

    def _update_beliefs_bne(self, beliefs: torch.Tensor, interactions: torch.Tensor, 
                           batch: EpisodeBatch, t: int) -> torch.Tensor:
        """
        使用BNE机制更新belief states
        
        Args:
            beliefs: (batch, n_agents, belief_dim)
            interactions: (batch, n_agents, n_agents)
            batch: Episode batch
            t: Time step
            
        Returns:
            updated beliefs: (batch, n_agents, belief_dim)
        """
        influence_all = torch.bmm(interactions, beliefs)  # (batch, n_agents, belief_dim)
        diag_w = interactions.diagonal(dim1=1, dim2=2).unsqueeze(-1)  # (batch, n_agents, 1)
        other_influence = influence_all - diag_w * beliefs

        bne_update_rate = float(getattr(self, "bne_update_rate", 0.1))
        return beliefs + bne_update_rate * other_influence

    def _recompute_agent_outputs(self, updated_beliefs: torch.Tensor, 
                               batch: EpisodeBatch, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于更新的belief states重新计算agent outputs
        
        Args:
            updated_beliefs: (batch, n_agents, belief_dim)
            batch: Episode batch
            t: Time step
            
        Returns:
            Tuple of (prompt_embeddings, q_values)
        """
        batch_size, n_agents, belief_dim = updated_beliefs.shape
        
        obs_tokens = batch["obs"][:, t]  # (batch_size, n_agents, max_token_len)
        inputs = obs_tokens.reshape(batch_size * n_agents, -1)
        
        if hasattr(self.mac.agent, 'belief_network'):
            mask = torch.zeros(inputs.shape, dtype=torch.bool, device=self.device)
            
            belief_outputs = self.mac.agent.belief_network(inputs, mask)
            
            prompt_embeddings = belief_outputs['prompt_embedding'].view(batch_size, n_agents, -1)
            q_values = belief_outputs['q_value'].view(batch_size, n_agents, -1).squeeze(-1)
            
            return prompt_embeddings, q_values
        else:
            prompt_embeddings = torch.randn(batch_size, n_agents, 2, device=self.device)
            q_values = torch.mean(updated_beliefs, dim=-1)  # 简化的Q值计算
            
            return prompt_embeddings, q_values

    def _calculate_belief_network_loss(self, belief_states_stage1: torch.Tensor,
                                     belief_states_stage2: torch.Tensor,
                                     q_values_stage1: torch.Tensor,
                                     q_values_stage2: torch.Tensor,
                                     target_q_total: torch.Tensor,
                                     rewards: torch.Tensor,
                                     mask: torch.Tensor):
        """
        计算BeliefNetwork的损失
        包括：
        1. Stage 1的TD损失
        2. Stage 2的BNE一致性损失
        3. Belief状态的正则化
        
        Args:
            belief_states_stage1/stage2: (batch, seq_len, n_agents, belief_dim)
            q_values_stage1/stage2: (batch, seq_len, n_agents)
            target_q_total: (batch, seq_len)
            rewards: (batch, seq_len)
            mask: (batch, seq_len)
            
        Returns:
            total belief loss
        """
        batch_size, seq_len, n_agents = q_values_stage1.shape
        
        td_error_abs_mean = torch.tensor(0.0, device=self.device)
        q_tot_mean = torch.tensor(0.0, device=self.device)
        target_q_tot_mean = torch.tensor(0.0, device=self.device)
        if self.mixer is None or self.target_mixer is None:
            q_tot_stage1 = q_values_stage1.mean(dim=-1)  # (batch, seq_len)
            td_err_tot = (q_tot_stage1 - target_q_total.detach()) * mask  # (batch, seq_len)
            loss_td_stage1 = (td_err_tot ** 2).sum() / mask.sum().clamp(min=1e-6)
            denom = mask.sum().clamp(min=1e-6)
            td_error_abs_mean = td_err_tot.abs().sum() / denom
            q_tot_mean = (q_tot_stage1 * mask).sum() / denom
            target_q_tot_mean = (target_q_total.detach() * mask).sum() / denom
        else:
            target_q_expanded = target_q_total.unsqueeze(-1).expand(-1, -1, n_agents)
            td_error_stage1 = (q_values_stage1 - target_q_expanded.detach()) * mask.unsqueeze(-1)
            loss_td_stage1 = (td_error_stage1 ** 2).sum() / mask.sum().clamp(min=1e-6)
            denom = mask.sum().clamp(min=1e-6)
            td_error_abs_mean = td_error_stage1.abs().sum() / denom
        
        q_mean_stage2 = q_values_stage2.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        consistency_error = (q_values_stage2 - q_mean_stage2) * mask.unsqueeze(-1)
        loss_bne_consistency = (consistency_error ** 2).sum() / mask.sum().clamp(min=1e-6)
        
        belief_evolution = belief_states_stage2 - belief_states_stage1
        evolution_norm = torch.norm(belief_evolution, p=2, dim=-1)  # (batch, seq_len, n_agents)
        target_evolution_norm = 0.1  # 期望的演化幅度
        evolution_loss = ((evolution_norm - target_evolution_norm) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1e-6)
        
        belief_reg_stage1 = torch.norm(belief_states_stage1, p=2, dim=-1).mean()
        belief_reg_stage2 = torch.norm(belief_states_stage2, p=2, dim=-1).mean()
        
        total_belief_loss = (
            loss_td_stage1 + 
            self.stage2_weight * loss_bne_consistency + 
            0.1 * evolution_loss + 
            0.01 * (belief_reg_stage1 + belief_reg_stage2)
        )

        components = {
            "loss_td_qtot": loss_td_stage1.detach(),
            "td_error_abs_mean": td_error_abs_mean.detach(),
            "q_tot_mean": q_tot_mean.detach(),
            "target_q_tot_mean": target_q_tot_mean.detach(),
            "loss_bne_consistency": loss_bne_consistency.detach(),
            "loss_belief_evolution": evolution_loss.detach(),
            "loss_belief_reg": (belief_reg_stage1 + belief_reg_stage2).detach(),
        }
        return total_belief_loss, components

    def eval_td_metrics(self, batch: EpisodeBatch) -> Dict[str, float]:
        """
        Stage4 RL eval metrics (no_grad):
        Compute the actually-optimized TD loss on global Q_tot (mixer-disabled uses mean aggregation).
        This is meant for test-mode evaluation / TensorBoard sanity, NOT for training updates.
        """
        import torch
        out: Dict[str, float] = {
            "test_loss_td_qtot": float("nan"),
            "test_td_error_abs_mean": float("nan"),
            "test_q_tot_mean": float("nan"),
            "test_target_q_tot_mean": float("nan"),
            "test_td_steps": 0.0,
        }
        if batch is None:
            return out
        try:
            if hasattr(batch, "to"):
                batch = batch.to(self.device)
        except Exception:
            pass

        try:
            rewards = batch["reward"][:, :-1].to(self.device)
            terminated = batch["terminated"][:, :-1].float().to(self.device)
            mask = batch["filled"][:, :-1].float().to(self.device)
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        except Exception:
            return out

        try:
            if hasattr(self.mac, "init_hidden"):
                self.mac.init_hidden(batch.batch_size)
            if hasattr(self.target_mac, "init_hidden"):
                self.target_mac.init_hidden(batch.batch_size)
        except Exception:
            pass

        def _select_chosen_q(q_all: torch.Tensor, actions_t: torch.Tensor) -> torch.Tensor:
            if actions_t.ndim == 3 and actions_t.shape[-1] == 1:
                a = actions_t.long()
            elif actions_t.ndim == 2:
                a = actions_t.long().unsqueeze(-1)
            else:
                a = actions_t.long().reshape(actions_t.shape[0], actions_t.shape[1], 1)
            return q_all.gather(-1, a).squeeze(-1)

        list_q = []
        list_q_next = []
        T = int(batch.max_seq_length - 1)
        with torch.no_grad():
            for t in range(T):
                agent_outs_t, mac_info_t = self.mac.forward(batch, t, train_mode=True)
                try:
                    if isinstance(agent_outs_t, torch.Tensor) and agent_outs_t.ndim == 3 and agent_outs_t.shape[-1] > 1 and ("actions" in batch.scheme):
                        a_t = batch["actions"][:, t].to(self.device)
                        list_q.append(_select_chosen_q(agent_outs_t, a_t))
                    else:
                        list_q.append(mac_info_t["q_values"])
                except Exception:
                    list_q.append(mac_info_t["q_values"])

                agent_outs_next, mac_info_next = self.target_mac.forward(batch, t + 1, train_mode=True)
                try:
                    if isinstance(agent_outs_next, torch.Tensor) and agent_outs_next.ndim == 3 and agent_outs_next.shape[-1] > 1:
                        list_q_next.append(agent_outs_next.max(dim=-1)[0])
                    else:
                        list_q_next.append(mac_info_next["q_values"])
                except Exception:
                    list_q_next.append(mac_info_next["q_values"])

        try:
            q_values = torch.stack(list_q, dim=1)  # (bs, T, n_agents)
            q_next = torch.stack(list_q_next, dim=1)  # (bs, T, n_agents)
        except Exception:
            return out

        q_tot = q_values.mean(dim=-1)  # (bs, T)
        q_tot_next = q_next.mean(dim=-1)  # (bs, T)

        rewards_flat = rewards.reshape(-1)
        terminated_flat = terminated.reshape(-1)
        mask_flat = mask.reshape(-1)
        q_tot_flat = q_tot.reshape(-1)
        q_tot_next_flat = q_tot_next.reshape(-1)

        target = rewards_flat + self.gamma * (1 - terminated_flat) * q_tot_next_flat
        td_err = (q_tot_flat - target.detach()) * mask_flat
        denom = torch.clamp(mask_flat.sum(), min=1e-6)
        loss = (td_err ** 2).sum() / denom
        out["test_td_steps"] = float(denom.detach().item())
        out["test_loss_td_qtot"] = float(loss.detach().item())
        out["test_td_error_abs_mean"] = float((td_err.abs().sum() / denom).detach().item())
        out["test_q_tot_mean"] = float(((q_tot_flat * mask_flat).sum() / denom).detach().item())
        out["test_target_q_tot_mean"] = float(((target.detach() * mask_flat).sum() / denom).detach().item())
        return out

    def _calculate_encoder_loss(self, belief_states_stage1: torch.Tensor,
                              belief_states_stage2: torch.Tensor,
                              group_repr_stage1: torch.Tensor,
                              group_repr_stage2: torch.Tensor) -> torch.Tensor:
        """
        计算BeliefEncoder的损失
        
        Args:
            belief_states_stage1/stage2: (batch, seq_len, n_agents, belief_dim)
            group_repr_stage1/stage2: (batch, seq_len, group_dim)
            
        Returns:
            encoder loss
        """
        batch_size, seq_len, n_agents, belief_dim = belief_states_stage1.shape
        
        beliefs_stage1_flat = belief_states_stage1.reshape(-1, n_agents, belief_dim)
        beliefs_stage2_flat = belief_states_stage2.reshape(-1, n_agents, belief_dim)
        
        recomputed_group_repr_stage1 = self.belief_encoder(beliefs_stage1_flat).reshape(batch_size, seq_len, -1)
        recomputed_group_repr_stage2 = self.belief_encoder(beliefs_stage2_flat).reshape(batch_size, seq_len, -1)
        
        consistency_loss_stage1 = F.mse_loss(recomputed_group_repr_stage1, group_repr_stage1)
        consistency_loss_stage2 = F.mse_loss(recomputed_group_repr_stage2, group_repr_stage2)
        
        evolution_loss = F.mse_loss(group_repr_stage2, group_repr_stage1)
        
        group_repr_stage2_norm = F.normalize(group_repr_stage2.reshape(-1, group_repr_stage2.shape[-1]), p=2, dim=-1)
        diversity_matrix = torch.mm(group_repr_stage2_norm, group_repr_stage2_norm.t())
        diversity_loss = torch.mean(torch.abs(diversity_matrix - torch.eye(diversity_matrix.shape[0], device=self.device)))
        
        total_encoder_loss = (
            consistency_loss_stage1 + consistency_loss_stage2 + 
            0.1 * evolution_loss + 
            0.01 * diversity_loss
        )
        
        return total_encoder_loss

    def _update_targets(self):
        """Update target networks with current network parameters."""
        if self.target_mixer and self.mixer:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.target_belief_encoder and self.belief_encoder:
            self.target_belief_encoder.load_state_dict(self.belief_encoder.state_dict())
        if self.target_mac and self.mac:
            self.target_mac.load_state_dict(self.mac.state_dict())

    def cuda(self):
        """Move all components to CUDA."""
        self.mac.cuda()
        if self.target_mac:
            self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
        if self.target_mixer is not None:
            self.target_mixer.cuda()
        if self.belief_encoder is not None: 
            self.belief_encoder.cuda()
        if self.target_belief_encoder is not None: 
            self.target_belief_encoder.cuda()

    def save_models(self, path: str):
        """Save all model components."""
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), f"{path}/mixer.th")
        if self.belief_encoder is not None and not hasattr(self.mac, 'belief_encoder'):
             torch.save(self.belief_encoder.state_dict(), f"{path}/belief_encoder.th")
        
        if self.belief_optimizer:
            torch.save(self.belief_optimizer.state_dict(), f"{path}/belief_opt.pth")
        if self.encoder_optimizer:
            torch.save(self.encoder_optimizer.state_dict(), f"{path}/encoder_opt.pth")
        if self.mixer_optimizer:
            torch.save(self.mixer_optimizer.state_dict(), f"{path}/mixer_opt.pth")

    def load_models(self, path: str):
        """Load all model components."""
        self.mac.load_models(path)
        if self.mixer is not None and os.path.exists(f"{path}/mixer.th"):
            self.mixer.load_state_dict(torch.load(f"{path}/mixer.th", map_location=lambda storage, loc: storage))
        
        if self.belief_encoder is not None and not hasattr(self.mac, 'belief_encoder') and os.path.exists(f"{path}/belief_encoder.th"):
            self.belief_encoder.load_state_dict(torch.load(f"{path}/belief_encoder.th", map_location=lambda storage, loc: storage))

        self._update_targets()

        if self.belief_optimizer and os.path.exists(f"{path}/belief_opt.pth"):
            try:
                self.belief_optimizer.load_state_dict(torch.load(f"{path}/belief_opt.pth"))
            except Exception as e:
                self.logger.warning(f"Skipping belief optimizer state load due to mismatch: {e}")
        if self.encoder_optimizer and os.path.exists(f"{path}/encoder_opt.pth"):
            try:
                self.encoder_optimizer.load_state_dict(torch.load(f"{path}/encoder_opt.pth"))
            except Exception as e:
                self.logger.warning(f"Skipping encoder optimizer state load due to mismatch: {e}")
        if self.mixer_optimizer and os.path.exists(f"{path}/mixer_opt.pth"):
            try:
                self.mixer_optimizer.load_state_dict(torch.load(f"{path}/mixer_opt.pth"))
            except Exception as e:
                self.logger.warning(f"Skipping mixer optimizer state load due to mismatch: {e}")
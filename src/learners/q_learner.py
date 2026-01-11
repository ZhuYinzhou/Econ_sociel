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
        # Correctly access use_cuda attribute
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        
        self.last_target_update_episode = 0
        self.log_stats_t = -getattr(args, "learner_log_interval", 100) - 1

        # Initialize ECON Network Components
        self.mixer: Optional[LLMQMixer] = None
        self.target_mixer: Optional[LLMQMixer] = None
        self.belief_encoder: Optional[BeliefEncoder] = None
        self.target_belief_encoder: Optional[BeliefEncoder] = None
        self.target_mac = None

        # Parameter Groups
        self.belief_net_params: List = []
        self.encoder_params: List = []
        self.mixer_params: List = []

        # Optimizers
        self.belief_optimizer: Optional[Adam] = None
        self.encoder_optimizer: Optional[Adam] = None
        self.mixer_optimizer: Optional[Adam] = None

        # Loss Weights
        self.gamma = getattr(args, "gamma", 0.99)
        self.lambda_e = getattr(args, "lambda_e", 0.1)
        self.lambda_sd = getattr(args, "lambda_sd", 0.1)
        self.lambda_m = getattr(args, "lambda_m", 0.1)
        self.lambda_belief = getattr(args.loss, "belief_weight", 0.1) if hasattr(args, 'loss') else 0.1
        # latent z supervision
        self.z_loss_weight = getattr(args, "z_loss_weight", 0.0)
        # innovation: z(t)->z(t+1) transition loss via BeliefEncoder.population_update_head
        self.z_transition_loss_weight = getattr(args, "z_transition_loss_weight", 0.0)
        self.z_head: Optional[nn.Module] = None
        # Encoder-only mode (for pure supervised z-transition training on offline datasets)
        self.train_encoder_only: bool = bool(getattr(args, "train_encoder_only", False))
        # Belief-network supervised mode (for offline stance-id classification on HF datasets)
        self.train_belief_supervised: bool = bool(getattr(args, "train_belief_supervised", False))
        # Stage1/2 option: explicitly freeze BeliefEncoder so supervised training does NOT update it
        # (also reduces accidental grads / optimizer coupling)
        self.freeze_belief_encoder_in_supervised: bool = bool(
            getattr(args, "freeze_belief_encoder_in_supervised", False)
        )
        self.belief_supervised_optimizer: Optional[Adam] = None
        
        # BNE协调参数
        self.bne_max_iterations = getattr(args, "bne_max_iterations", 5)
        self.bne_convergence_threshold = getattr(args, "bne_convergence_threshold", 0.01)
        self.stage2_weight = getattr(args, "stage2_weight", 0.3)  # Stage 2在总损失中的权重
        
        # Initialize networks and optimizers
        self._initialize_networks_and_optimizers(args)

    def _initialize_networks_and_optimizers(self, args: Any):
        # Initialize Mixer (CentralizedMixingNetwork)
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

        # Initialize Belief Encoder
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
            # Stage1/2: hard-freeze encoder weights if requested
            if self.train_belief_supervised and self.freeze_belief_encoder_in_supervised:
                try:
                    for p in self.belief_encoder.parameters():
                        p.requires_grad = False
                    self.logger.info("Froze BeliefEncoder parameters for supervised belief training (Stage1/2).")
                except Exception as e:
                    self.logger.warning(f"Failed to freeze BeliefEncoder params in supervised mode: {e}")

            # Stage3a: train ONLY population_update_head (+ stage_embed / mixing gate) for z_transition
            # This keeps group_repr mapping fixed and avoids unintended updates to the whole encoder/agent.
            if self.train_encoder_only and bool(getattr(args, "train_population_update_head_only", False)):
                try:
                    # 1) Freeze all encoder params first
                    for p in self.belief_encoder.parameters():
                        p.requires_grad = False

                    # 2) Unfreeze population_update_head
                    puh = getattr(self.belief_encoder, "population_update_head", None)
                    if puh is None:
                        raise RuntimeError("train_population_update_head_only=True but belief_encoder.population_update_head is None")
                    for p in puh.parameters():
                        p.requires_grad = True

                    # 3) Unfreeze stage_embed if the head uses stage conditioning
                    if bool(getattr(self.belief_encoder, "population_update_use_stage", False)):
                        se = getattr(self.belief_encoder, "stage_embed", None)
                        if se is None:
                            raise RuntimeError(
                                "population_update_use_stage=True but belief_encoder.stage_embed is None. "
                                "Please ensure BeliefEncoder initializes stage_embed when stage conditioning is enabled."
                            )
                        for p in se.parameters():
                            p.requires_grad = True

                    # 4) Unfreeze mixing gate parameter if present (part of the update head dynamics)
                    mix_logit = getattr(self.belief_encoder, "population_update_mix_logit", None)
                    if isinstance(mix_logit, torch.nn.Parameter):
                        mix_logit.requires_grad = True

                    self.logger.info("Stage3a: Froze encoder except population_update_head (+stage_embed/+mix gate).")
                except Exception as e:
                    self.logger.warning(f"Stage3a: Failed to apply train_population_update_head_only freezing: {e}")

                # Also freeze agent to avoid wasting memory on gradients (encoder-only stage doesn't need it)
                try:
                    agent = getattr(self.mac, "agent", None)
                    if agent is not None and hasattr(agent, "parameters"):
                        for p in agent.parameters():
                            p.requires_grad = False
                        self.logger.info("Stage3a: Froze mac.agent parameters (encoder-only head training).")
                except Exception as e:
                    self.logger.warning(f"Stage3a: Failed to freeze mac.agent params: {e}")

            # Stage4 RL recommended freezing knobs (backward-compatible: default False)
            # - freeze_belief_encoder_in_rl: keep representation + z_transition stable under noisy reward
            if (not self.train_belief_supervised) and (not self.train_encoder_only) and bool(getattr(args, "freeze_belief_encoder_in_rl", False)):
                try:
                    for p in self.belief_encoder.parameters():
                        p.requires_grad = False
                    self.logger.info("Stage4: Froze BeliefEncoder parameters in RL mode.")
                except Exception as e:
                    self.logger.warning(f"Stage4: Failed to freeze BeliefEncoder in RL: {e}")

            # Track only trainable params for optimizers/clipping
            self.encoder_params = [p for p in self.belief_encoder.parameters() if p.requires_grad]
            self.target_belief_encoder = copy.deepcopy(self.belief_encoder)
            self.logger.info(f"BeliefEncoder trainable params: {len(self.encoder_params)}")
        else:
            self.encoder_params = []
            self.target_belief_encoder = None

        # Initialize latent z head: map group representation -> z ∈ Δ^3
        if self.z_loss_weight and self.z_loss_weight > 0 and self.belief_encoder is not None:
            belief_dim = getattr(args, "belief_dim", 128)
            self.z_head = nn.Linear(belief_dim, 3).to(self.device)
            self.encoder_params.extend(list(self.z_head.parameters()))
            self.logger.info("Initialized z_head for edge-population latent z supervision.")
            
        # Initialize Target MAC
        self.target_mac = copy.deepcopy(self.mac)

        # Collect parameters for Individual Belief Networks
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

        # Stage4 RL recommended freezing knobs for agent parts (backward-compatible defaults)
        if (not self.train_belief_supervised) and (not self.train_encoder_only):
            agent = getattr(self.mac, "agent", None)
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

        # Stage4 RL: train discrete policy heads (action_type_head / stance_head) with TD-style objectives.
        # Without this, the action heads may never receive optimizer updates in RL mode.
        try:
            train_heads_rl = bool(getattr(args, "train_policy_heads_in_rl", True))
            if train_heads_rl and (not self.train_belief_supervised) and (not self.train_encoder_only):
                agent = getattr(self.mac, "agent", None)
                if agent is not None:
                    # action_type_head is the 5-way action policy (post/retweet/reply/like/do_nothing)
                    if hasattr(agent, "action_type_head") and getattr(agent, "action_type_head") is not None:
                        self.belief_net_params.extend(list(agent.action_type_head.parameters()))
                    # stance_head predicts stance_id (3-way) when action expresses stance; optional in RL
                    if bool(getattr(args, "train_stance_head_in_rl", True)):
                        if hasattr(agent, "stance_head") and getattr(agent, "stance_head") is not None:
                            self.belief_net_params.extend(list(agent.stance_head.parameters()))
        except Exception as e:
            self.logger.warning(f"Failed to include policy head params for RL training: {e}")

        # Initialize Optimizers
        self.belief_optimizer = None
        if self.belief_net_params:
            self.belief_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.belief_net_params),
                lr=getattr(args, "belief_net_lr", args.lr),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )
        
        self.encoder_optimizer = None
        if self.encoder_params and self.belief_encoder:
            self.encoder_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.encoder_params),
                lr=getattr(args, "encoder_lr", args.lr),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )
        
        self.mixer_optimizer = None
        if self.mixer_params and self.mixer:
            # Optional: freeze mixer in RL (backward-compatible default False)
            if (not self.train_belief_supervised) and (not self.train_encoder_only) and bool(getattr(args, "freeze_mixer_in_rl", False)):
                try:
                    for p in self.mixer.parameters():
                        p.requires_grad = False
                    self.logger.info("Stage4: Froze mixer parameters in RL mode.")
                except Exception as e:
                    self.logger.warning(f"Stage4: Failed to freeze mixer in RL: {e}")
                self.mixer_params = [p for p in self.mixer.parameters() if p.requires_grad]

            self.mixer_optimizer = Adam(
                params=filter(lambda p: p.requires_grad, self.mixer_params),
                lr=getattr(args, "mixer_lr", args.lr),
                weight_decay=getattr(args, "weight_decay", 0.0)
            )

        # Optimizer for belief supervised training (includes discrete action head + belief network)
        if self.train_belief_supervised:
            try:
                agent = getattr(self.mac, "agent", None)
                if agent is not None and hasattr(agent, "parameters"):
                    # Stage3b: offline action imitation (behavior cloning)
                    # Freeze belief-related modules and only train the 5-way action_type_head.
                    if bool(getattr(args, "train_action_imitation", False)):
                        try:
                            # Freeze everything in agent first (safe default)
                            for p in agent.parameters():
                                p.requires_grad = False

                            # Unfreeze action_type_head
                            if hasattr(agent, "action_type_head") and getattr(agent, "action_type_head") is not None:
                                for p in agent.action_type_head.parameters():
                                    p.requires_grad = True
                            else:
                                raise RuntimeError("train_action_imitation=True but agent.action_type_head is missing/None")

                            # Explicitly keep belief_network + stance_head frozen (clarity)
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
                        params=filter(lambda p: p.requires_grad, agent.parameters()),
                        lr=getattr(args, "belief_net_lr", args.lr),
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

        # ==============================
        # Belief supervised (offline classification) mode
        # ==============================
        # Train discrete action head to match gt_action (stance_id) on HF datasets.
        if self.train_belief_supervised:
            if "gt_action" not in batch.scheme:
                self.logger.warning("train_belief_supervised=True but gt_action not in batch.scheme; skipping.")
                return {"status": "skipped_no_gt_action"}
            if self.belief_supervised_optimizer is None:
                self.logger.warning("train_belief_supervised=True but belief_supervised_optimizer is None; skipping.")
                return {"status": "skipped_no_supervised_optimizer"}

            if hasattr(self.mac, 'init_hidden'):
                self.mac.init_hidden(batch.batch_size)

            # Optional: micro-batch the forward pass to reduce peak GPU memory.
            # This is important when sequence length is large and effective batch is bs*n_agents.
            try:
                belief_sup_micro_bs = int(getattr(self.args, "belief_supervised_micro_batch_size", 0))
            except Exception:
                belief_sup_micro_bs = 0
            belief_sup_micro_bs = max(0, belief_sup_micro_bs)

            # Supervised CE over timesteps
            # Optional: class weights for imbalanced stance labels (e.g., [w0,w1,w2]).
            # This is useful when accuracy plateaus at the majority-class ratio (e.g., ~0.76).
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
            total_count = 0.0
            # Soft-label diagnostics (for verifying Scheme-B actually takes effect)
            soft_available_steps = 0  # timesteps where gt_action_dist has any valid mass
            soft_used_steps = 0       # timesteps where we actually used soft CE (not fallback hard CE)
            soft_p1_sum = 0.0         # mean mass on class-1 across valid rows (helps detect Oppose signal)
            soft_p_count = 0
            # debug diagnostics (helps determine if model is actually learning)
            dbg_entropy_sum = 0.0
            dbg_entropy_count = 0
            dbg_maxprob_sum = 0.0
            dbg_maxprob_count = 0
            dbg_logit_abs_sum = 0.0
            dbg_logit_std_sum = 0.0
            dbg_logit_count = 0
            # debug: predicted/gt class distribution (helps detect majority-class collapse / argmax tie-to-0)
            pred_counts = None  # type: ignore
            gt_counts = None  # type: ignore
            # debug: per-class correctness (helps diagnose "class-0 never predicted" vs "class-0 always wrong")
            correct_counts = None  # type: ignore
            # Zero grads once; we will backward through micro-batches and step once.
            self.belief_supervised_optimizer.zero_grad()

            for t in range(batch.max_seq_length - 1):
                bs_total = int(batch.batch_size)
                if belief_sup_micro_bs <= 0 or belief_sup_micro_bs >= bs_total:
                    spans = [(0, bs_total)]
                else:
                    spans = [(i, min(i + belief_sup_micro_bs, bs_total)) for i in range(0, bs_total, belief_sup_micro_bs)]

                # global normalization across the full logical batch (exact gradient equivalence)
                m_t_full = mask[:, t]
                if m_t_full.ndim > 1:
                    m_t_full = m_t_full.view(bs_total)
                total_w = torch.clamp(m_t_full.float().sum() * float(self.args.n_agents), min=1.0).to(self.device)

                # Loss for logging (detached)
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

                    y_exp = y.unsqueeze(1).expand(bs, na).reshape(-1)
                    logits_flat = agent_logits.reshape(bs * na, nc)

                    # Debug: entropy / confidence (no grad)
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

                    use_soft = bool(getattr(self.args, "belief_supervised_use_soft_labels", False)) and ("gt_action_dist" in batch.scheme)
                    if use_soft:
                        # Soft CE sum (normalized by total_w for exactness)
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
                            psl = torch.clamp(psl.float(), min=0.0)
                            ps = psl.sum(dim=-1, keepdim=True)
                            valid = (ps.squeeze(-1) > 0)
                            if bool(valid.any().item()):
                                soft_available_steps += 1
                                try:
                                    if nc > 1:
                                        soft_p1_sum += float(psl[valid, 1].mean().item())
                                        soft_p_count += 1
                                except Exception:
                                    pass
                                psl = torch.where(ps > 0, psl / ps, torch.full_like(psl, 1.0 / float(nc)))
                                p_exp = psl.unsqueeze(1).expand(bs, na, nc).reshape(bs * na, nc)
                                logp = F.log_softmax(logits_flat, dim=-1)
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
                        ce = F.cross_entropy(logits_flat, y_exp, reduction="none", weight=ce_weight)
                        if not torch.isfinite(ce).all():
                            self.logger.warning(f"belief_supervised: non-finite CE at t={t}; skipping slice {s0}:{s1}.")
                            continue
                        loss_sum = (ce * w).sum()
                        (loss_sum / total_w).backward()
                        loss_t_val += float((loss_sum.detach() / total_w).item())

                    # accuracy + class counts (masked)
                    pred = logits_flat.argmax(dim=-1)
                    total_correct += float(((pred == y_exp).float() * w).sum().item())
                    total_count += float(w.sum().item())
                    try:
                        with torch.no_grad():
                            if pred_counts is None:
                                pred_counts = torch.zeros(nc, device=self.device, dtype=torch.float32)
                            if gt_counts is None:
                                gt_counts = torch.zeros(nc, device=self.device, dtype=torch.float32)
                            if correct_counts is None:
                                correct_counts = torch.zeros(nc, device=self.device, dtype=torch.float32)
                            m = (w > 0.5)
                            if bool(m.any().item()):
                                pc = torch.bincount(pred[m], minlength=nc).float()
                                gc = torch.bincount(y_exp[m], minlength=nc).float()
                                pred_counts[: pc.numel()] += pc
                                gt_counts[: gc.numel()] += gc
                                corr = (pred == y_exp) & m
                                if bool(corr.any().item()):
                                    cc = torch.bincount(y_exp[corr], minlength=nc).float()
                                    correct_counts[: cc.numel()] += cc
                    except Exception:
                        pass

                # accumulate (detached) per-t loss for logging
                total_loss = total_loss + torch.tensor(float(loss_t_val), device=self.device, dtype=torch.float32)

            # mean over timesteps (avoid dependence on seq_len)
            steps = max(1, int(batch.max_seq_length - 1))
            total_loss = total_loss / float(steps)

            # Safety: if loss is NaN/Inf (often caused by degenerate all-masked attention),
            # skip optimizer step to avoid corrupting weights.
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

            # gradients already accumulated via (loss_sum/total_w).backward() per micro-batch above
            # clip agent grads
            try:
                agent = getattr(self.mac, "agent", None)
                if agent is not None and hasattr(agent, "parameters"):
                    torch.nn.utils.clip_grad_norm_(list(agent.parameters()), 10.0)
            except Exception:
                pass
            # Debug: gradient norm (to detect frozen/zero-grad situations)
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

            acc = (total_correct / max(1.0, total_count)) if total_count > 0 else 0.0
            # finalize debug distributions
            pred_frac = [float("nan")] * int(nc)
            gt_frac = [float("nan")] * int(nc)
            # per-class precision/recall (masked)
            recall = [float("nan")] * int(nc)
            precision = [float("nan")] * int(nc)
            # raw counts for debugging (masked)
            pred_cnt = [0.0] * int(nc)
            gt_cnt = [0.0] * int(nc)
            correct_cnt = [0.0] * int(nc)
            has_gt = [0.0] * int(nc)
            try:
                if isinstance(pred_counts, torch.Tensor) and pred_counts.sum().item() > 0:
                    pf = (pred_counts / pred_counts.sum()).detach().cpu().tolist()
                    pred_frac = [float(x) for x in pf]
                if isinstance(gt_counts, torch.Tensor) and gt_counts.sum().item() > 0:
                    gf = (gt_counts / gt_counts.sum()).detach().cpu().tolist()
                    gt_frac = [float(x) for x in gf]
                if isinstance(correct_counts, torch.Tensor):
                    # recall_c = correct_c / gt_c
                    if isinstance(gt_counts, torch.Tensor):
                        for i in range(int(nc)):
                            gti = float(gt_counts[i].item())
                            ci = float(correct_counts[i].item())
                            gt_cnt[i] = gti
                            correct_cnt[i] = ci
                            has_gt[i] = 1.0 if gti > 0 else 0.0
                            recall[i] = (ci / gti) if gti > 0 else 0.0
                    # precision_c = correct_c / pred_c
                    if isinstance(pred_counts, torch.Tensor):
                        for i in range(int(nc)):
                            pi = float(pred_counts[i].item())
                            ci = float(correct_counts[i].item())
                            pred_cnt[i] = pi
                            # (correct_cnt already filled above if gt_counts existed; keep safe)
                            if correct_cnt[i] == 0.0:
                                correct_cnt[i] = ci
                            precision[i] = (ci / pi) if pi > 0 else 0.0
            except Exception:
                pass
            return {
                "status": "belief_supervised",
                "loss_total": float(total_loss.item()),
                # keep keys consistent for TB
                "loss_belief": float(total_loss.item()),
                "loss_encoder": 0.0,
                "loss_mixer": 0.0,
                "belief_sup_acc": float(acc),
                # how many (masked) samples actually contributed to CE/acc this update
                # (if this is ~1, pred*_frac will naturally jump between 0/1)
                "belief_sup_effective_count": float(total_count),
                "reward_mean": float(rewards.mean().item()) if rewards.numel() > 0 else 0.0,
                # diagnostics (high-signal when loss/acc look stuck)
                "belief_sup_grad_norm": float(grad_norm),
                "belief_sup_entropy": float(dbg_entropy_sum / max(1, dbg_entropy_count)) if dbg_entropy_count > 0 else float("nan"),
                "belief_sup_maxprob": float(dbg_maxprob_sum / max(1, dbg_maxprob_count)) if dbg_maxprob_count > 0 else float("nan"),
                "belief_sup_logit_abs_mean": float(dbg_logit_abs_sum / max(1, dbg_logit_count)) if dbg_logit_count > 0 else float("nan"),
                "belief_sup_logit_std": float(dbg_logit_std_sum / max(1, dbg_logit_count)) if dbg_logit_count > 0 else float("nan"),
                # soft-label diagnostics: verify Scheme-B is actually used
                "belief_sup_soft_available_frac": float(soft_available_steps / float(steps)) if steps > 0 else 0.0,
                "belief_sup_soft_used_frac": float(soft_used_steps / float(steps)) if steps > 0 else 0.0,
                "belief_sup_soft_p1_mean": float(soft_p1_sum / max(1, soft_p_count)) if soft_p_count > 0 else float("nan"),
                # class distribution (first 3 are most relevant for stance K=3)
                "belief_sup_pred0_frac": float(pred_frac[0]) if len(pred_frac) > 0 else float("nan"),
                "belief_sup_pred1_frac": float(pred_frac[1]) if len(pred_frac) > 1 else float("nan"),
                "belief_sup_pred2_frac": float(pred_frac[2]) if len(pred_frac) > 2 else float("nan"),
                "belief_sup_gt0_frac": float(gt_frac[0]) if len(gt_frac) > 0 else float("nan"),
                "belief_sup_gt1_frac": float(gt_frac[1]) if len(gt_frac) > 1 else float("nan"),
                "belief_sup_gt2_frac": float(gt_frac[2]) if len(gt_frac) > 2 else float("nan"),
                # per-class recall/precision (for stance K=3)
                "belief_sup_recall0": float(recall[0]) if len(recall) > 0 else float("nan"),
                "belief_sup_recall1": float(recall[1]) if len(recall) > 1 else float("nan"),
                "belief_sup_recall2": float(recall[2]) if len(recall) > 2 else float("nan"),
                "belief_sup_precision0": float(precision[0]) if len(precision) > 0 else float("nan"),
                "belief_sup_precision1": float(precision[1]) if len(precision) > 1 else float("nan"),
                "belief_sup_precision2": float(precision[2]) if len(precision) > 2 else float("nan"),
                # counts + validity flags (helps interpret recall/precision when a class is absent in current batch)
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
                # record the *effective* CE class weights (so we can verify config takes effect in TensorBoard)
                "belief_sup_ce_w0": float(ce_weight_list[0]) if isinstance(ce_weight_list, list) and len(ce_weight_list) > 0 else float("nan"),
                "belief_sup_ce_w1": float(ce_weight_list[1]) if isinstance(ce_weight_list, list) and len(ce_weight_list) > 1 else float("nan"),
                "belief_sup_ce_w2": float(ce_weight_list[2]) if isinstance(ce_weight_list, list) and len(ce_weight_list) > 2 else float("nan"),
            }

        # ==============================
        # Encoder-only supervised mode
        # ==============================
        # Used for stage-3 z(t)->z(t+1) training: avoid TD/mixer/belief losses and only train encoder heads.
        if self.train_encoder_only:
            if self.belief_encoder is None:
                self.logger.warning("train_encoder_only=True but belief_encoder is None; skipping.")
                return {"status": "skipped_encoder_none"}

            # Collect group representations from MAC forward (no mixer needed)
            if hasattr(self.mac, 'init_hidden'):
                self.mac.init_hidden(batch.batch_size)

            group_repr_list = []
            # batch.max_seq_length includes final dummy step; we train on [: -1]
            for t in range(batch.max_seq_length - 1):
                _, mac_info_t = self.mac.forward(batch, t, train_mode=True)
                gr = mac_info_t.get("group_repr")
                if gr is None:
                    # fallback: recompute from belief states if present
                    bs_t = mac_info_t.get("belief_states")
                    if bs_t is not None and callable(getattr(self.belief_encoder, "__call__", None)):
                        try:
                            gr = self.belief_encoder(bs_t)
                        except Exception:
                            gr = None
                if gr is None:
                    # last resort: zeros (keeps run alive but training signal weak)
                    gr = torch.zeros(batch.batch_size, getattr(self.args, "belief_dim", 128), device=self.device)
                group_repr_list.append(gr)

            group_representation_seq = torch.stack(group_repr_list, dim=1)  # (bs, seq, belief_dim)

            # Encoder supervised losses
            encoder_loss = torch.tensor(0.0, device=self.device)
            z_loss = torch.tensor(0.0, device=self.device)
            z_tr_loss = torch.tensor(0.0, device=self.device)

            # Optional: z head supervision (group_repr -> z_target)
            if self.z_head is not None and "z_target" in batch.scheme and "z_mask" in batch.scheme:
                z_logits = self.z_head(group_representation_seq)  # (bs, seq, 3)
                z_logp = F.log_softmax(z_logits, dim=-1)
                z_target = batch["z_target"][:, :-1].to(self.device)  # (bs, seq, K)
                z_mask = batch["z_mask"][:, :-1].to(self.device)      # (bs, seq, 1)
                z_mask = z_mask * mask.unsqueeze(-1)
                # normalize target if it is distribution-like
                z_target = torch.clamp(z_target, min=0.0)
                z_sum = z_target.sum(dim=-1, keepdim=True)
                # if K==3, normalize; otherwise keep as-is
                if z_target.shape[-1] == 3:
                    z_target = torch.where(z_sum > 0, z_target / z_sum, torch.full_like(z_target, 1.0 / 3.0))
                kl = F.kl_div(z_logp, z_target, reduction="none").sum(dim=-1, keepdim=True)
                denom = torch.clamp(z_mask.sum(), min=1.0)
                z_loss = (kl * z_mask).sum() / denom
                encoder_loss = encoder_loss + self.z_loss_weight * z_loss

            # Main: z(t)->z(t+1) transition supervision via BeliefEncoder head
            try:
                if (
                    self.z_transition_loss_weight
                    and self.z_transition_loss_weight > 0
                    and hasattr(self.belief_encoder, "compute_loss")
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

                    z_tr_loss = self.belief_encoder.compute_loss(
                        z_t_flat,
                        z_target_flat,
                        z_mask_flat,
                        group_repr=gr_flat,
                        stage_t=st_flat,
                        loss_type=getattr(self.args, "z_transition_loss_type", "kl"),
                    )
                    encoder_loss = encoder_loss + self.z_transition_loss_weight * z_tr_loss
            except Exception as e:
                self.logger.warning(f"train_encoder_only: z_transition_loss skipped due to error: {e}")

            # Optimize encoder params only
            if self.encoder_optimizer:
                self.encoder_optimizer.zero_grad()
                encoder_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder_params, 10.0)
                self.encoder_optimizer.step()

            train_stats = {
                "status": "encoder_only",
                "loss_total": float(encoder_loss.item()),
                "loss_encoder": float(encoder_loss.item()),
                "reward_mean": float(rewards.mean().item()) if rewards.numel() > 0 else 0.0,
            }
            if self.z_head is not None:
                train_stats["loss_z"] = float(z_loss.item())
            if self.z_transition_loss_weight and self.z_transition_loss_weight > 0:
                train_stats["loss_z_transition"] = float(z_tr_loss.item())
            return train_stats

        # ==============================
        # Full RL/mixer mode (default)
        # ==============================
        if self.mixer is None:
            self.logger.warning("Mixer is None, training will be skipped.")
            return {"status": "skipped_mixer_none"}

        if hasattr(self.mac, 'init_hidden'):
            self.mac.init_hidden(batch.batch_size)
        if hasattr(self.target_mac, 'init_hidden'):
            self.target_mac.init_hidden(batch.batch_size)

        # ===========================================
        # Stage 1: Individual Belief Formation
        # ===========================================
        
        # Helper: select scalar Q for executed actions from per-action Q tensor.
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

        # Collect data from forward passes - Stage 1
        list_belief_states_stage1, list_prompt_embeddings_stage1, list_local_q_values_stage1, list_group_repr_stage1 = [], [], [], []
        list_belief_states_stage1_next, list_prompt_embeddings_stage1_next, list_local_q_values_stage1_next, list_group_repr_stage1_next = [], [], [], []
        
        # Store commitment features if available in batch
        list_commitment_features_t = [] 
        has_commitment_features_in_batch = "commitment_embedding" in batch.scheme
        
        self.logger.debug(f"Commitment embedding in batch scheme: {has_commitment_features_in_batch}")
        if has_commitment_features_in_batch:
            self.logger.debug(f"Commitment embedding scheme: {batch.scheme['commitment_embedding']}")

        # Stage 1: Forward pass through time steps for individual belief formation
        self.logger.debug("Starting Stage 1: Individual belief formation")
        for t in range(batch.max_seq_length - 1):
            agent_outs_t, mac_info_t = self.mac.forward(batch, t, train_mode=True)
            list_belief_states_stage1.append(mac_info_t["belief_states"])
            list_prompt_embeddings_stage1.append(mac_info_t["prompt_embeddings"])
            # For RL with discrete actions, prefer chosen-action Q derived from per-action outputs.
            # This ensures action_type_head participates in TD updates (Stage4 objective).
            try:
                if isinstance(agent_outs_t, torch.Tensor) and agent_outs_t.ndim == 3 and agent_outs_t.shape[-1] > 1:
                    a_t = batch["actions"][:, t].to(self.device)  # (bs, n_agents, 1)
                    list_local_q_values_stage1.append(_select_chosen_q(agent_outs_t, a_t))
                else:
                    list_local_q_values_stage1.append(mac_info_t["q_values"])
            except Exception:
                list_local_q_values_stage1.append(mac_info_t["q_values"])
            list_group_repr_stage1.append(mac_info_t["group_repr"])

            # 修复：正确处理commitment_embedding
            if has_commitment_features_in_batch:
                if t < batch.max_seq_length - 1:  # 确保时间步有效
                    try:
                        commitment_emb_t = batch["commitment_embedding"][:, t]
                        list_commitment_features_t.append(commitment_emb_t)
                        self.logger.debug(f"Added commitment_embedding at t={t}, shape: {commitment_emb_t.shape}")
                    except (KeyError, IndexError) as e:
                        self.logger.warning(f"Failed to get commitment_embedding at t={t}: {e}")
                        # 创建dummy commitment embedding
                        dummy_emb = torch.zeros(batch.batch_size, self.args.commitment_embedding_dim, device=self.device)
                        list_commitment_features_t.append(dummy_emb)
                        self.logger.debug(f"Created dummy commitment_embedding at t={t}")

            target_agent_outs_next, target_mac_info_t_next = self.target_mac.forward(batch, t + 1, train_mode=True)
            list_belief_states_stage1_next.append(target_mac_info_t_next["belief_states"])
            list_prompt_embeddings_stage1_next.append(target_mac_info_t_next["prompt_embeddings"])
            # Target: max over actions for next state when per-action outputs are available.
            try:
                if isinstance(target_agent_outs_next, torch.Tensor) and target_agent_outs_next.ndim == 3 and target_agent_outs_next.shape[-1] > 1:
                    list_local_q_values_stage1_next.append(target_agent_outs_next.max(dim=-1)[0])
                else:
                    list_local_q_values_stage1_next.append(target_mac_info_t_next["q_values"])
            except Exception:
                list_local_q_values_stage1_next.append(target_mac_info_t_next["q_values"])
            list_group_repr_stage1_next.append(target_mac_info_t_next["group_repr"])

        # Stack temporal data for Stage 1
        belief_states_stage1_stacked = torch.stack(list_belief_states_stage1, dim=1)
        prompt_embeddings_stage1_stacked = torch.stack(list_prompt_embeddings_stage1, dim=1)
        local_q_values_stage1_stacked = torch.stack(list_local_q_values_stage1, dim=1)
        group_representation_stage1_stacked = torch.stack(list_group_repr_stage1, dim=1)

        belief_states_stage1_next_stacked = torch.stack(list_belief_states_stage1_next, dim=1)
        local_q_values_stage1_next_stacked = torch.stack(list_local_q_values_stage1_next, dim=1)

        # ===========================================
        # Stage 2: BNE Coordination
        # ===========================================
        
        self.logger.debug("Starting Stage 2: BNE coordination")
        belief_states_stage2, prompt_embeddings_stage2, local_q_values_stage2, group_representation_stage2 = self._perform_bne_coordination(
            belief_states_stage1_stacked,
            prompt_embeddings_stage1_stacked,
            local_q_values_stage1_stacked,
            group_representation_stage1_stacked,
            batch
        )

        # 修复：确保commitment_features正确处理
        commitment_features_t_stacked = None
        if has_commitment_features_in_batch and list_commitment_features_t:
            try:
                commitment_features_t_stacked = torch.stack(list_commitment_features_t, dim=1)
                self.logger.debug(f"Stacked commitment_features shape: {commitment_features_t_stacked.shape}")
            except Exception as e:
                self.logger.warning(f"Failed to stack commitment_features: {e}")
                # 创建dummy commitment features
                commitment_features_t_stacked = torch.zeros(
                    batch.batch_size, batch.max_seq_length - 1, self.args.commitment_embedding_dim, 
                    device=self.device
                )
                self.logger.debug(f"Created dummy commitment_features_t_stacked shape: {commitment_features_t_stacked.shape}")
        elif has_commitment_features_in_batch:
            # 如果scheme中有commitment_embedding但list为空，创建dummy
            commitment_features_t_stacked = torch.zeros(
                batch.batch_size, batch.max_seq_length - 1, self.args.commitment_embedding_dim, 
                device=self.device
            )
            self.logger.debug(f"Created dummy commitment_features (empty list) shape: {commitment_features_t_stacked.shape}")

        bs_x_seq_len = batch.batch_size * (batch.max_seq_length - 1)

        # ===========================================
        # Loss Calculation
        # ===========================================

        # 使用Stage 2的结果进行mixer计算
        prompt_embeddings_stage2_flat = prompt_embeddings_stage2.reshape(bs_x_seq_len, self.n_agents, -1)
        local_q_values_stage2_flat = local_q_values_stage2.reshape(bs_x_seq_len, self.n_agents)
        group_representation_stage2_flat = group_representation_stage2.reshape(bs_x_seq_len, -1)

        # Target values using Stage 1 (more stable)
        local_q_values_stage1_next_flat = local_q_values_stage1_next_stacked.reshape(bs_x_seq_len, self.n_agents)

        # 修复：正确处理commitment_features_flat
        commitment_features_flat = None
        if commitment_features_t_stacked is not None:
            commitment_features_flat = commitment_features_t_stacked.reshape(bs_x_seq_len, -1)
            self.logger.debug(f"Flattened commitment_features shape: {commitment_features_flat.shape}")

        # Forward pass through mixer using Stage 2 coordinated values
        mixer_results_stage2 = self.mixer(
            local_q_values=local_q_values_stage2_flat,
            prompt_embeddings=prompt_embeddings_stage2_flat,
            group_representation=group_representation_stage2_flat
        )
        q_total_stage2_flat = mixer_results_stage2["Q_tot"] 

        # Target mixer forward pass using Stage 1 next values
        # 需要计算target的group representation
        target_group_repr_next = self.target_belief_encoder(belief_states_stage1_next_stacked.reshape(bs_x_seq_len, self.n_agents, -1)).reshape(bs_x_seq_len, -1)
        target_prompt_embeddings_next_flat = torch.stack(list_prompt_embeddings_stage1_next, dim=1).reshape(bs_x_seq_len, self.n_agents, -1)
        
        target_mixer_results_next = self.target_mixer(
            local_q_values=local_q_values_stage1_next_flat,
            prompt_embeddings=target_prompt_embeddings_next_flat,
            group_representation=target_group_repr_next
        )
        q_total_target_next_flat = target_mixer_results_next["Q_tot"].detach()

        # Prepare reward/termination/mask as 1D vectors (N,) to avoid accidental (N,N) broadcasting
        rewards_flat = rewards.reshape(bs_x_seq_len)
        terminated_flat = terminated.reshape(bs_x_seq_len)
        mask_flat = mask.reshape(bs_x_seq_len)

        # Calculate target Q-values
        target_q_total_flat = rewards_flat + self.gamma * (1 - terminated_flat) * q_total_target_next_flat

        # ===========================================
        # BeliefNetwork Loss Calculation
        # ===========================================
        
        belief_loss = self._calculate_belief_network_loss(
            belief_states_stage1_stacked,
            belief_states_stage2,
            local_q_values_stage1_stacked,
            local_q_values_stage2,
            target_q_total_flat.reshape(batch.batch_size, batch.max_seq_length - 1),
            rewards.squeeze(-1),
            mask.squeeze(-1)
        )

        # ===========================================
        # Mixer Loss Calculation
        # ===========================================
        
        F_i_for_LSD = mixer_results_stage2.get("F_i_for_LSD")
        
        # 调试信息
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

        # ===========================================
        # BeliefEncoder Loss
        # ===========================================
        
        encoder_loss = self._calculate_encoder_loss(
            belief_states_stage1_stacked,
            belief_states_stage2,
            group_representation_stage1_stacked,
            group_representation_stage2
        )

        # ===========================================
        # Latent z Supervision (edge users macro dist)
        # ===========================================
        z_loss = torch.tensor(0.0, device=self.device)
        if self.z_head is not None and "z_target" in batch.scheme and "z_mask" in batch.scheme:
            # group_representation_stage2: (bs, seq, belief_dim)
            z_logits = self.z_head(group_representation_stage2)  # (bs, seq, 3)
            z_logp = F.log_softmax(z_logits, dim=-1)
            z_target = batch["z_target"][:, :-1].to(self.device)  # (bs, seq, 3)
            z_mask = batch["z_mask"][:, :-1].to(self.device)      # (bs, seq, 1)
            # also respect filled mask
            z_mask = z_mask * mask.unsqueeze(-1)
            # normalize target for numerical stability
            z_target = torch.clamp(z_target, min=0.0)
            z_sum = z_target.sum(dim=-1, keepdim=True)
            z_target = torch.where(z_sum > 0, z_target / z_sum, torch.full_like(z_target, 1.0 / 3.0))

            # KL(target || pred) = sum target * (log target - log pred)
            # Use torch.kl_div with log-prob input
            kl = F.kl_div(z_logp, z_target, reduction="none").sum(dim=-1, keepdim=True)  # (bs, seq, 1)
            denom = torch.clamp(z_mask.sum(), min=1.0)
            z_loss = (kl * z_mask).sum() / denom
            encoder_loss = encoder_loss + self.z_loss_weight * z_loss

        # ===========================================
        # Innovation: z(t)->z(t+1) transition supervision via BeliefEncoder head
        # ===========================================
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

                # group_representation_stage2: (bs, seq, belief_dim)
                gr_seq = group_representation_stage2

                bs, seq_len, k = z_t_seq.shape
                z_t_flat = z_t_seq.reshape(bs * seq_len, k)
                z_target_flat = z_target_seq.reshape(bs * seq_len, k)
                z_mask_flat = z_mask_seq.reshape(bs * seq_len)
                gr_flat = gr_seq.reshape(bs * seq_len, -1)
                st_flat = stage_t_seq.reshape(bs * seq_len, -1) if stage_t_seq is not None else None

                z_tr_loss = self.belief_encoder.compute_loss(
                    z_t_flat,
                    z_target_flat,
                    z_mask_flat,
                    group_repr=gr_flat,
                    stage_t=st_flat,
                    loss_type=getattr(self.args, "z_transition_loss_type", "kl"),
                )
                encoder_loss = encoder_loss + self.z_transition_loss_weight * z_tr_loss
        except Exception as e:
            self.logger.warning(f"z_transition_loss skipped due to error: {e}")

        # ===========================================
        # Network Optimization
        # ===========================================
        # IMPORTANT:
        # Do NOT step one optimizer before backpropagating other losses from the same forward graph.
        # Optimizer.step() modifies parameters in-place and can break autograd when retain_graph=True.
        total_loss = belief_loss + encoder_loss + total_mix_loss

        if self.belief_optimizer:
            self.belief_optimizer.zero_grad()
        if self.encoder_optimizer:
            self.encoder_optimizer.zero_grad()
        if self.mixer_optimizer:
            self.mixer_optimizer.zero_grad()

        total_loss.backward()

        # Clip gradients per parameter group (optional but keeps previous behavior)
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

        # Update target networks periodically
        if episode_num - self.last_target_update_episode >= getattr(self.args, "target_update_interval", 200):
            self._update_targets()
            self.last_target_update_episode = episode_num

        # Prepare training statistics
        train_stats = {
            "loss_total": total_loss.item(),
            "loss_belief": belief_loss.item(),
            "loss_encoder": encoder_loss.item(),
            "loss_mixer": total_mix_loss.item(),
            "q_total_stage1_mean": torch.stack(list_local_q_values_stage1).mean().item(),
            "q_total_stage2_mean": local_q_values_stage2.mean().item(),
            "reward_mean": rewards_flat.mean().item(),
        }
        if self.z_head is not None:
            train_stats["loss_z"] = z_loss.item()
        if self.z_transition_loss_weight and self.z_transition_loss_weight > 0:
            train_stats["loss_z_transition"] = z_tr_loss.item()
        
        # Add individual loss components
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
        
        # 初始化Stage 2的状态为Stage 1的状态
        # NOTE: avoid inplace ops on tensors that require grad (breaks autograd).
        belief_states_current = belief_states_stage1.clone()
        prompt_embeddings_current = prompt_embeddings_stage1.clone()
        local_q_values_current = local_q_values_stage1.clone()
        group_representation_current = group_representation_stage1.clone()
        
        # BNE迭代更新
        for iteration in range(self.bne_max_iterations):
            belief_states_prev = belief_states_current.clone()

            # 为每个时间步计算BNE更新（out-of-place 组装，避免 inplace）
            new_beliefs_ts = []
            new_prompt_ts = []
            new_q_ts = []
            new_group_ts = []
            for t in range(seq_len):
                # 当前时间步的状态
                current_beliefs_t = belief_states_current[:, t]  # (batch, n_agents, belief_dim)
                current_group_repr_t = group_representation_current[:, t]  # (batch, group_dim)

                # 计算agent间的互动影响
                agent_interactions = self._calculate_agent_interactions(
                    current_beliefs_t, current_group_repr_t
                )

                # 更新belief states (BNE step)
                updated_beliefs_t = self._update_beliefs_bne(
                    current_beliefs_t, agent_interactions, batch, t
                )

                # 重新计算prompt embeddings和Q values
                updated_prompt_emb_t, updated_q_vals_t = self._recompute_agent_outputs(
                    updated_beliefs_t, batch, t
                )

                # 更新group representation
                updated_group_repr_t = self.belief_encoder(updated_beliefs_t)

                new_beliefs_ts.append(updated_beliefs_t)
                new_prompt_ts.append(updated_prompt_emb_t)
                new_q_ts.append(updated_q_vals_t)
                new_group_ts.append(updated_group_repr_t)

            belief_states_current = torch.stack(new_beliefs_ts, dim=1)
            prompt_embeddings_current = torch.stack(new_prompt_ts, dim=1)
            local_q_values_current = torch.stack(new_q_ts, dim=1)
            group_representation_current = torch.stack(new_group_ts, dim=1)
            
            # 检查收敛性
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
        
        # 计算agent间的相似性矩阵
        beliefs_normalized = F.normalize(beliefs, p=2, dim=-1)
        similarity_matrix = torch.bmm(beliefs_normalized, beliefs_normalized.transpose(-2, -1))
        
        # 加入group representation的影响
        group_influence = group_repr.unsqueeze(1).expand(-1, n_agents, -1)  # (batch, n_agents, group_dim)
        
        # 简化的互动权重计算
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
        # Vectorized BNE update (no in-place writes; keeps autograd happy)
        # interactions: (batch, n_agents, n_agents), beliefs: (batch, n_agents, belief_dim)
        # influence_all[i] = sum_j w_ij * b_j
        influence_all = torch.bmm(interactions, beliefs)  # (batch, n_agents, belief_dim)
        # remove self influence to match the original i!=j loop
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
        
        # 获取原始观察
        obs_tokens = batch["obs"][:, t]  # (batch_size, n_agents, max_token_len)
        inputs = obs_tokens.reshape(batch_size * n_agents, -1)
        
        # 通过belief network重新计算输出
        if hasattr(self.mac.agent, 'belief_network'):
            # 创建一个虚拟的mask
            mask = torch.zeros(inputs.shape, dtype=torch.bool, device=self.device)
            
            # 重新前向传播
            belief_outputs = self.mac.agent.belief_network(inputs, mask)
            
            # 重塑输出
            prompt_embeddings = belief_outputs['prompt_embedding'].view(batch_size, n_agents, -1)
            q_values = belief_outputs['q_value'].view(batch_size, n_agents, -1).squeeze(-1)
            
            return prompt_embeddings, q_values
        else:
            # 备用：如果无法重新计算，返回基于belief的近似值
            prompt_embeddings = torch.randn(batch_size, n_agents, 2, device=self.device)
            q_values = torch.mean(updated_beliefs, dim=-1)  # 简化的Q值计算
            
            return prompt_embeddings, q_values

    def _calculate_belief_network_loss(self, belief_states_stage1: torch.Tensor,
                                     belief_states_stage2: torch.Tensor,
                                     q_values_stage1: torch.Tensor,
                                     q_values_stage2: torch.Tensor,
                                     target_q_total: torch.Tensor,
                                     rewards: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
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
        
        # 1. Stage 1 TD Loss (个体学习)
        target_q_expanded = target_q_total.unsqueeze(-1).expand(-1, -1, n_agents)
        td_error_stage1 = (q_values_stage1 - target_q_expanded.detach()) * mask.unsqueeze(-1)
        loss_td_stage1 = (td_error_stage1 ** 2).sum() / mask.sum().clamp(min=1e-6)
        
        # 2. Stage 2 BNE Consistency Loss (协调一致性)
        # 衡量Stage 2中agents之间的Q值一致性
        q_mean_stage2 = q_values_stage2.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        consistency_error = (q_values_stage2 - q_mean_stage2) * mask.unsqueeze(-1)
        loss_bne_consistency = (consistency_error ** 2).sum() / mask.sum().clamp(min=1e-6)
        
        # 3. Belief Evolution Loss (衡量Stage 1到Stage 2的合理演化)
        belief_evolution = belief_states_stage2 - belief_states_stage1
        evolution_norm = torch.norm(belief_evolution, p=2, dim=-1)  # (batch, seq_len, n_agents)
        # 希望演化不要太剧烈，但也不要完全不变
        target_evolution_norm = 0.1  # 期望的演化幅度
        evolution_loss = ((evolution_norm - target_evolution_norm) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1e-6)
        
        # 4. Belief Regularization (防止belief过度复杂)
        belief_reg_stage1 = torch.norm(belief_states_stage1, p=2, dim=-1).mean()
        belief_reg_stage2 = torch.norm(belief_states_stage2, p=2, dim=-1).mean()
        
        # 总的BeliefNetwork损失
        total_belief_loss = (
            loss_td_stage1 + 
            self.stage2_weight * loss_bne_consistency + 
            0.1 * evolution_loss + 
            0.01 * (belief_reg_stage1 + belief_reg_stage2)
        )
        
        return total_belief_loss

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
        # 1. Representation Consistency Loss
        # 确保group representation能够很好地总结individual beliefs
        batch_size, seq_len, n_agents, belief_dim = belief_states_stage1.shape
        
        # 重新计算group representation以确保一致性
        beliefs_stage1_flat = belief_states_stage1.reshape(-1, n_agents, belief_dim)
        beliefs_stage2_flat = belief_states_stage2.reshape(-1, n_agents, belief_dim)
        
        recomputed_group_repr_stage1 = self.belief_encoder(beliefs_stage1_flat).reshape(batch_size, seq_len, -1)
        recomputed_group_repr_stage2 = self.belief_encoder(beliefs_stage2_flat).reshape(batch_size, seq_len, -1)
        
        # 一致性损失
        consistency_loss_stage1 = F.mse_loss(recomputed_group_repr_stage1, group_repr_stage1)
        consistency_loss_stage2 = F.mse_loss(recomputed_group_repr_stage2, group_repr_stage2)
        
        # 2. Evolution Smoothness Loss
        # 确保group representation的演化是平滑的
        evolution_loss = F.mse_loss(group_repr_stage2, group_repr_stage1)
        
        # 3. Representation Diversity Loss
        # 确保不同的belief组合产生不同的group representation
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
        
        # Save optimizers for checkpointing
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

        # Load optimizers if they exist
        if self.belief_optimizer and os.path.exists(f"{path}/belief_opt.pth"):
            self.belief_optimizer.load_state_dict(torch.load(f"{path}/belief_opt.pth"))
        if self.encoder_optimizer and os.path.exists(f"{path}/encoder_opt.pth"):
            self.encoder_optimizer.load_state_dict(torch.load(f"{path}/encoder_opt.pth"))
        if self.mixer_optimizer and os.path.exists(f"{path}/mixer_opt.pth"):
            self.mixer_optimizer.load_state_dict(torch.load(f"{path}/mixer_opt.pth"))
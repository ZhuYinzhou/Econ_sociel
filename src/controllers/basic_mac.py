import torch
import torch.nn as nn
from modules.agents.transformer_agent import LLMTransformerAgent
from modules.llm.llm_wrapper import ImprovedLLMWrapper, LLMConfig
from components.action_selectors import REGISTRY as action_REGISTRY
from torch.nn import functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from modules.belief_encoder import BeliefEncoder
from transformers import AutoTokenizer

class LLMBasicMAC:
    """
    Multi-Agent Controller coordinating Transformer-based LLM agents.
    Handles both coordinator and executor roles in the system.
    """
    def __init__(self, scheme: Dict, groups: Dict, args: Any):
        self.n_agents = args.n_agents
        self.args = args
        if not hasattr(self.args, "agent_output_type"):
            self.args.agent_output_type = "q_values"
        if not hasattr(self.args, "action_selector"):
            self.args.action_selector = "multinomial"
        if not hasattr(self.args, "use_causal_mask"):
            self.args.use_causal_mask = False
        if not hasattr(self.args, "max_seq_length"):
            self.args.max_seq_length = getattr(getattr(self.args, "env_args", object()), "max_question_length", 1024)
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        model_name = args.llm_model_name if hasattr(args, "llm_model_name") else "gpt2"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            logger.info(f"Successfully loaded tokenizer for {model_name} from local cache")
        except (OSError, ConnectionError, Exception) as e:
            logger.warning(f"Failed to load tokenizer for model '{model_name}' from cache: {e}")
            logger.info("Trying to create a simple tokenizer as fallback...")
            try:
                from transformers import GPT2Tokenizer
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
                logger.info("Using cached GPT-2 tokenizer as fallback")
            except Exception as e2:
                logger.error(f"Failed to load any tokenizer: {e2}")
                self.tokenizer = self._create_minimal_tokenizer()
                logger.info("Created minimal tokenizer as last resort")
                
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer missing pad_token, set to eos_token: {self.tokenizer.eos_token}")

        input_shape = self._get_input_shape(scheme) 
        
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        
        self.action_selector = action_REGISTRY[args.action_selector](args)
        
        common_llm_config = LLMConfig(
            api_key=args.together_api_key,
            model_name=args.coordinator_model,
            debug=getattr(args, "debug", False),
            max_retries=getattr(args, "llm_max_retries", 6),
            max_workers=getattr(args, "llm_max_workers", 5)
        )

        self.coordinator = ImprovedLLMWrapper(
            api_key=args.together_api_key,
            model_name=args.coordinator_model,
            belief_dim=args.belief_dim,
            debug=getattr(args, "debug", False)
        )
        
        self.belief_encoder = BeliefEncoder(
            belief_dim=args.belief_dim,
            n_agents=args.n_agents,
            n_heads=args.arch.attention_heads if hasattr(args, 'arch') and hasattr(args.arch, 'attention_heads') else 4,
            key_dim=args.arch.key_dim if hasattr(args, 'arch') and hasattr(args.arch, 'key_dim') else 64,
            device=self.device,
            population_belief_dim=getattr(args, "population_belief_dim", 3),
            use_population_token=getattr(args, "use_population_token", True),
            n_stages=getattr(getattr(args, "env_args", object()), "n_stages", 13),
            use_stage_token=getattr(args, "use_stage_token", False),
            use_brief_encoder=getattr(args, "use_brief_encoder", False),
            brief_encoder_input_dim=getattr(args, "brief_encoder_input_dim", getattr(getattr(args, "env_args", object()), "group_representation_dim", 128)),
            brief_encoder_hidden_dim=getattr(args, "brief_encoder_hidden_dim", 256),
            brief_encoder_use_stage=getattr(args, "brief_encoder_use_stage", False),
            use_population_update_head=getattr(args, "use_population_update_head", True),
            population_update_hidden_dim=getattr(args, "population_update_hidden_dim", 128),
            population_update_use_group_repr=getattr(args, "population_update_use_group_repr", True),
            population_update_use_stage=getattr(args, "population_update_use_stage", False),
            population_update_parametrization=getattr(args, "population_update_parametrization", "categorical"),
            dirichlet_alpha_min=getattr(args, "dirichlet_alpha_min", 1e-3),
            population_update_use_extra_cond=getattr(args, "population_update_use_extra_cond", False),
            population_update_extra_cond_dim=getattr(args, "population_update_extra_cond_dim", 0),
            population_update_residual_mixing=getattr(args, "population_update_residual_mixing", True),
            population_update_mixing_init=getattr(args, "population_update_mixing_init", 0.5),
            population_update_mixing_learnable=getattr(args, "population_update_mixing_learnable", True),
            secondary_action_dim=getattr(args, "secondary_action_dim", 5),
            use_secondary_action_head=getattr(args, "use_secondary_action_head", False),
            secondary_action_hidden_dim=getattr(args, "secondary_action_hidden_dim", 128),
            secondary_action_use_group_repr=getattr(args, "secondary_action_use_group_repr", True),
            secondary_action_use_stage=getattr(args, "secondary_action_use_stage", False),
            secondary_action_use_population=getattr(args, "secondary_action_use_population", True),
        )

        self.max_cache_size = getattr(args, 'max_cache_size', 1000)
        self.strategy_cache = {}
        
        self.setup_attention_masks()

    def _unwrap(self, m: Any) -> Any:
        return getattr(m, "module", m)

    @property
    def agent_module(self) -> Any:
        return self._unwrap(getattr(self, "agent", None))

    @property
    def belief_encoder_module(self) -> Any:
        return self._unwrap(getattr(self, "belief_encoder", None))
        
    def preprocess_observation(self, observation_text: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Tokenize and preprocess a single observation text string.
        Args:
            observation_text: The raw text of the observation (e.g., a question).
            max_length: Optional maximum length for padding/truncation. If None, uses args.max_question_length.
        Returns:
            A tensor of token IDs, padded/truncated to max_length.
        """
        if max_length is None:
            max_length = getattr(self.args.env_args, "max_question_length", 512)

        encoding = self.tokenizer(
            observation_text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt',
        )
        
        return encoding.input_ids.squeeze(0).to(self.device)

    def setup_attention_masks(self):
        """Setup reusable attention masks."""
        self.base_attention_mask = torch.zeros(
            (1, self.args.max_seq_length),
            dtype=torch.bool,
            device=self.device
        )
        
        if bool(getattr(self.args, "use_causal_mask", False)):
            mask = torch.triu(
                torch.ones(self.args.max_seq_length, self.args.max_seq_length),
                diagonal=1
            )
            self.causal_mask = mask.bool().to(self.device)

    def _build_agents(self, input_shape: int):
        """Initialize Transformer agents."""
        try:
            self.agent = LLMTransformerAgent(input_shape, self.args)
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise

    def init_hidden(self, batch_size: int):
        """
        Initialize hidden states (dummy method for interface compatibility).
        For transformers, we initialize positional embeddings instead.
        """
        if not hasattr(self, 'pos_embeddings'):
            self.pos_embeddings = torch.arange(
                0, self.args.max_seq_length,
                device=self.device
            ).unsqueeze(0).expand(batch_size, -1)

    def _build_inputs(self, batch: Any, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct inputs for transformer agents.
        
        Args:
            batch: Batch of episode data
            t: Current timestep
            
        Returns:
            Tuple containing:
                - Input tensor
                - Attention mask
        """
        bs = batch.batch_size
        
        obs_tokens = batch["obs"][:, t]  # (batch_size, n_agents, max_token_len)
        inputs = obs_tokens.reshape(bs * self.n_agents, -1)  # (batch_size * n_agents, max_token_len)
        
        seq_len = inputs.size(1)  # max_token_len
        
        if hasattr(self, 'tokenizer'):
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                pad_token_id = self.tokenizer.pad_token_id
            else:
                pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = 50256  # GPT2's eos_token_id
            
        mask = (inputs == pad_token_id)
        try:
            if isinstance(mask, torch.Tensor) and mask.ndim == 2:
                all_masked = mask.all(dim=1)  # (bs*n_agents,)
                if bool(all_masked.any().item()):
                    mask[all_masked, 0] = False
        except Exception:
            pass
        
        return inputs, mask

    def select_actions(self, ep_batch: Any, t_ep: int, t_env: int, 
                      raw_observation_text: Optional[str] = None,
                      bs: slice = slice(None), test_mode: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Select actions for all agents and generate LLM responses and commitment features.
        
        Args:
            ep_batch: Episode batch data. NOTE: ep_batch["obs"] is now expected to be tokenized IDs.
            t_ep: Current episode timestep
            t_env: Current environment timestep
            raw_observation_text: Raw observation text for LLM processing
            bs: Batch slice
            test_mode: Whether in test mode
            
        Returns:
            Tuple of (actions, info_dict)
        """
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        
        agent_outputs, agent_info = self.forward(ep_batch, t_ep, test_mode)
        
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        
        enable_llm_rollout = bool(getattr(self.args, "enable_llm_rollout", True))
        if raw_observation_text is not None and enable_llm_rollout:
            env_name = str(getattr(self.args, "env", "") or "").strip().lower()
            action_source = str(getattr(self.args, "env_action_source", "") or "").strip().lower()
            is_social = (env_name == "hisim_social_env") or (action_source in ("llm_response_0", "executor0", "executor_0", "response0"))

            if is_social:
                strategy = ""
            else:
                strategy = self._get_strategy(raw_observation_text)
            
            executor_responses = []
            forced_action_types = []
            forced_stance_ids = []
            action_type_names = ["post", "retweet", "reply", "like", "do_nothing"]
            stance_actions = {"post", "retweet", "reply"}

            ca = chosen_actions
            try:
                if isinstance(ca, torch.Tensor) and ca.ndim >= 2:
                    ca0 = ca[0]
                else:
                    ca0 = ca
            except Exception:
                ca0 = ca

            stance_q = agent_info.get("stance_action_q_values")
            try:
                if isinstance(stance_q, torch.Tensor) and stance_q.ndim >= 3:
                    stance_q0 = stance_q[0]
                else:
                    stance_q0 = stance_q
            except Exception:
                stance_q0 = stance_q

            for agent_id in range(self.n_agents):
                forced_at = "do_nothing"
                forced_sid = None

                try:
                    if isinstance(ca0, torch.Tensor):
                        aid = int(ca0[agent_id].item()) if ca0.numel() > agent_id else 4
                    elif isinstance(ca0, (list, tuple)):
                        aid = int(ca0[agent_id]) if agent_id < len(ca0) else 4
                    else:
                        aid = int(ca0)
                    if 0 <= aid < len(action_type_names):
                        forced_at = action_type_names[aid]
                except Exception:
                    forced_at = "do_nothing"

                if forced_at in stance_actions:
                    try:
                        if isinstance(stance_q0, torch.Tensor) and stance_q0.ndim == 2 and stance_q0.shape[0] > agent_id:
                            forced_sid = int(stance_q0[agent_id].argmax(dim=-1).item())
                        else:
                            forced_sid = 0
                    except Exception:
                        forced_sid = 0

                forced_action_types.append(forced_at)
                forced_stance_ids.append(forced_sid)

                if is_social:
                    try:
                        response = self.agent_module.generate_answer(
                            question=raw_observation_text,
                            strategy=strategy,
                            forced_action_type=forced_at,
                            forced_stance_id=forced_sid,
                        )
                    except TypeError:
                        response = self.agent_module.generate_answer(
                            question=raw_observation_text,
                            strategy=strategy,
                        )
                else:
                    response = self.agent_module.generate_answer(
                        question=raw_observation_text,
                        strategy=strategy,
                    )
                executor_responses.append(response)

            if not is_social:
                raise RuntimeError(
                    "Non-social LLM commitment branch has been removed. "
                    "Use hisim_social_env (is_social=True) or restore the old commitment pipeline if needed."
                )
            agent_info.update({
                "strategy": strategy,
                "llm_responses": executor_responses,
                "executor_responses": executor_responses,
                "forced_action_types": forced_action_types,
                "forced_stance_ids": forced_stance_ids,
                "commitment": "",
                "commitment_text": "",
                "commitment_embedding": None
            })
        
        return chosen_actions, agent_info

    def forward(self, ep_batch: Any, t: int, test_mode: bool = False, train_mode: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through all agents.
        
        Args:
            ep_batch: Episode batch data
            t: Current timestep
            test_mode: Whether in test mode
            train_mode: Whether in training mode (for compatibility)
            
        Returns:
            Tuple of (agent_outputs, info_dict)
        """
        actual_test_mode = test_mode if not train_mode else False
        
        inputs, mask = self._build_inputs(ep_batch, t)

        population_belief = None
        stage_t = None
        try:
            if hasattr(ep_batch, "scheme"):
                if "belief_pre_population_z" in ep_batch.scheme:
                    population_belief = ep_batch["belief_pre_population_z"][:, t]  # (bs, K)
                elif "z_t" in ep_batch.scheme:
                    population_belief = ep_batch["z_t"][:, t]  # (bs, K)
                if "stage_t" in ep_batch.scheme:
                    stage_t = ep_batch["stage_t"][:, t]  # (bs,1) or (bs,)
        except Exception as e:
            logger.warning(f"Failed to fetch population_belief/stage_t from batch at t={t}: {e}")
            population_belief = None
            stage_t = None

        pop_flat = None
        st_flat = None
        try:
            if population_belief is not None and isinstance(population_belief, torch.Tensor):
                pb = population_belief
                if pb.ndim == 3 and pb.shape[1] == 1:
                    pb = pb.squeeze(1)
                if pb.ndim == 2 and pb.shape[0] == int(ep_batch.batch_size):
                    pop_flat = pb.unsqueeze(1).expand(int(ep_batch.batch_size), int(self.n_agents), int(pb.shape[-1])).reshape(int(ep_batch.batch_size) * int(self.n_agents), int(pb.shape[-1]))
        except Exception:
            pop_flat = None
        try:
            if stage_t is not None and isinstance(stage_t, torch.Tensor):
                st = stage_t
                if st.ndim == 2 and st.shape[1] == 1:
                    st = st.squeeze(1)
                if st.ndim == 1 and st.shape[0] == int(ep_batch.batch_size):
                    st_flat = st.unsqueeze(1).expand(int(ep_batch.batch_size), int(self.n_agents)).reshape(int(ep_batch.batch_size) * int(self.n_agents))
        except Exception:
            st_flat = None

        try:
            agent_outs, hidden_states = self.agent(
                inputs, mask,
                test_mode=actual_test_mode,
                population_belief=pop_flat,
                stage_t=st_flat,
            )
        except TypeError:
            agent_outs, hidden_states = self.agent(inputs, mask, test_mode=actual_test_mode)
        
        batch_size = ep_batch.batch_size
        
        belief_states = agent_outs.get('belief_state', torch.zeros(batch_size * self.n_agents, self.args.belief_dim, device=self.device))
        prompt_embeddings = agent_outs.get('prompt_embedding', torch.zeros(batch_size * self.n_agents, 2, device=self.device))
        q_values = agent_outs.get('q_value', torch.zeros(batch_size * self.n_agents, 1, device=self.device))
        action_q_values = agent_outs.get('action_q_values')
        stance_action_q_values = agent_outs.get('stance_action_q_values')
        action_type_q_values = agent_outs.get('action_type_q_values')
        
        belief_states = belief_states.view(batch_size, self.n_agents, -1)
        prompt_embeddings = prompt_embeddings.view(batch_size, self.n_agents, -1)
        q_values = q_values.view(batch_size, self.n_agents, -1)
        q_values_scalar = q_values.squeeze(-1) if isinstance(q_values, torch.Tensor) and q_values.shape[-1] == 1 else q_values
        if isinstance(action_q_values, torch.Tensor):
            action_q_values = action_q_values.view(batch_size, self.n_agents, -1)
        if isinstance(stance_action_q_values, torch.Tensor):
            stance_action_q_values = stance_action_q_values.view(batch_size, self.n_agents, -1)
        if isinstance(action_type_q_values, torch.Tensor):
            action_type_q_values = action_type_q_values.view(batch_size, self.n_agents, -1)
        

        freeze_enc_sup = bool(getattr(self.args, "freeze_belief_encoder_in_supervised", False))
        is_sup = bool(getattr(self.args, "train_belief_supervised", False))
        use_no_grad = bool(train_mode and is_sup and freeze_enc_sup)

        group_representation = None
        use_nonparam_gr = bool(getattr(self.args, "use_nonparam_group_repr", False))
        use_brief_enc = bool(getattr(self.args, "use_brief_encoder", False))
        if use_nonparam_gr:
            try:
                if "group_representation" in ep_batch.scheme:
                    grb = ep_batch["group_representation"][:, t]  # (bs, D_raw)
                    if isinstance(grb, torch.Tensor):
                        raw_gr = grb.to(self.device).float()
                        if use_brief_enc:
                            try:
                                be = self.belief_encoder_module
                                if be is not None and hasattr(be, "encode_brief"):
                                    group_representation = be.encode_brief(raw_gr, stage_t=stage_t)
                                else:
                                    group_representation = raw_gr
                            except Exception as e:
                                logger.warning(f"BriefEncoder encode_brief failed, fallback to raw group_representation: {e}")
                                group_representation = raw_gr
                        else:
                            group_representation = raw_gr
            except Exception:
                group_representation = None

        if group_representation is None:
            try:
                if use_no_grad:
                    with torch.no_grad():
                        group_representation = self.belief_encoder(
                            belief_states,
                            population_belief=population_belief,
                            stage_t=stage_t,
                        )  # (batch, belief_dim)
                else:
                    group_representation = self.belief_encoder(
                        belief_states,
                        population_belief=population_belief,
                        stage_t=stage_t,
                    )  # (batch, belief_dim)
            except Exception as e:
                logger.warning(f"BeliefEncoder forward with population_belief failed, fallback to vanilla: {e}")
                if use_no_grad:
                    with torch.no_grad():
                        group_representation = self.belief_encoder(belief_states)
                else:
                    group_representation = self.belief_encoder(belief_states)

        secondary_z_next = None
        secondary_action_probs = None
        try:
            be = self.belief_encoder_module
            if population_belief is not None and be is not None and hasattr(be, "predict_next_population_belief"):
                secondary_z_next = be.predict_next_population_belief(
                    population_belief,
                    group_repr=group_representation,
                    stage_t=stage_t,
                    return_logits=False,
                )  # (bs, K)
        except Exception as e:
            logger.debug(f"Secondary z_next prediction skipped: {e}")
            secondary_z_next = None

        try:
            be = self.belief_encoder_module
            if population_belief is not None and be is not None and getattr(be, "secondary_action_head", None) is not None:
                secondary_action_probs = be.predict_secondary_action_probs(
                    z_t=population_belief,
                    group_repr=group_representation,
                    stage_t=stage_t,
                    return_logits=False,
                )  # (bs, A)
        except Exception as e:
            logger.debug(f"Secondary action probs prediction skipped: {e}")
            secondary_action_probs = None
        
        try:
            avail_actions_t = ep_batch["avail_actions"][:, t]  # (bs, n_agents, n_avail)
            n_avail = int(avail_actions_t.shape[-1])
        except Exception:
            avail_actions_t = None
            n_avail = 1

        agent_outputs = q_values  # (bs, n_agents, 1)

        aq_src = None
        if n_avail == 3 and isinstance(stance_action_q_values, torch.Tensor):
            aq_src = stance_action_q_values
        elif n_avail == 5 and isinstance(action_type_q_values, torch.Tensor):
            aq_src = action_type_q_values
        elif isinstance(action_q_values, torch.Tensor):
            aq_src = action_q_values

        if isinstance(aq_src, torch.Tensor):
            aq = aq_src
            if aq.ndim == 2:
                aq = aq.unsqueeze(0)
            if aq.shape[-1] > n_avail:
                aq = aq[..., :n_avail]
            elif aq.shape[-1] < n_avail:
                pad = torch.full((batch_size, self.n_agents, n_avail - aq.shape[-1]), -1e9, device=aq.device, dtype=aq.dtype)
                aq = torch.cat([aq, pad], dim=-1)
            agent_outputs = aq
        else:
            if n_avail > 1 and isinstance(q_values, torch.Tensor) and q_values.shape[-1] == 1:
                agent_outputs = q_values.repeat(1, 1, n_avail)
        
        info_dict = {
            "belief_states": belief_states,
            "prompt_embeddings": prompt_embeddings,
            "q_values": q_values_scalar,
            "action_q_values": action_q_values if isinstance(action_q_values, torch.Tensor) else None,
            "stance_action_q_values": stance_action_q_values if isinstance(stance_action_q_values, torch.Tensor) else None,
            "action_type_q_values": action_type_q_values if isinstance(action_type_q_values, torch.Tensor) else None,
            "group_repr": group_representation,
            "secondary_z_next": secondary_z_next,
            "secondary_action_probs": secondary_action_probs,
            "hidden_states": hidden_states
        }
        
        return agent_outputs, info_dict

    def _get_strategy(self, question: str) -> str:
        """
        Generate strategy using coordinator LLM with token limit.
        
        Args:
            question: Input question
            
        Returns:
            Generated strategy (limited to 50 tokens)
        """
        if question in self.strategy_cache:
            return self.strategy_cache[question]
        
        strategy_prompt = f"""You are a coordinator for mathematical problem-solving agents. Analyze this math problem and provide a clear solving strategy WITHOUT calculating the final answer.

Problem: {question}

REQUIREMENTS:
1. Your response must be EXACTLY 50 tokens or less
2. Provide step-by-step approach and methodology ONLY
3. DO NOT calculate numbers or provide the final answer
4. Focus on the solving process and required operations
5. Emphasize that final answer must be in \\boxed{{numerical_answer}} format

IMPORTANT: 
- Do NOT solve the problem yourself
- Only provide strategy and method
- Keep response within 50 tokens

Strategy (method only, no calculations):"""
        
        try:
            strategy = self.coordinator.generate_response(
                prompt=strategy_prompt,
                temperature=0.3,  # Lower temperature for more consistent strategies
                max_tokens=50  # Strict limit to prevent exceeding
            )
            
            logger.info(f"📋 COORDINATOR STRATEGY: {strategy}")
            
            if len(self.strategy_cache) < self.max_cache_size:
                self.strategy_cache[question] = strategy
            
            return strategy
            
        except Exception as e:
            logger.warning(f"Failed to generate strategy: {e}")
            fallback_strategy = "Solve step by step: 1) Identify given values 2) Apply operations 3) Present answer in \\boxed{{}} format"
            logger.info(f"📋 COORDINATOR STRATEGY (FALLBACK): {fallback_strategy}")
            return fallback_strategy


    def _get_input_shape(self, scheme: Dict) -> int:
        """
        Get input shape for agents based on observation scheme.
        
        Args:
            scheme: Data scheme dictionary
            
        Returns:
            Input shape for agents
        """
        if hasattr(self.args, 'vocab_size'):
            return self.args.vocab_size
        else:
            return 50257

    def _get_default_actions(self, bs: slice, 
                           avail_actions: torch.Tensor) -> torch.Tensor:
        """Get default actions when agent forward fails."""
        batch_size = avail_actions.shape[0]
        random_actions = torch.randint(0, 2, (batch_size, self.n_agents), device=self.device)
        return random_actions

    def _get_default_outputs(self, ep_batch: Any) -> torch.Tensor:
        """Get default outputs when forward pass fails."""
        batch_size = ep_batch.batch_size
        return torch.zeros(batch_size, self.n_agents, self.args.n_actions, device=self.device)

    def cuda(self):
        """Move all components to CUDA."""
        try:
            if getattr(self, "agent", None) is not None and hasattr(self.agent, "cuda"):
                self.agent.cuda()
        except Exception as e:
            logger.warning(f"Failed to cuda() agent: {e}")
        try:
            if getattr(self, "belief_encoder", None) is not None and hasattr(self.belief_encoder, "cuda"):
                self.belief_encoder.cuda()
        except Exception as e:
            logger.warning(f"Failed to cuda() belief_encoder: {e}")
        for name in ("coordinator",):
            obj = getattr(self, name, None)
            if obj is None:
                continue
            fn = getattr(obj, "cuda", None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    logger.debug(f"Skipping {name}.cuda() due to error: {e}")

    def state_dict(self) -> Dict[str, Any]:
        """
        Provide a minimal state dict so learners can sync target networks via:
        target_mac.load_state_dict(mac.state_dict()).
        """
        sd: Dict[str, Any] = {}
        try:
            am = self.agent_module
            if am is not None and hasattr(am, "state_dict"):
                sd["agent"] = am.state_dict()
        except Exception as e:
            logger.warning(f"Failed to get agent state_dict: {e}")
        try:
            bm = self.belief_encoder_module
            if bm is not None and hasattr(bm, "state_dict"):
                sd["belief_encoder"] = bm.state_dict()
        except Exception as e:
            logger.warning(f"Failed to get belief_encoder state_dict: {e}")
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Load state produced by state_dict()."""
        if not isinstance(state_dict, dict):
            raise TypeError(f"LLMBasicMAC.load_state_dict expects dict, got: {type(state_dict)}")
        if "agent" in state_dict and self.agent_module is not None and hasattr(self.agent_module, "load_state_dict"):
            try:
                self.agent_module.load_state_dict(state_dict["agent"], strict=strict)
            except TypeError:
                self.agent_module.load_state_dict(state_dict["agent"])
        if "belief_encoder" in state_dict and self.belief_encoder_module is not None and hasattr(self.belief_encoder_module, "load_state_dict"):
            try:
                self.belief_encoder_module.load_state_dict(state_dict["belief_encoder"], strict=strict)
            except TypeError:
                self.belief_encoder_module.load_state_dict(state_dict["belief_encoder"])
        return self

    def save_models(self, path: str):
        """Save all model components."""
        self.agent_module.save_models(path)
        try:
            bm = self.belief_encoder_module
            if bm is not None:
                torch.save(bm.state_dict(), f"{path}/belief_encoder.th")
        except Exception as e:
            logger.warning(f"Failed to save belief_encoder: {e}")

    def load_models(self, path: str):
        """Load all model components."""
        self.agent_module.load_models(path)
        try:
            import os

            p = f"{path}/belief_encoder.th"
            bm = self.belief_encoder_module
            if bm is not None and os.path.exists(p):
                sd = torch.load(p, map_location=lambda storage, loc: storage)
                try:
                    bm.load_state_dict(sd, strict=True)
                except Exception as e_strict:
                    logger.warning(f"Strict load for belief_encoder failed ({e_strict}); retrying strict=False")
                    try:
                        cur = bm.state_dict()
                        filtered = {}
                        skipped_mismatch = 0
                        skipped_missing = 0
                        for k, v in sd.items():
                            if k not in cur:
                                skipped_missing += 1
                                continue
                            try:
                                if tuple(cur[k].shape) != tuple(v.shape):
                                    if (
                                        isinstance(k, str)
                                        and k.endswith("stage_embed.weight")
                                        and hasattr(cur[k], "shape")
                                        and hasattr(v, "shape")
                                        and (len(cur[k].shape) == 2)
                                        and (len(v.shape) == 2)
                                        and (int(cur[k].shape[1]) == int(v.shape[1]))
                                    ):
                                        try:
                                            tgt = cur[k].detach().clone()
                                            src = v.detach().clone() if hasattr(v, "detach") else v
                                            n = min(int(tgt.shape[0]), int(src.shape[0]))
                                            tgt[:n, :] = src[:n, :]
                                            filtered[k] = tgt
                                            continue
                                        except Exception:
                                            skipped_mismatch += 1
                                            continue
                                    skipped_mismatch += 1
                                    continue
                            except Exception:
                                skipped_mismatch += 1
                                continue
                            filtered[k] = v
                        missing, unexpected = bm.load_state_dict(filtered, strict=False)
                        logger.warning(
                            "BeliefEncoder partial-load (shape-filtered) applied: "
                            f"kept={len(filtered)}/{len(sd)}, skipped_missing={skipped_missing}, skipped_mismatch={skipped_mismatch}. "
                            f"missing_after={len(missing)}, unexpected_after={len(unexpected)}"
                        )
                    except Exception as e_partial:
                        logger.warning(f"BeliefEncoder partial-load failed ({e_partial}); retrying strict=False without filtering.")
                        bm.load_state_dict(sd, strict=False)
        except Exception as e:
            logger.warning(f"Failed to load belief_encoder: {e}")

    def _create_minimal_tokenizer(self):
        """Create a minimal tokenizer."""
        class MinimalTokenizer:
            class _Encoding(dict):
                """Minimal BatchEncoding-like container supporting both dict and attribute access."""
                @property
                def input_ids(self):
                    return self["input_ids"]

                @property
                def attention_mask(self):
                    return self.get("attention_mask", None)

            def __init__(self):
                self.vocab = {chr(i): i for i in range(32, 127)}  # ASCII printable characters
                self.vocab.update({'[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3})
                self.pad_token = '[PAD]'
                self.eos_token = '[EOS]'
                self.pad_token_id = 0
                self.eos_token_id = 3
                self.vocab_size = len(self.vocab)
                
            def __call__(
                self,
                text,
                max_length=None,
                padding=True,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
                return_attention_mask=False,
                **kwargs,
            ):
                if isinstance(text, str):
                    text = [text]
                
                tokenized = []
                attn_masks = []
                for t in text:
                    if max_length is not None and max_length > 0:
                        t = t[: max(0, max_length - 1)]

                    tokens = [self.vocab.get(c, 1) for c in t]
                    if add_special_tokens:
                        tokens.append(self.eos_token_id)  # EOS token
                    
                    if max_length and padding:
                        if len(tokens) < max_length:
                            tokens.extend([0] * (max_length - len(tokens)))  # PAD tokens
                        tokens = tokens[:max_length]
                    elif max_length and truncation:
                        tokens = tokens[:max_length]
                    
                    tokenized.append(tokens)
                    if return_attention_mask:
                        attn_masks.append([0 if tok == self.pad_token_id else 1 for tok in tokens])
                
                if return_tensors == "pt":
                    import torch
                    enc = self._Encoding({"input_ids": torch.tensor(tokenized, dtype=torch.long)})
                    if return_attention_mask:
                        enc["attention_mask"] = torch.tensor(attn_masks, dtype=torch.long)
                    return enc
                enc = self._Encoding({"input_ids": tokenized})
                if return_attention_mask:
                    enc["attention_mask"] = attn_masks
                return enc

            def encode(self, text, add_special_tokens=True, max_length=None, truncation=False, **kwargs):
                """HF-like encode() used by _truncate_to_tokens."""
                enc = self(
                    text,
                    max_length=max_length,
                    padding=False,
                    truncation=truncation,
                    return_tensors=None,
                    add_special_tokens=add_special_tokens,
                    return_attention_mask=False,
                )
                ids = enc["input_ids"]
                if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                    return ids[0]
                return ids
                
            def decode(self, token_ids, skip_special_tokens=True):
                if hasattr(token_ids, 'tolist'):
                    token_ids = token_ids.tolist()
                
                text = ""
                reverse_vocab = {v: k for k, v in self.vocab.items()}
                for token_id in token_ids:
                    char = reverse_vocab.get(token_id, '[UNK]')
                    if skip_special_tokens and char in ['[PAD]', '[UNK]', '[BOS]', '[EOS]']:
                        continue
                    text += char
                return text
        
        logger.warning("Using minimal character-level tokenizer - this may affect performance")
        return MinimalTokenizer()

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        截断文本到指定的token数量限制
        
        Args:
            text: 要截断的文本
            max_tokens: 最大token数量
            
        Returns:
            截断后的文本
        """
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                
                logger.debug(f"Truncated text from {len(tokens)} to {len(truncated_tokens)} tokens")
                return truncated_text
            else:
                return text
                
        except Exception as e:
            logger.warning(f"Error during token truncation: {e}")
            words = text.split()
            if len(words) > max_tokens:
                return " ".join(words[:max_tokens])
            return text
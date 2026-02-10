import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from modules.llm.llm_wrapper import ImprovedLLMWrapper
from loguru import logger

class TransformerMixingNetwork(nn.Module):
    """Transformer-based mixing network for Q-value integration."""
    def __init__(self, args: Any):
        super().__init__()
        
        self.entity_embedding = nn.Linear(args.state_dim, args.entity_dim)
        
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, args.max_seq_length, args.entity_dim)
        )
        
        self.layer_norm = nn.LayerNorm(args.entity_dim, eps=args.layer_norm_epsilon)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=args.entity_dim,
            num_heads=args.attention_heads,
            dropout=args.dropout_rate,
            batch_first=True
        )
        
        self.ff_network = nn.Sequential(
            nn.Linear(args.entity_dim, args.feedforward_size),
            nn.ReLU(),
            nn.Dropout(args.dropout_rate),
            nn.Linear(args.feedforward_size, args.entity_dim),
            nn.Dropout(args.dropout_rate)
        )

    def forward(self, states: torch.Tensor, agent_qs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the mixing network.
        
        Args:
            states: State representations
            agent_qs: Agent Q-values
            mask: Optional attention mask
            
        Returns:
            Mixed Q-values
        """
        state_embed = self.entity_embedding(states)
        if len(state_embed.shape) == 2:
            state_embed = state_embed.unsqueeze(1)
            
        state_embed = state_embed + self.pos_encoder[:, :state_embed.size(1)]
        
        state_embed = self.layer_norm(state_embed)
        
        state_attended, _ = self.attention(
            state_embed, state_embed, state_embed,
            key_padding_mask=mask
        )
        
        state_mixed = self.layer_norm(state_embed + state_attended)
        
        state_final = self.layer_norm(state_mixed + self.ff_network(state_mixed))
        
        return state_final

class LLMQMixer(nn.Module):
    """
    Q-Mixer with transformer architecture and LLM integration.
    Handles Q-value mixing and commitment generation.
    ECON Adaptation: This mixer will be adapted to process prompt embeddings,
    group representation, and local Q-values as per the ECON paper.
    """
    def __init__(self, args: Any):
        """
        Initialize the LLM Q-Mixer.
        
        Args:
            args: Configuration arguments containing model parameters
        """
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.mixing_embed_dim = getattr(args, "mixing_embed_dim", 32) # Default from typical QMIX
        self.entity_dim = getattr(args, "entity_dim", 64) # General purpose dimension
        self.belief_dim = args.belief_dim # Dimension of E^t
        self.mlp_hidden_size = getattr(args, "mlp_hidden_size", 128)
        self.layer_norm_epsilon = getattr(args, "layer_norm_epsilon", 1e-5)
        self.dropout_rate = getattr(args, "dropout_rate", 0.1)
        self.commitment_embedding_dim = getattr(args, "commitment_embedding_dim", self.entity_dim)

        self.prompt_embedding_dim = 2 
        
        self.prompt_attention_heads = getattr(args, "prompt_attention_heads", 4)
        self.prompt_self_attention = nn.MultiheadAttention(
            embed_dim=self.prompt_embedding_dim, 
            num_heads=self.prompt_attention_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.w_i_projection_dim = getattr(args, "w_i_dim", self.entity_dim // 2)
        self.project_w_i = nn.Linear(self.prompt_embedding_dim, self.w_i_projection_dim)

        self.F_i_input_dim = self.w_i_projection_dim + self.belief_dim
        self.F_i_dim = getattr(args, "F_i_dim", self.entity_dim)
        self.feature_transform_F_i = nn.Sequential(
            nn.Linear(self.F_i_input_dim, self.F_i_dim),
            nn.ReLU(),
            nn.LayerNorm(self.F_i_dim, eps=self.layer_norm_epsilon) 
        )

        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.belief_dim, self.mlp_hidden_size), 
            nn.ReLU(),
            nn.LayerNorm(self.mlp_hidden_size, eps=self.layer_norm_epsilon),
            nn.Linear(self.mlp_hidden_size, self.n_agents * self.mixing_embed_dim)
        )
        
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(self.belief_dim, self.mlp_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.mlp_hidden_size, eps=self.layer_norm_epsilon),
            nn.Linear(self.mlp_hidden_size, self.mixing_embed_dim)
        )
        
        self.hyper_b_1 = nn.Linear(self.belief_dim, self.mixing_embed_dim)
        self.hyper_b_2 = nn.Sequential( 
            nn.Linear(self.belief_dim, self.mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size // 2, 1)
        )
        
        self.coordinator = ImprovedLLMWrapper(
            api_key=args.together_api_key,
            model_name=args.coordinator_model
        )
        
        self.commitment_eval_input_dim = getattr(args, "commitment_eval_input_dim", self.entity_dim * 2)
        self.commitment_eval = nn.Sequential(
            nn.Linear(self.commitment_eval_input_dim, self.mlp_hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.mlp_hidden_size, eps=self.layer_norm_epsilon),
            nn.Linear(self.mlp_hidden_size, 1)
        )
        
        self.F_i_project_for_LSD = nn.Linear(self.F_i_dim, self.commitment_embedding_dim)

        self.reward_weight_network = nn.Sequential(
            nn.Linear(self.belief_dim, self.mlp_hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(self.mlp_hidden_size // 2, eps=self.layer_norm_epsilon),
            nn.Linear(self.mlp_hidden_size // 2, 3) 
        )
        
        self.max_cache_size = getattr(args, 'max_cache_size', 1000)
        self.commitment_cache = {}
        
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def forward(self, 
                local_q_values: torch.Tensor,      # {Q_i^t} from BeliefNetwork
                prompt_embeddings: torch.Tensor, # {e_i^t} from BeliefNetwork, shape: (batch, n_agents, 2)
                group_representation: torch.Tensor,# E^t from BeliefEncoder, shape: (batch, belief_dim)
                agent_raw_outputs: Optional[List[List[str]]] = None, # List per batch item, then list of strings {u_i^t}
                commitment_text_features: Optional[torch.Tensor] = None # Optional pre-computed C features for L_SD
                ) -> Dict[str, Any]:
        """
        Forward pass for ECON's CentralizedMixingNetwork.
        
        Args:
            local_q_values: (batch_size, n_agents) - Q_i^t(τ_i^t, e_i^t; φ_i)
            prompt_embeddings: (batch_size, n_agents, 2) - e_i^t = [T_i, p_i]
            group_representation: (batch_size, belief_dim) - E^t
            agent_raw_outputs: Optional: List (batch_size) of lists of strings from executor LLMs.
            commitment_text_features: Optional pre-computed features of coordinator's commitment C for L_SD.
                                 Shape: (batch_size, commitment_embedding_dim)
            
        Returns:
            Dictionary containing:
                - Q_tot: Global Q-value (batch_size)
                - F_i_transformed: Transformed features F_i (batch, n_agents, F_i_dim)
                - generated_commitment_text: Generated commitment string (List[str] for batch, or None)
                - other relevant tensors for loss calculation
        """
        batch_size = local_q_values.size(0)
        
        attn_output_w, _ = self.prompt_self_attention(prompt_embeddings, prompt_embeddings, prompt_embeddings)
        w_i_projected = self.project_w_i(attn_output_w) # (batch, n_agents, w_i_projection_dim)

        E_t_expanded = group_representation.unsqueeze(1).repeat(1, self.n_agents, 1) 
        combined_for_F_i = torch.cat([w_i_projected, E_t_expanded], dim=-1)
        
        combined_for_F_i_flat = combined_for_F_i.view(batch_size * self.n_agents, -1)
        F_i_transformed_flat = self.feature_transform_F_i(combined_for_F_i_flat)
        F_i_transformed = F_i_transformed_flat.view(batch_size, self.n_agents, self.F_i_dim)

        agent_qs_reshaped = local_q_values.unsqueeze(-1) # (batch, n_agents, 1)

        w1 = torch.abs(self.hyper_w_1(group_representation)) 
        w1 = w1.view(batch_size, self.n_agents, self.mixing_embed_dim)
        
        b1 = self.hyper_b_1(group_representation) 
        b1 = b1.view(batch_size, 1, self.mixing_embed_dim)

        hidden = F.elu(torch.sum(agent_qs_reshaped * w1, dim=1, keepdim=True) + b1) # (batch, 1, mixing_embed_dim)

        w2 = torch.abs(self.hyper_w_2(group_representation))
        w2 = w2.view(batch_size, self.mixing_embed_dim, 1) 

        b2 = self.hyper_b_2(group_representation)
        b2 = b2.view(batch_size, 1, 1)

        Q_tot = torch.bmm(hidden, w2) + b2 
        Q_tot = Q_tot.squeeze(-1).squeeze(-1) # (batch_size)

        generated_commitment_texts = None
        if agent_raw_outputs is not None:
            generated_commitment_texts = []
            for i in range(batch_size):
                current_E_t = group_representation[i:i+1] # Keep batch dim for coordinator if it expects it
                commitment_str = self._generate_commitment(
                    agent_raw_outputs=agent_raw_outputs[i], # List[str] for current batch item
                )
                generated_commitment_texts.append(commitment_str)

        outputs = {
            "Q_tot": Q_tot,                           
            "F_i_transformed": F_i_transformed,       
            "local_q_values": local_q_values,         
            "generated_commitment_text": generated_commitment_texts, 
        }
        
        F_i_for_LSD = self.F_i_project_for_LSD(F_i_transformed) 
        outputs["F_i_for_LSD"] = F_i_for_LSD

        return outputs

    def _generate_commitment(self,
                           agent_raw_outputs: List[str],
                           group_representation_single: Optional[torch.Tensor] = None) -> str:
        """
        Generates a commitment for a single batch item using the Coordinator LLM.
        """
        formatted_answers = self._format_answers(agent_raw_outputs)
        prompt = f"The following answers were provided by executor agents:\n{formatted_answers}\nSynthesize a final commitment."
        
        commitment = self.coordinator.generate_response(
            question=prompt,
            temperature=getattr(self.args, "coordinator_temperature", 0.7),
            repetition_penalty=getattr(self.args, "coordinator_repetition_penalty", 1.0)
        )
        return commitment
            
    def _format_answers(self, answers: List[str]) -> str:
        """Format agent answers for the prompt."""
        return "\n".join([f"Agent {i+1}: {answer}" for i, answer in enumerate(answers)])

    def calculate_mix_loss(self, 
                           Q_tot: torch.Tensor, 
                           local_q_values: torch.Tensor,
                           F_i_for_LSD: Optional[torch.Tensor], # Projected F_i, (batch, n_agents, C_embed_dim)
                           commitment_text_features: Optional[torch.Tensor], # C features, (batch, C_embed_dim)
                           target_Q_tot: torch.Tensor,
                           rewards_total: torch.Tensor, 
                           gamma: float,
                           lambda_sd: float,
                           lambda_m: float,
                           terminated: torch.Tensor, # Batch of booleans indicating termination
                           mask_flat: torch.Tensor # Flattened mask for TD loss calculation
                           ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates all components of the L_mix loss for ECON.
        L_mix(φ) = L_TD^tot(φ) + L_SD + λ_m Σ_i ||Q_i^t - Q_tot^t||^2
        
        Args:
            Q_tot: Current total Q value from the mixer (batch_size)
            local_q_values: Q_i^t from belief networks (batch_size, n_agents)
            F_i_for_LSD: Projected F_i features for L_SD (batch, n_agents, commitment_embedding_dim) or None.
            commitment_text_features: Features of coordinator's commitment C (batch, commitment_embedding_dim) or None.
            target_Q_tot: Target Q_tot (e.g., from target network) (batch_size)
            rewards_total: Global reward r_tot (batch_size)
            gamma: Discount factor.
            lambda_sd: Weight for L_SD.
            lambda_m: Weight for local-global Q consistency loss.
            terminated: Tensor indicating if the episode terminated for each batch item (batch_size).
            mask_flat: Flattened mask for TD loss calculation
            
        Returns:
            total_mix_loss: The L_mix loss.
            loss_components: Dictionary of individual loss values.
        """
        loss_components = {}

        if isinstance(mask_flat, torch.Tensor) and mask_flat.ndim > 1:
            mask_flat = mask_flat.view(-1)
        if isinstance(rewards_total, torch.Tensor) and rewards_total.ndim > 1:
            rewards_total = rewards_total.view(-1)
        if isinstance(terminated, torch.Tensor) and terminated.ndim > 1:
            terminated = terminated.view(-1)
        if isinstance(target_Q_tot, torch.Tensor) and target_Q_tot.ndim > 1:
            target_Q_tot = target_Q_tot.view(-1)
        if isinstance(Q_tot, torch.Tensor) and Q_tot.ndim > 1:
            Q_tot = Q_tot.view(-1)

        td_target = rewards_total + gamma * target_Q_tot * (1 - terminated.float())
        
        td_error_tot = (Q_tot - td_target.detach()) # expected (N,)
        masked_td_error_tot = td_error_tot * mask_flat 
        loss_td_tot = (masked_td_error_tot**2).sum() / mask_flat.sum().clamp(min=1e-6) # clamp for stability
        loss_components["L_TD_tot"] = loss_td_tot

        loss_sd = torch.tensor(0.0, device=self.device)
        if lambda_sd > 0 and F_i_for_LSD is not None and commitment_text_features is not None:
            C_embedding_expanded = commitment_text_features.unsqueeze(1)
            
            cosine_sim = F.cosine_similarity(F_i_for_LSD, C_embedding_expanded, dim=-1) # (batch, n_agents)
            
            loss_sd_per_agent = (1.0 - cosine_sim).pow(2)
            loss_sd = loss_sd_per_agent.sum(dim=1).mean() # Sum over agents, mean over batch items
        loss_components["L_SD"] = loss_sd

        loss_q_consistency = torch.tensor(0.0, device=self.device)
        if lambda_m > 0:
            Q_tot_expanded = Q_tot.unsqueeze(1).repeat(1, self.n_agents) # (batch, n_agents)
            loss_q_consistency = (local_q_values - Q_tot_expanded.detach()).pow(2).sum(dim=1).mean()
        loss_components["L_Q_consistency"] = loss_q_consistency

        total_mix_loss = loss_td_tot + lambda_sd * loss_sd + lambda_m * loss_q_consistency
        loss_components["L_mix_total"] = total_mix_loss

        if not torch.isfinite(total_mix_loss):
            logger.warning(f"[LLMQMixer] total_mix_loss is not finite: {total_mix_loss}. "
                           f"Shapes: Q_tot={tuple(Q_tot.shape)}, td_target={tuple(td_target.shape)}, mask={tuple(mask_flat.shape)}")
        
        return total_mix_loss, loss_components

    def calculate_similarity_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.tensor(0.0, device=self.device)
    
    def calculate_reward_diff_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.tensor(0.0, device=self.device)
    
    def _get_default_outputs(self, agent_qs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return default outputs when errors occur."""
        return {'q_tot': agent_qs.mean(dim=1)}
    
    def _get_default_commitment(self, answers: List[str]) -> str:
        """
        Return a default commitment when commitment generation fails.
        
        Args:
            answers: Agent answers
            
        Returns:
            Default commitment
        """
        return "Based on the available information, I need more context to provide a definitive answer."
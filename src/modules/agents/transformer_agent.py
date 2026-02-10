import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
from modules.llm.llm_wrapper import ImprovedLLMWrapper
import logging
import json
import re
import os

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    位置编码层，为 Transformer 输入提供位置信息。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerBlock(nn.Module):
    """
    自注意力 Transformer 块，基础构建块。
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_mask = None
        key_padding_mask = None
        if mask is not None:
            if isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.shape[0] == x.shape[0]:
                key_padding_mask = mask
            elif isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.shape[0] == mask.shape[1]:
                attn_mask = mask

        attended, _ = self.attention(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        
        attended = self.norm1(x + attended)
        
        ff_output = self.feed_forward(attended)
        
        output = self.norm2(attended + ff_output)
        
        return output

class BeliefNetwork(nn.Module):
    """
    个体置信网络 B_i，用于维护和更新智能体的置信状态 b_i。
    根据ECON论文，此网络接收局部轨迹 τ_i^t 和当前观察 o_i^t，
    输出置信状态 b_i, prompt embedding e_i = [T_i, p_i], 和局部 Q值 Q_i^t。
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, belief_dim: int, 
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
                 T_min: float = 0.1, T_max: float = 2.0, p_min: float = 0.1, p_max: float = 0.9,
                 vocab_size: int = 50257):  # 添加词汇表大小参数，GPT2的默认值
        super(BeliefNetwork, self).__init__()
        
        self.observation_dim = observation_dim  # 这是max_token_length
        self.belief_dim = belief_dim
        self.T_min = T_min
        self.T_max = T_max
        self.p_min = p_min
        self.p_max = p_max
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                ff_dim=hidden_dim * 4, # Standard practice for ff_dim
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        self.belief_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Added a layer for more capacity
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim)
        )
        
        self.temp_projection = nn.Linear(belief_dim, 1) # W_T b_i + b_T
        self.penalty_projection = nn.Linear(belief_dim, 1) # W_p b_i + b_p
        
        self.q_network = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出标量 Q 值
        )
        
    def forward(self, token_ids: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播，从token IDs生成置信状态、提示嵌入和 Q 值。
        
        Args:
            token_ids: Token IDs张量，形状为 (batch_size, seq_len) 或 (batch_size, 1, seq_len)
            mask: 可选的注意力掩码
            
        Returns:
            包含置信状态 b_i、缩放后的提示嵌入 e_i = [T_i, p_i] 和局部 Q值 Q_i^t 的字典。
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
        if token_ids.ndim == 3:
            token_ids = token_ids.squeeze(1)    # (batch, 1, seq_len) -> (batch, seq_len)
        
        x = self.token_embedding(token_ids.long())  # (batch_size, seq_len, hidden_dim)
        
        x = self.pos_encoder(x) # x is (batch, seq_len, hidden_dim)
        
        for layer in self.transformer_layers:
            x = layer(x, mask) # x is (batch, seq_len, hidden_dim)
        
        if mask is not None:
            valid_lengths = (~mask).sum(dim=1)  # 每个序列的有效长度
            batch_indices = torch.arange(x.size(0), device=x.device)
            last_valid_indices = (valid_lengths - 1).clamp(min=0)
            processed_sequence = x[batch_indices, last_valid_indices]  # (batch, hidden_dim)
        else:
            processed_sequence = x[:, -1]  # (batch, hidden_dim)
            
        belief_state = self.belief_projection(processed_sequence) # (batch, belief_dim)
        
        
        temp_logit = self.temp_projection(belief_state) # (batch, 1)
        penalty_logit = self.penalty_projection(belief_state) # (batch, 1)
        
        temperature = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(temp_logit)
        penalty = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(penalty_logit)
        
        prompt_embedding_scaled = torch.cat([temperature, penalty], dim=1)
        
        q_value = self.q_network(belief_state) # (batch, 1)
        
        return {
            'belief_state': belief_state,          # b_i
            'prompt_embedding': prompt_embedding_scaled, # e_i = [T_i, p_i]
            'q_value': q_value,                    # Q_i^t
            'temp_logit': temp_logit,              # 原始温度 logit
            'penalty_logit': penalty_logit         # 原始惩罚 logit
        }

class LLMTransformerAgent(nn.Module):
    """
    基于 Transformer 的 LLM 智能体，维护置信状态并生成动态提示嵌入。
    """
    def __init__(self, input_shape: int, args: Any): # input_shape 现在代表 observation_dim + action_dim 或仅 observation_dim
        super(LLMTransformerAgent, self).__init__()
        
        self.args = args
        self.belief_dim = args.belief_dim
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        sampling_cfg = getattr(args, "sampling", None)
        if sampling_cfg is None:
            try:
                from types import SimpleNamespace
                sampling_cfg = SimpleNamespace()
                setattr(args, "sampling", sampling_cfg)
            except Exception:
                sampling_cfg = None

        self.T_min = getattr(sampling_cfg, 'temperature_min', 0.1) if sampling_cfg is not None else 0.1
        self.T_max = getattr(sampling_cfg, 'temperature_max', 2.0) if sampling_cfg is not None else 2.0
        self.p_min = getattr(sampling_cfg, 'p_min', 0.1) if sampling_cfg is not None else 0.1
        self.p_max = getattr(sampling_cfg, 'p_max', 0.9) if sampling_cfg is not None else 0.9
        
        max_token_len = getattr(args.env_args, "max_question_length", 512)
        belief_network_input_dim = max_token_len  # 使用tokenized观察的长度

        self.belief_network = BeliefNetwork(
            observation_dim=belief_network_input_dim, # 明确传递观察维度
            action_dim=0, # 假设动作已包含在观察中或不直接作为独立输入给BeliefNetwork基础嵌入层
            hidden_dim=getattr(args.arch, 'entity_dim', 256),
            belief_dim=self.belief_dim,
            n_heads=getattr(args.arch, 'attention_heads', 4),
            n_layers=getattr(args.arch, 'transformer_blocks', 2),
            dropout=getattr(args.arch, 'dropout_rate', 0.1),
            T_min=self.T_min, T_max=self.T_max, 
            p_min=self.p_min, p_max=self.p_max,
            vocab_size=getattr(args, 'vocab_size', 50257)  # 添加词汇表大小
        )
        
        self.stance_n_actions = int(getattr(args, "stance_n_actions", 3))
        self.stance_n_actions = max(1, self.stance_n_actions)
        self.action_type_n_actions = int(getattr(args, "action_type_n_actions", getattr(args, "n_actions", 5)))
        self.action_type_n_actions = max(1, self.action_type_n_actions)

        self.stance_head = nn.Linear(self.belief_dim, self.stance_n_actions)
        self.action_type_head = nn.Linear(self.belief_dim, self.action_type_n_actions)
        self.output_network = self.action_type_head

        self.s3b_bias_enabled = bool(getattr(args, "s3b_bias_enabled", False))
        self.s3b_bias_mode = str(getattr(args, "s3b_bias_mode", "diff01") or "diff01").strip().lower()
        self.s3b_bias_alpha = float(getattr(args, "s3b_bias_alpha", 1.0))
        self.s3b_bias_alpha_trainable = bool(getattr(args, "s3b_bias_alpha_trainable", False))
        self.s3b_bias_alpha_max = float(getattr(args, "s3b_bias_alpha_max", 0.0) or 0.0)
        self.s3b_bias_gate_by_stance = bool(getattr(args, "s3b_bias_gate_by_stance", False))
        self.s3b_bias_stance_min_conf = float(getattr(args, "s3b_bias_stance_min_conf", 0.5) or 0.5)
        self.s3b_prior_head = None
        if self.s3b_bias_enabled:
            self.s3b_prior_head = nn.Linear(self.belief_dim, self.action_type_n_actions)
            for p in self.s3b_prior_head.parameters():
                p.requires_grad = False
            if self.s3b_bias_alpha_trainable:
                alpha0 = float(max(self.s3b_bias_alpha, 1e-6))
                self.s3b_bias_alpha_param = nn.Parameter(torch.tensor(float(np.log(alpha0))))
            else:
                self.s3b_bias_alpha_param = None
        self.last_s3b_bias_alpha = float("nan")
        self.last_s3b_bias_logit_mean = float("nan")
        self.last_s3b_bias_logit_std = float("nan")
        self.last_s3b_bias_applied_frac = float("nan")

        self.use_population_belief_in_action_head = bool(getattr(args, "use_population_belief_in_action_head", False))
        try:
            self.population_belief_dim = int(
                getattr(args, "population_belief_dim", getattr(getattr(args, "env_args", None), "population_belief_dim", 3))
            )
        except Exception:
            self.population_belief_dim = 3
        self.population_belief_dim = max(1, int(self.population_belief_dim))
        self.population_belief_proj = None
        self.population_belief_gate_logit = None
        if self.use_population_belief_in_action_head:
            self.population_belief_proj = nn.Sequential(
                nn.Linear(self.population_belief_dim, self.belief_dim),
                nn.Tanh(),
            )
            self.population_belief_gate_logit = nn.Parameter(torch.tensor(-2.0, device=self.device))
        
        self.llm_wrapper = ImprovedLLMWrapper(
            api_key=args.together_api_key,
            model_name=args.executor_model,
            belief_dim=self.belief_dim # LLM Wrapper 可能也需要信念状态
        )
        
        self.current_prompt_embedding_tensor = torch.tensor([ (self.T_min + self.T_max) / 2, (self.p_min + self.p_max) / 2 ], device=self.device) # (2,)
        
        self.current_prompt_embedding = {
            'temperature': (self.T_min + self.T_max) / 2,
            'repetition_penalty': (self.p_min + self.p_max) / 2
        }
        
    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        test_mode: bool = False,
        hidden_state: Optional[torch.Tensor] = None,
        temperature: Optional[Any] = None,
        repetition_penalty: Optional[Any] = None,
        population_belief: Optional[torch.Tensor] = None,
        stage_t: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播，生成动作概率、置信状态和提示嵌入。
        
        Args:
            inputs: 代表 τ_i^t 和 o_i^t 的输入张量
            mask: 可选的注意力掩码
            test_mode: 是否处于测试模式
            hidden_state: (兼容参数) 不再使用
            temperature: 可选覆盖温度（float / tensor），形状支持 (bs,) / (bs,1) / 标量
            repetition_penalty: 可选覆盖重复惩罚（float / tensor），形状支持 (bs,) / (bs,1) / 标量
            
        Returns:
            (outputs_dict, hidden_state=None)
        """
        freeze_bn_rl = bool(getattr(self.args, "freeze_belief_network_in_rl", False))
        use_no_grad_bn = bool(self.training) and (not bool(test_mode)) and freeze_bn_rl
        if use_no_grad_bn:
            with torch.no_grad():
                belief_outputs = self.belief_network(inputs, mask)
        else:
            belief_outputs = self.belief_network(inputs, mask)
        
        belief_state = belief_outputs['belief_state']
        prompt_embedding = belief_outputs['prompt_embedding'] # e_i
        local_q_value = belief_outputs['q_value'] # Q_i^t
        temp_logit = belief_outputs['temp_logit'] # 原始温度 logit
        penalty_logit = belief_outputs['penalty_logit'] # 原始惩罚 logit

        if self.use_population_belief_in_action_head and (self.population_belief_proj is not None) and (population_belief is not None):
            try:
                z = population_belief
                if isinstance(z, torch.Tensor):
                    z = z.to(self.device, dtype=belief_state.dtype)
                    if z.ndim == 1:
                        z = z.view(-1, 1)
                    if z.ndim == 3 and z.shape[1] == 1:
                        z = z.squeeze(1)
                    if z.ndim == 2 and z.shape[1] != int(self.population_belief_dim):
                        z = z.reshape(z.shape[0], int(self.population_belief_dim))
                    if z.ndim == 2 and belief_state.ndim == 2 and z.shape[0] == belief_state.shape[0]:
                        try:
                            if bool(self.training) and (not bool(test_mode)):
                                p_drop = float(getattr(self.args, "s3b_z_drop_prob", 0.0))
                                p_shuffle = float(getattr(self.args, "s3b_z_shuffle_prob", 0.0))
                                p_drop = max(0.0, min(1.0, p_drop))
                                p_shuffle = max(0.0, min(1.0, p_shuffle))
                                if p_shuffle > 0 and torch.rand((), device=z.device).item() < p_shuffle:
                                    perm = torch.randperm(z.shape[0], device=z.device)
                                    z = z[perm]
                                if p_drop > 0:
                                    mask = (torch.rand((z.shape[0], 1), device=z.device) < p_drop)
                                    z = torch.where(mask, torch.zeros_like(z), z)
                        except Exception:
                            pass

                        z_emb = self.population_belief_proj(z)
                        gate = torch.sigmoid(self.population_belief_gate_logit) if self.population_belief_gate_logit is not None else 1.0
                        belief_state = belief_state + gate * z_emb
            except Exception:
                pass
        
        bs = int(belief_state.shape[0]) if isinstance(belief_state, torch.Tensor) else 1

        def _coerce_param(v: Any, *, default_val: float, lo: float, hi: float) -> torch.Tensor:
            if v is None:
                t = torch.full((bs, 1), float(default_val), device=self.device, dtype=belief_state.dtype)
            elif isinstance(v, torch.Tensor):
                t = v.to(self.device, dtype=belief_state.dtype)
                if t.ndim == 0:
                    t = t.view(1, 1).expand(bs, 1)
                elif t.ndim == 1:
                    t = t.view(-1, 1)
                elif t.ndim == 2 and t.shape[1] == 1:
                    pass
                else:
                    t = t.reshape(bs, 1)
            else:
                t = torch.full((bs, 1), float(v), device=self.device, dtype=belief_state.dtype)
            return torch.clamp(t, min=float(lo), max=float(hi))

        if temperature is not None or repetition_penalty is not None:
            temp_t = _coerce_param(
                temperature,
                default_val=float(prompt_embedding[:, 0].mean().item()) if bs > 0 else float((self.T_min + self.T_max) / 2),
                lo=float(self.T_min),
                hi=float(self.T_max),
            )
            pen_t = _coerce_param(
                repetition_penalty,
                default_val=float(prompt_embedding[:, 1].mean().item()) if bs > 0 else float((self.p_min + self.p_max) / 2),
                lo=float(self.p_min),
                hi=float(self.p_max),
            )
            prompt_embedding = torch.cat([temp_t, pen_t], dim=1)  # (bs,2)
        elif test_mode:
            temp_t = torch.full((bs, 1), float((self.T_min + self.T_max) / 2), device=self.device, dtype=belief_state.dtype)
            pen_t = torch.full((bs, 1), float((self.p_min + self.p_max) / 2), device=self.device, dtype=belief_state.dtype)
            prompt_embedding = torch.cat([temp_t, pen_t], dim=1)

        if prompt_embedding is not None and isinstance(prompt_embedding, torch.Tensor) and prompt_embedding.numel() >= 2:
            try:
                self.current_prompt_embedding_tensor = prompt_embedding[0].detach().clone()
            except Exception:
                pass


        stance_action_q_values = self.stance_head(belief_state)
        action_type_q_values = self.action_type_head(belief_state)

        try:
            if self.s3b_bias_enabled and (not bool(getattr(self.args, "train_action_imitation", False))):
                if isinstance(self.s3b_prior_head, nn.Module):
                    with torch.no_grad():
                        prior_logits = self.s3b_prior_head(belief_state)
                    if prior_logits.shape[-1] >= 2:
                        bias = (prior_logits[:, 1] - prior_logits[:, 0]).view(-1)
                        apply_mask = torch.ones_like(bias, dtype=torch.float32)
                        if self.s3b_bias_gate_by_stance and isinstance(stance_action_q_values, torch.Tensor):
                            try:
                                sp = torch.softmax(stance_action_q_values, dim=-1)
                                maxp = sp.max(dim=-1)[0].view(-1)
                                apply_mask = (maxp >= float(self.s3b_bias_stance_min_conf)).float()
                            except Exception:
                                apply_mask = torch.ones_like(bias, dtype=torch.float32)
                        if self.s3b_bias_alpha_trainable and isinstance(self.s3b_bias_alpha_param, torch.Tensor):
                            alpha_val = torch.exp(self.s3b_bias_alpha_param)
                        else:
                            alpha_val = torch.tensor(float(max(self.s3b_bias_alpha, 0.0)), device=bias.device)
                        if float(self.s3b_bias_alpha_max) > 0:
                            alpha_val = torch.clamp(alpha_val, max=float(self.s3b_bias_alpha_max))
                        action_type_q_values = action_type_q_values.clone()
                        adj = alpha_val * bias * apply_mask
                        action_type_q_values[:, 1] = action_type_q_values[:, 1] + adj
                        action_type_q_values[:, 0] = action_type_q_values[:, 0] - adj
                        try:
                            self.last_s3b_bias_alpha = float(alpha_val.detach().item())
                            if bool(apply_mask.any().item()):
                                m = apply_mask > 0.5
                                bsel = bias[m]
                                self.last_s3b_bias_logit_mean = float(bsel.mean().item())
                                self.last_s3b_bias_logit_std = float(bsel.std(unbiased=False).item())
                            else:
                                self.last_s3b_bias_logit_mean = float("nan")
                                self.last_s3b_bias_logit_std = float("nan")
                            self.last_s3b_bias_applied_frac = float(apply_mask.mean().item())
                        except Exception:
                            pass
        except Exception:
            pass
        
        outputs = {
            "action_q_values": action_type_q_values,
            "stance_action_q_values": stance_action_q_values,
            "action_type_q_values": action_type_q_values,
            "belief_state": belief_state,       # b_i
            "prompt_embedding": prompt_embedding, # e_i = [T_i, p_i]
            "q_value": local_q_value,           # Q_i^t - 保持与BeliefNetwork一致的字段名
            "raw_prompt_embed_params": torch.cat([temp_logit, penalty_logit], dim=1) # 保存原始 logits, 用于可能的后续分析或损失计算
        }
        
        return outputs, None
    
    def generate_answer(
        self,
        question: str,
        strategy: str,
        belief_state: Optional[torch.Tensor] = None,  # 可选，因为agent内部会生成
        temperature: Optional[float] = None,  # 将从 self.current_prompt_embedding_tensor 获取
        repetition_penalty: Optional[float] = None,  # 对应论文的 p_i
        forced_action_type: Optional[str] = None,
        forced_stance_id: Optional[int] = None,
    ) -> str:
        """
        使用 LLM 生成答案，基于当前的置信状态和动态生成的提示嵌入。
        
        Args:
            question: 输入的问题
            strategy: 协调器提供的策略
            belief_state: (可选) 当前的置信状态，如果未提供，agent将尝试使用其内部状态
            temperature: (可选) 覆盖动态生成的温度
            repetition_penalty: (可选) 覆盖动态生成的重复惩罚 p_i
            
        Returns:
            LLM 生成的答案字符串
        """
        
        current_temp = self.current_prompt_embedding_tensor[0].item()
        current_penalty = self.current_prompt_embedding_tensor[1].item()

        final_temp = temperature if temperature is not None else current_temp
        final_penalty = repetition_penalty if repetition_penalty is not None else current_penalty
        
        self.current_prompt_embedding['temperature'] = final_temp
        self.current_prompt_embedding['repetition_penalty'] = final_penalty

        fa = str(forced_action_type).strip().lower() if forced_action_type is not None else ""
        fs = None
        try:
            fs = int(forced_stance_id) if forced_stance_id is not None else None
        except Exception:
            fs = None

        constraint_block = ""
        if fa:
            constraint_block += "\nPOLICY CONSTRAINTS (MUST FOLLOW):\n"
            constraint_block += f'- You MUST output action_type exactly "{fa}".\n'
            if fa in ("post", "retweet", "reply"):
                constraint_block += f"- You MUST output stance_id exactly {int(fs) if fs is not None else 0}.\n"
                constraint_block += "- You MUST output a non-empty post_text consistent with the observation.\n"
            else:
                constraint_block += '- You MUST output stance_id as null (or 0 if you cannot output null).\n'
                constraint_block += '- You MUST output post_text as an empty string.\n'

        executor_prompt = f"""You are simulating a Twitter-like social media user in a multi-agent system.

Context (observation):
{question}

Coordinator hint (optional):
{strategy}

TASK:
- Choose EXACTLY ONE action for the current user at the current stage.
{constraint_block}

OUTPUT FORMAT (STRICT):
- Output JSON ONLY (no markdown, no extra text).
- Keys must be exactly:
  - "action_type": one of ["post","retweet","reply","like","do_nothing"]
  - "stance_id": integer stance class id
  - "post_text": string tweet content

RULES:
- If action_type in ["post","retweet","reply"]:
  - "stance_id" is REQUIRED and must be a valid integer id as specified in the context.
  - "post_text" is REQUIRED and should be a single tweet (concise, realistic).
- If action_type in ["like","do_nothing"]:
  - "stance_id" MUST be null (or 0 if you cannot output null).
  - "post_text" MUST be an empty string.
- Do NOT include any other keys.

Return JSON only:"""

        answer = self.llm_wrapper.generate_response(
            prompt=executor_prompt,
            strategy=None,  # Strategy is already included in the prompt
            temperature=final_temp,
            repetition_penalty=final_penalty,
            max_tokens=int(getattr(self.args, "max_answer_tokens", 256)),
            response_format={"type": "json_object"} if bool(getattr(self.args, "llm_response_format_json", False)) else None,
        )
        
        fixed = self._ensure_social_json(answer, forced_action_type=fa if fa else None, forced_stance_id=fs)
        return fixed

    def _ensure_social_json(self, s: Any, forced_action_type: Optional[str] = None, forced_stance_id: Optional[int] = None) -> str:
        """尽量把模型输出修复为 {"action_type": str, "stance_id": int|None, "post_text": str} 的 JSON 字符串。"""
        allowed_actions = {"post", "retweet", "reply", "like", "do_nothing"}
        stance_actions = {"post", "retweet", "reply"}

        def _coerce_action_type(obj: Dict[str, Any]) -> str:
            at = str(obj.get("action_type") or obj.get("action") or "").strip().lower()
            if not at:
                if ("stance_id" in obj) or ("post_text" in obj) or ("text" in obj) or ("tweet" in obj):
                    at = "post"
                else:
                    at = "do_nothing"
            if at not in allowed_actions:
                at = "do_nothing"
            return at

        def _coerce_stance_id(v: Any) -> Optional[int]:
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                return None

        def _coerce_post_text(obj: Dict[str, Any]) -> str:
            return str(obj.get("post_text") or obj.get("text") or obj.get("tweet") or "")

        def _normalize(obj: Dict[str, Any]) -> str:
            at = _coerce_action_type(obj)
            fa = str(forced_action_type).strip().lower() if forced_action_type else ""
            if fa in allowed_actions:
                at = fa
            if at in stance_actions:
                sid = forced_stance_id if forced_stance_id is not None else _coerce_stance_id(obj.get("stance_id"))
                txt = _coerce_post_text(obj)
                if sid is None:
                    sid = 0
                return json.dumps({"action_type": at, "stance_id": int(sid), "post_text": str(txt)}, ensure_ascii=False)
            return json.dumps({"action_type": at, "stance_id": None, "post_text": ""}, ensure_ascii=False)

        try:
            if isinstance(s, dict):
                return _normalize(s)
        except Exception:
            pass

        ss = str(s or "").strip()
        if not ss:
            return json.dumps({"action_type": "do_nothing", "stance_id": None, "post_text": ""}, ensure_ascii=False)

        try:
            obj = json.loads(ss)
            if isinstance(obj, dict):
                return _normalize(obj)
        except Exception:
            pass

        m = re.search(r"\{[\s\S]*\}", ss)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return _normalize(obj)
            except Exception:
                pass

        sid = 0
        mid = re.search(r"stance_id\s*[:=]\s*(-?\d+)", ss)
        if mid:
            try:
                sid = int(mid.group(1))
            except Exception:
                sid = 0

        txt = ss
        if len(txt) > 800:
            txt = txt[:800]
        return json.dumps({"action_type": "post", "stance_id": int(sid), "post_text": str(txt)}, ensure_ascii=False)
        
    def save_models(self, path: str):
        """
        保存模型参数。
        
        Args:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/agent.th")
    
    def load_models(self, path: str):
        """
        加载模型参数。
        
        Args:
            path: 加载路径
        """
        agent_path = f"{path}/agent.th"
        if os.path.exists(agent_path):
            sd = torch.load(agent_path, map_location=self.device)
            try:
                missing, unexpected = self.load_state_dict(sd, strict=False)
                if missing:
                    logger.warning(f"Agent checkpoint missing keys (ignored): {missing[:20]}{'...' if len(missing) > 20 else ''}")
                if unexpected:
                    logger.warning(f"Agent checkpoint unexpected keys (ignored): {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")
            except Exception:
                self.load_state_dict(sd, strict=False)
            try:
                if self.s3b_bias_enabled and isinstance(self.s3b_prior_head, nn.Module):
                    self.s3b_prior_head.load_state_dict(self.action_type_head.state_dict(), strict=True)
            except Exception:
                pass
            return
        bn = f"{path}/belief_network.th"
        on = f"{path}/output_network.th"
        if os.path.exists(bn):
            self.belief_network.load_state_dict(torch.load(bn, map_location=self.device))
        if os.path.exists(on):
            try:
                self.output_network.load_state_dict(torch.load(on, map_location=self.device), strict=False)
            except Exception:
                try:
                    self.action_type_head.load_state_dict(torch.load(on, map_location=self.device), strict=False)
                except Exception:
                    pass

        try:
            if self.s3b_bias_enabled and isinstance(self.s3b_prior_head, nn.Module):
                self.s3b_prior_head.load_state_dict(self.action_type_head.state_dict(), strict=True)
        except Exception:
            pass
    
    def cuda(self):
        """
        将模型参数移动到 CUDA 设备上。
        """
        self.to(self.device)
        return self
        
    def init_hidden(self):
        """
        初始化隐藏状态（在 Transformer 中不使用，为接口兼容性而保留）。
        """
        return torch.zeros(1, device=self.device) 
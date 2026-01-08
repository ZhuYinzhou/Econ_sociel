import torch
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
        # 自注意力层
        # 处理注意力掩码：PyTorch的MultiheadAttention期望(seq_len, seq_len)形状的2D掩码
        attn_mask = None
        if mask is not None:
            if mask.ndim == 2 and mask.shape[0] == x.shape[0]:  # (batch_size, seq_len)
                # 转换为(seq_len, seq_len)，对于每个batch都相同
                seq_len = mask.shape[1]
                # 这里我们假设mask是一个padding mask，True表示需要忽略的位置
                # 对于自注意力，我们需要一个(seq_len, seq_len)的掩码
                # 简单处理：如果某个位置在任何batch中被mask，则在所有位置都mask它
                # 或者我们可以直接使用第一个batch的mask
                position_mask = mask[0]  # 取第一个batch的mask (seq_len,)
                # 创建(seq_len, seq_len)的掩码
                attn_mask = position_mask.unsqueeze(0).expand(seq_len, -1) | position_mask.unsqueeze(1).expand(-1, seq_len)
            elif mask.ndim == 2 and mask.shape[0] == mask.shape[1]:  # 已经是(seq_len, seq_len)
                attn_mask = mask
        
        attended, _ = self.attention(x, x, x, attn_mask=attn_mask)
        
        # 残差连接和层归一化
        attended = self.norm1(x + attended)
        
        # 前馈网络
        ff_output = self.feed_forward(attended)
        
        # 残差连接和层归一化
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
        
        # 保存参数
        self.observation_dim = observation_dim  # 这是max_token_length
        self.belief_dim = belief_dim
        self.T_min = T_min
        self.T_max = T_max
        self.p_min = p_min
        self.p_max = p_max
        
        # Token嵌入层：将token IDs转换为dense vectors
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer 层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                ff_dim=hidden_dim * 4, # Standard practice for ff_dim
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # 输出映射层（从 hidden_dim 到 belief_dim）
        self.belief_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Added a layer for more capacity
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim)
        )
        
        # Prompt embedding 参数生成网络 (对应 W_T, b_T, W_p, b_p)
        # 为 T 和 p 分别创建线性层，更接近论文公式
        self.temp_projection = nn.Linear(belief_dim, 1) # W_T b_i + b_T
        self.penalty_projection = nn.Linear(belief_dim, 1) # W_p b_i + b_p
        
        # Q 值预测网络 (参数 φ_i)
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
        # 确保token_ids是正确的形状
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
        if token_ids.ndim == 3:
            token_ids = token_ids.squeeze(1)    # (batch, 1, seq_len) -> (batch, seq_len)
        
        # Token嵌入
        x = self.token_embedding(token_ids.long())  # (batch_size, seq_len, hidden_dim)
        
        # 位置编码
        x = self.pos_encoder(x) # x is (batch, seq_len, hidden_dim)
        
        # 应用 Transformer 层
        for layer in self.transformer_layers:
            x = layer(x, mask) # x is (batch, seq_len, hidden_dim)
        
        # 序列池化：取最后一个有效token的输出作为序列的总结
        # 这里我们简单地取最后一个时间步的输出
        if mask is not None:
            # 如果有mask，找到每个序列中最后一个有效token
            # mask为True表示需要忽略的位置
            valid_lengths = (~mask).sum(dim=1)  # 每个序列的有效长度
            batch_indices = torch.arange(x.size(0), device=x.device)
            last_valid_indices = (valid_lengths - 1).clamp(min=0)
            processed_sequence = x[batch_indices, last_valid_indices]  # (batch, hidden_dim)
        else:
            # 没有mask，直接取最后一个位置
            processed_sequence = x[:, -1]  # (batch, hidden_dim)
            
        # 生成置信状态 b_i
        belief_state = self.belief_projection(processed_sequence) # (batch, belief_dim)
        
        # 生成 prompt embedding e_i = [T_i, p_i]
        # T_i = T_min + (T_max - T_min) * σ(W_T b_i + b_T)
        # p_i = p_min + (p_max - p_min) * σ(W_p b_i + b_p)
        
        temp_logit = self.temp_projection(belief_state) # (batch, 1)
        penalty_logit = self.penalty_projection(belief_state) # (batch, 1)
        
        temperature = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(temp_logit)
        penalty = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(penalty_logit)
        
        # prompt_embedding_scaled 形状: (batch_size, 2)
        prompt_embedding_scaled = torch.cat([temperature, penalty], dim=1)
        
        # 生成 Q 值 Q_i^t
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
        
        # 参数设置
        self.args = args
        # self.input_shape = input_shape # 改为直接使用 args 中的维度或推断
        self.belief_dim = args.belief_dim
        # 正确访问use_cuda属性
        use_cuda = hasattr(args, 'system') and hasattr(args.system, 'use_cuda') and args.system.use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        # 温度和 top_p 范围 (从 args 或 BeliefNetwork 的默认值获取)
        # NOTE: 旧/精简配置可能没有 sampling 节点；这里提供向后兼容默认值
        sampling_cfg = getattr(args, "sampling", None)
        if sampling_cfg is None:
            try:
                from types import SimpleNamespace
                sampling_cfg = SimpleNamespace()
                # 尽量写回 args，方便其他模块统一读取
                setattr(args, "sampling", sampling_cfg)
            except Exception:
                sampling_cfg = None

        self.T_min = getattr(sampling_cfg, 'temperature_min', 0.1) if sampling_cfg is not None else 0.1
        self.T_max = getattr(sampling_cfg, 'temperature_max', 2.0) if sampling_cfg is not None else 2.0
        self.p_min = getattr(sampling_cfg, 'p_min', 0.1) if sampling_cfg is not None else 0.1
        self.p_max = getattr(sampling_cfg, 'p_max', 0.9) if sampling_cfg is not None else 0.9
        
        # 初始化个体置信网络
        # 输入应该是tokenized观察的长度，而不是state_shape
        # 从配置中获取观察的最大token长度
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
        
        # 输出层（生成动作概率） - 这部分可能需要重新审视
        # 在ECON中，动作是 prompt_embedding e_i。这里的 output_network 可能是用于离散动作空间。
        # 如果框架完全依赖 e_i 作为动作，这个网络可能不需要或用途不同。
        # 保留它以防万一，但标记为需要根据ECON的整体动作选择机制来确认。
        self.output_network = nn.Sequential(
            nn.Linear(self.belief_dim, args.n_actions)
        )
        
        # 初始化 LLM 包装器
        self.llm_wrapper = ImprovedLLMWrapper(
            api_key=args.together_api_key,
            model_name=args.executor_model,
            belief_dim=self.belief_dim # LLM Wrapper 可能也需要信念状态
        )
        
        # 缓存最新的提示嵌入
        self.current_prompt_embedding_tensor = torch.tensor([ (self.T_min + self.T_max) / 2, (self.p_min + self.p_max) / 2 ], device=self.device) # (2,)
        
        # 初始化提示嵌入字典（用于日志和调试）
        self.current_prompt_embedding = {
            'temperature': (self.T_min + self.T_max) / 2,
            'repetition_penalty': (self.p_min + self.p_max) / 2
        }
        
    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        test_mode: bool = False,
        # 兼容旧签名：hidden_state 不再使用，统一返回 None
        hidden_state: Optional[torch.Tensor] = None,
        # 社交媒体任务：允许外部覆盖采样参数（温度/重复惩罚）
        temperature: Optional[Any] = None,
        repetition_penalty: Optional[Any] = None,
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
        # 通过置信网络获取置信状态, prompt embedding e_i, 和局部 Q值 Q_i^t
        # 'inputs' 参数现在被理解为 local_history_obs
        belief_outputs = self.belief_network(inputs, mask)
        
        belief_state = belief_outputs['belief_state']
        prompt_embedding = belief_outputs['prompt_embedding'] # e_i
        local_q_value = belief_outputs['q_value'] # Q_i^t
        temp_logit = belief_outputs['temp_logit'] # 原始温度 logit
        penalty_logit = belief_outputs['penalty_logit'] # 原始惩罚 logit
        
        # 覆盖采样参数：temperature / repetition_penalty
        # 优先级：显式 override > test_mode 固定值 > 网络输出
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
                    # 尝试 squeeze 到 (bs,1)
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
            # 测试模式默认用中值（与先前行为一致）
            temp_t = torch.full((bs, 1), float((self.T_min + self.T_max) / 2), device=self.device, dtype=belief_state.dtype)
            pen_t = torch.full((bs, 1), float((self.p_min + self.p_max) / 2), device=self.device, dtype=belief_state.dtype)
            prompt_embedding = torch.cat([temp_t, pen_t], dim=1)

        # 更新缓存的提示嵌入 (张量形式)：使用当前 forward 实际采用的 prompt_embedding
        if prompt_embedding is not None and isinstance(prompt_embedding, torch.Tensor) and prompt_embedding.numel() >= 2:
            try:
                self.current_prompt_embedding_tensor = prompt_embedding[0].detach().clone()
            except Exception:
                pass


        # ECON 中的动作是 prompt_embedding e_i
        # output_network 的作用需要根据具体场景确定。
        # 如果是用于从 belief_state 派生其他类型的动作（例如离散动作选择），则保留。
        # 如果ECON框架中的动作 *仅仅* 是 e_i，那么 action_q_values 可能不直接用于最终动作选择，
        # 或者 Q_i^t 本身就是针对 e_i 的价值评估。
        action_q_values = self.output_network(belief_state) # 这可能是用于辅助任务或不同的动作空间
        
        outputs = {
            "action_q_values": action_q_values, # 可能是辅助输出
            "belief_state": belief_state,       # b_i
            "prompt_embedding": prompt_embedding, # e_i = [T_i, p_i]
            "q_value": local_q_value,           # Q_i^t - 保持与BeliefNetwork一致的字段名
            "raw_prompt_embed_params": torch.cat([temp_logit, penalty_logit], dim=1) # 保存原始 logits, 用于可能的后续分析或损失计算
        }
        
        # 统一 hidden_state 返回为 None（Transformer 不需要 recurrent hidden）
        return outputs, None
    
    def generate_answer(self, question: str, strategy: str, 
                       belief_state: Optional[torch.Tensor] = None, # 可选，因为agent内部会生成
                       temperature: Optional[float] = None, # 将从 self.current_prompt_embedding_tensor 获取
                       repetition_penalty: Optional[float] = None) -> str: # 对应论文的 p_i
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
        
        # 获取当前的提示嵌入参数
        # self.current_prompt_embedding_tensor 存储 [T_i, p_i]
        current_temp = self.current_prompt_embedding_tensor[0].item()
        current_penalty = self.current_prompt_embedding_tensor[1].item()

        final_temp = temperature if temperature is not None else current_temp
        final_penalty = repetition_penalty if repetition_penalty is not None else current_penalty
        
        # 更新 current_prompt_embedding 字典，主要用于日志或调试，实际参数传递给LLM
        self.current_prompt_embedding['temperature'] = final_temp
        # 论文中 p_i 是 penalty threshold for repetition.
        # 假设 ImprovedLLMWrapper 有一个 repetition_penalty 参数。
        self.current_prompt_embedding['repetition_penalty'] = final_penalty

        # 社交媒体仿真：executor prompt 要求输出严格 JSON（action_type + stance_id + post_text）
        # 注意：环境 observation 本身已包含 persona/neighbor/population 信息与 stance_id 映射。
        # 这里做“最后一道闸门”，确保输出格式可被 env 解析。
        executor_prompt = f"""You are simulating a Twitter-like social media user in a multi-agent system.

Context (observation):
{question}

Coordinator hint (optional):
{strategy}

TASK:
- Choose EXACTLY ONE action for the current user at the current stage.

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
            # For social-media simulation, optionally request strict JSON from OpenAI-compatible providers.
            response_format={"type": "json_object"} if bool(getattr(self.args, "llm_response_format_json", False)) else None,
        )
        
        # 轻量校验/修复：确保返回可解析 JSON（env 侧也有解析，但这里尽量提高成功率）
        fixed = self._ensure_social_json(answer)
        return fixed

    def _ensure_social_json(self, s: Any) -> str:
        """尽量把模型输出修复为 {"action_type": str, "stance_id": int|None, "post_text": str} 的 JSON 字符串。"""
        allowed_actions = {"post", "retweet", "reply", "like", "do_nothing"}
        stance_actions = {"post", "retweet", "reply"}

        def _coerce_action_type(obj: Dict[str, Any]) -> str:
            at = str(obj.get("action_type") or obj.get("action") or "").strip().lower()
            if not at:
                # backward-compat: stance/text implies "post"
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
            if at in stance_actions:
                sid = _coerce_stance_id(obj.get("stance_id"))
                txt = _coerce_post_text(obj)
                if sid is None:
                    sid = 0
                return json.dumps({"action_type": at, "stance_id": int(sid), "post_text": str(txt)}, ensure_ascii=False)
            # like / do_nothing: no stance expressed
            return json.dumps({"action_type": at, "stance_id": None, "post_text": ""}, ensure_ascii=False)

        try:
            if isinstance(s, dict):
                return _normalize(s)
        except Exception:
            pass

        ss = str(s or "").strip()
        if not ss:
            return json.dumps({"action_type": "do_nothing", "stance_id": None, "post_text": ""}, ensure_ascii=False)

        # try parse as json directly
        try:
            obj = json.loads(ss)
            if isinstance(obj, dict):
                return _normalize(obj)
        except Exception:
            pass

        # try extract a json object substring
        m = re.search(r"\{[\s\S]*\}", ss)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return _normalize(obj)
            except Exception:
                pass

        # fallback: extract stance_id integer if present
        sid = 0
        mid = re.search(r"stance_id\s*[:=]\s*(-?\d+)", ss)
        if mid:
            try:
                sid = int(mid.group(1))
            except Exception:
                sid = 0

        # use remaining text as tweet content (truncate)
        txt = ss
        if len(txt) > 800:
            txt = txt[:800]
        # fallback implies stance-only, treat as post
        return json.dumps({"action_type": "post", "stance_id": int(sid), "post_text": str(txt)}, ensure_ascii=False)
        
    def save_models(self, path: str):
        """
        保存模型参数。
        
        Args:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        # 统一保存整个 agent，包含 belief_network/output_network 及其它参数
        torch.save(self.state_dict(), f"{path}/agent.th")
    
    def load_models(self, path: str):
        """
        加载模型参数。
        
        Args:
            path: 加载路径
        """
        agent_path = f"{path}/agent.th"
        if os.path.exists(agent_path):
            self.load_state_dict(torch.load(agent_path, map_location=self.device))
            return
        # 兼容旧 checkpoint 命名
        bn = f"{path}/belief_network.th"
        on = f"{path}/output_network.th"
        if os.path.exists(bn):
            self.belief_network.load_state_dict(torch.load(bn, map_location=self.device))
        if os.path.exists(on):
            self.output_network.load_state_dict(torch.load(on, map_location=self.device))
    
    def cuda(self):
        """
        将模型参数移动到 CUDA 设备上。
        """
        # 统一用 self.to(self.device)
        self.to(self.device)
        return self
        
    def init_hidden(self):
        """
        初始化隐藏状态（在 Transformer 中不使用，为接口兼容性而保留）。
        """
        # Transformer 不需要隐藏状态
        return torch.zeros(1, device=self.device) 
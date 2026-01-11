import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional


class BeliefEncoder(nn.Module):
    """
    共享置信编码器，聚合所有智能体的置信状态以产生群体表征。
    
    根据 ECON 框架，此模块接收所有智能体的置信状态 b_i，使用多头注意力
    机制处理它们，并输出一个群体级表征 E。
    """
    
    def __init__(
        self,
        belief_dim: int,
        n_agents: int,
        n_heads: int = 4,
        key_dim: int = 64,
        device: torch.device = None,
        # --- extensions for HiSim social simulation ---
        population_belief_dim: int = 3,
        use_population_token: bool = True,
        n_stages: int = 13,
        use_stage_token: bool = False,
        # --- population belief update head (z(t) -> z(t+1)) ---
        use_population_update_head: bool = True,
        population_update_hidden_dim: int = 128,
        population_update_use_group_repr: bool = True,
        population_update_use_stage: bool = False,
        # --- optional structured conditioning vector for z-transition ---
        population_update_use_extra_cond: bool = False,
        population_update_extra_cond_dim: int = 0,
        # residual mixing: z_next = mix * z_hat + (1-mix) * z_t
        population_update_residual_mixing: bool = True,
        population_update_mixing_init: float = 0.5,
        population_update_mixing_learnable: bool = True,
        # --- secondary user action belief head (optional) ---
        secondary_action_dim: int = 5,
        use_secondary_action_head: bool = False,
        secondary_action_hidden_dim: int = 128,
        secondary_action_use_group_repr: bool = True,
        secondary_action_use_stage: bool = False,
        secondary_action_use_population: bool = True,
    ):
        """
        初始化置信编码器。
        
        Args:
            belief_dim: 置信状态的维度
            n_agents: 智能体数量
            n_heads: 注意力头数量
            key_dim: 每个注意力头的维度
            device: 计算设备
            population_belief_dim: 边缘用户的 latent population belief z 的维度（默认 3 类 stance）
            use_population_token: 是否将 population belief 作为额外 token 融入注意力聚合
            n_stages: stage 数（用于可选的 stage token）
            use_stage_token: 是否加入 stage token（用于时序条件化；默认关闭以保持行为稳定）
        """
        super(BeliefEncoder, self).__init__()
        
        self.belief_dim = belief_dim
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.population_belief_dim = int(population_belief_dim)
        self.use_population_token = bool(use_population_token)
        self.n_stages = int(n_stages)
        self.use_stage_token = bool(use_stage_token)

        # population belief update head config
        self.use_population_update_head = bool(use_population_update_head)
        self.population_update_hidden_dim = int(population_update_hidden_dim)
        self.population_update_use_group_repr = bool(population_update_use_group_repr)
        self.population_update_use_stage = bool(population_update_use_stage)
        self.population_update_use_extra_cond = bool(population_update_use_extra_cond)
        self.population_update_extra_cond_dim = int(population_update_extra_cond_dim)
        self.population_update_residual_mixing = bool(population_update_residual_mixing)
        self.population_update_mixing_init = float(population_update_mixing_init)
        self.population_update_mixing_learnable = bool(population_update_mixing_learnable)

        # secondary action belief head config
        self.secondary_action_dim = int(secondary_action_dim)
        self.use_secondary_action_head = bool(use_secondary_action_head)
        self.secondary_action_hidden_dim = int(secondary_action_hidden_dim)
        self.secondary_action_use_group_repr = bool(secondary_action_use_group_repr)
        self.secondary_action_use_stage = bool(secondary_action_use_stage)
        self.secondary_action_use_population = bool(secondary_action_use_population)
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=belief_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # 将 population_belief(z) 投影到 belief_dim 作为 token
        if self.use_population_token:
            self.population_proj = nn.Sequential(
                nn.Linear(self.population_belief_dim, belief_dim),
                nn.LayerNorm(belief_dim),
                nn.Tanh(),
            )
        else:
            self.population_proj = None

        # stage embedding (用于显式时序条件化)
        #
        # IMPORTANT:
        # - 历史上 `stage_embed` 只在 use_stage_token=True 时初始化（用于 attention token）。
        # - 但 Stage3a 希望 population_update_head 显式带 stage 条件，此时即使不把 stage 作为 token
        #   融入 attention（use_stage_token=False），population_update_head 仍需要 stage_embed。
        need_stage_embed = bool(self.use_stage_token or self.population_update_use_stage or self.secondary_action_use_stage)
        if need_stage_embed:
            # stage_t 预期为 [0..n_stages-1]，额外留一个 padding/unknown
            self.stage_embed = nn.Embedding(max(1, self.n_stages) + 1, belief_dim)
        else:
            self.stage_embed = None

        # ===== Population Belief Update Head: z(t) -> z(t+1) =====
        # 说明：
        # - 输入默认包含 z(t)；可选拼接 group_repr（核心用户聚合的 belief）与 stage embedding
        # - 输出 logits，再 softmax 得到 z(t+1) ∈ Δ^K
        self.population_update_head: Optional[nn.Module]
        if self.use_population_update_head:
            in_dim = self.population_belief_dim
            if self.population_update_use_group_repr:
                in_dim += belief_dim
            if self.population_update_use_stage:
                in_dim += belief_dim
            if self.population_update_use_extra_cond:
                in_dim += max(0, int(self.population_update_extra_cond_dim))
            hid = max(8, self.population_update_hidden_dim)
            self.population_update_head = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
                nn.Linear(hid, self.population_belief_dim),
            )
            # residual mixing gate（可选）
            if self.population_update_residual_mixing:
                if self.population_update_mixing_learnable:
                    # 全局可学习标量 gate，经 sigmoid 映射到 (0,1)
                    init = torch.tensor(self.population_update_mixing_init).clamp(0.0, 1.0)
                    # 把 init 反推到 logit 空间，避免训练初期饱和
                    eps = 1e-6
                    init = torch.clamp(init, eps, 1 - eps)
                    init_logit = torch.log(init / (1 - init))
                    self.population_update_mix_logit = nn.Parameter(init_logit)
                else:
                    self.register_buffer(
                        "population_update_mix_const",
                        torch.tensor(self.population_update_mixing_init).clamp(0.0, 1.0),
                        persistent=False,
                    )
                    self.population_update_mix_logit = None
            else:
                self.population_update_mix_logit = None
        else:
            self.population_update_head = None

        # ===== Secondary Action Belief Head: predict secondary users' action-type distribution =====
        # 输入默认包含 z(t)；可选拼接 group_repr（核心用户聚合的 belief）与 stage embedding
        self.secondary_action_head: Optional[nn.Module]
        if self.use_secondary_action_head:
            in_dim = 0
            if self.secondary_action_use_population:
                in_dim += self.population_belief_dim
            if self.secondary_action_use_group_repr:
                in_dim += belief_dim
            if self.secondary_action_use_stage:
                in_dim += belief_dim
            hid = max(8, self.secondary_action_hidden_dim)
            self.secondary_action_head = nn.Sequential(
                nn.Linear(in_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
                nn.Linear(hid, self.secondary_action_dim),
            )
        else:
            self.secondary_action_head = None
        
        # 输出投影层
        self.out_proj = nn.Linear(belief_dim, belief_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(belief_dim)
        
        # 前馈网络
        self.feedforward = nn.Sequential(
            nn.Linear(belief_dim, 4 * belief_dim),
            nn.ReLU(),
            nn.Linear(4 * belief_dim, belief_dim)
        )
        
        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(belief_dim)
        
    def forward(
        self,
        belief_states: torch.Tensor,
        *,
        population_belief: Optional[torch.Tensor] = None,
        stage_t: Optional[torch.Tensor] = None,
        return_tokens: bool = False,
    ) -> torch.Tensor:
        """
        前向传播，聚合智能体置信状态产生群体表征。

        兼容旧接口：只传 belief_states 也可运行。
        为 HiSim 社交仿真扩展：可选传入 population_belief(z) 作为额外 token，显式建模 700 边缘用户的 latent population belief。
        
        Args:
            belief_states: 所有智能体的置信状态 [batch_size, n_agents, belief_dim]
            population_belief: population belief z（例如 3 类 stance 分布）[batch_size, K]
            stage_t: stage index [batch_size] 或 [batch_size, 1]（可选）
            return_tokens: True 时返回 dict（含 group_repr/tokens）；False 时仅返回 group_repr
            
        Returns:
            群体表征 E [batch_size, belief_dim]（或 return_tokens=True 时返回 dict）
        """
        if belief_states.ndim != 3:
            raise ValueError(f"belief_states 期望形状 [bs, n_agents, belief_dim]，实际={tuple(belief_states.shape)}")
        batch_size = belief_states.shape[0]
        tokens = belief_states  # (bs, n_agents, belief_dim)

        extra_tokens: List[torch.Tensor] = []

        # population token
        if self.use_population_token and self.population_proj is not None and population_belief is not None:
            if population_belief.ndim == 1:
                population_belief = population_belief.unsqueeze(0)  # (K,) -> (1,K)
            # 允许 (bs,1,K)
            if population_belief.ndim == 3 and population_belief.shape[1] == 1:
                population_belief = population_belief.squeeze(1)
            pop_token = self.population_proj(population_belief.to(tokens.device, dtype=tokens.dtype)).unsqueeze(1)  # (bs,1,belief_dim)
            extra_tokens.append(pop_token)

        # stage token
        if self.use_stage_token and self.stage_embed is not None and stage_t is not None:
            st = stage_t
            if st.ndim == 2 and st.shape[1] == 1:
                st = st.squeeze(1)
            if st.ndim != 1:
                raise ValueError(f"stage_t 期望形状 [bs] 或 [bs,1]，实际={tuple(stage_t.shape)}")
            st = st.to(tokens.device, dtype=torch.long).clamp(min=0, max=self.n_stages)
            st_token = self.stage_embed(st).unsqueeze(1)  # (bs,1,belief_dim)
            extra_tokens.append(st_token)

        if extra_tokens:
            tokens = torch.cat([tokens] + extra_tokens, dim=1)  # (bs, n_agents + n_extra, belief_dim)
        
        # 应用多头注意力
        # tokens: [batch_size, n_tokens, belief_dim]
        attn_output, _ = self.multihead_attn(
            query=tokens,
            key=tokens,
            value=tokens
        )
        # attn_output: [batch_size, n_tokens, belief_dim]
        
        # 残差连接和层归一化
        attn_output = tokens + attn_output
        attn_output = self.layer_norm(attn_output)
        
        # 前馈网络
        ff_output = self.feedforward(attn_output)
        
        # 残差连接和层归一化
        ff_output = attn_output + ff_output
        ff_output = self.final_layer_norm(ff_output)
        
        # 聚合所有智能体的表征，生成群体表征
        # 默认：只对 agent tokens 做平均池化（不把 pop/stage token 的表征混进来，稳定性更好）
        group_repr = ff_output[:, : self.n_agents].mean(dim=1)  # [batch_size, belief_dim]
        
        # 输出投影
        group_repr = self.out_proj(group_repr)

        if return_tokens:
            return {
                "group_repr": group_repr,
                "tokens": ff_output,
            }

        return group_repr

    def predict_next_population_belief(
        self,
        z_t: torch.Tensor,
        *,
        group_repr: Optional[torch.Tensor] = None,
        stage_t: Optional[torch.Tensor] = None,
        extra_cond: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Population Belief Update Head：根据当前 z(t)（可选条件 group_repr / stage）预测 z(t+1)。

        Args:
            z_t: [bs, K] 或 [K]
            group_repr: [bs, belief_dim]（可选）
            stage_t: [bs] 或 [bs,1]（可选；仅当 population_update_use_stage=True 时使用）
            return_logits: True 返回 logits；False 返回 softmax 后的概率分布

        Returns:
            z_next: [bs, K]（logits 或 probs）
        """
        if self.population_update_head is None:
            raise RuntimeError("population_update_head 未启用：请设置 use_population_update_head=True")

        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if z_t.ndim != 2:
            raise ValueError(f"z_t 期望形状 [bs,K] 或 [K]，实际={tuple(z_t.shape)}")
        if z_t.size(-1) != self.population_belief_dim:
            raise ValueError(f"z_t 最后一维应为 K={self.population_belief_dim}，实际={z_t.size(-1)}")

        parts = [z_t]

        if self.population_update_use_group_repr:
            if group_repr is None:
                raise ValueError("population_update_use_group_repr=True 但未提供 group_repr")
            if group_repr.ndim == 1:
                group_repr = group_repr.unsqueeze(0)
            if group_repr.ndim != 2 or group_repr.size(-1) != self.belief_dim:
                raise ValueError(f"group_repr 期望形状 [bs, belief_dim]，实际={tuple(group_repr.shape)}")
            parts.append(group_repr)

        if self.population_update_use_stage:
            if stage_t is None or self.stage_embed is None:
                raise ValueError("population_update_use_stage=True 但未提供 stage_t 或 stage_embed 未初始化")
            st = stage_t
            if st.ndim == 2 and st.shape[1] == 1:
                st = st.squeeze(1)
            if st.ndim != 1:
                raise ValueError(f"stage_t 期望形状 [bs] 或 [bs,1]，实际={tuple(stage_t.shape)}")
            st = st.to(z_t.device, dtype=torch.long).clamp(min=0, max=self.n_stages)
            st_tok = self.stage_embed(st)
            parts.append(st_tok)

        if self.population_update_use_extra_cond and extra_cond is not None:
            ec = extra_cond
            if ec.ndim == 1:
                ec = ec.unsqueeze(0)
            if ec.ndim != 2:
                raise ValueError(f"extra_cond 期望形状 [bs, D]，实际={tuple(extra_cond.shape)}")
            # Best-effort check on dim if configured
            if int(self.population_update_extra_cond_dim) > 0 and ec.size(-1) != int(self.population_update_extra_cond_dim):
                raise ValueError(
                    f"extra_cond dim 不匹配: expect D={int(self.population_update_extra_cond_dim)} got {ec.size(-1)}"
                )
            parts.append(ec)

        x = torch.cat(parts, dim=-1)
        logits = self.population_update_head(x)
        if return_logits:
            return logits
        # === continuous scalar mode: K=1 means z ∈ [-1,1] ===
        if int(self.population_belief_dim) == 1:
            # tanh squashes to [-1,1]
            z_hat = torch.tanh(logits)
        else:
            z_hat = F.softmax(logits, dim=-1)

        # residual mixing (convex combination keeps simplex)
        if self.population_update_residual_mixing:
            if int(self.population_belief_dim) == 1:
                # scalar convex mixing, clamp to [-1,1]
                z_in = torch.clamp(z_t, min=-1.0, max=1.0)
            else:
                z_in = torch.clamp(z_t, min=0.0)
                z_in = z_in / torch.clamp(z_in.sum(dim=-1, keepdim=True), min=1e-8)
            if self.population_update_mixing_learnable and self.population_update_mix_logit is not None:
                mix = torch.sigmoid(self.population_update_mix_logit)  # scalar
            else:
                mix = getattr(self, "population_update_mix_const", torch.tensor(self.population_update_mixing_init, device=z_hat.device))
            # broadcast scalar -> [bs,1]
            mix = mix.to(z_hat.device, dtype=z_hat.dtype).view(1, 1)
            z_out = mix * z_hat + (1.0 - mix) * z_in
            if int(self.population_belief_dim) == 1:
                return torch.clamp(z_out, min=-1.0, max=1.0)
            # safety renorm (categorical simplex)
            return z_out / torch.clamp(z_out.sum(dim=-1, keepdim=True), min=1e-8)

        return z_hat

    def predict_secondary_action_probs(
        self,
        *,
        z_t: Optional[torch.Tensor] = None,
        group_repr: Optional[torch.Tensor] = None,
        stage_t: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Secondary Action Belief Head：预测次要用户的 action_type 分布（例如 5 类：post/retweet/reply/like/do_nothing）。

        Args:
            z_t: [bs, K] 或 [K]（可选；仅当 secondary_action_use_population=True 时需要）
            group_repr: [bs, belief_dim]（可选；仅当 secondary_action_use_group_repr=True 时需要）
            stage_t: [bs] 或 [bs,1]（可选；仅当 secondary_action_use_stage=True 时需要）
            return_logits: True 返回 logits；False 返回 softmax 后的概率分布

        Returns:
            action_probs/logits: [bs, A]
        """
        if self.secondary_action_head is None:
            raise RuntimeError("secondary_action_head 未启用：请设置 use_secondary_action_head=True")

        xs: List[torch.Tensor] = []
        device = None
        dtype = None

        if self.secondary_action_use_population:
            if z_t is None:
                raise ValueError("secondary_action_use_population=True 但未提供 z_t")
            zz = z_t
            if zz.ndim == 1:
                zz = zz.unsqueeze(0)
            if zz.ndim != 2 or zz.size(-1) != self.population_belief_dim:
                raise ValueError(f"z_t 期望 [bs,K]，实际={tuple(zz.shape)}")
            device = zz.device
            dtype = zz.dtype
            xs.append(zz)

        if self.secondary_action_use_group_repr:
            if group_repr is None:
                raise ValueError("secondary_action_use_group_repr=True 但未提供 group_repr")
            gg = group_repr
            if gg.ndim == 1:
                gg = gg.unsqueeze(0)
            if gg.ndim != 2 or gg.size(-1) != self.belief_dim:
                raise ValueError(f"group_repr 期望 [bs, belief_dim]，实际={tuple(gg.shape)}")
            device = gg.device if device is None else device
            dtype = gg.dtype if dtype is None else dtype
            xs.append(gg)

        if self.secondary_action_use_stage:
            if stage_t is None or self.stage_embed is None:
                raise ValueError("secondary_action_use_stage=True 但未提供 stage_t 或 stage_embed 未初始化")
            st = stage_t
            if st.ndim == 2 and st.shape[1] == 1:
                st = st.squeeze(1)
            if st.ndim != 1:
                raise ValueError(f"stage_t 期望形状 [bs] 或 [bs,1]，实际={tuple(stage_t.shape)}")
            st = st.to(device if device is not None else gg.device, dtype=torch.long).clamp(min=0, max=self.n_stages)
            st_tok = self.stage_embed(st)
            if device is not None:
                st_tok = st_tok.to(device=device)
            if dtype is not None:
                st_tok = st_tok.to(dtype=dtype)
            xs.append(st_tok)

        if not xs:
            raise ValueError("Secondary action head received no inputs (check config flags).")

        x = torch.cat(xs, dim=-1)
        logits = self.secondary_action_head(x)
        if return_logits:
            return logits
        return F.softmax(logits, dim=-1)
    
    def compute_td_style_loss(
        self,
        td_loss_tot: torch.Tensor,
        td_losses_i: List[torch.Tensor],
        lambda_e: float,
    ) -> torch.Tensor:
        """
        兼容保留：TD-style encoder loss（旧版 ECON 论文形式）。
        你当前任务主用的是 belief supervision（见 compute_loss）。
        """
        sum_local_td_losses = sum(td_losses_i)
        return td_loss_tot + lambda_e * sum_local_td_losses

    def compute_loss(
        self,
        z_t: torch.Tensor,          # [bs, K]
        z_target: torch.Tensor,     # [bs, K]
        z_mask: torch.Tensor,       # [bs] or [bs,1]
        *,
        group_repr: Optional[torch.Tensor] = None,
        stage_t: Optional[torch.Tensor] = None,
        extra_cond: Optional[torch.Tensor] = None,
        loss_type: str = "kl",
    ) -> torch.Tensor:
        """
        belief supervision loss：用 PopulationBeliefUpdateHead 预测 z(t+1)，并用 KL/CE 监督到 target。
        """
        z_pred = self.predict_next_population_belief(
            z_t,
            group_repr=group_repr,
            stage_t=stage_t,
            extra_cond=extra_cond,
            return_logits=False,
        )
        return self.compute_population_belief_loss(z_pred, z_target, z_mask, loss_type=loss_type)

    def compute_population_belief_loss(
        self,
        z_pred: torch.Tensor,     # [bs, K] (probs)
        z_target: torch.Tensor,   # [bs, K] (probs)
        z_mask: torch.Tensor,     # [bs] or [bs,1]
        loss_type: str = "kl",
    ) -> torch.Tensor:
        """
        KL/CE supervision loss with mask.
        - KL: KL(target || pred)
        - CE: cross entropy with soft target
        """
        eps = 1e-8
        if z_pred.ndim == 1:
            z_pred = z_pred.unsqueeze(0)
        if z_target.ndim == 1:
            z_target = z_target.unsqueeze(0)
        if z_pred.ndim != 2 or z_target.ndim != 2:
            raise ValueError(f"z_pred/z_target 期望 [bs,K]，实际 z_pred={tuple(z_pred.shape)} z_target={tuple(z_target.shape)}")
        if z_pred.size(-1) != z_target.size(-1):
            raise ValueError(f"z_pred/z_target K 不一致: {z_pred.size(-1)} vs {z_target.size(-1)}")

        # === continuous scalar mode: K=1 -> regression in [-1,1] ===
        if int(z_pred.size(-1)) == 1:
            # coerce mask
            if z_mask.ndim == 2 and z_mask.shape[-1] == 1:
                z_mask = z_mask.squeeze(-1)
            z_mask = z_mask.to(z_pred.device, dtype=z_pred.dtype)

            # clamp to valid range
            z_pred = torch.clamp(z_pred, min=-1.0, max=1.0)
            z_target = torch.clamp(z_target, min=-1.0, max=1.0)
            # choose regression loss
            lt = str(loss_type or "mse").lower()
            if lt in ("smooth_l1", "huber"):
                per = F.smooth_l1_loss(z_pred, z_target, reduction="none").squeeze(-1)
            else:
                per = (z_pred - z_target).pow(2).squeeze(-1)
            per = per * z_mask
            return per.sum() / (z_mask.sum() + eps)

        # normalize for safety (categorical simplex)
        z_pred = torch.clamp(z_pred, min=0.0)
        z_target = torch.clamp(z_target, min=0.0)
        z_pred = z_pred / torch.clamp(z_pred.sum(dim=-1, keepdim=True), min=eps)
        z_target = z_target / torch.clamp(z_target.sum(dim=-1, keepdim=True), min=eps)

        if z_mask.ndim == 2 and z_mask.shape[-1] == 1:
            z_mask = z_mask.squeeze(-1)
        z_mask = z_mask.to(z_pred.device, dtype=z_pred.dtype)

        lt = str(loss_type or "kl").lower()
        if lt == "kl":
            loss = F.kl_div((z_pred + eps).log(), z_target, reduction="none").sum(dim=-1)
        else:  # CE (soft target)
            loss = -(z_target * (z_pred + eps).log()).sum(dim=-1)

        loss = loss * z_mask
        return loss.sum() / (z_mask.sum() + eps)
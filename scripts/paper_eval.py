"""
Paper evaluation script for HiSim/ECON social simulation (Stage4-style rollout).

Goals (paper-facing):
  (A) Population-level accuracy:
      - KL(z_GT || z_sim), JS(z_GT, z_sim) at stage boundaries (z_mask==1)
      - Stage-wise error curves and aggregated mean/std
      - Macro-trend alignment over time: entropy(z), polarization(z) correlation + DTW
  (B) Action–Outcome consistency (counterfactual / sensitivity):
      - Intervene on core-user action distribution (e.g., retweet vs post) at evaluation time
        by overriding chosen_actions, then measure response of z_sim trajectory.
  (C) Micro-level sanity:
      - Action distribution entropy/mode-frac and mode-collapse frequency (from chosen actions).
  (D) Efficiency:
      - Trainable / total parameter counts
      - Wall-clock inference time per episode (best-effort)

Note on timestep semantics (paper §2.1):
- At stage t, core actions happen and population responds to form z_t.
- Therefore env_info["z_target"] corresponds to stage t (not t+1) in the current codebase.

This script intentionally avoids any training-only dependencies and can run on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _normalize_prob(p: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(list(p), dtype=np.float64)
    a = np.clip(a, 0.0, None)
    s = float(a.sum())
    if not math.isfinite(s) or s <= 0:
        return np.full_like(a, 1.0 / max(1, a.size))
    a = a / s
    a = np.clip(a, eps, None)
    a = a / float(a.sum())
    return a


def kl_div(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """KL(p || q)."""
    p = _normalize_prob(p, eps=eps)
    q = _normalize_prob(q, eps=eps)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def js_div(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence (symmetric)."""
    p = _normalize_prob(p, eps=eps)
    q = _normalize_prob(q, eps=eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps=eps) + 0.5 * kl_div(q, m, eps=eps)


def entropy(p: Sequence[float], eps: float = 1e-12) -> float:
    p = _normalize_prob(p, eps=eps)
    return float(-np.sum(p * np.log(p + eps)))


def polarization_index(p: Sequence[float], eps: float = 1e-12) -> float:
    """
    A simple, bounded polarization proxy in [0,1]:
      1 - H(p)/log(K)
    """
    p = _normalize_prob(p, eps=eps)
    k = max(1, int(p.size))
    h = float(-np.sum(p * np.log(p + eps)))
    return float(1.0 - (h / max(eps, math.log(k))))


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    # simple rank (no tie correction; good enough for paper trend sanity)
    order = a.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def dtw_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """Classic O(T^2) DTW with absolute distance for 1D sequences."""
    x = list(map(float, x))
    y = list(map(float, y))
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan")
    dp = np.full((n + 1, m + 1), float("inf"), dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def _count_params(module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    try:
        for p in module.parameters():
            n = int(p.numel())
            total += n
            if bool(getattr(p, "requires_grad", False)):
                trainable += n
    except Exception:
        pass
    return total, trainable


@dataclass
class ZPoint:
    stage_t: int
    z_pred: List[float]
    z_gt: List[float]
    kl_gt_pred: float
    js_gt_pred: float
    ent_pred: float
    ent_gt: float
    pol_pred: float
    pol_gt: float
    labeled_edge_n: int


@dataclass
class ZStageAgg:
    """
    One aggregated macro point per stage/round (within a single episode).
    We aggregate multiple z points that share the same stage_t (if any).
    """
    stage_t: int
    z_pred_mean: List[float]
    z_gt_mean: List[float]
    labeled_edge_n_sum: int


def _extract_z_series(env_infos: List[Dict[str, Any]]) -> List[ZPoint]:
    out: List[ZPoint] = []
    for info in env_infos:
        if not isinstance(info, dict):
            continue
        z_mask = _safe_float(info.get("z_mask", 0.0), 0.0)
        if z_mask <= 0.5:
            continue
        z_pred = info.get("z_pred", None)
        z_gt = info.get("z_target", None)
        if not isinstance(z_pred, (list, tuple)) or not isinstance(z_gt, (list, tuple)):
            continue
        if len(z_pred) < 2 or len(z_gt) < 2:
            continue
        try:
            t = int(info.get("t", -1))
        except Exception:
            t = -1
        labeled_edge_n = int(info.get("z_target_labeled_edge_n", 0) or 0)
        k = kl_div(z_gt, z_pred)
        j = js_div(z_gt, z_pred)
        ep = entropy(z_pred)
        eg = entropy(z_gt)
        pp = polarization_index(z_pred)
        pg = polarization_index(z_gt)
        out.append(
            ZPoint(
                stage_t=t,
                z_pred=[float(x) for x in z_pred],
                z_gt=[float(x) for x in z_gt],
                kl_gt_pred=float(k),
                js_gt_pred=float(j),
                ent_pred=float(ep),
                ent_gt=float(eg),
                pol_pred=float(pp),
                pol_gt=float(pg),
                labeled_edge_n=labeled_edge_n,
            )
        )
    return out


def _aggregate_z_by_stage(zpts: List[ZPoint]) -> List[ZStageAgg]:
    """
    Aggregate z points by stage_t within ONE episode to form a macro curve over time/rounds.
    This is the recommended "macro-by-round" evaluation unit.
    """
    if not zpts:
        return []
    by_t: Dict[int, List[ZPoint]] = {}
    for p in zpts:
        by_t.setdefault(int(p.stage_t), []).append(p)
    out: List[ZStageAgg] = []
    for t in sorted(by_t.keys()):
        pts = by_t[t]
        # mean z vectors
        zp = np.mean(np.asarray([p.z_pred for p in pts], dtype=np.float64), axis=0)
        zg = np.mean(np.asarray([p.z_gt for p in pts], dtype=np.float64), axis=0)
        out.append(
            ZStageAgg(
                stage_t=int(t),
                z_pred_mean=[float(x) for x in zp.tolist()],
                z_gt_mean=[float(x) for x in zg.tolist()],
                labeled_edge_n_sum=int(sum(int(getattr(p, "labeled_edge_n", 0) or 0) for p in pts)),
            )
        )
    return out


def _summarize_z_stage_curve(stages: List[ZStageAgg]) -> Dict[str, Any]:
    """
    Macro-curve evaluation over stage_t (within ONE episode):
    - div: KL/JS between aggregated z_gt_mean vs z_pred_mean at each stage
    - bias: mean offsets on entropy/polarization (pred_mean - gt_mean)
    - Corr/DTW: alignment of entropy/polarization curves over time
    """
    if not stages:
        return {
            "n_stages": 0,
            "kl_mean": float("nan"),
            "kl_std": float("nan"),
            "js_mean": float("nan"),
            "js_std": float("nan"),
        }

    stage_t = [int(s.stage_t) for s in stages]
    zpred = [list(s.z_pred_mean) for s in stages]
    zgt = [list(s.z_gt_mean) for s in stages]
    edge_sum = [int(s.labeled_edge_n_sum) for s in stages]

    kls = [kl_div(g, p) for g, p in zip(zgt, zpred)]
    jss = [js_div(g, p) for g, p in zip(zgt, zpred)]

    ent_pred = [entropy(p) for p in zpred]
    ent_gt = [entropy(g) for g in zgt]
    pol_pred = [polarization_index(p) for p in zpred]
    pol_gt = [polarization_index(g) for g in zgt]

    ent_pearson = _pearson(ent_pred, ent_gt)
    ent_spearman = _spearman(ent_pred, ent_gt)
    pol_pearson = _pearson(pol_pred, pol_gt)
    pol_spearman = _spearman(pol_pred, pol_gt)
    ent_dtw = dtw_distance(ent_pred, ent_gt)
    pol_dtw = dtw_distance(pol_pred, pol_gt)

    return {
        "n_stages": int(len(stages)),
        "kl_mean": float(np.mean(kls)),
        "kl_std": float(np.std(kls)),
        "js_mean": float(np.mean(jss)),
        "js_std": float(np.std(jss)),
        "entropy_pred_mean": float(np.mean(ent_pred)),
        "entropy_gt_mean": float(np.mean(ent_gt)),
        "polarization_pred_mean": float(np.mean(pol_pred)),
        "polarization_gt_mean": float(np.mean(pol_gt)),
        "bias_entropy_mean": float(np.mean(ent_pred) - np.mean(ent_gt)),
        "bias_polarization_mean": float(np.mean(pol_pred) - np.mean(pol_gt)),
        "entropy_pearson": float(ent_pearson),
        "entropy_spearman": float(ent_spearman),
        "entropy_dtw": float(ent_dtw),
        "polarization_pearson": float(pol_pearson),
        "polarization_spearman": float(pol_spearman),
        "polarization_dtw": float(pol_dtw),
        # curves (for plotting; per-episode)
        "stage_t": stage_t,
        "stage_kl": [float(x) for x in kls],
        "stage_js": [float(x) for x in jss],
        "stage_labeled_edge_n_sum": edge_sum,
        "entropy_pred": [float(x) for x in ent_pred],
        "entropy_gt": [float(x) for x in ent_gt],
        "polarization_pred": [float(x) for x in pol_pred],
        "polarization_gt": [float(x) for x in pol_gt],
    }


def _extract_actions_from_batch(batch: Any, n_actions: int = 5) -> Dict[str, Any]:
    """
    Best-effort read actions chosen by MAC from EpisodeBatch.
    Returns per-step action entropy/mode_frac and overall counts.
    """
    out: Dict[str, Any] = {}
    if batch is None:
        return out
    try:
        acts = batch["actions"]  # expected torch.Tensor
    except Exception:
        return out
    try:
        import torch  # local import to allow running parts without torch

        if not isinstance(acts, torch.Tensor):
            return out
        # common shapes:
        # - (bs, T, n_agents, 1) or (bs, T, n_agents)
        a = acts.detach().cpu()
        if a.ndim == 4 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim != 3:
            return out
        bs, T, n_agents = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
        a = a.reshape(bs * T * n_agents)
        a = a.numpy().astype(np.int64, copy=False)
        a = np.clip(a, 0, max(0, int(n_actions) - 1))
        counts = np.bincount(a, minlength=int(n_actions)).astype(np.int64)
        out["action_counts"] = counts.tolist()
        out["action_freq"] = (counts / max(1, int(counts.sum()))).tolist()

        # per-step stats (stage-level)
        per_step_entropy: List[float] = []
        per_step_mode_frac: List[float] = []
        per_step_counts: List[List[int]] = []
        for t in range(T):
            at = acts.detach().cpu()
            if at.ndim == 4 and at.shape[-1] == 1:
                at = at[..., 0]
            at = at[:, t, :]  # (bs, n_agents)
            flat = at.reshape(-1).numpy().astype(np.int64, copy=False)
            flat = np.clip(flat, 0, max(0, int(n_actions) - 1))
            c = np.bincount(flat, minlength=int(n_actions)).astype(np.int64)
            p = (c / max(1, int(c.sum()))).astype(np.float64)
            h = float(-np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0)))
            mf = float(c.max() / max(1, int(c.sum())))
            per_step_entropy.append(h)
            per_step_mode_frac.append(mf)
            per_step_counts.append(c.tolist())
        out["per_step_action_entropy"] = per_step_entropy
        out["per_step_action_mode_frac"] = per_step_mode_frac
        out["per_step_action_counts"] = per_step_counts

        # mode collapse frequency (sanity)
        out["mode_collapse_frac_gt095"] = float(np.mean([1.0 if x >= 0.95 else 0.0 for x in per_step_mode_frac])) if per_step_mode_frac else float("nan")
    except Exception:
        return out
    return out


def _rollout_policy_once(runner, test_mode: bool = True) -> Tuple[Any, List[Dict[str, Any]], float]:
    t0 = time.time()
    batch = runner.run(test_mode=test_mode)
    dt = time.time() - t0
    infos = getattr(runner, "last_env_infos", [])
    if not isinstance(infos, list):
        infos = []
    return batch, infos, float(dt)


def _with_action_intervention(runner, probs: Sequence[float]):
    """
    Patch runner.mac.select_actions to override chosen_actions by sampling from `probs`.
    This will change the per-agent discrete actions, which the env uses to build per-core-user actions.
    """
    import torch  # required

    probs_t = torch.tensor(list(probs), dtype=torch.float32)
    probs_t = torch.clamp(probs_t, min=0.0)
    probs_t = probs_t / torch.clamp(probs_t.sum(), min=1e-12)

    mac = runner.mac
    orig = mac.select_actions

    def _wrapped(ep_batch, t_ep: int, t_env: int, raw_observation_text=None, bs=slice(None), test_mode: bool = False):
        actions, info = orig(ep_batch, t_ep=t_ep, t_env=t_env, raw_observation_text=raw_observation_text, bs=bs, test_mode=test_mode)
        try:
            if isinstance(actions, torch.Tensor) and actions.ndim >= 2:
                bs0 = int(actions.shape[0])
                n_agents = int(actions.shape[1])
                samp = torch.multinomial(probs_t.to(actions.device), num_samples=bs0 * n_agents, replacement=True)
                samp = samp.view(bs0, n_agents)
                return samp, info
        except Exception:
            pass
        return actions, info

    return orig, _wrapped


def _summarize_z(zpts: List[ZPoint]) -> Dict[str, Any]:
    if not zpts:
        return {
            "z_eval_steps": 0,
            "kl_mean": float("nan"),
            "kl_std": float("nan"),
            "js_mean": float("nan"),
            "js_std": float("nan"),
        }
    kls = [p.kl_gt_pred for p in zpts]
    jss = [p.js_gt_pred for p in zpts]
    ent_pred = [p.ent_pred for p in zpts]
    ent_gt = [p.ent_gt for p in zpts]
    pol_pred = [p.pol_pred for p in zpts]
    pol_gt = [p.pol_gt for p in zpts]

    # trend alignment
    ent_pearson = _pearson(ent_pred, ent_gt)
    ent_spearman = _spearman(ent_pred, ent_gt)
    pol_pearson = _pearson(pol_pred, pol_gt)
    pol_spearman = _spearman(pol_pred, pol_gt)
    ent_dtw = dtw_distance(ent_pred, ent_gt)
    pol_dtw = dtw_distance(pol_pred, pol_gt)

    # stage-wise curves (sorted by stage_t)
    zpts_sorted = sorted(zpts, key=lambda x: x.stage_t)
    stage_t = [int(p.stage_t) for p in zpts_sorted]
    stage_kl = [float(p.kl_gt_pred) for p in zpts_sorted]
    stage_js = [float(p.js_gt_pred) for p in zpts_sorted]
    stage_edge_n = [int(p.labeled_edge_n) for p in zpts_sorted]

    return {
        "z_eval_steps": int(len(zpts)),
        "kl_mean": float(np.mean(kls)),
        "kl_std": float(np.std(kls)),
        "js_mean": float(np.mean(jss)),
        "js_std": float(np.std(jss)),
        "entropy_pred_mean": float(np.mean(ent_pred)),
        "entropy_gt_mean": float(np.mean(ent_gt)),
        "polarization_pred_mean": float(np.mean(pol_pred)),
        "polarization_gt_mean": float(np.mean(pol_gt)),
        # Bias proxies (systematic offset): pred_mean - gt_mean
        "bias_entropy_mean": float(np.mean(ent_pred) - np.mean(ent_gt)),
        "bias_polarization_mean": float(np.mean(pol_pred) - np.mean(pol_gt)),
        "entropy_pearson": float(ent_pearson),
        "entropy_spearman": float(ent_spearman),
        "entropy_dtw": float(ent_dtw),
        "polarization_pearson": float(pol_pearson),
        "polarization_spearman": float(pol_spearman),
        "polarization_dtw": float(pol_dtw),
        "stage_t": stage_t,
        "stage_kl": stage_kl,
        "stage_js": stage_js,
        "stage_labeled_edge_n": stage_edge_n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., hisim_stage4.yaml)")
    ap.add_argument("--ckpt", type=str, default="", help="Checkpoint directory (expects agent.th / belief_encoder.th)")
    ap.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="Force CPU mode")
    ap.add_argument("--out_json", type=str, default="paper_eval_results.json")
    ap.add_argument("--out_csv", type=str, default="paper_eval_stagewise.csv")
    ap.add_argument("--max_core_users", type=int, default=0, help="Optional cap to speed up eval (0 = keep config)")
    ap.add_argument("--n_stages", type=int, default=0, help="Optional override n_stages for eval (0 = keep config)")

    # Intervention grid: vary retweet probability vs post (action ids: post=0, retweet=1)
    ap.add_argument("--do_intervention", action="store_true", help="Run action->outcome sensitivity grid")
    ap.add_argument("--intervention_grid", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Comma list of retweet probs")
    ap.add_argument("--intervention_episodes", type=int, default=10, help="Episodes per intervention point")
    args = ap.parse_args()

    # Import project code (relative to ECON/ directory)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(repo_root, "src")
    import sys

    sys.path.insert(0, src_dir)
    from train import load_config, setup_experiment  # type: ignore

    cfg = load_config(str(args.config))
    cfg.system.seed = int(args.seed)
    cfg.enable_llm_rollout = False  # for stable eval and cost
    cfg.test_nepisode = int(max(1, args.episodes))
    if args.ckpt:
        cfg.load_model_path = str(args.ckpt)
    if args.cpu:
        cfg.system.use_cuda = False
        cfg.system.device_num = 0

    # optional speed knobs
    if hasattr(cfg, "env_args"):
        if int(args.max_core_users) > 0:
            cfg.env_args.max_core_users = int(args.max_core_users)
            cfg.env_args.expected_core_users = int(args.max_core_users)
        if int(args.n_stages) > 0:
            cfg.env_args.n_stages = int(args.n_stages)

    runner, mac, learner, logger, device = setup_experiment(cfg)

    # Params
    tot_agent, tr_agent = _count_params(getattr(mac, "agent_module", mac.agent))
    tot_be, tr_be = _count_params(getattr(mac, "belief_encoder_module", None) or getattr(mac, "belief_encoder", None) or mac.agent)
    params_info = {
        "agent_total_params": int(tot_agent),
        "agent_trainable_params": int(tr_agent),
        "belief_encoder_total_params": int(tot_be),
        "belief_encoder_trainable_params": int(tr_be),
        "device": str(device),
    }

    # === Base evaluation (policy rollout) ===
    # Note:
    # - We keep TWO summaries:
    #   (1) global_concat: concatenate all episodes' z points and compute KL/JS/DTW/Corr once (length-dependent DTW!)
    #   (2) per_episode: compute metrics per episode then average across episodes (recommended for comparison)
    all_zpts_concat: List[ZPoint] = []
    per_ep_summaries: List[Dict[str, Any]] = []
    per_ep_stage_summaries: List[Dict[str, Any]] = []
    all_dt: List[float] = []
    last_batch = None
    last_infos: List[Dict[str, Any]] = []
    for _ in range(int(args.episodes)):
        batch, infos, dt = _rollout_policy_once(runner, test_mode=True)
        all_dt.append(dt)
        last_batch = batch
        last_infos = infos
        zpts_i = _extract_z_series(infos)
        all_zpts_concat.extend(zpts_i)
        per_ep_summaries.append(_summarize_z(zpts_i))
        # New: macro curve by stage_t within episode
        stages_i = _aggregate_z_by_stage(zpts_i)
        per_ep_stage_summaries.append(_summarize_z_stage_curve(stages_i))

    z_summary_concat = _summarize_z(all_zpts_concat)

    # Per-episode aggregation (more comparable across runs; DTW/Corr not inflated by concatenation length)
    def _mean_std(key: str) -> Tuple[float, float]:
        xs = []
        for s in per_ep_summaries:
            v = s.get(key, float("nan"))
            try:
                v = float(v)
            except Exception:
                v = float("nan")
            if math.isfinite(v):
                xs.append(v)
        if not xs:
            return float("nan"), float("nan")
        return float(np.mean(xs)), float(np.std(xs))

    pop_per_ep = {
        "episodes": int(args.episodes),
        "z_eval_steps_mean": _mean_std("z_eval_steps")[0],
        "kl_mean_mean": _mean_std("kl_mean")[0],
        "kl_mean_std": _mean_std("kl_mean")[1],
        "js_mean_mean": _mean_std("js_mean")[0],
        "js_mean_std": _mean_std("js_mean")[1],
        "bias_entropy_mean_mean": _mean_std("bias_entropy_mean")[0],
        "bias_entropy_mean_std": _mean_std("bias_entropy_mean")[1],
        "bias_polarization_mean_mean": _mean_std("bias_polarization_mean")[0],
        "bias_polarization_mean_std": _mean_std("bias_polarization_mean")[1],
        "entropy_pearson_mean": _mean_std("entropy_pearson")[0],
        "entropy_spearman_mean": _mean_std("entropy_spearman")[0],
        "entropy_dtw_mean": _mean_std("entropy_dtw")[0],
        "polarization_pearson_mean": _mean_std("polarization_pearson")[0],
        "polarization_spearman_mean": _mean_std("polarization_spearman")[0],
        "polarization_dtw_mean": _mean_std("polarization_dtw")[0],
        "note": (
            "Per-episode averages (recommended). "
            "DTW here is computed per episode (short sequences), so values are comparable across runs."
        ),
    }

    # New: per-episode macro-curve (by stage_t) aggregation
    def _mean_std_stage(key: str) -> Tuple[float, float]:
        xs = []
        for s in per_ep_stage_summaries:
            v = s.get(key, float("nan"))
            try:
                v = float(v)
            except Exception:
                v = float("nan")
            if math.isfinite(v):
                xs.append(v)
        if not xs:
            return float("nan"), float("nan")
        return float(np.mean(xs)), float(np.std(xs))

    pop_per_ep_by_stage = {
        "episodes": int(args.episodes),
        "n_stages_mean": _mean_std_stage("n_stages")[0],
        "kl_mean_mean": _mean_std_stage("kl_mean")[0],
        "kl_mean_std": _mean_std_stage("kl_mean")[1],
        "js_mean_mean": _mean_std_stage("js_mean")[0],
        "js_mean_std": _mean_std_stage("js_mean")[1],
        "bias_entropy_mean_mean": _mean_std_stage("bias_entropy_mean")[0],
        "bias_entropy_mean_std": _mean_std_stage("bias_entropy_mean")[1],
        "bias_polarization_mean_mean": _mean_std_stage("bias_polarization_mean")[0],
        "bias_polarization_mean_std": _mean_std_stage("bias_polarization_mean")[1],
        "entropy_pearson_mean": _mean_std_stage("entropy_pearson")[0],
        "entropy_spearman_mean": _mean_std_stage("entropy_spearman")[0],
        "entropy_dtw_mean": _mean_std_stage("entropy_dtw")[0],
        "polarization_pearson_mean": _mean_std_stage("polarization_pearson")[0],
        "polarization_spearman_mean": _mean_std_stage("polarization_spearman")[0],
        "polarization_dtw_mean": _mean_std_stage("polarization_dtw")[0],
        "note": (
            "Macro-curve by stage_t within each episode (time/round-based). "
            "Recommended when you want turn-wise macro curves before computing div/DTW/Corr."
        ),
    }
    action_summary = _extract_actions_from_batch(last_batch, n_actions=int(getattr(cfg, "n_actions", 5)))

    results: Dict[str, Any] = {
        "meta": {
            "config": str(args.config),
            "ckpt": str(args.ckpt) if args.ckpt else "",
            "episodes": int(args.episodes),
            "seed": int(args.seed),
            "cpu": bool(args.cpu),
        },
        "params": params_info,
        "timing": {
            "episode_time_sec_mean": float(np.mean(all_dt)) if all_dt else float("nan"),
            "episode_time_sec_std": float(np.std(all_dt)) if all_dt else float("nan"),
        },
        # Backward compatible key: keep the original (global concatenation) behavior.
        "population_metrics": z_summary_concat,
        # Recommended for comparisons (matches the user's expected magnitudes for DTW/bias/div in many setups).
        "population_metrics_per_episode": pop_per_ep,
        # New: macro curve (by stage_t) per-episode summary (recommended for HiSim-style macro comparisons).
        "population_metrics_per_episode_by_stage": pop_per_ep_by_stage,
        "micro_sanity": action_summary,
    }

    # === Intervention / sensitivity grid ===
    if bool(args.do_intervention):
        try:
            grid = [float(x.strip()) for x in str(args.intervention_grid).split(",") if x.strip()]
        except Exception:
            grid = [0.1, 0.3, 0.5, 0.7, 0.9]
        grid = [min(0.999, max(0.001, x)) for x in grid]

        intervention_rows: List[Dict[str, Any]] = []
        for pr in grid:
            # action probs over 5 actions: [post, retweet, reply, like, do_nothing]
            probs = [1.0 - pr, pr, 0.0, 0.0, 0.0]
            orig_sel, wrapped = _with_action_intervention(runner, probs)
            runner.mac.select_actions = wrapped  # type: ignore
            try:
                zpts_i: List[ZPoint] = []
                for _ in range(int(args.intervention_episodes)):
                    _, infos, _dt = _rollout_policy_once(runner, test_mode=True)
                    zpts_i.extend(_extract_z_series(infos))
                summ_i = _summarize_z(zpts_i)
                # response proxies: final-stage z_pred (use last zpt in time)
                zpts_sorted = sorted(zpts_i, key=lambda z: z.stage_t)
                z_last = zpts_sorted[-1].z_pred if zpts_sorted else []
                # common: report Support prob (id=2) if K=3
                z_support = float(_normalize_prob(z_last)[2]) if len(z_last) >= 3 else float("nan")
                intervention_rows.append(
                    {
                        "retweet_prob": float(pr),
                        "post_prob": float(1.0 - pr),
                        "z_eval_steps": int(summ_i.get("z_eval_steps", 0)),
                        "kl_mean": float(summ_i.get("kl_mean", float("nan"))),
                        "js_mean": float(summ_i.get("js_mean", float("nan"))),
                        "polarization_pred_mean": float(summ_i.get("polarization_pred_mean", float("nan"))),
                        "entropy_pred_mean": float(summ_i.get("entropy_pred_mean", float("nan"))),
                        "z_pred_support_last": float(z_support),
                    }
                )
            finally:
                runner.mac.select_actions = orig_sel  # type: ignore

        # sensitivity summary (monotonicity + slope)
        xs = [r["retweet_prob"] for r in intervention_rows]
        ys = [r.get("z_pred_support_last", float("nan")) for r in intervention_rows]
        xs2 = [x for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(float(y))]
        ys2 = [float(y) for y in ys if math.isfinite(float(y))]
        sens = {
            "spearman_retweetprob_vs_support_last": float(_spearman(xs2, ys2)) if len(xs2) >= 2 else float("nan"),
            "pearson_retweetprob_vs_support_last": float(_pearson(xs2, ys2)) if len(xs2) >= 2 else float("nan"),
        }
        results["intervention"] = {
            "grid": intervention_rows,
            "sensitivity": sens,
            "note": "Intervention overrides core-user action_type distribution by sampling chosen_actions; stance ids still come from model stance head.",
        }

    # Write outputs
    out_json = str(args.out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Stage-wise CSV (z points from last episode for quick plotting)
    out_csv = str(args.out_csv)
    try:
        z_last = _extract_z_series(last_infos)
        z_last = sorted(z_last, key=lambda z: z.stage_t)
        # write minimal csv without pandas
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("stage_t,kl_gt_pred,js_gt_pred,ent_pred,ent_gt,pol_pred,pol_gt,labeled_edge_n\n")
            for p in z_last:
                f.write(
                    f"{p.stage_t},{p.kl_gt_pred:.8f},{p.js_gt_pred:.8f},{p.ent_pred:.8f},{p.ent_gt:.8f},{p.pol_pred:.8f},{p.pol_gt:.8f},{p.labeled_edge_n}\n"
                )
    except Exception:
        pass

    print(f"[OK] Wrote JSON: {out_json}")
    print(f"[OK] Wrote CSV:  {out_csv}")
    print("[Summary] population KL mean:", results.get("population_metrics", {}).get("kl_mean"))
    print("[Summary] population JS mean:", results.get("population_metrics", {}).get("js_mean"))
    if bool(args.do_intervention):
        print("[Summary] intervention spearman(retweet->support_last):", results.get("intervention", {}).get("sensitivity", {}).get("spearman_retweetprob_vs_support_last"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


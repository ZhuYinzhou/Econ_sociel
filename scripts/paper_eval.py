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
      - Stance accuracy (1/2/3/all) using the same t+1 imitation-style alignment as the env reward.
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
from pathlib import Path
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

    def _mean2(a: float, b: float) -> float:
        try:
            fa = float(a)
            fb = float(b)
        except Exception:
            return float("nan")
        if (not math.isfinite(fa)) and (not math.isfinite(fb)):
            return float("nan")
        if not math.isfinite(fa):
            return float(fb)
        if not math.isfinite(fb):
            return float(fa)
        return 0.5 * (fa + fb)

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
        "corr_entropy": float(ent_pearson),
        "corr_polarization": float(pol_pearson),
        "corr_mean": float(_mean2(ent_pearson, pol_pearson)),
        "entropy_pearson": float(ent_pearson),
        "entropy_spearman": float(ent_spearman),
        "entropy_dtw": float(ent_dtw),
        "polarization_pearson": float(pol_pearson),
        "polarization_spearman": float(pol_spearman),
        "polarization_dtw": float(pol_dtw),
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

        out["mode_collapse_frac_gt095"] = float(np.mean([1.0 if x >= 0.95 else 0.0 for x in per_step_mode_frac])) if per_step_mode_frac else float("nan")
    except Exception:
        return out
    return out


def _extract_stance_accuracy_from_env(env: Any, n_stances: int = 3, gt_offset: int = 1) -> Dict[str, Any]:
    """
    Compute stance accuracy from env internal traces.

    Why env-based?
    - In sync-stage mode, per-user (pred,gt) stance ids are not necessarily emitted in env_infos.
    - However, env always stores predicted posts in env.core_posts[(user, t)] and exposes _gt_for(user, t_gt).

    Alignment (IMPORTANT, matches hisim_social_env reward semantics):
      pred at stage t is evaluated against ground-truth at stage (t + gt_offset), default gt_offset=1.

    Returns:
      - stance_acc_all: overall accuracy over valid stance-expressing actions
      - stance_acc_1/2/3: per-class accuracy by GT stance id (reported as 1..K for paper readability)
      - stance_n_all / stance_n_1/2/3: denominators
    """
    out: Dict[str, Any] = {}
    k = max(1, int(n_stances))
    try:
        posts = getattr(env, "core_posts", None)
        gt_for = getattr(env, "_gt_for", None)
        n_stages = int(getattr(env, "n_stages", 0) or 0)
    except Exception:
        posts = None
        gt_for = None
        n_stages = 0
    if not isinstance(posts, dict) or not callable(gt_for):
        return out

    stance_actions = {"post", "retweet", "reply"}
    total = 0
    correct = 0
    denom_by_gt = [0 for _ in range(k)]
    corr_by_gt = [0 for _ in range(k)]

    for key, p in posts.items():
        if not (isinstance(key, tuple) and len(key) >= 2):
            continue
        user, t_pred = key[0], key[1]
        if not isinstance(p, dict):
            continue
        at = str(p.get("action_type") or "").strip().lower()
        if at not in stance_actions:
            continue
        sid = p.get("stance_id", None)
        try:
            sid_i = int(sid) if sid is not None else None
        except Exception:
            sid_i = None
        if sid_i is None or sid_i < 0 or sid_i >= k:
            continue

        try:
            t_gt = int(t_pred) + int(gt_offset)
        except Exception:
            continue
        if n_stages > 0 and t_gt >= n_stages:
            continue

        try:
            gt_sid, _gt_lab, _gt_text = gt_for(str(user), int(t_gt))
        except Exception:
            gt_sid = None
        if gt_sid is None:
            continue
        try:
            gt_i = int(gt_sid)
        except Exception:
            continue
        if gt_i < 0 or gt_i >= k:
            continue

        total += 1
        denom_by_gt[gt_i] += 1
        if int(sid_i) == int(gt_i):
            correct += 1
            corr_by_gt[gt_i] += 1

    def _acc(n: int, d: int) -> float:
        return float(n) / float(d) if d > 0 else float("nan")

    out["stance_n_all"] = int(total)
    out["stance_acc_all"] = _acc(correct, total)
    for j in range(min(3, k)):
        out[f"stance_n_{j+1}"] = int(denom_by_gt[j])
        out[f"stance_acc_{j+1}"] = _acc(corr_by_gt[j], denom_by_gt[j])
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

    ent_pearson = _pearson(ent_pred, ent_gt)
    ent_spearman = _spearman(ent_pred, ent_gt)
    pol_pearson = _pearson(pol_pred, pol_gt)
    pol_spearman = _spearman(pol_pred, pol_gt)
    ent_dtw = dtw_distance(ent_pred, ent_gt)
    pol_dtw = dtw_distance(pol_pred, pol_gt)

    def _mean2(a: float, b: float) -> float:
        try:
            fa = float(a)
            fb = float(b)
        except Exception:
            return float("nan")
        if (not math.isfinite(fa)) and (not math.isfinite(fb)):
            return float("nan")
        if not math.isfinite(fa):
            return float(fb)
        if not math.isfinite(fb):
            return float(fa)
        return 0.5 * (fa + fb)

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
        "bias_entropy_mean": float(np.mean(ent_pred) - np.mean(ent_gt)),
        "bias_polarization_mean": float(np.mean(pol_pred) - np.mean(pol_gt)),
        "corr_entropy": float(ent_pearson),
        "corr_polarization": float(pol_pearson),
        "corr_mean": float(_mean2(ent_pearson, pol_pearson)),
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
    ap.add_argument(
        "--stagewise_source",
        type=str,
        default="last",
        choices=["last", "all_mean"],
        help=(
            "How to build the stage-wise CSV curve. "
            "'last' uses z points from the last episode only (backward compatible). "
            "'all_mean' aggregates z across ALL episodes by stage_t and writes mean z_pred/z_gt per stage."
        ),
    )
    ap.add_argument("--max_core_users", type=int, default=0, help="Optional cap to speed up eval (0 = keep config)")
    ap.add_argument("--n_stages", type=int, default=0, help="Optional override n_stages for eval (0 = keep config)")

    ap.add_argument("--do_intervention", action="store_true", help="Run action->outcome sensitivity grid")
    ap.add_argument("--intervention_grid", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Comma list of retweet probs")
    ap.add_argument("--intervention_episodes", type=int, default=10, help="Episodes per intervention point")
    args = ap.parse_args()

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

    if hasattr(cfg, "env_args"):
        if int(args.max_core_users) > 0:
            cfg.env_args.max_core_users = int(args.max_core_users)
            cfg.env_args.expected_core_users = int(args.max_core_users)
        if int(args.n_stages) > 0:
            cfg.env_args.n_stages = int(args.n_stages)

    runner, mac, learner, logger, device = setup_experiment(cfg)

    tot_agent, tr_agent = _count_params(getattr(mac, "agent_module", mac.agent))
    tot_be, tr_be = _count_params(getattr(mac, "belief_encoder_module", None) or getattr(mac, "belief_encoder", None) or mac.agent)
    params_info = {
        "agent_total_params": int(tot_agent),
        "agent_trainable_params": int(tr_agent),
        "belief_encoder_total_params": int(tot_be),
        "belief_encoder_trainable_params": int(tr_be),
        "device": str(device),
    }

    all_zpts_concat: List[ZPoint] = []
    per_ep_summaries: List[Dict[str, Any]] = []
    per_ep_stage_summaries: List[Dict[str, Any]] = []
    per_ep_stance: List[Dict[str, Any]] = []
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
        stages_i = _aggregate_z_by_stage(zpts_i)
        per_ep_stage_summaries.append(_summarize_z_stage_curve(stages_i))
        try:
            per_ep_stance.append(_extract_stance_accuracy_from_env(getattr(runner, "env", None), n_stances=3, gt_offset=1))
        except Exception:
            per_ep_stance.append({})

    z_summary_concat = _summarize_z(all_zpts_concat)

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
        "corr_entropy_mean": _mean_std("corr_entropy")[0],
        "corr_polarization_mean": _mean_std("corr_polarization")[0],
        "corr_mean_mean": _mean_std("corr_mean")[0],
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
        "corr_entropy_mean": _mean_std_stage("corr_entropy")[0],
        "corr_polarization_mean": _mean_std_stage("corr_polarization")[0],
        "corr_mean_mean": _mean_std_stage("corr_mean")[0],
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

    def _mean_stance(key: str) -> float:
        xs = []
        for s in per_ep_stance:
            v = s.get(key, float("nan"))
            try:
                v = float(v)
            except Exception:
                v = float("nan")
            if math.isfinite(v):
                xs.append(v)
        return float(np.mean(xs)) if xs else float("nan")

    stance_acc = {
        "stance_acc_all": _mean_stance("stance_acc_all"),
        "stance_acc_1": _mean_stance("stance_acc_1"),
        "stance_acc_2": _mean_stance("stance_acc_2"),
        "stance_acc_3": _mean_stance("stance_acc_3"),
        "note": "Stance accuracy computed from env.core_posts and env._gt_for(user, t+1), matching env imitation-style reward semantics.",
    }

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
        "population_metrics": z_summary_concat,
        "population_metrics_per_episode": pop_per_ep,
        "population_metrics_per_episode_by_stage": pop_per_ep_by_stage,
        "micro_sanity": {
            **(action_summary or {}),
            "stance_accuracy": stance_acc,
        },
    }

    if bool(args.do_intervention):
        try:
            grid = [float(x.strip()) for x in str(args.intervention_grid).split(",") if x.strip()]
        except Exception:
            grid = [0.1, 0.3, 0.5, 0.7, 0.9]
        grid = [min(0.999, max(0.001, x)) for x in grid]

        intervention_rows: List[Dict[str, Any]] = []
        for pr in grid:
            probs = [1.0 - pr, pr, 0.0, 0.0, 0.0]
            orig_sel, wrapped = _with_action_intervention(runner, probs)
            runner.mac.select_actions = wrapped  # type: ignore
            try:
                zpts_i: List[ZPoint] = []
                for _ in range(int(args.intervention_episodes)):
                    _, infos, _dt = _rollout_policy_once(runner, test_mode=True)
                    zpts_i.extend(_extract_z_series(infos))
                summ_i = _summarize_z(zpts_i)
                zpts_sorted = sorted(zpts_i, key=lambda z: z.stage_t)
                z_last = zpts_sorted[-1].z_pred if zpts_sorted else []
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

    out_json = str(args.out_json)
    try:
        Path(out_json).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    out_csv = str(args.out_csv)
    try:
        Path(out_csv).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        stagewise_source = str(getattr(args, "stagewise_source", "last") or "last").strip().lower()
        stagewise_source = stagewise_source if stagewise_source in ("last", "all_mean") else "last"

        with open(out_csv, "w", encoding="utf-8") as f:
            f.write(
                "stage_t,"
                "z_pred_neutral,z_pred_oppose,z_pred_support,"
                "z_gt_neutral,z_gt_oppose,z_gt_support,"
                "kl_gt_pred,js_gt_pred,ent_pred,ent_gt,pol_pred,pol_gt,"
                "labeled_edge_n\n"
            )

            if stagewise_source == "all_mean":
                stages_all = _aggregate_z_by_stage(all_zpts_concat)
                for s in stages_all:
                    zp = _normalize_prob(list(s.z_pred_mean))
                    zg = _normalize_prob(list(s.z_gt_mean))
                    k = kl_div(zg, zp)
                    j = js_div(zg, zp)
                    ep = entropy(zp)
                    eg = entropy(zg)
                    pp = polarization_index(zp)
                    pg = polarization_index(zg)
                    f.write(
                        f"{int(s.stage_t)},"
                        f"{float(zp[0]):.8f},{float(zp[1]):.8f},{float(zp[2]):.8f},"
                        f"{float(zg[0]):.8f},{float(zg[1]):.8f},{float(zg[2]):.8f},"
                        f"{float(k):.8f},{float(j):.8f},{float(ep):.8f},{float(eg):.8f},{float(pp):.8f},{float(pg):.8f},"
                        f"{int(getattr(s, 'labeled_edge_n_sum', 0) or 0)}\n"
                    )
            else:
                z_last = _extract_z_series(last_infos)
                z_last = sorted(z_last, key=lambda z: z.stage_t)
                for p in z_last:
                    zp = _normalize_prob(list(p.z_pred))
                    zg = _normalize_prob(list(p.z_gt))
                    f.write(
                        f"{int(p.stage_t)},"
                        f"{float(zp[0]):.8f},{float(zp[1]):.8f},{float(zp[2]):.8f},"
                        f"{float(zg[0]):.8f},{float(zg[1]):.8f},{float(zg[2]):.8f},"
                        f"{float(p.kl_gt_pred):.8f},{float(p.js_gt_pred):.8f},{float(p.ent_pred):.8f},{float(p.ent_gt):.8f},{float(p.pol_pred):.8f},{float(p.pol_gt):.8f},"
                        f"{int(p.labeled_edge_n)}\n"
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


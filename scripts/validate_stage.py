#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证五个阶段（S1/S2/S3a/S3b/S4）的“关键卡点”是否过关。

目标（按你的要求）：

Stage1（Core belief）必须停下来验证：
- core_stance_acc（eval split）
- belief_sup_acc（train）
- majority baseline + margin
- train/eval gap 不大；曲线不发散/不剧烈震荡
- 混淆矩阵（至少脑内判断）=> 这里给出真实 confusion matrix（从 eval 跑出来的 boxed-id 预测/GT）

Stage3a（z transition）必须单独验证：
- train/loss_z_transition（KL/CE）稳定下降（从日志看趋势）
- held-out split（validation/test）做 no_grad eval loss（不依赖 run_test 的 legacy schema）
- identity baseline 必须对比：
    KL(z_target || z_t)  vs  KL(z_target || z_pred)
- 防退化：z_pred entropy / maxprob / mean probs（避免退化成均值分布）

用法示例：
  # Stage1：用 Stage1 的 config + checkpoint + logdir 验证
  conda run -n HiSim python ECON/scripts/validate_stage1_s3a.py \\
    --mode stage1 \\
    --config /home/zhuyinzhou/MAS/ECON/examples/configs/hisim_belief_core_stable.yaml \\
    --ckpt /data/zhuyinzhou/ECON/models/checkpoints_s1_e1e2_stable/episode_45000 \\
    --logdir /home/zhuyinzhou/MAS/ECON/logs/hisim-belief-core-stable \\
    --eval_split test \\
    --eval_episodes 200

  # Stage3a：用 Stage3a 的 config + checkpoint + logdir 验证（held-out split no_grad）
  conda run -n HiSim python ECON/scripts/validate_stage1_s3a.py \\
    --mode stage3a \\
    --config /home/zhuyinzhou/MAS/ECON/examples/configs/hisim_z_transition_dist_cond.yaml \\
    --ckpt /data/zhuyinzhou/ECON/models/checkpoints_s3a_e/episode_50000 \\
    --logdir /home/zhuyinzhou/MAS/ECON/logs/hisim-z-transition-s3a-e \\
    --eval_split test \\
    --eval_episodes 200

  # Stage2：非核心用户 belief（K=3 stance）同 Stage1 的 boxed-id eval + majority baseline
  conda run -n HiSim python ECON/scripts/validate_stage1_s3a.py \\
    --mode stage2 \\
    --config /home/zhuyinzhou/MAS/ECON/examples/configs/hisim_belief_noncore_stage2.yaml \\
    --ckpt /data/zhuyinzhou/ECON/models/checkpoints_s2/episode_50000 \\
    --logdir /home/zhuyinzhou/MAS/ECON/logs/hisim-belief-noncore-stage2 \\
    --eval_split test \\
    --eval_episodes 200

  # Stage3b：action imitation（K=5），并支持 masked supervision（例如只评估 post/retweet）
  conda run -n HiSim python ECON/scripts/validate_stage1_s3a.py \\
    --mode stage3b \\
    --config /home/zhuyinzhou/MAS/ECON/examples/configs/hisim_action_imitation_core.yaml \\
    --ckpt /data/zhuyinzhou/ECON/models/checkpoints_s3b/final \\
    --logdir /home/zhuyinzhou/MAS/ECON/logs/hisim-action-imitation-s3b \\
    --eval_split test \\
    --eval_episodes 200

  # Stage4：online RL（hisim_social_env），跑若干 test episodes 打印 return/z_kl 等诊断（PASS/FAIL 以“可稳定跑完+指标有限”为主）
  conda run -n HiSim python ECON/scripts/validate_stage1_s3a.py \\
    --mode stage4 \\
    --config /home/zhuyinzhou/MAS/ECON/examples/configs/hisim_social_stage4_zreward.yaml \\
    --ckpt /data/zhuyinzhou/ECON/models/checkpoints_s4/final \\
    --logdir /home/zhuyinzhou/MAS/ECON/logs/hisim-social-stage4-zreward \\
    --eval_split test \\
    --eval_episodes 5

注意：
- 本脚本会启动 runner 并跑 eval episodes（不会训练），用于拿到真实预测与 held-out eval loss。
- 需要在包含 torch/datasets 的环境里运行（推荐 conda env: HiSim）。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _boxed_int(s: Any) -> Optional[int]:
    try:
        if not isinstance(s, str):
            return None
        m = re.search(r"\\boxed\{\s*([-+]?\d+)\s*\}", s)
        if not m:
            m = re.search(r"boxed\{\s*([-+]?\d+)\s*\}", s)
        if not m:
            return None
        return int(m.group(1))
    except Exception:
        return None


def _read_metrics_jsonl(metrics_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(metrics_path):
        return rows
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _extract_series(metrics_rows: List[Dict[str, Any]], metric_name: str) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for r in metrics_rows:
        if r.get("metric") != metric_name:
            continue
        try:
            step = int(r.get("step"))
            v = float(r.get("value"))
        except Exception:
            continue
        out.append((step, v))
    out.sort(key=lambda x: x[0])
    return out


def _moving_avg(xs: List[float], k: int = 200) -> float:
    if not xs:
        return float("nan")
    k = max(1, int(k))
    w = xs[-k:] if len(xs) >= k else xs
    return float(sum(w) / float(len(w)))


@dataclass
class Stage1EvalResult:
    eval_acc: float
    majority_baseline: float
    confusion: List[List[int]]  # 3x3
    n: int


def _majority_baseline_from_dataset(cfg: Any, split: str) -> Optional[float]:
    """
    从 HF dataset 的 answer(\\boxed{id}) 统计 majority baseline（不跑模型）。
    """
    try:
        from datasets import load_from_disk, concatenate_datasets  # type: ignore
    except Exception:
        return None

    try:
        env_args = getattr(cfg, "env_args", None)
        if env_args is None:
            return None
        paths = getattr(env_args, "hf_dataset_path", None)
        if paths is None:
            return None
        ds_sources: List[str] = []
        if isinstance(paths, list):
            ds_sources = [str(x) for x in paths if str(x).strip()]
        else:
            ds_sources = [str(paths)]
        parts = []
        for p in ds_sources:
            if not os.path.isdir(p):
                continue
            dd = load_from_disk(p)
            if hasattr(dd, "keys"):
                if split not in dd:
                    continue
                parts.append(dd[split])
            else:
                parts.append(dd)
        if not parts:
            return None
        dset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        k = int(getattr(getattr(cfg, "env_args", None), "n_actions", 3))
        k = max(1, k)
        cnt = [0 for _ in range(k)]
        n = 0
        for ex in dset:
            a = ex.get(getattr(env_args, "answer_field_name", "answer"), ex.get("answer", ""))
            y = _boxed_int(a)
            if y is None or y < 0 or y >= k:
                continue
            cnt[int(y)] += 1
            n += 1
        if n <= 0:
            return None
        maj = max(cnt) / float(n)
        return float(maj)
    except Exception:
        return None


def _stage1_eval_confusion(cfg: Any, ckpt: str, eval_split: str, eval_episodes: int) -> Stage1EvalResult:
    """
    通过 runner 在 eval split 上跑若干 episodes，基于 env_info 的 boxed-id 预测/GT 计算 acc 与 confusion matrix。
    """
    # import project training harness
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from train import setup_experiment  # type: ignore

    cfg2 = copy.deepcopy(cfg)
    # enforce eval split for this run
    try:
        if hasattr(cfg2, "env_args") and hasattr(cfg2.env_args, "dataset_split"):
            cfg2.env_args.dataset_split = str(eval_split)
    except Exception:
        pass
    # load ckpt
    cfg2.load_model_path = str(ckpt)
    runner, _mac, _learner, logger, _device = setup_experiment(cfg2)

    k = int(getattr(getattr(cfg2, "env_args", None), "n_actions", 3))
    k = max(1, k)
    if k < 3:
        # still build square confusion
        k = 3
    conf = [[0 for _ in range(k)] for _ in range(k)]
    correct = 0
    total = 0
    n_ep = max(1, int(eval_episodes))
    for _ in range(n_ep):
        _batch = runner.run(test_mode=True)
        infos = getattr(runner, "last_env_infos", None)
        if not isinstance(infos, list):
            continue
        for info in infos:
            if not isinstance(info, dict):
                continue
            gt = _boxed_int(info.get("ground_truth_answer", "")) or _boxed_int(info.get("ground_truth", ""))
            pr = _boxed_int(info.get("llm_answer", "")) or _boxed_int(info.get("answer", ""))
            if gt is None or pr is None:
                continue
            if gt < 0 or gt >= k or pr < 0 or pr >= k:
                continue
            conf[int(gt)][int(pr)] += 1
            total += 1
            if int(gt) == int(pr):
                correct += 1

    eval_acc = float(correct) / float(total) if total > 0 else 0.0
    majority = _majority_baseline_from_dataset(cfg2, eval_split)
    if majority is None:
        majority = 0.0
        logger.warning("Stage1: failed to compute majority baseline from dataset; defaulting to 0.0")
    return Stage1EvalResult(eval_acc=eval_acc, majority_baseline=float(majority), confusion=conf, n=total)

def _majority_baseline_from_dataset_masked(cfg: Any, split: str, *, only_ids: Optional[set]) -> Optional[float]:
    """
    Stage3b: 从 HF dataset 的 answer(\\boxed{id}) 统计 masked majority baseline。
    仅统计 gt ∈ only_ids 的样本；若 only_ids=None 则等价于普通 majority。
    """
    try:
        from datasets import load_from_disk, concatenate_datasets  # type: ignore
    except Exception:
        return None
    try:
        env_args = getattr(cfg, "env_args", None)
        if env_args is None:
            return None
        paths = getattr(env_args, "hf_dataset_path", None)
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            return None
        parts = []
        for p in paths:
            if not isinstance(p, str) or (not os.path.isdir(p)):
                continue
            dd = load_from_disk(p)
            if hasattr(dd, "keys"):
                if split not in dd:
                    continue
                parts.append(dd[split])
            else:
                parts.append(dd)
        if not parts:
            return None
        dset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        k = int(getattr(getattr(cfg, "env_args", None), "n_actions", 5))
        k = max(1, k)
        cnt = [0 for _ in range(k)]
        n = 0
        for ex in dset:
            a = ex.get(getattr(env_args, "answer_field_name", "answer"), ex.get("answer", ""))
            y = _boxed_int(a)
            if y is None or y < 0 or y >= k:
                continue
            if isinstance(only_ids, set) and len(only_ids) > 0 and int(y) not in only_ids:
                continue
            cnt[int(y)] += 1
            n += 1
        if n <= 0:
            return None
        maj = max(cnt) / float(n)
        return float(maj)
    except Exception:
        return None


def _stage2_eval_confusion(cfg: Any, ckpt: str, eval_split: str, eval_episodes: int) -> Stage1EvalResult:
    """
    Stage2 与 Stage1 相同：K=3 stance boxed-id accuracy + confusion matrix（只是数据是 noncore）。
    """
    return _stage1_eval_confusion(cfg, ckpt, eval_split, eval_episodes)


def _stage3b_eval_action_imitation(cfg: Any, ckpt: str, eval_split: str, eval_episodes: int) -> Dict[str, Any]:
    """
    Stage3b: 在 HF dataset eval split 上做 boxed-id action accuracy。
    若 config.action_imitation_supervised_action_ids 存在，则计算 masked accuracy（只评估这些 label）。
    """
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from train import setup_experiment  # type: ignore

    cfg2 = copy.deepcopy(cfg)
    try:
        if hasattr(cfg2, "env_args") and hasattr(cfg2.env_args, "dataset_split"):
            cfg2.env_args.dataset_split = str(eval_split)
    except Exception:
        pass
    cfg2.load_model_path = str(ckpt)
    runner, _mac, _learner, logger, _device = setup_experiment(cfg2)

    # supervised ids for masked eval
    sup_ids = None
    try:
        only_ids = getattr(cfg2, "action_imitation_supervised_action_ids", None)
        if isinstance(only_ids, (list, tuple)) and len(only_ids) > 0:
            sup_ids = set(int(x) for x in only_ids)
    except Exception:
        sup_ids = None

    # Determine K
    try:
        k = int(getattr(getattr(cfg2, "env_args", None), "n_actions", getattr(cfg2, "n_actions", 5)))
    except Exception:
        k = 5
    k = max(1, k)

    # confusion for masked rows only (size k x k)
    conf = [[0 for _ in range(k)] for _ in range(k)]
    total = 0
    correct = 0
    skipped_unsup = 0

    n_ep = max(1, int(eval_episodes))
    for _ in range(n_ep):
        _batch = runner.run(test_mode=True)
        infos = getattr(runner, "last_env_infos", None)
        if not isinstance(infos, list):
            continue
        for info in infos:
            if not isinstance(info, dict):
                continue
            gt = _boxed_int(info.get("ground_truth_answer", "")) or _boxed_int(info.get("ground_truth", ""))
            pr = _boxed_int(info.get("llm_answer", "")) or _boxed_int(info.get("answer", ""))
            if gt is None:
                continue
            if isinstance(sup_ids, set) and len(sup_ids) > 0 and int(gt) not in sup_ids:
                skipped_unsup += 1
                continue
            if pr is None:
                # count as incorrect (still increases total)
                pr = -1
            if gt < 0 or gt >= k:
                continue
            if pr < 0 or pr >= k:
                # treat as invalid prediction; still count total
                total += 1
                continue
            conf[int(gt)][int(pr)] += 1
            total += 1
            if int(gt) == int(pr):
                correct += 1

    acc = float(correct) / float(total) if total > 0 else 0.0
    # masked baseline
    maj = _majority_baseline_from_dataset_masked(cfg2, eval_split, only_ids=sup_ids)
    if maj is None:
        maj = 0.0
        logger.warning("Stage3b: failed to compute masked majority baseline from dataset; defaulting to 0.0")

    denom = float(total + skipped_unsup)
    coverage = float(total) / denom if denom > 0 else 0.0
    skipped_ratio = float(skipped_unsup) / denom if denom > 0 else 0.0

    return {
        "eval_acc_masked": float(acc),
        "majority_baseline_masked": float(maj),
        "margin": float(acc - float(maj)),
        "n_masked": int(total),
        "n_skipped_unsup": int(skipped_unsup),
        "coverage": float(coverage),
        "skipped_ratio": float(skipped_ratio),
        "confusion": conf,
        "k": int(k),
        "sup_ids": sorted(list(sup_ids)) if isinstance(sup_ids, set) else None,
    }


def _stage4_eval_online(cfg: Any, ckpt: str, eval_episodes: int) -> Dict[str, Any]:
    """
    Stage4: online env sanity + key metrics.
    PASS/FAIL 以“可稳定跑完且指标为有限值”为主（RL 的目标阈值依赖 reward 配置与训练进度，不强行 hard gate）。
    """
    import math
    import numpy as np

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from train import setup_experiment  # type: ignore

    cfg2 = copy.deepcopy(cfg)
    cfg2.load_model_path = str(ckpt)
    runner, _mac, _learner, _logger, _device = setup_experiment(cfg2)

    returns: List[float] = []
    z_kl_list: List[float] = []
    z_steps = 0

    def _renorm(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        p = np.clip(p, 0.0, None)
        s = float(np.sum(p))
        if s <= eps:
            return np.ones_like(p) / float(len(p))
        return p / s

    def _kl(tgt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
        t = _renorm(tgt, eps=eps)
        q = _renorm(pred, eps=eps)
        return float(np.sum(t * (np.log(t + eps) - np.log(q + eps))))

    n_ep = max(1, int(eval_episodes))
    for _ in range(n_ep):
        batch = runner.run(test_mode=True)
        if batch is None:
            continue
        try:
            ep_ret = float(batch["reward"].sum().item())
        except Exception:
            ep_ret = 0.0
        returns.append(ep_ret)
        infos = getattr(runner, "last_env_infos", None)
        if not isinstance(infos, list):
            continue
        for info in infos:
            if not isinstance(info, dict):
                continue
            try:
                zm = float(info.get("z_mask", 0.0))
            except Exception:
                zm = 0.0
            if zm <= 0.5:
                continue
            zp = info.get("z_pred")
            zt = info.get("z_target")
            if isinstance(zp, list) and isinstance(zt, list) and len(zp) == len(zt) and len(zp) > 1:
                z_steps += 1
                try:
                    z_kl_list.append(_kl(np.array(zt, dtype=np.float32), np.array(zp, dtype=np.float32)))
                except Exception:
                    continue

    ret_mean = float(np.mean(returns)) if returns else 0.0
    ret_std = float(np.std(returns)) if returns else 0.0
    z_kl = float(np.mean(z_kl_list)) if z_kl_list else float("nan")

    ok = True
    if not math.isfinite(ret_mean):
        ok = False
    # If z_kl is present, require it finite.
    if z_steps > 0 and (not math.isfinite(z_kl)):
        ok = False

    return {
        "ok": bool(ok),
        "test_return_mean": float(ret_mean),
        "test_return_std": float(ret_std),
        "test_episodes": int(len(returns)),
        "z_kl": float(z_kl),
        "z_eval_steps": int(z_steps),
    }

def _stage3a_eval_z_transition(cfg: Any, ckpt: str, eval_split: str, eval_episodes: int) -> Dict[str, float]:
    """
    held-out split 上 no_grad eval：
    - loss_z_transition (KL/CE)
    - KL(z_target||z_t) vs KL(z_target||z_pred)
    - z_pred entropy/maxprob/mean probs
    - per-stage z_delta buckets
    - ablation: no-stage, no-group-repr
    """
    import torch
    import torch.nn.functional as F  # noqa: F401

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from train import setup_experiment  # type: ignore

    cfg2 = copy.deepcopy(cfg)
    try:
        if hasattr(cfg2, "env_args") and hasattr(cfg2.env_args, "dataset_split"):
            cfg2.env_args.dataset_split = str(eval_split)
    except Exception:
        pass
    cfg2.load_model_path = str(ckpt)
    runner, mac, _learner, logger, device = setup_experiment(cfg2)
    be = getattr(mac, "belief_encoder", None)
    if be is None or (not hasattr(be, "predict_next_population_belief")):
        raise RuntimeError("Stage3a eval requires BeliefEncoder with predict_next_population_belief().")

    def _renorm(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        p = torch.clamp(p, min=0.0)
        return p / torch.clamp(p.sum(dim=-1, keepdim=True), min=eps)

    def _kl_tgt_pred(tgt: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        t = _renorm(tgt, eps=eps)
        q = _renorm(pred, eps=eps)
        return torch.sum(t * (torch.log(t + eps) - torch.log(q + eps)), dim=-1)

    sum_mask = 0.0
    sum_loss = 0.0
    sum_kl_tgt_zt = 0.0
    sum_kl_tgt_zp = 0.0
    sum_ent = 0.0
    sum_maxp = 0.0
    sum_p0 = 0.0
    sum_p1 = 0.0
    sum_p2 = 0.0
    sum_dz_pred = 0.0
    sum_dz_tgt = 0.0
    sum_kl_nostage = 0.0
    sum_kl_nogr = 0.0
    sum_kl_randstage = 0.0
    sum_kl_shiftstage = 0.0
    # fixed-stage probes help diagnose "stage embeddings for late stages are untrained/misaligned"
    fixed_stage_ids = list(getattr(cfg2, "stage3a_fixed_stages", [11, 12]) or [11, 12])
    fixed_stage_ids = [int(x) for x in fixed_stage_ids if str(x).strip() != ""]
    fixed_stage_sums: Dict[int, float] = {int(s): 0.0 for s in fixed_stage_ids}

    stage_sum_mask: Dict[int, float] = {}
    stage_sum_dz_pred: Dict[int, float] = {}
    stage_sum_dz_tgt: Dict[int, float] = {}

    n_ep = max(1, int(eval_episodes))
    for _ in range(n_ep):
        batch = runner.run(test_mode=True)
        if batch is None:
            continue
        # Stage3a: 必须有至少 2 个时间步，否则 [:, :-1] 会变空，eval 结果没有意义
        try:
            seq_len = int(batch["z_t"].shape[1])
        except Exception as e:
            raise RuntimeError(f"Stage3a eval: failed to read batch['z_t'].shape[1]: {e}")
        if seq_len < 2:
            raise AssertionError(f"Stage3a eval requires seq_len>=2, got seq_len={seq_len}. Check dataset episode length.")
        # (bs, seq, K)
        z_t = batch["z_t"][:, :-1].to(device)
        z_target = batch["z_target"][:, :-1].to(device)
        z_mask = batch["z_mask"][:, :-1].to(device)  # (bs, seq, 1)
        stage_t = batch["stage_t"][:, :-1].to(device) if "stage_t" in batch.scheme else None
        gr = batch["group_representation"][:, :-1].to(device) if "group_representation" in batch.scheme else None

        bs0, sl0, k0 = z_t.shape
        N = bs0 * sl0
        zt = z_t.reshape(N, k0)
        ztar = z_target.reshape(N, k0)
        zm = z_mask.reshape(N).to(dtype=torch.float32).clamp(min=0.0, max=1.0)
        denom = float(zm.sum().item())
        if denom <= 0:
            continue
        grf = gr.reshape(N, -1) if gr is not None else None
        stf = stage_t.reshape(N, -1) if stage_t is not None else None

        with torch.no_grad():
            zpred = be.predict_next_population_belief(zt, group_repr=grf, stage_t=stf, return_logits=False)
            loss = be.compute_population_belief_loss(
                zpred, ztar, zm, loss_type=str(getattr(cfg2, "z_transition_loss_type", "kl") or "kl")
            )
            kl_zt = _kl_tgt_pred(ztar, zt)
            kl_zp = _kl_tgt_pred(ztar, zpred)

            dz_pred = torch.norm((zpred - zt), p=2, dim=-1)
            dz_tgt = torch.norm((ztar - zt), p=2, dim=-1)

            sum_loss += float(loss.item()) * denom
            sum_kl_tgt_zt += float((kl_zt * zm).sum().item())
            sum_kl_tgt_zp += float((kl_zp * zm).sum().item())
            sum_dz_pred += float((dz_pred * zm).sum().item())
            sum_dz_tgt += float((dz_tgt * zm).sum().item())
            sum_mask += denom

            if int(zpred.shape[-1]) == 3:
                zp = _renorm(zpred)
                ent = -torch.sum(zp * torch.log(zp + 1e-8), dim=-1)
                mx = torch.max(zp, dim=-1)[0]
                sum_ent += float((ent * zm).sum().item())
                sum_maxp += float((mx * zm).sum().item())
                sum_p0 += float((zp[:, 0] * zm).sum().item())
                sum_p1 += float((zp[:, 1] * zm).sum().item())
                sum_p2 += float((zp[:, 2] * zm).sum().item())

            # per-stage buckets
            if stf is not None:
                st1 = stf.reshape(-1).to(dtype=torch.long)
                for s in torch.unique(st1).tolist():
                    try:
                        si = int(s)
                    except Exception:
                        continue
                    sel = (st1 == si)
                    if not bool(sel.any().item()):
                        continue
                    m_s = zm[sel]
                    d_s = float(m_s.sum().item())
                    if d_s <= 0:
                        continue
                    stage_sum_mask[si] = float(stage_sum_mask.get(si, 0.0) + d_s)
                    stage_sum_dz_pred[si] = float(stage_sum_dz_pred.get(si, 0.0) + float((dz_pred[sel] * m_s).sum().item()))
                    stage_sum_dz_tgt[si] = float(stage_sum_dz_tgt.get(si, 0.0) + float((dz_tgt[sel] * m_s).sum().item()))

            # ablation: no-stage
            if stf is not None:
                st0 = torch.zeros_like(stf)
                zpred_nostage = be.predict_next_population_belief(zt, group_repr=grf, stage_t=st0, return_logits=False)
                kl_nostage = _kl_tgt_pred(ztar, zpred_nostage)
                sum_kl_nostage += float((kl_nostage * zm).sum().item())

                # ablation: random-stage (shuffle stage_t across steps; preserves marginal distribution)
                perm = torch.randperm(stf.shape[0], device=stf.device)
                st_rand = stf[perm]
                zpred_randstage = be.predict_next_population_belief(zt, group_repr=grf, stage_t=st_rand, return_logits=False)
                kl_rand = _kl_tgt_pred(ztar, zpred_randstage)
                sum_kl_randstage += float((kl_rand * zm).sum().item())

                # ablation: shift-stage (use stage_t + 1, clamped to n_stages)
                # Note: BeliefEncoder internally clamps to [0..n_stages]; we clamp here for clarity.
                nst = int(getattr(be, "n_stages", 13))
                st_shift = (stf.to(dtype=torch.long) + 1).clamp(min=0, max=nst).to(dtype=stf.dtype)
                zpred_shiftstage = be.predict_next_population_belief(zt, group_repr=grf, stage_t=st_shift, return_logits=False)
                kl_shift = _kl_tgt_pred(ztar, zpred_shiftstage)
                sum_kl_shiftstage += float((kl_shift * zm).sum().item())

                # probes: fixed-stage constants (e.g., stage 11/12)
                if fixed_stage_ids:
                    for sid in fixed_stage_ids:
                        s_clamped = int(max(0, min(int(sid), int(nst))))
                        st_fixed = torch.full(stf.shape, s_clamped, device=stf.device, dtype=stf.dtype)
                        zpred_fixed = be.predict_next_population_belief(
                            zt, group_repr=grf, stage_t=st_fixed, return_logits=False
                        )
                        kl_fixed = _kl_tgt_pred(ztar, zpred_fixed)
                        fixed_stage_sums[int(sid)] = float(
                            fixed_stage_sums.get(int(sid), 0.0) + float((kl_fixed * zm).sum().item())
                        )

            # sensitivity: no-group-repr
            if grf is not None:
                gr0 = torch.zeros_like(grf)
                zpred_nogr = be.predict_next_population_belief(zt, group_repr=gr0, stage_t=stf, return_logits=False)
                kl_nogr = _kl_tgt_pred(ztar, zpred_nogr)
                sum_kl_nogr += float((kl_nogr * zm).sum().item())

    if sum_mask <= 0:
        raise RuntimeError("Stage3a eval: no valid masked steps (z_mask sum is 0). Check dataset z_mask or eval_episodes.")

    out: Dict[str, float] = {}
    out["eval_loss_z_transition"] = float(sum_loss / sum_mask)
    out["eval_kl_target_zt"] = float(sum_kl_tgt_zt / sum_mask)
    out["eval_kl_target_zpred"] = float(sum_kl_tgt_zp / sum_mask)
    out["eval_z_pred_minus_z_t_l2"] = float(sum_dz_pred / sum_mask)
    out["eval_z_target_minus_z_t_l2"] = float(sum_dz_tgt / sum_mask)
    out["eval_z_pred_entropy"] = float(sum_ent / sum_mask) if sum_ent > 0 else 0.0
    out["eval_z_pred_maxprob"] = float(sum_maxp / sum_mask) if sum_maxp > 0 else 0.0
    out["eval_z_pred_p0_mean"] = float(sum_p0 / sum_mask) if sum_p0 > 0 else 0.0
    out["eval_z_pred_p1_mean"] = float(sum_p1 / sum_mask) if sum_p1 > 0 else 0.0
    out["eval_z_pred_p2_mean"] = float(sum_p2 / sum_mask) if sum_p2 > 0 else 0.0
    out["eval_kl_target_zpred_nostage"] = float(sum_kl_nostage / sum_mask) if sum_kl_nostage > 0 else 0.0
    out["eval_kl_target_zpred_nogr"] = float(sum_kl_nogr / sum_mask) if sum_kl_nogr > 0 else 0.0
    out["eval_kl_target_zpred_randstage"] = float(sum_kl_randstage / sum_mask) if sum_kl_randstage > 0 else 0.0
    out["eval_kl_target_zpred_shiftstage"] = float(sum_kl_shiftstage / sum_mask) if sum_kl_shiftstage > 0 else 0.0
    for sid in fixed_stage_ids:
        s = int(sid)
        v = float(fixed_stage_sums.get(s, 0.0))
        if v > 0:
            out[f"eval_kl_target_zpred_fixedstage{s}"] = float(v / sum_mask)
    out["eval_mask_sum"] = float(sum_mask)
    # per-stage buckets
    for s, m in stage_sum_mask.items():
        if m <= 0:
            continue
        out[f"eval_z_pred_delta_l2_stage{s}"] = float(stage_sum_dz_pred.get(s, 0.0) / m)
        out[f"eval_z_target_delta_l2_stage{s}"] = float(stage_sum_dz_tgt.get(s, 0.0) / m)
        out[f"eval_stage_mask_sum_stage{s}"] = float(m)
    return out


def _find_metrics_jsonl(logdir: str) -> Optional[str]:
    # common layout: <logdir>/<experiment_name>/metrics.jsonl OR <logdir>/metrics.jsonl
    if not logdir:
        return None
    cand = []
    cand.append(os.path.join(logdir, "metrics.jsonl"))
    # scan one-level subdirs
    try:
        for name in os.listdir(logdir):
            p = os.path.join(logdir, name, "metrics.jsonl")
            cand.append(p)
    except Exception:
        pass
    for p in cand:
        if os.path.exists(p):
            return p
    return None


def _ckpt_must_have(ckpt_dir: str, filenames: List[str]) -> None:
    missing = []
    for fn in filenames:
        p = os.path.join(ckpt_dir, fn)
        if not os.path.exists(p):
            missing.append(fn)
    if missing:
        raise FileNotFoundError(f"ckpt={ckpt_dir} missing required files: {missing}")


def _stage_counts_from_hf(cfg: Any) -> Dict[str, Dict[int, int]]:
    """
    统计 HF dataset 各 split 的 stage_t 分布（用于检测 train/test 是否按 stage 切分导致 OOD）。
    返回: {split: {stage: count}}
    """
    try:
        from datasets import load_from_disk, concatenate_datasets  # type: ignore
    except Exception:
        return {}
    env_args = getattr(cfg, "env_args", None)
    if env_args is None:
        return {}
    paths = getattr(env_args, "hf_dataset_path", None)
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list):
        return {}
    paths = [p for p in paths if isinstance(p, str) and os.path.isdir(p)]
    if not paths:
        return {}

    out: Dict[str, Dict[int, int]] = {}
    for split in ["train", "validation", "test"]:
        parts = []
        for p in paths:
            try:
                dd = load_from_disk(p)
            except Exception:
                continue
            if hasattr(dd, "keys") and split in dd:
                parts.append(dd[split])
        if not parts:
            continue
        ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        cnt: Dict[int, int] = {}
        for ex in ds:
            st = ex.get("stage_t", None)
            if st is None:
                continue
            try:
                st_i = int(st)
            except Exception:
                continue
            cnt[st_i] = cnt.get(st_i, 0) + 1
        out[split] = cnt
    return out


def _print_and_assert_stage_flags(cfg: Any, mode: str) -> None:
    """
    打印并硬断言关键阶段/冻结开关，避免 config–ckpt 用错（这是最常见的 silent failure）。
    """
    def _gf(name: str, default: Any = None) -> Any:
        return getattr(cfg, name, default)

    flags = {
        "train_belief_supervised": bool(_gf("train_belief_supervised", False)),
        "train_encoder_only": bool(_gf("train_encoder_only", False)),
        "train_action_imitation": bool(_gf("train_action_imitation", False)),
        "train_population_update_head_only": bool(_gf("train_population_update_head_only", False)),
        "freeze_belief_encoder_in_supervised": bool(_gf("freeze_belief_encoder_in_supervised", False)),
        "freeze_belief_encoder_in_rl": bool(_gf("freeze_belief_encoder_in_rl", False)),
        "freeze_belief_network_in_rl": bool(_gf("freeze_belief_network_in_rl", False)),
        "z_transition_loss_weight": float(_gf("z_transition_loss_weight", 0.0) or 0.0),
        "z_transition_loss_type": str(_gf("z_transition_loss_type", "kl") or "kl"),
    }
    print("\n=== Config stage/freeze flags (sanity) ===")
    for k in sorted(flags.keys()):
        print(f"- {k}: {flags[k]}")

    if mode == "stage1":
        # Stage1 应该是 belief supervised（非 encoder-only / 非 imitation）
        if (not flags["train_belief_supervised"]) or flags["train_encoder_only"] or flags["train_action_imitation"]:
            raise AssertionError(
                "Stage1 config sanity failed: expected train_belief_supervised=True AND "
                "train_encoder_only=False AND train_action_imitation=False"
            )
        # 强制 test_mode argmax（非 sampling）：依赖 MultinomialActionSelector(test_mode && test_greedy -> epsilon=0)
        # 这里不强制 action_selector 名称，但会在运行前把 test_greedy/epsilon 置为确定性。
        return

    if mode == "stage2":
        # Stage2 也是 belief supervised（noncore），同样不应该是 encoder-only / imitation
        if (not flags["train_belief_supervised"]) or flags["train_encoder_only"] or flags["train_action_imitation"]:
            raise AssertionError(
                "Stage2 config sanity failed: expected train_belief_supervised=True AND "
                "train_encoder_only=False AND train_action_imitation=False"
            )
        return

    if mode == "stage3b":
        # Stage3b: belief_supervised mode + train_action_imitation
        if (not flags["train_belief_supervised"]) or flags["train_encoder_only"] or (not flags["train_action_imitation"]):
            raise AssertionError(
                "Stage3b config sanity failed: expected train_belief_supervised=True AND "
                "train_encoder_only=False AND train_action_imitation=True"
            )
        return

    if mode == "stage4":
        # Stage4: neither supervised nor encoder-only by default (online RL)
        if flags["train_belief_supervised"] or flags["train_encoder_only"]:
            raise AssertionError(
                "Stage4 config sanity failed: expected train_belief_supervised=False AND train_encoder_only=False"
            )
        return

    # stage3a
    if mode == "stage3a":
        # Stage3a 必须是 encoder-only 且有 z_transition loss
        if (not flags["train_encoder_only"]) or flags["train_belief_supervised"] or flags["train_action_imitation"]:
            raise AssertionError(
                "Stage3a config sanity failed: expected train_encoder_only=True AND "
                "train_belief_supervised=False AND train_action_imitation=False"
            )
        if flags["z_transition_loss_weight"] <= 0:
            raise AssertionError("Stage3a config sanity failed: z_transition_loss_weight must be > 0")
        # 强烈建议：只训练 update head；这里不强制，但打印出来帮助发现不一致
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["stage1", "stage2", "stage3a", "stage3b", "stage4"])
    ap.add_argument("--config", type=str, required=True, help="YAML config path (stage1 or stage3a).")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint directory (expects agent.th / belief_encoder.th).")
    ap.add_argument("--logdir", type=str, default="", help="Log directory to read metrics.jsonl for trend checks.")
    ap.add_argument("--eval_split", type=str, default="test", choices=["train", "validation", "test"])
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--min_margin", type=float, default=0.05, help="Stage1: required margin over majority baseline.")
    ap.add_argument(
        "--stage3a_fixed_stages",
        type=str,
        default="11,12",
        help="Stage3a: comma-separated fixed stage ids for diagnostic probes (e.g., '11,12').",
    )
    args = ap.parse_args()

    # load config using src/train.py loader (supports SimpleNamespace)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from train import load_config  # type: ignore

    cfg = load_config(str(args.config))
    # Stage3a diagnostic probe stages (passed via cfg for simplicity)
    try:
        fixed = [int(x) for x in str(args.stage3a_fixed_stages).split(",") if str(x).strip() != ""]
    except Exception:
        fixed = [11, 12]
    cfg.stage3a_fixed_stages = fixed
    _print_and_assert_stage_flags(cfg, str(args.mode))

    # --- trend checks from logs (best-effort) ---
    if args.logdir:
        mp = _find_metrics_jsonl(str(args.logdir))
        if mp:
            rows = _read_metrics_jsonl(mp)
            if args.mode == "stage1":
                tr = _extract_series(rows, "train/belief_sup_acc")
                te = _extract_series(rows, "test/core_stance_acc")
                if tr:
                    print(f"[Stage1][log] train/belief_sup_acc: first={tr[0][1]:.4f} last={tr[-1][1]:.4f} ma200={_moving_avg([v for _,v in tr],200):.4f}")
                if te:
                    print(f"[Stage1][log] test/core_stance_acc: first={te[0][1]:.4f} last={te[-1][1]:.4f} ma50={_moving_avg([v for _,v in te],50):.4f}")
            else:
                tr = _extract_series(rows, "train/loss_z_transition")
                if tr:
                    print(f"[Stage3a][log] train/loss_z_transition: first={tr[0][1]:.6f} last={tr[-1][1]:.6f} ma200={_moving_avg([v for _,v in tr],200):.6f}")
        else:
            print(f"[WARN] 未找到 metrics.jsonl 于 logdir={args.logdir}（趋势检查跳过）")

    if args.mode == "stage1":
        # --- 强制 test_mode 使用 argmax（非 sampling） ---
        # 1) 固定 action selector 相关随机性（MultinomialActionSelector 在 test_mode && test_greedy 下会 epsilon=0 -> 纯 argmax）
        cfg.test_greedy = True
        cfg.epsilon_start = 0.0
        cfg.epsilon_finish = 0.0
        cfg.epsilon_anneal_time = 1
        # 2) 明确 action_selector（避免 config 里被改成其它采样器）
        cfg.action_selector = str(getattr(cfg, "action_selector", "multinomial") or "multinomial")
        if str(cfg.action_selector).strip().lower() != "multinomial":
            raise AssertionError(f"Stage1: for deterministic argmax, require action_selector=multinomial, got {cfg.action_selector}")

        _ckpt_must_have(str(args.ckpt), ["agent.th"])
        res = _stage1_eval_confusion(cfg, str(args.ckpt), str(args.eval_split), int(args.eval_episodes))
        print("")
        print("=== Stage1(core belief) eval ===")
        print(f"- eval_split: {args.eval_split}")
        print(f"- action_selector: {cfg.action_selector} (forced deterministic via test_greedy/epsilon=0)")
        print(f"- eval_acc: {res.eval_acc:.4f}")
        print(f"- majority_baseline: {res.majority_baseline:.4f}")
        print(f"- margin: {res.eval_acc - res.majority_baseline:+.4f}")
        print(f"- n_samples: {res.n}")
        print("")
        print("Confusion matrix (rows=GT, cols=Pred):")
        for row in res.confusion[:3]:
            print("  " + " ".join([f"{int(x):6d}" for x in row[:3]]))

        # pass/fail gate
        ok = True
        if res.eval_acc < (res.majority_baseline + float(args.min_margin)):
            ok = False
            print(f"\n[FAIL] Stage1 eval_acc<{res.majority_baseline + float(args.min_margin):.4f} (majority+margin). 建议不要继续后续阶段。")
        else:
            print("\n[PASS] Stage1 达到 majority baseline + margin。")
        return 0 if ok else 2

    if args.mode == "stage2":
        cfg.test_greedy = True
        cfg.epsilon_start = 0.0
        cfg.epsilon_finish = 0.0
        cfg.epsilon_anneal_time = 1
        cfg.action_selector = str(getattr(cfg, "action_selector", "multinomial") or "multinomial")
        if str(cfg.action_selector).strip().lower() != "multinomial":
            raise AssertionError(f"Stage2: for deterministic argmax, require action_selector=multinomial, got {cfg.action_selector}")

        _ckpt_must_have(str(args.ckpt), ["agent.th"])
        res = _stage2_eval_confusion(cfg, str(args.ckpt), str(args.eval_split), int(args.eval_episodes))
        print("")
        print("=== Stage2(noncore belief) eval ===")
        print(f"- eval_split: {args.eval_split}")
        print(f"- action_selector: {cfg.action_selector} (forced deterministic via test_greedy/epsilon=0)")
        print(f"- eval_acc: {res.eval_acc:.4f}")
        print(f"- majority_baseline: {res.majority_baseline:.4f}")
        print(f"- margin: {res.eval_acc - res.majority_baseline:+.4f}")
        print(f"- n_samples: {res.n}")
        print("")
        print("Confusion matrix (rows=GT, cols=Pred):")
        for row in res.confusion[:3]:
            print("  " + " ".join([f"{int(x):6d}" for x in row[:3]]))

        ok = True
        if res.eval_acc < (res.majority_baseline + float(args.min_margin)):
            ok = False
            print(f"\n[FAIL] Stage2 eval_acc<{res.majority_baseline + float(args.min_margin):.4f} (majority+margin).")
        else:
            print("\n[PASS] Stage2 达到 majority baseline + margin。")
        return 0 if ok else 2

    if args.mode == "stage3b":
        _ckpt_must_have(str(args.ckpt), ["agent.th"])
        out = _stage3b_eval_action_imitation(cfg, str(args.ckpt), str(args.eval_split), int(args.eval_episodes))
        print("")
        print("=== Stage3b(action imitation) eval ===")
        print(f"- eval_split: {args.eval_split}")
        if out.get("sup_ids") is not None:
            print(f"- masked_supervised_ids: {out.get('sup_ids')}")
        print(f"- eval_acc_masked: {out['eval_acc_masked']:.4f}")
        print(f"- majority_baseline_masked: {out['majority_baseline_masked']:.4f}")
        print(f"- margin: {out['margin']:+.4f}")
        print(f"- n_masked: {out['n_masked']}")
        print(f"- n_skipped_unsup: {out['n_skipped_unsup']}")
        print(f"- coverage: {out['coverage']:.4f}")
        print(f"- skipped_ratio: {out['skipped_ratio']:.4f}")
        print("")
        print("Confusion matrix (rows=GT, cols=Pred) over MASKED rows:")
        conf = out["confusion"]
        k = int(out["k"])
        for i in range(min(k, 5)):
            row = conf[i]
            print("  " + " ".join([f"{int(x):6d}" for x in row[: min(k, 5)]]))

        ok = True
        # Minimal gate: masked acc should beat masked majority + margin, and coverage should not be 0.
        if float(out["n_masked"]) <= 0:
            ok = False
            print("\n[FAIL] Stage3b: masked eval has zero labeled samples. Check supervised ids / dataset.")
        elif float(out["eval_acc_masked"]) < (float(out["majority_baseline_masked"]) + float(args.min_margin)):
            ok = False
            print("\n[FAIL] Stage3b: eval_acc_masked < majority_baseline_masked + margin.")
        else:
            print("\n[PASS] Stage3b: masked accuracy 达到 majority baseline + margin。")
        return 0 if ok else 4

    if args.mode == "stage4":
        # Stage4 online sanity: must at least have agent; belief_encoder is strongly recommended when using z dynamics.
        # We don't hard-require belief_encoder.th because some ablations might not save it, but we warn if missing.
        _ckpt_must_have(str(args.ckpt), ["agent.th"])
        if not os.path.exists(os.path.join(str(args.ckpt), "belief_encoder.th")):
            print("[WARN] Stage4 ckpt missing belief_encoder.th (z-dynamics / secondary sim may not work as intended).")

        out = _stage4_eval_online(cfg, str(args.ckpt), int(args.eval_episodes))
        print("")
        print("=== Stage4(online RL) eval ===")
        print(f"- test_episodes: {out.get('test_episodes', 0)}")
        print(f"- test_return_mean: {out.get('test_return_mean', 0.0):.6f}")
        print(f"- test_return_std: {out.get('test_return_std', 0.0):.6f}")
        print(f"- z_eval_steps: {out.get('z_eval_steps', 0)}")
        zkl = out.get("z_kl", float("nan"))
        try:
            if zkl == zkl:
                print(f"- z_kl: {float(zkl):.6f}")
            else:
                print("- z_kl: nan")
        except Exception:
            print("- z_kl: nan")

        if bool(out.get("ok", False)):
            print("\n[PASS] Stage4: can run stable test episodes; metrics are finite.")
            return 0
        print("\n[FAIL] Stage4: unstable run or non-finite metrics. Check env config / ckpt / reward settings.")
        return 5

    # stage3a
    # --- dataset stage distribution sanity (detect OOD-by-stage split) ---
    stage_cnts = _stage_counts_from_hf(cfg)
    if stage_cnts:
        print("\n=== Stage3a dataset stage_t distribution (from HF) ===")
        for sp in ["train", "validation", "test"]:
            if sp in stage_cnts:
                items = sorted(stage_cnts[sp].items())
                print(f"- {sp}: {items[:30]}")
        tr_stages = set(stage_cnts.get("train", {}).keys())
        ev_stages = set(stage_cnts.get(str(args.eval_split), {}).keys())
        if tr_stages and ev_stages and (not ev_stages.issubset(tr_stages)):
            missing = sorted(list(ev_stages - tr_stages))
            print(
                f"[WARN] eval_split={args.eval_split} contains stages not present in train: {missing}. "
                "This is an OOD-by-stage split; stage_embed for these stages is likely untrained -> "
                "with-stage KL can look worse than nostage/fixedstage0 even if training loss decreases."
            )

    _ckpt_must_have(str(args.ckpt), ["belief_encoder.th"])
    out = _stage3a_eval_z_transition(cfg, str(args.ckpt), str(args.eval_split), int(args.eval_episodes))
    print("")
    print("=== Stage3a(z_transition) held-out eval (no_grad) ===")
    print(f"- eval_split: {args.eval_split}")
    for k in (
        "eval_loss_z_transition",
        "eval_kl_target_zt",
        "eval_kl_target_zpred",
        "eval_z_pred_entropy",
        "eval_z_pred_maxprob",
        "eval_z_pred_minus_z_t_l2",
        "eval_z_target_minus_z_t_l2",
        "eval_kl_target_zpred_nostage",
        "eval_kl_target_zpred_randstage",
        "eval_kl_target_zpred_shiftstage",
        "eval_kl_target_zpred_nogr",
        "eval_mask_sum",
    ):
        if k in out:
            print(f"- {k}: {out[k]:.6f}")
    # print fixed-stage probes (if present)
    fixed_keys = sorted([k for k in out.keys() if k.startswith("eval_kl_target_zpred_fixedstage")])
    for k in fixed_keys:
        print(f"- {k}: {out[k]:.6f}")
    # a few stage buckets (print only those present)
    stages = sorted({int(k.split("stage")[-1]) for k in out.keys() if k.startswith("eval_z_pred_delta_l2_stage")})
    if stages:
        print("\nPer-stage z_delta (masked mean L2):")
        for s in stages:
            kp = f"eval_z_pred_delta_l2_stage{s}"
            kt = f"eval_z_target_delta_l2_stage{s}"
            print(f"  stage{s}: pred={out.get(kp, 0.0):.6f} | target={out.get(kt, 0.0):.6f}")

    # pass/fail heuristic (经验): model KL 应 <= identity baseline KL，且不爆炸；entropy 不应逼近 ln(3) 且 maxprob 不应 ~1/3
    ok = True
    if out.get("eval_kl_target_zpred", 1e9) > (out.get("eval_kl_target_zt", 0.0) + 1e-6):
        # model worse than identity baseline
        ok = False
        print("\n[FAIL] Stage3a: KL(z_target||z_pred) > KL(z_target||z_t)（模型比 identity baseline 更差）。Stage4 很可能不稳。")
    else:
        print("\n[PASS] Stage3a: 模型 KL 优于/不差于 identity baseline（这是必要条件）。")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())


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
            gt = _boxed_int(info.get("ground_truth_answer", ""))
            if gt is None:
                gt = _boxed_int(info.get("ground_truth", ""))
            pr = _boxed_int(info.get("llm_answer", ""))
            if pr is None:
                pr = _boxed_int(info.get("answer", ""))
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
    import numpy as np

    cfg2 = copy.deepcopy(cfg)
    try:
        if hasattr(cfg2, "env_args") and hasattr(cfg2.env_args, "dataset_split"):
            cfg2.env_args.dataset_split = str(eval_split)
    except Exception:
        pass
    # Make evaluation deterministic/stable: disable random sampling over the HF dataset split.
    # This avoids noisy preference_BCE estimates due to different sampled subsets across runs.
    try:
        if hasattr(cfg2, "env_args") and hasattr(cfg2.env_args, "use_random_sampling"):
            cfg2.env_args.use_random_sampling = False
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
    # Stage3b option: binary(0/1) imitation prior.
    try:
        if bool(getattr(cfg2, "train_action_imitation", False)) and bool(getattr(cfg2, "action_imitation_binary_01", False)):
            k = 2
    except Exception:
        pass
    # Stage3b option: preference scorer (retweet vs post)
    try:
        s3b_preference_scorer = bool(getattr(cfg2, "s3b_preference_scorer", False))
    except Exception:
        s3b_preference_scorer = False
    k = max(1, k)

    # confusion for masked rows only (size k x k)
    conf = [[0 for _ in range(k)] for _ in range(k)]
    total = 0
    correct = 0
    skipped_unsup = 0
    # full marginal stats over ALL labels (including unsupervised ones)
    gt_counts_all = [0 for _ in range(k)]
    pred_counts_all = [0 for _ in range(k)]
    invalid_pred = 0
    # preference scorer metrics
    pref_p1_targets = []
    pref_p1_preds = []
    # debug counters for preference scorer
    dbg_pref = bool(getattr(cfg2, "debug_preference", False))
    dbg_pref_total_infos = 0
    dbg_pref_has_dist = 0
    dbg_pref_denom_pos = 0
    dbg_pref_has_bias = 0
    dbg_pref_added = 0
    dbg_pref_examples = []

    n_ep = max(1, int(eval_episodes))
    for _ in range(n_ep):
        _batch = runner.run(test_mode=True)
        infos = getattr(runner, "last_env_infos", None)
        if not isinstance(infos, list):
            continue
        for info in infos:
            if not isinstance(info, dict):
                continue
            if dbg_pref and s3b_preference_scorer:
                dbg_pref_total_infos += 1
            gt = _boxed_int(info.get("ground_truth_answer", ""))
            if gt is None:
                gt = _boxed_int(info.get("ground_truth", ""))
            pr = _boxed_int(info.get("llm_answer", ""))
            if pr is None:
                pr = _boxed_int(info.get("answer", ""))
            if gt is None:
                continue
            # preference scorer: use target_distribution_prob if present
            try:
                if s3b_preference_scorer and isinstance(info, dict):
                    dist = info.get("target_distribution_prob")
                    if isinstance(dist, dict) and ("0" in dist or "1" in dist):
                        if dbg_pref:
                            dbg_pref_has_dist += 1
                        p0 = float(dist.get("0", 0.0) or 0.0)
                        p1 = float(dist.get("1", 0.0) or 0.0)
                        denom = p0 + p1
                        if denom > 0:
                            if dbg_pref:
                                dbg_pref_denom_pos += 1
                            p1_t = p1 / denom
                            # predicted p1 from bias logit (if provided)
                            bl = info.get("pref_bias_logit", None)
                            if bl is not None:
                                if dbg_pref:
                                    dbg_pref_has_bias += 1
                                p1_pred = 1.0 / (1.0 + np.exp(-float(bl)))
                                pref_p1_targets.append(p1_t)
                                pref_p1_preds.append(float(p1_pred))
                                if dbg_pref:
                                    dbg_pref_added += 1
                            else:
                                if dbg_pref and len(dbg_pref_examples) < 5:
                                    dbg_pref_examples.append(
                                        {
                                            "why": "missing_pref_bias_logit",
                                            "gt": int(gt) if gt is not None else None,
                                            "pr": int(pr) if pr is not None else None,
                                            "target_distribution_prob": dist,
                                            "has_pref_p0": ("pref_p0" in info),
                                            "has_pref_p1": ("pref_p1" in info),
                                            "keys": sorted(list(info.keys()))[:40],
                                        }
                                    )
                        else:
                            if dbg_pref and len(dbg_pref_examples) < 5:
                                dbg_pref_examples.append(
                                    {
                                        "why": "p0_p1_denom_zero",
                                        "gt": int(gt) if gt is not None else None,
                                        "pr": int(pr) if pr is not None else None,
                                        "target_distribution_prob": dist,
                                        "keys": sorted(list(info.keys()))[:40],
                                    }
                                )
                    else:
                        if dbg_pref and len(dbg_pref_examples) < 5:
                            dbg_pref_examples.append(
                                {
                                    "why": "missing_or_unexpected_target_distribution_prob",
                                    "gt": int(gt) if gt is not None else None,
                                    "pr": int(pr) if pr is not None else None,
                                    "target_distribution_prob_type": str(type(dist)),
                                    "target_distribution_prob": dist,
                                    "keys": sorted(list(info.keys()))[:40],
                                }
                            )
            except Exception:
                pass
            # full marginal counts (for collapse detection)
            try:
                if 0 <= int(gt) < k:
                    gt_counts_all[int(gt)] += 1
            except Exception:
                pass
            try:
                if pr is not None and 0 <= int(pr) < k:
                    pred_counts_all[int(pr)] += 1
                elif pr is not None:
                    invalid_pred += 1
            except Exception:
                pass
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

    # === Sanity metrics for collapse / marginals ===
    # We compute marginal distributions over the whole eval split (including unsupervised labels).
    try:
        import numpy as np

        gt = np.array(gt_counts_all, dtype=np.float64)
        pr = np.array(pred_counts_all, dtype=np.float64)
        gt_frac = (gt / max(1.0, float(gt.sum()))).tolist()
        pr_frac = (pr / max(1.0, float(pr.sum()))).tolist()
        ent_pred = _entropy_np(pr_frac)
        ent_gt = _entropy_np(gt_frac)
        kl_pred_gt = _kl_np(pr_frac, gt_frac)
        mode_frac = float(np.max(pr / max(1.0, float(pr.sum())))) if float(pr.sum()) > 0 else 0.0

        # In binary-01 mode (k=2), "unsup" fractions are not meaningful.
        unsup_pred_frac = float("nan")
        unsup_gt_frac = float("nan")
        if k > 2:
            unsup_ids = None
            if isinstance(sup_ids, set) and len(sup_ids) > 0:
                unsup_ids = [i for i in range(k) if i not in sup_ids]
            unsup_pred_frac = float(sum(pr_frac[i] for i in (unsup_ids or []))) if unsup_ids else float("nan")
            unsup_gt_frac = float(sum(gt_frac[i] for i in (unsup_ids or []))) if unsup_ids else float("nan")
    except Exception:
        gt_frac = None
        pr_frac = None
        ent_pred = float("nan")
        ent_gt = float("nan")
        kl_pred_gt = float("nan")
        mode_frac = float("nan")
        unsup_pred_frac = float("nan")
        unsup_gt_frac = float("nan")

    # preference scorer diagnostics (retweet vs post)
    pref_bce = float("nan")
    pref_bce_baseline = float("nan")
    pref_corr = float("nan")
    pref_margin = float("nan")
    try:
        if len(pref_p1_targets) > 0:
            import numpy as np
            t = np.array(pref_p1_targets, dtype=np.float64)
            p = np.array(pref_p1_preds, dtype=np.float64)
            eps = 1e-8
            p = np.clip(p, eps, 1.0 - eps)
            pref_bce = float(np.mean(-(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))))
            # baseline: constant mean(p1)
            m = float(np.mean(t))
            m = float(min(1.0 - eps, max(eps, m)))
            pref_bce_baseline = float(np.mean(-(t * np.log(m) + (1.0 - t) * np.log(1.0 - m))))
            pref_margin = float(pref_bce_baseline - pref_bce)
            if t.size >= 2 and np.std(t) > 0 and np.std(p) > 0:
                pref_corr = float(np.corrcoef(t, p)[0, 1])
    except Exception:
        pass

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
        # collapse/marginal sanity
        "pred_counts_all": pred_counts_all,
        "gt_counts_all": gt_counts_all,
        "invalid_pred": int(invalid_pred),
        "pred_frac_all": pr_frac,
        "gt_frac_all": gt_frac,
        "pred_entropy": float(ent_pred),
        "gt_entropy": float(ent_gt),
        "pred_kl_gt": float(kl_pred_gt),
        "pred_mode_frac": float(mode_frac),
        "unsup_pred_frac": float(unsup_pred_frac),
        "unsup_gt_frac": float(unsup_gt_frac),
        "s3b_preference_scorer": bool(s3b_preference_scorer),
        "pref_n": int(len(pref_p1_targets)),
        "pref_bce": float(pref_bce),
        "pref_bce_baseline": float(pref_bce_baseline),
        "pref_bce_margin": float(pref_margin),
        "pref_corr": float(pref_corr),
        # debug: why pref_n is 0
        "debug_preference": bool(dbg_pref),
        "debug_pref_total_infos": int(dbg_pref_total_infos),
        "debug_pref_has_dist": int(dbg_pref_has_dist),
        "debug_pref_denom_pos": int(dbg_pref_denom_pos),
        "debug_pref_has_bias": int(dbg_pref_has_bias),
        "debug_pref_added": int(dbg_pref_added),
        "debug_pref_examples": dbg_pref_examples,
    }


def _safe_softmax_np(x, axis=-1, eps: float = 1e-8):
    import numpy as np
    xx = np.array(x, dtype=np.float64)
    xx = xx - np.max(xx, axis=axis, keepdims=True)
    ex = np.exp(xx)
    s = np.sum(ex, axis=axis, keepdims=True)
    return ex / (s + eps)


def _entropy_np(p, axis=-1, eps: float = 1e-8) -> float:
    import numpy as np
    pp = np.clip(np.array(p, dtype=np.float64), eps, 1.0)
    ent = -np.sum(pp * np.log(pp), axis=axis)
    return float(np.mean(ent))


def _kl_np(p, q, eps: float = 1e-8) -> float:
    import numpy as np
    pp = np.clip(np.array(p, dtype=np.float64), eps, 1.0)
    qq = np.clip(np.array(q, dtype=np.float64), eps, 1.0)
    pp = pp / np.sum(pp)
    qq = qq / np.sum(qq)
    return float(np.sum(pp * (np.log(pp) - np.log(qq))))


def _policy_stats_from_runner(runner: Any, batch: Any, *, k_override: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Compute policy distribution stats from the current runner/batch at t=0:
    - mean action distribution over agents (softmax of action_type_q_values if available)
    - entropy of that distribution
    """
    try:
        import numpy as np
        mac = getattr(runner, "mac", None)
        if mac is None:
            return None
        outs, info = mac.forward(batch, t=0, test_mode=True)  # outs: (bs,n_agents,n_avail)
        if outs is None:
            return None
        if hasattr(outs, "detach"):
            outs_np = outs.detach().float().cpu().numpy()
        else:
            outs_np = np.array(outs, dtype=np.float64)
        # expect bs==1
        if outs_np.ndim == 3 and outs_np.shape[0] >= 1:
            logits = outs_np[0]  # (n_agents, n_actions)
        elif outs_np.ndim == 2:
            logits = outs_np
        else:
            return None
        try:
            if isinstance(k_override, int) and k_override > 0 and logits.shape[-1] >= int(k_override):
                logits = logits[:, : int(k_override)]
        except Exception:
            pass
        p_agents = _safe_softmax_np(logits, axis=-1)
        p_mean = np.mean(p_agents, axis=0)  # (n_actions,)
        return {
            "p_mean": p_mean.tolist(),
            "entropy_mean": _entropy_np(p_mean),
        }
    except Exception:
        return None


def _stage3b_compare_z_ablations(cfg: Any, ckpt: str, eval_split: str, eval_episodes: int, *, shuffle_seed: int = 0) -> Dict[str, Any]:
    """
    Sanity check for "z_t is really used by policy":
    Run the SAME ckpt under three modes:
      - z_ablation_mode=none (with z_t)
      - z_ablation_mode=zero (z_t=0)
      - z_ablation_mode=shuffle (z_t mismatched)
    Report:
      - mean action distribution p(a)
      - entropy H(p)
      - KL(p_with_z || p_zero), KL(p_with_z || p_shuffle)
    """
    import numpy as np
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from train import setup_experiment  # type: ignore

    modes = ["none", "zero", "shuffle"]
    out_by_mode: Dict[str, Any] = {}
    # binary(0/1) prior: compare action distribution only over first 2 classes
    k_override = None
    try:
        if bool(getattr(cfg, "train_action_imitation", False)) and bool(getattr(cfg, "action_imitation_binary_01", False)):
            k_override = 2
    except Exception:
        k_override = None
    for m in modes:
        cfg2 = copy.deepcopy(cfg)
        try:
            if hasattr(cfg2, "env_args") and hasattr(cfg2.env_args, "dataset_split"):
                cfg2.env_args.dataset_split = str(eval_split)
        except Exception:
            pass
        cfg2.load_model_path = str(ckpt)
        cfg2.z_ablation_mode = str(m)
        cfg2.z_shuffle_seed = int(shuffle_seed)
        runner, _mac, _learner, _logger, _device = setup_experiment(cfg2)

        ps = []
        ents = []
        n_ok = 0
        for _ in range(max(1, int(eval_episodes))):
            batch = runner.run(test_mode=True)
            st = _policy_stats_from_runner(runner, batch, k_override=k_override)
            if not isinstance(st, dict):
                continue
            p = st.get("p_mean")
            if not isinstance(p, list) or len(p) <= 0:
                continue
            ps.append(np.array(p, dtype=np.float64))
            ents.append(float(st.get("entropy_mean", 0.0)))
            n_ok += 1
        if n_ok <= 0:
            out_by_mode[m] = {"n": 0}
        else:
            p_mean = np.mean(np.stack(ps, axis=0), axis=0)
            p_mean = p_mean / max(1e-8, float(np.sum(p_mean)))
            out_by_mode[m] = {
                "n": int(n_ok),
                "p_mean": p_mean.tolist(),
                "entropy_mean": float(np.mean(ents)),
            }

    # Compare KLs (with_z vs ablations)
    p0 = out_by_mode.get("none", {}).get("p_mean")
    if isinstance(p0, list) and len(p0) > 0:
        pz = np.array(p0, dtype=np.float64)
        if isinstance(out_by_mode.get("zero", {}).get("p_mean"), list):
            p_zero = np.array(out_by_mode["zero"]["p_mean"], dtype=np.float64)
            out_by_mode["kl_withz_vs_zero"] = float(_kl_np(pz, p_zero))
        if isinstance(out_by_mode.get("shuffle", {}).get("p_mean"), list):
            p_shuf = np.array(out_by_mode["shuffle"]["p_mean"], dtype=np.float64)
            out_by_mode["kl_withz_vs_shuffle"] = float(_kl_np(pz, p_shuf))
    return out_by_mode


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
    # Eval policy mode: greedy (deterministic) vs stochastic (train-like exploration).
    try:
        pol = str(getattr(cfg2, "stage4_eval_policy", "greedy") or "greedy").strip().lower()
    except Exception:
        pol = "greedy"
    if pol in ("stochastic", "sample", "sampling", "trainlike", "train_like"):
        # Keep config exploration knobs; ensure test_greedy is off so selector can sample.
        try:
            cfg2.test_greedy = False
        except Exception:
            pass
    else:
        # Deterministic evaluation: disable exploration in action selector (if any)
        try:
            cfg2.test_greedy = True
            cfg2.epsilon_start = 0.0
            cfg2.epsilon_finish = 0.0
            cfg2.epsilon_anneal_time = 1
        except Exception:
            pass
    runner, _mac, _learner, _logger, _device = setup_experiment(cfg2)
    return _stage4_eval_online_runner(runner, cfg2, int(eval_episodes))


def _stage4_eval_online_runner(runner: Any, cfg2: Any, eval_episodes: int) -> Dict[str, Any]:
    """
    Stage4 eval core loop given an already-constructed runner.
    This is required so we can fairly compare different *policies* (ckpt vs random)
    under the exact same env/config without losing patches by re-calling setup_experiment().
    """
    import math
    import numpy as np

    returns: List[float] = []
    z_kl_list: List[float] = []
    z_steps = 0

    # Extra paper-facing diagnostics for Stage4
    # - action histogram / entropy (micro-level sanity)
    # - z trajectory early/late means (macro trend direction)
    action_counts = None  # np.ndarray (K,)
    action_steps = 0
    z_pred_traj: List[np.ndarray] = []
    z_gt_traj: List[np.ndarray] = []

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
        # Greedy eval uses test_mode=True; stochastic eval uses test_mode=False to keep exploration (temperature/epsilon).
        try:
            pol = str(getattr(cfg2, "stage4_eval_policy", "greedy") or "greedy").strip().lower()
        except Exception:
            pol = "greedy"
        run_test_mode = False if pol in ("stochastic", "sample", "sampling", "trainlike", "train_like") else True
        batch = runner.run(test_mode=bool(run_test_mode))
        if batch is None:
            continue
        try:
            ep_ret = float(batch["reward"].sum().item())
        except Exception:
            ep_ret = 0.0
        returns.append(ep_ret)

        # action histogram from EpisodeBatch (best-effort)
        try:
            import torch

            if action_counts is None:
                k = int(getattr(getattr(cfg2, "env_args", None), "n_actions", getattr(cfg2, "n_actions", 5)))
                k = max(1, k)
                action_counts = np.zeros((k,), dtype=np.int64)
            acts = batch["actions"]
            if isinstance(acts, torch.Tensor):
                a = acts.detach().cpu()
                # shapes: (bs,T,n_agents,1) or (bs,T,n_agents)
                if a.ndim == 4 and a.shape[-1] == 1:
                    a = a[..., 0]
                if a.ndim == 3:
                    flat = a.reshape(-1).numpy().astype(np.int64, copy=False)
                    flat = np.clip(flat, 0, int(action_counts.shape[0]) - 1)
                    action_counts += np.bincount(flat, minlength=int(action_counts.shape[0])).astype(np.int64)
                    action_steps += int(flat.size)
        except Exception:
            pass

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
                    zt_arr = np.array(zt, dtype=np.float32)
                    zp_arr = np.array(zp, dtype=np.float32)
                    z_kl_list.append(_kl(zt_arr, zp_arr))
                    # trajectory buffers (for early/late mean print)
                    z_pred_traj.append(_renorm(zp_arr))
                    z_gt_traj.append(_renorm(zt_arr))
                except Exception:
                    continue

    ret_mean = float(np.mean(returns)) if returns else 0.0
    ret_std = float(np.std(returns)) if returns else 0.0
    z_kl = float(np.mean(z_kl_list)) if z_kl_list else float("nan")

    # action distribution diagnostics
    act_entropy = float("nan")
    act_mode_frac = float("nan")
    act_hist = None
    try:
        if isinstance(action_counts, np.ndarray) and int(action_counts.sum()) > 0:
            p = action_counts.astype(np.float64) / float(max(1, int(action_counts.sum())))
            eps = 1e-8
            act_entropy = float(-np.sum(np.clip(p, eps, 1.0) * np.log(np.clip(p, eps, 1.0))))
            act_mode_frac = float(np.max(p))
            act_hist = action_counts.tolist()
    except Exception:
        act_entropy = float("nan")
        act_mode_frac = float("nan")
        act_hist = None

    # z trajectory summary (early vs late)
    z_pred_early = None
    z_pred_late = None
    z_gt_early = None
    z_gt_late = None
    try:
        ktraj = int(getattr(cfg2, "stage4_z_traj_k", 3) or 3)
    except Exception:
        ktraj = 3
    ktraj = max(1, int(ktraj))
    try:
        if len(z_pred_traj) > 0:
            early = z_pred_traj[:ktraj]
            late = z_pred_traj[-ktraj:] if len(z_pred_traj) >= ktraj else z_pred_traj
            z_pred_early = np.mean(np.stack(early, axis=0), axis=0).tolist()
            z_pred_late = np.mean(np.stack(late, axis=0), axis=0).tolist()
        if len(z_gt_traj) > 0:
            early = z_gt_traj[:ktraj]
            late = z_gt_traj[-ktraj:] if len(z_gt_traj) >= ktraj else z_gt_traj
            z_gt_early = np.mean(np.stack(early, axis=0), axis=0).tolist()
            z_gt_late = np.mean(np.stack(late, axis=0), axis=0).tolist()
    except Exception:
        z_pred_early = None
        z_pred_late = None
        z_gt_early = None
        z_gt_late = None

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
        # paper-facing diagnostics
        "action_entropy": float(act_entropy),
        "action_mode_frac": float(act_mode_frac),
        "action_hist": act_hist,
        "action_total_samples": int(action_steps),
        "z_pred_mean_early": z_pred_early,
        "z_pred_mean_late": z_pred_late,
        "z_gt_mean_early": z_gt_early,
        "z_gt_mean_late": z_gt_late,
    }


def _stage4_eval_online_with_policy(cfg: Any, ckpt: str, eval_episodes: int, *, policy_mode: str = "ckpt") -> Dict[str, Any]:
    """
    Stage4 eval wrapper:
    - policy_mode="ckpt": use policy from checkpoint (default)
    - policy_mode="random": override MAC action selection with random uniform actions (same env/config)
    This enables "frozen vs trained" comparisons under identical eval config.
    """
    policy_mode = str(policy_mode or "ckpt").strip().lower()
    if policy_mode not in ("ckpt", "checkpoint", "trained", "init", "random"):
        policy_mode = "ckpt"
    policy_mode_norm = "random" if policy_mode == "random" else "ckpt"

    # Setup runner once (loads ckpt into MAC/BeliefEncoder)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from train import setup_experiment  # type: ignore
    import torch

    cfg2 = copy.deepcopy(cfg)
    cfg2.load_model_path = str(ckpt)
    try:
        cfg2.test_greedy = True
        cfg2.epsilon_start = 0.0
        cfg2.epsilon_finish = 0.0
        cfg2.epsilon_anneal_time = 1
    except Exception:
        pass
    runner, _mac, _learner, _logger, _device = setup_experiment(cfg2)

    if policy_mode_norm == "ckpt":
        return _stage4_eval_online_runner(runner, cfg2, int(eval_episodes))

    # Random policy: patch mac.select_actions to sample uniformly from available actions
    mac = runner.mac
    orig = mac.select_actions

    def _rand_select_actions(ep_batch, t_ep: int, t_env: int, raw_observation_text=None, bs=slice(None), test_mode: bool = True):
        avail = ep_batch["avail_actions"][:, t_ep]  # (bs,n_agents,n_actions)
        # avail is 0/1; sample uniform over avail
        probs = avail.float()
        probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-12)
        bs0, n_agents, n_act = probs.shape
        samp = torch.multinomial(probs.view(-1, n_act), 1).view(bs0, n_agents)
        # keep info from original forward pass (for logging/secondary signals)
        try:
            _outs, info = mac.forward(ep_batch, t_ep, test_mode=True)
        except Exception:
            info = {}
        return samp, info

    try:
        mac.select_actions = _rand_select_actions  # type: ignore
        return _stage4_eval_online_runner(runner, cfg2, int(eval_episodes))
    finally:
        mac.select_actions = orig  # type: ignore


def _stage4_debug_alignment(cfg: Any, ckpt: str, *, max_steps: int = 15) -> None:
    """
    Stage4 engineering debug (one episode):
    Print per-step alignment to diagnose time/semantics bugs.
    Focuses on:
      - batch stage_t / z_t (policy input)
      - chosen action histogram per step (over n_agents)
      - env info: t, z_mask, z_target (for t+1), z_pred (post-update), secondary_z_next injected
      - consistency: || normalize(z_pred) - normalize(secondary_z_next) ||_2
    """
    import numpy as np
    import torch

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from train import setup_experiment  # type: ignore

    cfg2 = copy.deepcopy(cfg)
    cfg2.load_model_path = str(ckpt)
    # deterministic for debugging
    try:
        cfg2.test_greedy = True
        cfg2.epsilon_start = 0.0
        cfg2.epsilon_finish = 0.0
        cfg2.epsilon_anneal_time = 1
    except Exception:
        pass

    runner, _mac, _learner, _logger, _device = setup_experiment(cfg2)
    batch = runner.run(test_mode=True)
    infos = getattr(runner, "last_env_infos", None)
    if not isinstance(infos, list) or not infos:
        print("[stage4_debug_alignment] No env infos captured.")
        return

    # Best-effort: read stage_t and z_t inputs stored in EpisodeBatch
    zt = None
    st = None
    try:
        if hasattr(batch, "scheme") and "z_t" in batch.scheme:
            zt = batch["z_t"]
        elif hasattr(batch, "scheme") and "belief_pre_population_z" in batch.scheme:
            zt = batch["belief_pre_population_z"]
    except Exception:
        zt = None
    try:
        if hasattr(batch, "scheme") and "stage_t" in batch.scheme:
            st = batch["stage_t"]
    except Exception:
        st = None

    acts = None
    try:
        acts = batch["actions"]
    except Exception:
        acts = None

    k_actions = int(getattr(getattr(cfg2, "env_args", None), "n_actions", getattr(cfg2, "n_actions", 5)))
    k_actions = max(1, k_actions)

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

    print("\n=== Stage4 alignment debug (one episode) ===")
    print(f"- ckpt: {ckpt}")
    print(f"- steps_in_infos: {len(infos)}")
    print("Note: in HiSim sync mode, env_info.t is the stage index BEFORE increment; z_target is for stage t+1.")

    n = min(int(max_steps), len(infos))
    for i in range(n):
        info = infos[i] if isinstance(infos[i], dict) else {}
        try:
            it = int(info.get("t", -1))
        except Exception:
            it = -1
        zm = float(info.get("z_mask", 0.0) or 0.0)
        edge_n = int(info.get("z_target_labeled_edge_n", 0) or 0)

        st_i = None
        try:
            if isinstance(st, torch.Tensor):
                x = st[0, i]
                st_i = int(x.view(-1)[0].item())
        except Exception:
            st_i = None

        zt_i = None
        try:
            if isinstance(zt, torch.Tensor):
                zz = zt[0, i].detach().cpu().numpy().reshape(-1)
                zt_i = [float(x) for x in zz.tolist()]
        except Exception:
            zt_i = None

        ah = None
        try:
            if isinstance(acts, torch.Tensor):
                a = acts.detach().cpu()
                if a.ndim == 4 and a.shape[-1] == 1:
                    a = a[..., 0]
                if a.ndim == 3:
                    flat = a[0, i, :].reshape(-1).numpy().astype(np.int64, copy=False)
                    flat = np.clip(flat, 0, k_actions - 1)
                    ah = np.bincount(flat, minlength=k_actions).astype(np.int64).tolist()
        except Exception:
            ah = None

        z_pred = info.get("z_pred", None)
        z_tgt = info.get("z_target", None)
        kl = float("nan")
        try:
            if isinstance(z_pred, list) and isinstance(z_tgt, list) and len(z_pred) == len(z_tgt) and len(z_pred) > 1:
                kl = _kl(np.array(z_tgt, dtype=np.float32), np.array(z_pred, dtype=np.float32))
        except Exception:
            kl = float("nan")

        sec = info.get("secondary_z_next", None)
        sec_src = info.get("secondary_z_next_source", None)
        dz = float("nan")
        try:
            if isinstance(sec, torch.Tensor):
                secv = sec.detach().cpu().numpy().reshape(-1)
            elif isinstance(sec, (list, tuple)):
                secv = np.array(list(sec), dtype=np.float32).reshape(-1)
            else:
                secv = None
            if secv is not None and isinstance(z_pred, list) and len(z_pred) == int(secv.size):
                dz = float(np.linalg.norm(_renorm(np.array(z_pred, dtype=np.float32)) - _renorm(secv)))
        except Exception:
            dz = float("nan")

        print(f"\n[{i:02d}] info.t={it:2d} batch.stage_t={st_i} z_mask={zm:.1f} edge_n={edge_n:4d} KL={kl:.6f} dz(pred,sec)={dz:.6f}")
        print(f"     action_hist={ah}")
        print(f"     z_t_input={zt_i}")
        print(f"     z_pred={z_pred}")
        print(f"     z_target(t)={z_tgt}")
        if isinstance(sec, torch.Tensor):
            sec_print = sec.detach().cpu().numpy().reshape(-1).tolist()
        else:
            sec_print = sec
        print(f"     secondary_z_next(src={sec_src})={sec_print}")
        # Directly inspect env.core_posts at stage=info.t (should match action_hist semantics)
        try:
            env = getattr(runner, "env", None)
            cps = getattr(env, "core_posts", None)
            core_users = getattr(env, "core_users", None)
            if isinstance(cps, dict) and isinstance(core_users, list) and it >= 0:
                cnt = {"post": 0, "retweet": 0, "reply": 0, "like": 0, "do_nothing": 0, "other": 0}
                for u in core_users:
                    p = cps.get((str(u), int(it)))
                    if not isinstance(p, dict):
                        continue
                    at = str(p.get("action_type") or "do_nothing").strip().lower()
                    if at in cnt:
                        cnt[at] += 1
                    else:
                        cnt["other"] += 1
                print(f"     env.core_posts@stage{it}.action_type_counts={cnt}")
        except Exception:
            pass
        try:
            grn = info.get("group_representation_next", None)
            if isinstance(grn, (list, tuple)) and len(grn) > 0:
                print(f"     env.group_representation_next[:8]={list(grn)[:8]}")
        except Exception:
            pass


def _stage4_compare_z_ablations(cfg: Any, ckpt: str, eval_episodes: int, *, shuffle_seed: int = 0) -> Dict[str, Any]:
    """
    Stage4 sanity check:
    Compare early test returns under different z_ablation_mode values.
    This diagnoses whether z_t affects action selection (policy input), while env dynamics remain the same.
    """
    modes = ["none", "zero", "shuffle"]
    out: Dict[str, Any] = {}
    for m in modes:
        cfg2 = copy.deepcopy(cfg)
        cfg2.load_model_path = str(ckpt)
        cfg2.z_ablation_mode = str(m)
        cfg2.z_shuffle_seed = int(shuffle_seed)
        res = _stage4_eval_online(cfg2, str(ckpt), int(eval_episodes))
        out[m] = res
    # simple "slope" proxy: difference vs with-z
    try:
        rz = float(out.get("none", {}).get("test_return_mean", 0.0))
        out["delta_return_zero_minus_withz"] = float(out.get("zero", {}).get("test_return_mean", 0.0) - rz)
        out["delta_return_shuffle_minus_withz"] = float(out.get("shuffle", {}).get("test_return_mean", 0.0) - rz)
    except Exception:
        pass
    return out

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
    sum_alpha0 = 0.0
    sum_alpha0_tgt = 0.0
    sum_dir_ent = 0.0
    sum_dir_varsum = 0.0
    # fixed-stage probes help diagnose "stage embeddings for late stages are untrained/misaligned"
    fixed_stage_ids = list(getattr(cfg2, "stage3a_fixed_stages", [11, 12]) or [11, 12])
    fixed_stage_ids = [int(x) for x in fixed_stage_ids if str(x).strip() != ""]
    fixed_stage_sums: Dict[int, float] = {int(s): 0.0 for s in fixed_stage_ids}

    stage_sum_mask: Dict[int, float] = {}
    stage_sum_dz_pred: Dict[int, float] = {}
    stage_sum_dz_tgt: Dict[int, float] = {}

    n_ep = max(1, int(eval_episodes))
    lt = str(getattr(cfg2, "z_transition_loss_type", "kl") or "kl").strip().lower()
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
        # optional per-sample alpha0_target inferred by env/dataset
        a0_tgt = None
        try:
            if "z_alpha0_target" in batch.scheme:
                a0_tgt = batch["z_alpha0_target"][:, :-1].reshape(N).to(device)
                a0_tgt = a0_tgt.to(dtype=torch.float32)
        except Exception:
            a0_tgt = None
        grf = gr.reshape(N, -1) if gr is not None else None
        stf = stage_t.reshape(N, -1) if stage_t is not None else None

        with torch.no_grad():
            if lt.startswith("dirichlet"):
                if not hasattr(be, "predict_next_population_belief_alpha"):
                    raise RuntimeError("validate_stage stage3a: dirichlet requested but BeliefEncoder lacks predict_next_population_belief_alpha().")
                if not hasattr(be, "compute_population_belief_loss_dirichlet_kl"):
                    raise RuntimeError("validate_stage stage3a: dirichlet requested but BeliefEncoder lacks compute_population_belief_loss_dirichlet_kl().")
                alpha_pred = be.predict_next_population_belief_alpha(zt, group_repr=grf, stage_t=stf)
                zpred = be.population_belief_mean_from_alpha(alpha_pred)
                loss = be.compute_population_belief_loss_dirichlet_kl(
                    alpha_pred,
                    ztar,
                    zm,
                    alpha0_target=(a0_tgt if a0_tgt is not None else float(getattr(cfg2, "dirichlet_alpha0_target", 10.0))),
                )
                try:
                    a0 = alpha_pred.sum(dim=-1)
                    sum_alpha0 += float((a0 * zm).sum().item())
                    if a0_tgt is not None:
                        sum_alpha0_tgt += float((a0_tgt.reshape(-1) * zm).sum().item())
                    # Dirichlet uncertainty stats
                    try:
                        k_dir = int(alpha_pred.shape[-1])
                        ap = torch.clamp(alpha_pred, min=float(getattr(be, "dirichlet_alpha_min", 1e-6)))
                        ap0 = torch.clamp(ap.sum(dim=-1), min=1e-6)
                        logB = torch.sum(torch.lgamma(ap), dim=-1) - torch.lgamma(ap0)
                        ent = logB + (ap0 - float(k_dir)) * torch.digamma(ap0) - torch.sum((ap - 1.0) * torch.digamma(ap), dim=-1)
                        sum_dir_ent += float((ent * zm).sum().item())
                        ap0u = ap0.unsqueeze(-1)
                        var = (ap * (ap0u - ap)) / (ap0u * ap0u * (ap0u + 1.0))
                        var_sum = var.sum(dim=-1)
                        sum_dir_varsum += float((var_sum * zm).sum().item())
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                zpred = be.predict_next_population_belief(zt, group_repr=grf, stage_t=stf, return_logits=False)
                loss = be.compute_population_belief_loss(zpred, ztar, zm, loss_type=lt)
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
    if lt.startswith("dirichlet") and sum_alpha0 > 0:
        out["eval_z_pred_alpha0_mean"] = float(sum_alpha0 / sum_mask)
        if sum_alpha0_tgt > 0:
            out["eval_z_target_alpha0_mean"] = float(sum_alpha0_tgt / sum_mask)
        if sum_dir_ent > 0:
            out["eval_z_pred_dirichlet_entropy"] = float(sum_dir_ent / sum_mask)
        if sum_dir_varsum > 0:
            out["eval_z_pred_dirichlet_varsum"] = float(sum_dir_varsum / sum_mask)
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
    ap.add_argument("--seed", type=int, default=42, help="Seed for eval runs (best-effort).")
    ap.add_argument("--min_margin", type=float, default=0.05, help="Stage1: required margin over majority baseline.")
    ap.add_argument(
        "--pref_min_margin",
        type=float,
        default=0.005,
        help="Stage3b(preference scorer): required margin in (baseline_BCE - model_BCE). Default is small because BCE improvements are subtle.",
    )
    ap.add_argument(
        "--stage3b_eval_sampling",
        action="store_true",
        help="Stage3b: evaluate with sampling (do not force argmax); useful to inspect sampling behavior.",
    )
    ap.add_argument(
        "--debug_preference",
        action="store_true",
        help="Stage3b: preference scorer debug. Print why preference_eval_n could be zero (missing dist/bias logit).",
    )
    ap.add_argument(
        "--compare_z",
        action="store_true",
        help="Sanity check: compare z_t usage by running the SAME ckpt with z_ablation_mode in {none,zero,shuffle}. "
             "For stage3b prints action distribution entropy/KL; for stage4 prints early return deltas.",
    )
    ap.add_argument("--z_shuffle_seed", type=int, default=0, help="Seed for z_t shuffle ablation (diagnostic only).")
    # Stage4: frozen vs trained comparisons (paper-facing)
    ap.add_argument("--stage4_ref_ckpt", type=str, default="", help="Stage4: reference ckpt for frozen/init baseline (optional).")
    ap.add_argument(
        "--stage4_ref_policy",
        type=str,
        default="ckpt",
        choices=["ckpt", "random"],
        help="Stage4: reference policy mode. 'ckpt' uses stage4_ref_ckpt weights; 'random' overrides to random actions.",
    )
    ap.add_argument(
        "--stage4_eval_policy",
        type=str,
        default="greedy",
        choices=["greedy", "stochastic"],
        help=(
            "Stage4: evaluation action-selection mode. "
            "'greedy' disables exploration (epsilon=0, argmax) for deterministic reporting; "
            "'stochastic' keeps config exploration (temperature/epsilon) to approximate training rollout metrics."
        ),
    )
    ap.add_argument(
        "--stage4_z_traj_k",
        type=int,
        default=3,
        help="Stage4: number of early/late z points to average for trajectory print (default 3).",
    )
    ap.add_argument(
        "--stage4_debug_alignment",
        action="store_true",
        help="Stage4: engineering debug. Print per-step alignment table (actions/z_t/secondary_z_next/z_pred/z_target).",
    )
    ap.add_argument(
        "--stage4_debug_steps",
        type=int,
        default=15,
        help="Stage4: number of steps to print for --stage4_debug_alignment (default 15).",
    )
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
    # Best-effort set eval seed (env / numpy / torch handled inside project setup)
    try:
        if hasattr(cfg, "system") and hasattr(cfg.system, "seed"):
            cfg.system.seed = int(args.seed)
    except Exception:
        pass
    # Stage4: trajectory averaging window
    try:
        cfg.stage4_z_traj_k = int(args.stage4_z_traj_k)
    except Exception:
        pass
    # Stage4: eval policy mode (greedy vs stochastic)
    try:
        cfg.stage4_eval_policy = str(getattr(args, "stage4_eval_policy", "greedy") or "greedy").strip().lower()
    except Exception:
        cfg.stage4_eval_policy = "greedy"
    # pass debug flags through cfg (used by stage3b eval)
    try:
        cfg.debug_preference = bool(getattr(args, "debug_preference", False))
    except Exception:
        pass
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
        # For validation, default to deterministic boxed action selection to avoid sampling noise.
        # You can opt-in to sampling via --stage3b_eval_sampling.
        try:
            if bool(getattr(args, "stage3b_eval_sampling", False)):
                # keep config as-is (sampling)
                pass
            else:
                cfg.s3b_boxed_action_selection = "argmax"
                cfg.s3b_boxed_action_temperature = 1.0
                cfg.s3b_boxed_action_epsilon = 0.0
        except Exception:
            pass
        _ckpt_must_have(str(args.ckpt), ["agent.th"])
        if bool(getattr(args, "compare_z", False)):
            outz = _stage3b_compare_z_ablations(cfg, str(args.ckpt), str(args.eval_split), int(args.eval_episodes), shuffle_seed=int(args.z_shuffle_seed))
            print("")
            print("=== Stage3b z_t usage sanity (same ckpt; z ablations) ===")
            for m in ["none", "zero", "shuffle"]:
                r = outz.get(m, {})
                print(f"- mode={m}: n={int(r.get('n',0))} entropy_mean={float(r.get('entropy_mean',0.0)):.6f}")
                pm = r.get("p_mean")
                if isinstance(pm, list) and len(pm) <= 8:
                    print(f"  p_mean={['{:.3f}'.format(float(x)) for x in pm]}")
            if "kl_withz_vs_zero" in outz:
                print(f"- KL(with_z || z=0): {float(outz['kl_withz_vs_zero']):.6f}")
            if "kl_withz_vs_shuffle" in outz:
                print(f"- KL(with_z || z=shuffle): {float(outz['kl_withz_vs_shuffle']):.6f}")
        out = _stage3b_eval_action_imitation(cfg, str(args.ckpt), str(args.eval_split), int(args.eval_episodes))
        print("")
        print("=== Stage3b(action imitation) eval ===")
        print(f"- eval_split: {args.eval_split}")
        if bool(getattr(args, "stage3b_eval_sampling", False)):
            print("- boxed_action_selection: sampling (from config)")
        else:
            print("- boxed_action_selection: argmax (forced for stable eval)")
        print("- boxed_action_selection: argmax (forced for stable eval)")
        if out.get("sup_ids") is not None:
            print(f"- masked_supervised_ids: {out.get('sup_ids')}")
        print(f"- eval_acc_masked: {out['eval_acc_masked']:.4f}")
        print(f"- majority_baseline_masked: {out['majority_baseline_masked']:.4f}")
        print(f"- margin: {out['margin']:+.4f}")
        if bool(out.get("s3b_preference_scorer", False)):
            print(f"- preference_eval_n: {int(out.get('pref_n', 0))}")
            print(f"- preference_bce: {float(out.get('pref_bce', float('nan'))):.6f}")
            print(f"- preference_bce_baseline: {float(out.get('pref_bce_baseline', float('nan'))):.6f}")
            print(f"- preference_bce_margin(baseline - model): {float(out.get('pref_bce_margin', float('nan'))):+.6f}")
            print(f"- preference_corr: {float(out.get('pref_corr', float('nan'))):.4f}")
            if bool(out.get("debug_preference", False)):
                print(f"- [debug] total_env_infos: {int(out.get('debug_pref_total_infos', 0))}")
                print(f"- [debug] has_target_dist_dict: {int(out.get('debug_pref_has_dist', 0))}")
                print(f"- [debug] p0+p1>0: {int(out.get('debug_pref_denom_pos', 0))}")
                print(f"- [debug] has_pref_bias_logit: {int(out.get('debug_pref_has_bias', 0))}")
                print(f"- [debug] pref_pairs_added: {int(out.get('debug_pref_added', 0))}")
                exs = out.get("debug_pref_examples", None)
                if isinstance(exs, list) and len(exs) > 0:
                    print(f"- [debug] examples(first {min(5, len(exs))}):")
                    for ex in exs[:5]:
                        try:
                            why = ex.get('why')
                            keys = ex.get('keys')
                            print(f"  - why={why} keys={keys}")
                            if 'target_distribution_prob' in ex:
                                print(f"    target_distribution_prob={ex.get('target_distribution_prob')}")
                            if 'target_distribution_prob_type' in ex:
                                print(f"    target_distribution_prob_type={ex.get('target_distribution_prob_type')}")
                        except Exception:
                            pass
        print(f"- n_masked: {out['n_masked']}")
        print(f"- n_skipped_unsup: {out['n_skipped_unsup']}")
        print(f"- coverage: {out['coverage']:.4f}")
        print(f"- skipped_ratio: {out['skipped_ratio']:.4f}")
        # Collapse sanity (marginal distribution over ALL labels, including unsupervised)
        try:
            if out.get("pred_entropy") == out.get("pred_entropy"):
                print(f"- pred_entropy(all labels): {float(out.get('pred_entropy',0.0)):.6f}")
            if out.get("pred_mode_frac") == out.get("pred_mode_frac"):
                print(f"- pred_mode_frac(all labels): {float(out.get('pred_mode_frac',0.0)):.6f}")
            if out.get("pred_kl_gt") == out.get("pred_kl_gt"):
                print(f"- KL(pred_marginal || gt_marginal): {float(out.get('pred_kl_gt',0.0)):.6f}")
            if out.get("unsup_pred_frac") == out.get("unsup_pred_frac"):
                print(f"- unsup_pred_frac(sum over unsup labels): {float(out.get('unsup_pred_frac',0.0)):.6f}")
            if out.get("unsup_gt_frac") == out.get("unsup_gt_frac"):
                print(f"- unsup_gt_frac(sum over unsup labels): {float(out.get('unsup_gt_frac',0.0)):.6f}")
        except Exception:
            pass
        print("")
        print("Confusion matrix (rows=GT, cols=Pred) over MASKED rows:")
        conf = out["confusion"]
        k = int(out["k"])
        for i in range(min(k, 5)):
            row = conf[i]
            print("  " + " ".join([f"{int(x):6d}" for x in row[: min(k, 5)]]))

        ok = True
        # Gate depends on whether S3b is a classifier or preference scorer.
        if bool(out.get("s3b_preference_scorer", False)):
            if float(out.get("pref_n", 0)) <= 0:
                ok = False
                print("\n[FAIL] Stage3b: preference eval has zero valid samples (missing target_distribution_prob?).")
            else:
                # prefer lower BCE than baseline by a margin
                try:
                    bce = float(out.get("pref_bce", float("nan")))
                    bce_base = float(out.get("pref_bce_baseline", float("nan")))
                    if not (bce == bce and bce_base == bce_base):
                        ok = False
                        print("\n[FAIL] Stage3b: preference BCE is NaN.")
                    elif bce > (bce_base - float(getattr(args, "pref_min_margin", args.min_margin))):
                        ok = False
                        need = float(getattr(args, "pref_min_margin", args.min_margin))
                        print(f"\n[FAIL] Stage3b: preference BCE not better than baseline by margin={need:.6f}.")
                    else:
                        need = float(getattr(args, "pref_min_margin", args.min_margin))
                        print(f"\n[PASS] Stage3b: preference BCE beats baseline by margin={need:.6f}.")
                except Exception:
                    ok = False
                    print("\n[FAIL] Stage3b: preference BCE evaluation error.")
        else:
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

        if bool(getattr(args, "stage4_debug_alignment", False)):
            _stage4_debug_alignment(cfg, str(args.ckpt), max_steps=int(getattr(args, "stage4_debug_steps", 15)))
            return 0

        if bool(getattr(args, "compare_z", False)):
            outz = _stage4_compare_z_ablations(cfg, str(args.ckpt), int(args.eval_episodes), shuffle_seed=int(args.z_shuffle_seed))
            print("")
            print("=== Stage4 z_t usage sanity (same ckpt; z ablations) ===")
            for m in ["none", "zero", "shuffle"]:
                r = outz.get(m, {})
                print(f"- mode={m}: return_mean={float(r.get('test_return_mean',0.0)):.6f} std={float(r.get('test_return_std',0.0)):.6f} z_kl={float(r.get('z_kl', float('nan'))):.6f}")
            if "delta_return_zero_minus_withz" in outz:
                print(f"- Δreturn(z=0 - with_z): {float(outz['delta_return_zero_minus_withz']):+.6f}")
            if "delta_return_shuffle_minus_withz" in outz:
                print(f"- Δreturn(z=shuffle - with_z): {float(outz['delta_return_shuffle_minus_withz']):+.6f}")

        # Main eval (trained ckpt)
        out = _stage4_eval_online_with_policy(cfg, str(args.ckpt), int(args.eval_episodes), policy_mode="ckpt")

        def _print_stage4_block(title: str, r: Dict[str, Any]) -> None:
            print("")
            print(f"=== Stage4(online RL) eval: {title} ===")
            print(f"- test_episodes: {r.get('test_episodes', 0)}")
            print(f"- test_return_mean: {r.get('test_return_mean', 0.0):.6f}")
            print(f"- test_return_std: {r.get('test_return_std', 0.0):.6f}")
            print(f"- z_eval_steps: {r.get('z_eval_steps', 0)}")
            zkl = r.get("z_kl", float("nan"))
            try:
                if zkl == zkl:
                    print(f"- z_kl: {float(zkl):.6f}")
                else:
                    print("- z_kl: nan")
            except Exception:
                print("- z_kl: nan")
            # micro-level action sanity
            try:
                ae = r.get("action_entropy", float("nan"))
                am = r.get("action_mode_frac", float("nan"))
                if ae == ae:
                    print(f"- action_entropy: {float(ae):.6f}")
                if am == am:
                    print(f"- action_mode_frac: {float(am):.6f}")
                ah = r.get("action_hist", None)
                if isinstance(ah, list) and len(ah) > 0:
                    print(f"- action_hist(counts): {ah}")
            except Exception:
                pass
            # z trajectory directionality
            try:
                z0 = r.get("z_pred_mean_early", None)
                z1 = r.get("z_pred_mean_late", None)
                if isinstance(z0, list) and isinstance(z1, list):
                    print(f"- z_pred_mean_early: {[float(x) for x in z0]}")
                    print(f"- z_pred_mean_late:  {[float(x) for x in z1]}")
                g0 = r.get("z_gt_mean_early", None)
                g1 = r.get("z_gt_mean_late", None)
                if isinstance(g0, list) and isinstance(g1, list):
                    print(f"- z_gt_mean_early:   {[float(x) for x in g0]}")
                    print(f"- z_gt_mean_late:    {[float(x) for x in g1]}")
            except Exception:
                pass

        # Optional: reference baseline (frozen/init or random policy)
        ref_ckpt = str(getattr(args, "stage4_ref_ckpt", "") or "").strip()
        if ref_ckpt:
            ref_mode = str(getattr(args, "stage4_ref_policy", "ckpt") or "ckpt").strip().lower()
            ref_out = _stage4_eval_online_with_policy(cfg, ref_ckpt, int(args.eval_episodes), policy_mode=ref_mode)
            _print_stage4_block(f"REF({ref_mode}) ckpt={ref_ckpt}", ref_out)
            _print_stage4_block(f"TRAINED ckpt={args.ckpt}", out)
            # Key deltas (paper: show there is a gap)
            try:
                d_ret = float(out.get("test_return_mean", 0.0)) - float(ref_out.get("test_return_mean", 0.0))
                d_kl = float(ref_out.get("z_kl", float("nan"))) - float(out.get("z_kl", float("nan")))
                print("")
                print("=== Stage4 compare (TRAINED - REF) ===")
                print(f"- Δreturn_mean (trained - ref): {d_ret:+.6f} (higher is better)")
                if d_kl == d_kl:
                    print(f"- Δz_kl (ref - trained): {d_kl:+.6f} (positive means trained improved z-match)")
            except Exception:
                pass
        else:
            _print_stage4_block(f"ckpt={args.ckpt}", out)

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
        # Dirichlet-specific (may be absent if loss_type is not dirichlet_* or ckpt doesn't support it)
        "eval_z_target_alpha0_mean",
        "eval_z_pred_alpha0_mean",
        "eval_z_pred_dirichlet_entropy",
        "eval_z_pred_dirichlet_varsum",
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


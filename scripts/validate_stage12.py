#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage1/Stage2 training validation script for the HiSim belief pipeline.

Validations covered (per user request):
1) Training is learning: training loss down / acc up; check minority-class behavior.
2) Eval-set effect: confirm test/* metrics exist; summarize eval metrics.
3) Stage2 inherits & improves (A/B): compare Stage2(with Stage1 init) vs Stage2(baseline, no init) if provided.
4) Regression: evaluate Stage2 checkpoint on CORE dataset split to ensure Stage2 didn't destroy Stage1 capability.

This script is designed to be:
- fast on large metrics.jsonl (streams + only parses needed metrics)
- safe (doesn't overwrite existing logs; eval runs go to /tmp by default)

Usage examples:
  # 1) summarize existing runs (no extra eval)
  conda run -n HiSim python ECON/scripts/validate_stage12.py \
    --stage1-logdir /home/zhuyinzhou/MAS/ECON/logs/hisim-belief-core-stable \
    --stage2-logdir /home/zhuyinzhou/MAS/ECON/logs/hisim-belief-noncore-stage2

  # 2) add regression eval on core using Stage1/Stage2 checkpoints
  PYTHONPATH=/home/zhuyinzhou/MAS/ECON/src conda run --no-capture-output -n HiSim python ECON/scripts/validate_stage12.py \
    --stage1-logdir /home/zhuyinzhou/MAS/ECON/logs/hisim-belief-core-stable \
    --stage2-logdir /home/zhuyinzhou/MAS/ECON/logs/hisim-belief-noncore-stage2 \
    --stage1-ckpt /data/zhuyinzhou/ECON/models/checkpoints_s1_e1e2_stable/episode_45000 \
    --stage2-ckpt /data/zhuyinzhou/ECON/models/checkpoints_s2_e/final \
    --core-eval-config /home/zhuyinzhou/MAS/ECON/examples/configs/hisim_belief_core_stable.yaml \
    --eval-episodes 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


# -----------------------
# Metrics parsing helpers
# -----------------------

MetricPoint = Tuple[int, float]  # (step, value)


@dataclass
class MetricSummary:
    name: str
    count: int
    first: Optional[MetricPoint]
    last: Optional[MetricPoint]
    min_v: Optional[float]
    max_v: Optional[float]
    early_mean: Optional[float]
    late_mean: Optional[float]


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    if v in (float("inf"), float("-inf")):
        return None
    return v


def stream_metric_summaries(
    metrics_jsonl_path: str,
    wanted: Iterable[str],
    early_k: int = 200,
    late_k: int = 200,
    max_lines: int = -1,
) -> Dict[str, MetricSummary]:
    """
    Stream-parse a metrics.jsonl and compute summaries for selected metric keys.
    Keeps only small buffers (early_k + late_k per metric).
    """
    wanted_set = set(str(x) for x in wanted)
    early_buf: Dict[str, List[float]] = {k: [] for k in wanted_set}
    late_buf: Dict[str, Deque[float]] = {k: deque(maxlen=max(1, int(late_k))) for k in wanted_set}

    first: Dict[str, Optional[MetricPoint]] = {k: None for k in wanted_set}
    last: Dict[str, Optional[MetricPoint]] = {k: None for k in wanted_set}
    min_v: Dict[str, Optional[float]] = {k: None for k in wanted_set}
    max_v: Dict[str, Optional[float]] = {k: None for k in wanted_set}
    count: Dict[str, int] = {k: 0 for k in wanted_set}

    n = 0
    with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            if max_lines > 0 and n > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            m = obj.get("metric")
            if m not in wanted_set:
                continue
            step = obj.get("step")
            try:
                step_i = int(step)
            except Exception:
                continue
            v = _safe_float(obj.get("value"))
            if v is None:
                continue

            count[m] += 1
            if first[m] is None:
                first[m] = (step_i, v)
            last[m] = (step_i, v)
            if min_v[m] is None or v < float(min_v[m]):
                min_v[m] = v
            if max_v[m] is None or v > float(max_v[m]):
                max_v[m] = v

            if len(early_buf[m]) < int(early_k):
                early_buf[m].append(v)
            late_buf[m].append(v)

    out: Dict[str, MetricSummary] = {}
    for k in wanted_set:
        e = early_buf[k]
        l = list(late_buf[k])
        out[k] = MetricSummary(
            name=k,
            count=int(count[k]),
            first=first[k],
            last=last[k],
            min_v=min_v[k],
            max_v=max_v[k],
            early_mean=(sum(e) / len(e)) if e else None,
            late_mean=(sum(l) / len(l)) if l else None,
        )
    return out


def has_any_test_metrics(metrics_jsonl_path: str, max_lines: int = -1) -> bool:
    n = 0
    with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            if max_lines > 0 and n > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            m = obj.get("metric", "")
            if isinstance(m, str) and m.startswith("test/"):
                return True
    return False


def _fmt_point(p: Optional[MetricPoint]) -> str:
    if not p:
        return "None"
    return f"(step={p[0]}, value={p[1]:.6g})"


def _fmt_f(v: Optional[float]) -> str:
    if v is None:
        return "None"
    return f"{v:.6g}"


# -----------------------
# Eval (regression) helper
# -----------------------

def _import_econ_train_module(repo_src_dir: str):
    """
    Import ECON's src/train.py as a module named econ_train.
    """
    src_dir = os.path.abspath(repo_src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import importlib

    return importlib.import_module("train")


def eval_checkpoint_on_config(
    repo_src_dir: str,
    config_path: str,
    ckpt_dir: str,
    eval_split: str = "test",
    test_episodes: int = 200,
    cuda_visible_devices: Optional[str] = None,
    tmp_log_root: str = "/tmp/econ_stage12_validate",
) -> Dict[str, Any]:
    """
    Pure evaluation: load checkpoint, run run_test(), return stats dict.
    Writes temporary logs under tmp_log_root/<timestamp>/ to satisfy logger.
    """
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    econ_train = _import_econ_train_module(repo_src_dir)

    cfg = econ_train.load_config(config_path)
    # Override to evaluation-only behavior
    try:
        cfg.eval_dataset_split = str(eval_split)
    except Exception:
        pass
    try:
        cfg.test_nepisode = int(test_episodes)
    except Exception:
        pass
    cfg.load_model_path = str(ckpt_dir)
    # Ensure no expensive rollouts
    try:
        cfg.enable_llm_rollout = False
        cfg.together_api_key = ""
    except Exception:
        pass

    # Write eval logs to tmp dir to avoid polluting user's experiment logs
    ts = int(time.time())
    exp = f"eval_{os.path.basename(os.path.normpath(ckpt_dir))}_{ts}"
    try:
        if not hasattr(cfg, "logging"):
            from types import SimpleNamespace

            cfg.logging = SimpleNamespace()
        cfg.logging.log_path = os.path.join(tmp_log_root, exp)
        cfg.logging.experiment_name = exp
        cfg.logging.use_tensorboard = False
        cfg.logging.save_model = False
        # Keep checkpoint_path unused in eval
        cfg.logging.checkpoint_path = os.path.join(tmp_log_root, exp, "models")
    except Exception:
        pass

    runner, _mac, _learner, logger, _device = econ_train.setup_experiment(cfg)
    stats = econ_train.run_test(runner, logger, cfg)
    try:
        runner.close_env()
    except Exception:
        pass
    return stats


# -----------------------
# Report logic
# -----------------------

DEFAULT_TRAIN_METRICS = [
    "train/loss_total",
    "train/belief_sup_acc",
    "train/belief_sup_recall1",
    "train/belief_sup_precision1",
    "train/belief_sup_pred2_frac",
    "train/belief_sup_pred1_frac",
    "train/belief_sup_gt1_frac",
]

DEFAULT_TEST_METRICS = [
    "test/test_return_mean",
    "test/stance_recall0",
    "test/stance_recall1",
    "test/stance_recall2",
    "test/stance_precision0",
    "test/stance_precision1",
    "test/stance_precision2",
]


def summarize_run(name: str, logdir: str, early_k: int, late_k: int, max_lines: int) -> Dict[str, Any]:
    metrics_path = os.path.join(logdir, "metrics.jsonl")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"metrics.jsonl not found: {metrics_path}")

    wanted = list(DEFAULT_TRAIN_METRICS) + list(DEFAULT_TEST_METRICS)
    sums = stream_metric_summaries(metrics_path, wanted=wanted, early_k=early_k, late_k=late_k, max_lines=max_lines)

    ok_test = has_any_test_metrics(metrics_path, max_lines=max_lines)
    return {"name": name, "logdir": logdir, "metrics_path": metrics_path, "summaries": sums, "has_test": ok_test}


def print_learning_checks(run: Dict[str, Any]) -> None:
    sums: Dict[str, MetricSummary] = run["summaries"]
    print(f"\n## {run['name']}：训练是否在学习（train/*）")
    for k in DEFAULT_TRAIN_METRICS:
        s = sums.get(k)
        if not s or s.count <= 0:
            print(f"- {k}: MISSING")
            continue
        delta = None
        if s.early_mean is not None and s.late_mean is not None:
            delta = s.late_mean - s.early_mean
        print(
            f"- {k}: n={s.count} first={_fmt_point(s.first)} last={_fmt_point(s.last)} "
            f"early_mean={_fmt_f(s.early_mean)} late_mean={_fmt_f(s.late_mean)}"
            + (f" delta={delta:+.6g}" if delta is not None else "")
        )

    # Heuristic verdicts (lightweight, not a strict "pass/fail")
    loss = sums.get("train/loss_total")
    acc = sums.get("train/belief_sup_acc")
    if loss and loss.early_mean is not None and loss.late_mean is not None:
        if loss.late_mean < loss.early_mean:
            print("- 结论: loss_total ✅（后期均值低于前期）")
        else:
            print("- 结论: loss_total ⚠️（后期均值未低于前期；可能尚未收敛或学习率/数据问题）")
    if acc and acc.early_mean is not None and acc.late_mean is not None:
        if acc.late_mean > acc.early_mean:
            print("- 结论: belief_sup_acc ✅（后期均值高于前期）")
        else:
            print("- 结论: belief_sup_acc ⚠️（后期均值未高于前期）")


def print_eval_checks(run: Dict[str, Any]) -> None:
    sums: Dict[str, MetricSummary] = run["summaries"]
    print(f"\n## {run['name']}：评估集效果（test/*）")
    if not run["has_test"]:
        print("- ❌ 未发现任何 test/* 指标（TensorBoard 看不到完整评估指标就会“少一大截”）")
        print("  建议检查：test_interval 是否触发、eval_dataset_split 是否设置、训练是否跑到 episode > test_interval。")
        return
    print("- ✅ 已发现 test/* 指标")
    for k in DEFAULT_TEST_METRICS:
        s = sums.get(k)
        if not s or s.count <= 0:
            print(f"- {k}: MISSING")
            continue
        print(f"- {k}: n={s.count} last={_fmt_point(s.last)} min={_fmt_f(s.min_v)} max={_fmt_f(s.max_v)}")


def compare_two_runs(run_a: Dict[str, Any], run_b: Dict[str, Any], label_a: str, label_b: str) -> None:
    """
    Compare two runs on key test metrics (prefers test/test_return_mean).
    """
    print("\n## Stage2 继承+提升（A/B 对照）")
    if (not run_a.get("has_test")) or (not run_b.get("has_test")):
        print("- ❌ 至少有一个 run 缺少 test/* 指标，无法做严谨 A/B 对照。")
        return

    def _last(metric: str, r: Dict[str, Any]) -> Optional[float]:
        s = r["summaries"].get(metric)
        if not s or not s.last:
            return None
        return float(s.last[1])

    m = "test/test_return_mean"
    a = _last(m, run_a)
    b = _last(m, run_b)
    if a is not None and b is not None:
        diff = a - b
        print(f"- {m}（last）：{label_a}={a:.6g} vs {label_b}={b:.6g} | Δ={diff:+.6g}")
    else:
        print(f"- {m}：缺失，改用 stance_recall1/precision1 作为对照")

    for m2 in ("test/stance_recall1", "test/stance_precision1"):
        a2 = _last(m2, run_a)
        b2 = _last(m2, run_b)
        if a2 is not None and b2 is not None:
            print(f"- {m2}（last）：{label_a}={a2:.6g} vs {label_b}={b2:.6g} | Δ={a2-b2:+.6g}")


def main() -> int:
    p = argparse.ArgumentParser(description="Validate Stage1/Stage2 training effectiveness.")
    p.add_argument("--stage1-logdir", type=str, required=True, help="Stage1 run log dir containing metrics.jsonl")
    p.add_argument("--stage2-logdir", type=str, required=True, help="Stage2 run log dir containing metrics.jsonl")
    p.add_argument("--stage2-baseline-logdir", type=str, default="", help="Optional baseline Stage2 log dir (no Stage1 init) for A/B")

    p.add_argument("--early-k", type=int, default=200, help="How many early points per metric to average")
    p.add_argument("--late-k", type=int, default=200, help="How many late points per metric to average")
    p.add_argument("--max-lines", type=int, default=-1, help="Optional: cap lines read from metrics.jsonl (debug/perf)")

    # Regression eval
    p.add_argument("--stage1-ckpt", type=str, default="", help="Optional Stage1 checkpoint dir for regression eval baseline")
    p.add_argument("--stage2-ckpt", type=str, default="", help="Stage2 checkpoint dir (final) for regression eval on core")
    p.add_argument(
        "--core-eval-config",
        type=str,
        default="/home/zhuyinzhou/MAS/ECON/examples/configs/hisim_belief_core_stable.yaml",
        help="Config yaml for core dataset evaluation",
    )
    p.add_argument("--eval-split", type=str, default="test", help="Eval split name for HF dataset env")
    p.add_argument("--eval-episodes", type=int, default=200, help="Number of eval episodes")
    p.add_argument("--cuda-visible-devices", type=str, default=None, help="CUDA_VISIBLE_DEVICES for eval runs")
    p.add_argument(
        "--repo-src-dir",
        type=str,
        default="",
        help="Path to ECON/src (if empty, inferred from script location)",
    )
    args = p.parse_args()

    stage1 = summarize_run("Stage1(core)", args.stage1_logdir, args.early_k, args.late_k, args.max_lines)
    stage2 = summarize_run("Stage2(noncore)", args.stage2_logdir, args.early_k, args.late_k, args.max_lines)

    print("\n### 1) 训练过程是否真的在学习")
    print_learning_checks(stage1)
    print_learning_checks(stage2)

    print("\n### 2) 必须看评估集效果（test/*）")
    print_eval_checks(stage1)
    print_eval_checks(stage2)

    print("\n### 3) Stage2 是否“继承 + 提升”（A/B）")
    if args.stage2_baseline_logdir:
        base = summarize_run("Stage2(baseline, no init)", args.stage2_baseline_logdir, args.early_k, args.late_k, args.max_lines)
        compare_two_runs(stage2, base, "Stage2(init)", "Stage2(baseline)")
    else:
        print("- 跳过：未提供 --stage2-baseline-logdir（建议跑一个不加载 Stage1 的短 run 做对照）")

    print("\n### 4) 回归测试：Stage2 是否破坏了 Stage1(core) 能力")
    if not args.stage2_ckpt:
        print("- 跳过：未提供 --stage2-ckpt（建议填 Stage2 的 final checkpoint 目录）")
        return 0

    repo_src = args.repo_src_dir.strip()
    if not repo_src:
        here = os.path.abspath(os.path.dirname(__file__))
        repo_src = os.path.abspath(os.path.join(here, "..", "src"))

    print(f"- 使用 core-eval-config: {args.core_eval_config}")
    print(f"- eval_split={args.eval_split} eval_episodes={args.eval_episodes}")

    try:
        s2_core = eval_checkpoint_on_config(
            repo_src_dir=repo_src,
            config_path=args.core_eval_config,
            ckpt_dir=args.stage2_ckpt,
            eval_split=args.eval_split,
            test_episodes=int(args.eval_episodes),
            cuda_visible_devices=args.cuda_visible_devices,
        )
        print("- Stage2->core eval:", {k: s2_core.get(k) for k in ("test_return_mean", "core_stance_acc", "stance_recall1", "stance_precision1")})
    except Exception as e:
        print(f"- ❌ Stage2->core eval 失败：{e}")
        print("  建议：确保在 HiSim 环境运行、并设置 PYTHONPATH=/.../ECON/src；同时 core-eval-config 路径正确。")
        return 1

    if args.stage1_ckpt:
        try:
            s1_core = eval_checkpoint_on_config(
                repo_src_dir=repo_src,
                config_path=args.core_eval_config,
                ckpt_dir=args.stage1_ckpt,
                eval_split=args.eval_split,
                test_episodes=int(args.eval_episodes),
                cuda_visible_devices=args.cuda_visible_devices,
            )
            print("- Stage1->core eval:", {k: s1_core.get(k) for k in ("test_return_mean", "core_stance_acc", "stance_recall1", "stance_precision1")})
            # Simple regression signal on core_stance_acc
            try:
                a = float(s1_core.get("core_stance_acc", 0.0))
                b = float(s2_core.get("core_stance_acc", 0.0))
                print(f"- 回归对比(core_stance_acc): Stage2 - Stage1 = {b-a:+.6g}")
            except Exception:
                pass
        except Exception as e:
            print(f"- ⚠️ Stage1->core eval 失败（将只输出 Stage2->core）：{e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


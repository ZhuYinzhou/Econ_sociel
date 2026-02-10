#!/usr/bin/env python3
"""
Batch evaluation script for distillation experiments.

It runs ECON/scripts/paper_eval.py for multiple checkpoints (e.g., Final / s3a / s3b / s2),
then aggregates key macro metrics for your paper:
  - KL, JS (divergence between z_t GT vs Pred)
  - DTW (trend alignment over time; we report entropy/polarization DTW and their mean)
  - Corr (trend correlation; we report Spearman for entropy/polarization and their mean)

Why reuse paper_eval.py?
- Keeps the metric definitions consistent with your existing paper evaluation pipeline.
- Avoids duplicating env rollout / parsing logic.

Example:
  source /home/zhuyinzhou/miniconda3/etc/profile.d/conda.sh
  conda activate HiSim
  python /home/zhuyinzhou/MAS/ECON/scripts/distill_eval.py \
    --config /home/zhuyinzhou/MAS/ECON/examples/configs/hisim_stage4.yaml \
    --episodes 50 --seed 42 --stagewise_source all_mean \
    --run Final=/data/zhuyinzhou/ECON/models/checkpoints_s4/final \
    --run s3b=/data/zhuyinzhou/ECON/models/checkpoints_s3b/final \
    --run s3a=/data/zhuyinzhou/ECON/models/checkpoints_s3a/final \
    --run s2=/data/zhuyinzhou/ECON/models/checkpoints_s2/final \
    --out_csv /home/zhuyinzhou/paper/distill_eval_summary.csv \
    --out_dir /home/zhuyinzhou/paper/distill_eval_runs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunSpec:
    name: str
    ckpt: str


def _parse_run(s: str) -> RunSpec:
    if "=" not in s:
        raise ValueError(f"--run expects NAME=CKPT_PATH, got: {s}")
    name, ckpt = s.split("=", 1)
    name = str(name).strip()
    ckpt = str(ckpt).strip()
    if not name:
        raise ValueError(f"Empty run name in: {s}")
    if not ckpt:
        raise ValueError(f"Empty ckpt path in: {s}")
    return RunSpec(name=name, ckpt=ckpt)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _extract_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract macro metrics from paper_eval JSON.

    We prioritize per-episode-by-stage summary because it is closest to "macro curve by round".
    Fallback to concatenation summary if missing.
    """
    src = _get_nested(results, "population_metrics_per_episode_by_stage")
    src_name = "population_metrics_per_episode_by_stage"
    if not isinstance(src, dict):
        src = _get_nested(results, "population_metrics_per_episode")
        src_name = "population_metrics_per_episode"
    if not isinstance(src, dict):
        src = _get_nested(results, "population_metrics")
        src_name = "population_metrics"
    if not isinstance(src, dict):
        src = {}
        src_name = "missing"

    kl = _safe_float(src.get("kl_mean_mean", src.get("kl_mean")))
    js = _safe_float(src.get("js_mean_mean", src.get("js_mean")))

    ent_dtw = _safe_float(src.get("entropy_dtw_mean", src.get("entropy_dtw")))
    pol_dtw = _safe_float(src.get("polarization_dtw_mean", src.get("polarization_dtw")))
    ent_corr = _safe_float(src.get("corr_entropy_mean", src.get("corr_entropy", src.get("entropy_pearson_mean", src.get("entropy_pearson")))))
    pol_corr = _safe_float(
        src.get("corr_polarization_mean", src.get("corr_polarization", src.get("polarization_pearson_mean", src.get("polarization_pearson"))))
    )

    def _mean2(a: float, b: float) -> float:
        if a != a and b != b:  # both nan
            return float("nan")
        if a != a:
            return float(b)
        if b != b:
            return float(a)
        return 0.5 * (float(a) + float(b))

    dtw_mean = _mean2(ent_dtw, pol_dtw)
    corr_mean = _mean2(ent_corr, pol_corr)

    z_steps = src.get("z_eval_steps_mean", src.get("z_eval_steps"))
    n_stages = src.get("n_stages_mean", src.get("n_stages"))
    timing = _get_nested(results, "timing") if isinstance(_get_nested(results, "timing"), dict) else {}

    return {
        "_metric_source": src_name,
        "kl": kl,
        "js": js,
        "dtw_entropy": ent_dtw,
        "dtw_polarization": pol_dtw,
        "dtw_mean": dtw_mean,
        "corr_entropy": ent_corr,
        "corr_polarization": pol_corr,
        "corr_mean": corr_mean,
        "z_eval_steps": int(z_steps) if isinstance(z_steps, (int, float)) and float(z_steps) == float(z_steps) else "",
        "n_stages": float(n_stages) if isinstance(n_stages, (int, float)) else "",
        "episode_time_sec_mean": _safe_float(timing.get("episode_time_sec_mean")) if isinstance(timing, dict) else float("nan"),
        "episode_time_sec_std": _safe_float(timing.get("episode_time_sec_std")) if isinstance(timing, dict) else float("nan"),
    }


def _run_one(
    *,
    python: str,
    paper_eval_py: str,
    config: str,
    ckpt: str,
    episodes: int,
    seed: int,
    cpu: bool,
    stagewise_source: str,
    out_json: str,
    out_csv: str,
) -> None:
    cmd = [
        python,
        paper_eval_py,
        "--config",
        config,
        "--ckpt",
        ckpt,
        "--episodes",
        str(int(episodes)),
        "--seed",
        str(int(seed)),
        "--out_json",
        out_json,
        "--out_csv",
        out_csv,
        "--stagewise_source",
        stagewise_source,
    ]
    if bool(cpu):
        cmd.append("--cpu")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"[distill_eval] paper_eval failed for ckpt={ckpt}\n--- output ---\n{p.stdout}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--stagewise_source", type=str, default="all_mean", choices=["last", "all_mean"])
    ap.add_argument(
        "--run",
        type=str,
        action="append",
        default=[],
        help="Repeatable: NAME=CKPT_PATH (e.g., --run Final=/path --run s3a=/path2 ...)",
    )
    ap.add_argument("--out_dir", type=str, default="./distill_eval_runs", help="Directory to store per-run paper_eval outputs")
    ap.add_argument("--out_csv", type=str, default="./distill_eval_summary.csv")
    ap.add_argument("--out_json", type=str, default="./distill_eval_summary.json")
    ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to run paper_eval.py (use HiSim env python). Defaults to current interpreter.",
    )
    args = ap.parse_args()

    runs = [_parse_run(s) for s in (args.run or [])]
    if not runs:
        raise SystemExit("No runs provided. Use --run Final=/path --run s3a=/path ...")

    paper_eval_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "paper_eval.py"))
    if not os.path.exists(paper_eval_py):
        raise SystemExit(f"paper_eval.py not found at: {paper_eval_py}")

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    per_run: Dict[str, Any] = {}
    for r in runs:
        run_slug = r.name.replace(" ", "_")
        run_json = str(out_dir / f"{run_slug}.json")
        run_csv = str(out_dir / f"{run_slug}_stagewise.csv")

        _run_one(
            python=str(args.python),
            paper_eval_py=paper_eval_py,
            config=str(args.config),
            ckpt=str(r.ckpt),
            episodes=int(args.episodes),
            seed=int(args.seed),
            cpu=bool(args.cpu),
            stagewise_source=str(args.stagewise_source),
            out_json=run_json,
            out_csv=run_csv,
        )

        with open(run_json, "r", encoding="utf-8") as f:
            res = json.load(f)
        m = _extract_metrics(res if isinstance(res, dict) else {})

        row = {
            "name": r.name,
            "ckpt": str(r.ckpt),
            **m,
            "run_json": run_json,
            "run_stagewise_csv": run_csv,
        }
        rows.append(row)
        per_run[r.name] = row

        print(
            f"[OK] {r.name}: KL={row['kl']:.6f} JS={row['js']:.6f} "
            f"DTW(mean)={row['dtw_mean']:.4f} Corr(mean)={row['corr_mean']:.4f} "
            f"(src={row['_metric_source']})"
        )

    out_csv = Path(str(args.out_csv)).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "ckpt",
        "kl",
        "js",
        "dtw_entropy",
        "dtw_polarization",
        "dtw_mean",
        "corr_entropy",
        "corr_polarization",
        "corr_mean",
        "z_eval_steps",
        "n_stages",
        "episode_time_sec_mean",
        "episode_time_sec_std",
        "_metric_source",
        "run_json",
        "run_stagewise_csv",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"[OK] Wrote summary CSV: {str(out_csv)}")

    out_json = Path(str(args.out_json)).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"runs": per_run, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote summary JSON: {str(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


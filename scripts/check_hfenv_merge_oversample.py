#!/usr/bin/env python3
"""
Quick sanity-check for Stage-1/2 offline datasets:
- Merge multiple HuggingFace datasets (local save_to_disk dirs) via hf_dataset_path list
- Apply train-only oversampling for an imbalanced stance label (boxed id in answer)

Usage (example):
  python ECON/scripts/check_hfenv_merge_oversample.py \
    --paths /home/zhuyinzhou/MAS/ECON/data/stage_1_dataset_metoo_e1 \
            /home/zhuyinzhou/MAS/ECON/data/stage_1_dataset_metoo_e2 \
    --split train --filter core --oversample --target 0.05

Then run again with --split test to confirm oversampling does NOT apply.
"""

import argparse
import os
import re
import sys
from collections import Counter


def _count_boxed_labels(dataset_list, answer_field: str = "answer"):
    boxed = re.compile(r"\\boxed\{\s*([-+]?\d+)\s*\}")
    c = Counter()
    missing = 0
    for s in dataset_list:
        a = s.get(answer_field, "")
        if not isinstance(a, str):
            missing += 1
            continue
        m = boxed.search(a)
        if not m:
            missing += 1
            continue
        try:
            c[int(m.group(1))] += 1
        except Exception:
            missing += 1
    return c, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True, help="One or more hf_dataset_path entries (local dirs or hub names).")
    ap.add_argument("--split", default="train", help="Dataset split: train/validation/test")
    ap.add_argument("--filter", default="core", choices=["core", "noncore", "all"], help="Filter is_core_user")
    ap.add_argument("--n-actions", type=int, default=3)
    ap.add_argument("--oversample", action="store_true")
    ap.add_argument("--label-id", type=int, default=1)
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--max-mult", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Make sure ECON/ is on sys.path so we can import `src.*`
    econ_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if econ_dir not in sys.path:
        sys.path.insert(0, econ_dir)
    from src.envs.huggingface_dataset_env import HuggingFaceDatasetEnv  # type: ignore

    filt = None
    if args.filter == "core":
        filt = "core"
    elif args.filter == "noncore":
        filt = "noncore"
    else:
        filt = None

    env = HuggingFaceDatasetEnv(
        hf_dataset_path=list(args.paths),
        dataset_split=str(args.split),
        question_field_name="question",
        answer_field_name="answer",
        dataset_streaming=False,
        use_random_sampling=False,
        use_dataset_episode=False,
        filter_is_core_user=filt,
        n_actions=int(args.n_actions),
        oversample_enabled=bool(args.oversample),
        oversample_only_train=True,
        oversample_label_id=int(args.label_id),
        oversample_target_ratio=float(args.target),
        oversample_seed=int(args.seed),
        oversample_max_multiplier=int(args.max_mult),
        verbose_step_logging=False,
    )

    counts, missing = _count_boxed_labels(env.dataset_list, answer_field="answer")
    total = sum(counts.values())
    print(f"split={args.split} filter={args.filter} num_samples={env.num_samples}")
    print(f"label_counts={dict(counts)} total_boxed={total} missing_boxed={missing}")
    if total > 0 and int(args.label_id) in counts:
        print(f"label{args.label_id}_ratio={counts[int(args.label_id)]/total:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick sanity-check for Stage-1/2 offline datasets:
- Merge multiple HuggingFace datasets (local save_to_disk dirs) via hf_dataset_path list
- Apply train-only oversampling for an imbalanced stance label (boxed id in answer)

Usage (example):
  python ECON/scripts/check_hfenv_merge_oversample.py \
    --paths /home/zhuyinzhou/MAS/ECON/data/stage_1_dataset_metoo_e1 \
            /home/zhuyinzhou/MAS/ECON/data/stage_1_dataset_metoo_e2 \
    --split train --filter core --oversample --target 0.05

Then run again with --split test to confirm oversampling does NOT apply.
"""

import argparse
import os
import re
import sys
from collections import Counter


def _count_boxed_labels(dataset_list, answer_field: str = "answer"):
    boxed = re.compile(r"\\boxed\{\s*([-+]?\d+)\s*\}")
    c = Counter()
    missing = 0
    for s in dataset_list:
        a = s.get(answer_field, "")
        if not isinstance(a, str):
            missing += 1
            continue
        m = boxed.search(a)
        if not m:
            missing += 1
            continue
        try:
            c[int(m.group(1))] += 1
        except Exception:
            missing += 1
    return c, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True, help="One or more hf_dataset_path entries (local dirs or hub names).")
    ap.add_argument("--split", default="train", help="Dataset split: train/validation/test")
    ap.add_argument("--filter", default="core", choices=["core", "noncore", "all"], help="Filter is_core_user")
    ap.add_argument("--n-actions", type=int, default=3)
    ap.add_argument("--oversample", action="store_true")
    ap.add_argument("--label-id", type=int, default=1)
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--max-mult", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Make sure ECON/ is on sys.path so we can import `src.*`
    econ_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if econ_dir not in sys.path:
        sys.path.insert(0, econ_dir)
    from src.envs.huggingface_dataset_env import HuggingFaceDatasetEnv  # type: ignore

    filt = None
    if args.filter == "core":
        filt = "core"
    elif args.filter == "noncore":
        filt = "noncore"
    else:
        filt = None

    env = HuggingFaceDatasetEnv(
        hf_dataset_path=list(args.paths),
        dataset_split=str(args.split),
        question_field_name="question",
        answer_field_name="answer",
        dataset_streaming=False,
        use_random_sampling=False,
        use_dataset_episode=False,
        filter_is_core_user=filt,
        n_actions=int(args.n_actions),
        oversample_enabled=bool(args.oversample),
        oversample_only_train=True,
        oversample_label_id=int(args.label_id),
        oversample_target_ratio=float(args.target),
        oversample_seed=int(args.seed),
        oversample_max_multiplier=int(args.max_mult),
        verbose_step_logging=False,
    )

    counts, missing = _count_boxed_labels(env.dataset_list, answer_field="answer")
    total = sum(counts.values())
    print(f"split={args.split} filter={args.filter} num_samples={env.num_samples}")
    print(f"label_counts={dict(counts)} total_boxed={total} missing_boxed={missing}")
    if total > 0 and int(args.label_id) in counts:
        print(f"label{args.label_id}_ratio={counts[int(args.label_id)]/total:.4f}")


if __name__ == "__main__":
    main()


"""
Quick sanity-check for Stage-1/2 offline datasets:
- Merge multiple HuggingFace datasets (local save_to_disk dirs) via hf_dataset_path list
- Apply train-only oversampling for an imbalanced stance label (boxed id in answer)

Usage (example):
  python ECON/scripts/check_hfenv_merge_oversample.py \
    --paths /home/zhuyinzhou/MAS/ECON/data/stage_1_dataset_metoo_e1 \
            /home/zhuyinzhou/MAS/ECON/data/stage_1_dataset_metoo_e2 \
    --split train --filter core --oversample --target 0.05

Then run again with --split test to confirm oversampling does NOT apply.
"""

import argparse
import re
from collections import Counter


def _count_boxed_labels(dataset_list, answer_field: str = "answer"):
+    boxed = re.compile(r"\\boxed\{\s*([-+]?\d+)\s*\}")
    c = Counter()
    missing = 0
    for s in dataset_list:
        a = s.get(answer_field, "")
        if not isinstance(a, str):
            missing += 1
            continue
        m = boxed.search(a)
        if not m:
            missing += 1
            continue
        try:
            c[int(m.group(1))] += 1
        except Exception:
            missing += 1
    return c, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True, help="One or more hf_dataset_path entries (local dirs or hub names).")
    ap.add_argument("--split", default="train", help="Dataset split: train/validation/test")
    ap.add_argument("--filter", default="core", choices=["core", "noncore", "all"], help="Filter is_core_user")
    ap.add_argument("--n-actions", type=int, default=3)
    ap.add_argument("--oversample", action="store_true")
    ap.add_argument("--label-id", type=int, default=1)
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--max-mult", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Local import to keep this script runnable from repo root
    from ECON.src.envs.huggingface_dataset_env import HuggingFaceDatasetEnv  # type: ignore

    filt = None
    if args.filter == "core":
        filt = "core"
    elif args.filter == "noncore":
        filt = "noncore"
    else:
        filt = None

    env = HuggingFaceDatasetEnv(
        hf_dataset_path=list(args.paths),
        dataset_split=str(args.split),
        question_field_name="question",
        answer_field_name="answer",
        dataset_streaming=False,
        use_random_sampling=False,
        use_dataset_episode=False,
        filter_is_core_user=filt,
        n_actions=int(args.n_actions),
        oversample_enabled=bool(args.oversample),
        oversample_only_train=True,
        oversample_label_id=int(args.label_id),
        oversample_target_ratio=float(args.target),
        oversample_seed=int(args.seed),
        oversample_max_multiplier=int(args.max_mult),
        verbose_step_logging=False,
    )

    counts, missing = _count_boxed_labels(env.dataset_list, answer_field="answer")
    total = sum(counts.values())
    print(f"split={args.split} filter={args.filter} num_samples={env.num_samples}")
    print(f"label_counts={dict(counts)} total_boxed={total} missing_boxed={missing}")
    if total > 0 and int(args.label_id) in counts:
        print(f"label{args.label_id}_ratio={counts[int(args.label_id)]/total:.4f}")


if __name__ == "__main__":
    main()


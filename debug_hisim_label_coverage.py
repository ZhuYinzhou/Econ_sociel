#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速诊断 HiSim -> ECON belief 数据处理是否把某些标签“处理没了”。

它会统计：
1) macro 数据中 tweet-level stance_label 的总体分布
2) macro 数据中 user-stage majority label 的总体分布
3) 在给定 neighbor_mode(followers/following) + max_neighbor_users 下，
   neighbor_tp1 的 target_distribution 是否包含 Oppose(1)，以及最终 mode target_id 的分布

用法示例：
  python3 ECON/debug_hisim_label_coverage.py \
    --hisim-data-root /data/zhuyinzhou/HiSim/data \
    --topic metoo --event e2 \
    --neighbor-mode followers --max-neighbor-users 50

如果你怀疑方向错了，改成：
  --neighbor-mode following
"""

import argparse
import pickle
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


def _as_mapping(obj: Any) -> Optional[Dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            d = obj.to_dict()
            return d if isinstance(d, dict) else None
        except Exception:
            return None
    return None


def _normalize_label(s: Any) -> str:
    if s is None:
        return ""
    ss = str(s).strip()
    if not ss:
        return ""
    low = ss.lower().strip()
    mapping = {
        "neutral": "Neutral",
        "none": "Neutral",
        "unknown": "Neutral",
        "oppose": "Oppose",
        "against": "Oppose",
        "support": "Support",
        "favor": "Support",
    }
    return mapping.get(low, ss)


def _extract_label(tweet: Dict[str, Any]) -> Optional[str]:
    for k in ("stance_label", "stance", "label", "content_label", "behavior"):
        v = tweet.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s != "":
            lab = _normalize_label(s)
            return lab if lab else None
    return None


def _stage_label(stage_items: List[Any]) -> Optional[str]:
    labs: List[str] = []
    for it in stage_items:
        d = _as_mapping(it)
        if not d:
            continue
        lab = _extract_label(d)
        if lab:
            labs.append(lab)
    if not labs:
        return None
    return Counter(labs).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hisim-data-root", type=str, default="/data/zhuyinzhou/HiSim/data")
    ap.add_argument("--topic", type=str, required=True)
    ap.add_argument("--event", type=str, required=True)  # e1/e2/p1...
    ap.add_argument("--neighbor-mode", type=str, default="followers", choices=["followers", "following"])
    ap.add_argument("--max-neighbor-users", type=int, default=50)
    args = ap.parse_args()

    macro_path = os.path.join(args.hisim_data_root, "hisim_with_tweet", f"{args.topic}_macro_{args.event}.pkl")
    follower_path = os.path.join(args.hisim_data_root, "user_data", args.topic, "follower_dict.json")

    if not os.path.exists(macro_path):
        raise FileNotFoundError(macro_path)
    if not os.path.exists(follower_path):
        raise FileNotFoundError(follower_path)

    import json
    with open(follower_path, "r", encoding="utf-8") as f:
        follower_dict = json.load(f)
    if not isinstance(follower_dict, dict):
        raise RuntimeError("follower_dict.json not a dict")

    with open(macro_path, "rb") as f:
        macro = pickle.load(f)
    if not isinstance(macro, dict):
        raise RuntimeError("macro pickle not a dict")

    # Build following dict if requested
    following_dict: Dict[str, List[str]] = defaultdict(list)
    if args.neighbor_mode == "following":
        for u, followers in follower_dict.items():
            if not isinstance(followers, list):
                continue
            for fu in followers:
                if fu:
                    following_dict[str(fu)].append(str(u))

    tweet_counter = Counter()
    user_stage_counter = Counter()
    # neighbor_tp1 stats
    tp1_dist_has_oppose = 0
    tp1_total = 0
    tp1_mode_counter = Counter()
    tp1_dist_counter_sum = Counter()  # sum of neighbor labels across all samples

    for user, ud in macro.items():
        if not isinstance(ud, dict):
            continue
        # tweet-level labels
        for t, stage in ud.items():
            if not isinstance(stage, list):
                continue
            for it in stage:
                d = _as_mapping(it)
                if not d:
                    continue
                lab = _extract_label(d)
                if lab:
                    tweet_counter[lab] += 1

        # stage majority label
        for t in range(0, 14):
            st = ud.get(t) or []
            if not isinstance(st, list) or not st:
                continue
            labt = _stage_label(st)
            if labt:
                user_stage_counter[labt] += 1

        # neighbor_tp1 targets (t in 0..12)
        for t in range(0, 13):
            if (t not in ud) or ((t + 1) not in ud):
                continue
            # get neighbors
            if args.neighbor_mode == "following":
                neighbors = list(following_dict.get(str(user), []))
            else:
                neighbors = list(follower_dict.get(str(user), [])) if follower_dict else []
            neighbors = [nb for nb in neighbors if nb in macro]
            if args.max_neighbor_users > 0 and len(neighbors) > args.max_neighbor_users:
                neighbors = neighbors[: args.max_neighbor_users]
            if not neighbors:
                continue

            # distribution over neighbor stage labels at t+1
            dist = Counter()
            for nb in neighbors:
                nb_ud = macro.get(nb, {})
                if not isinstance(nb_ud, dict):
                    continue
                nb_tp1 = nb_ud.get(t + 1) or []
                if not isinstance(nb_tp1, list) or not nb_tp1:
                    continue
                nlab = _stage_label(nb_tp1)
                if nlab:
                    dist[nlab] += 1

            if not dist:
                continue
            tp1_total += 1
            tp1_dist_counter_sum.update(dist)
            if dist.get("Oppose", 0) > 0:
                tp1_dist_has_oppose += 1
            tp1_mode = dist.most_common(1)[0][0]
            tp1_mode_counter[tp1_mode] += 1

    def _fmt(counter: Counter) -> str:
        tot = sum(counter.values())
        if tot <= 0:
            return "{}"
        parts = []
        for k in ("Neutral", "Oppose", "Support"):
            parts.append(f"{k}:{counter.get(k,0)}({counter.get(k,0)/tot:.3f})")
        return ", ".join(parts) + f" | total={tot}"

    print("=== tweet-level stance_label distribution (macro) ===")
    print(_fmt(tweet_counter))
    print("\n=== user-stage majority label distribution (macro) ===")
    print(_fmt(user_stage_counter))
    print("\n=== neighbor_tp1 target stats ===")
    print(f"neighbor_mode={args.neighbor_mode} max_neighbor_users={args.max_neighbor_users}")
    print(f"tp1_total_samples={tp1_total}")
    print(f"samples_with_any_oppose_in_dist={tp1_dist_has_oppose} ({(tp1_dist_has_oppose/max(1,tp1_total)):.3f})")
    print("mode(target_label) distribution:")
    print(_fmt(tp1_mode_counter))
    print("sum of neighbor labels across all samples (dist mass):")
    print(_fmt(tp1_dist_counter_sum))


if __name__ == "__main__":
    main()


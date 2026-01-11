#!/usr/bin/env python3
"""
将 HiSim 的 **macro** 数据（例如 `metoo_macro_e1.pkl` / `metoo_macro_e2.pkl`）转换为
“信念网络（belief network）”可用的监督学习数据集。

你给定的 Pipeline（Algorithm）在本脚本里的落地实现（针对每个 topic 的 e1/e2 宏观数据）：

- 遍历 event ∈ {e1, e2}
- 遍历 user ∈ 1000 users
- 遍历 stage t ∈ [0, 12]（要求 t+1 存在且非空）
- 构造 UserState(u, t)
  - persona（来自 `HiSim/data/user_data/<topic>/role_desc_v2_clean.json`，若缺失则为空）
  - self tweets（stage t 的文本，或可选带历史摘要）
  - self stance（stage t 的 `stance_label` 多数投票/回退到首条）
- 聚合 NeighborState(u, t)
  - 邻居来自 `follower_dict.json`（该文件通常是 user -> followers 列表；覆盖用户可能少于 1000）
  - 聚合邻居在 stage t 的 tweet 文本与 stance 统计（若邻居在宏观数据中存在）
- 构造 BeliefInput(u, t)：将以上信息拼成一个 prompt
- 用 t+1 的真实数据构造 BeliefTarget：stage t+1 的 stance_label（多数投票）
- 存为一个训练样本：字段包含 `question` / `answer`（同时保留 `target_label` 等元信息）

输出：
- 使用 `datasets.DatasetDict.save_to_disk(output_dir)` 保存为本地 HuggingFace 数据集
- 额外保存 `label2id.json`/`stats.json`，便于训练/分析

说明：
- 宏观 pkl 的 tweet item 通常是 `pandas.Series`（可 `.to_dict()`），字段常见：
  `rawContent`, `stance_label`, `date`, `user` 等。
- 本脚本默认把 `answer` 做成 `\\boxed{<label_id>}`，兼容 ECON 现有的数字评估逻辑；
  同时也保留原始 `target_label` 字段用于更通用的训练。
"""

import os
import sys
import pickle
import json
import glob
import hashlib
import re
import random
from datetime import datetime, timezone
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, Literal

try:
    from datasets import Dataset, DatasetDict  # type: ignore
    _HAS_DATASETS = True
except Exception:
    Dataset = None  # type: ignore
    DatasetDict = None  # type: ignore
    _HAS_DATASETS = False


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stable_hash_to_bucket(s: str, buckets: int = 10) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % buckets


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_mapping(obj: Any) -> Optional[Dict[str, Any]]:
    """
    将 tweet item 尽量转成 dict。
    宏观 pkl 的 item 常见为 pandas.Series（有 to_dict），也可能本身就是 dict。
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            d = obj.to_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            return None
    return None


def _normalize_label(s: Any) -> str:
    """
    与 `ECON/src/envs/hisim_social_env.py::_normalize_label` 对齐：
    把可能形态的 stance label 归一化到稳定字符串。
    """
    if s is None:
        return ""
    ss = str(s).strip()
    if not ss:
        return ""
    ss2 = re.sub(r"\s+", " ", ss).strip()
    low = ss2.lower()
    mapping = {
        "neutral": "Neutral",
        "none": "Neutral",
        "unknown": "Neutral",
        "oppose": "Oppose",
        "against": "Oppose",
        "support": "Support",
        "favor": "Support",
    }
    return mapping.get(low, ss2)


def _extract_text(tweet: Dict[str, Any]) -> str:
    for k in ("rawContent", "content", "text", "full_text", "message"):
        v = tweet.get(k)
        if v:
            return str(v).strip()
    return ""

def _infer_action_type_from_macro_tweet(tweet: Dict[str, Any]) -> str:
    """
    Infer a 5-way action_type from HiSim macro tweet-like dict.
    Aligns with hisim_social_env.action_types:
      ["post","retweet","reply","like","do_nothing"]

    Notes:
    - Macro data usually cannot observe "like" or "do_nothing" explicitly.
    - We map quotedTweet to retweet (user request: quote ≈ retweet for action-type).
    """
    # If there is no text, treat as do_nothing
    txt = _extract_text(tweet)
    if not txt:
        return "do_nothing"

    # Prefer explicit structure
    try:
        if tweet.get("retweetedTweet") is not None:
            return "retweet"
        # quotedTweet 表示引用 -> 映射为 retweet（合并到转发类）
        if tweet.get("quotedTweet") is not None:
            return "retweet"
    except Exception:
        pass

    s = str(txt).lstrip()
    # Fallback to common Twitter conventions
    if s.startswith("RT @") or s.startswith("RT@"):
        return "retweet"
    if s.startswith("@"):
        return "reply"
    return "post"


def _action_counts_from_stage(stage_items: List[Any]) -> List[int]:
    """Return action counts in fixed order [post,retweet,reply,like,do_nothing]."""
    order = ["post", "retweet", "reply", "like", "do_nothing"]
    idx = {a: i for i, a in enumerate(order)}
    counts = [0, 0, 0, 0, 0]
    for it in stage_items:
        d = _as_mapping(it)
        if not d:
            continue
        at = _infer_action_type_from_macro_tweet(d)
        counts[idx.get(at, 0)] += 1
    return counts


def _counts_to_ratio(counts: List[int]) -> List[float]:
    tot = float(sum(int(x) for x in counts))
    if tot <= 0:
        return [0.0 for _ in counts]
    return [float(x) / tot for x in counts]


def _stance_id_from_label(label: Optional[str], label2id: Dict[str, int]) -> int:
    if not label:
        return -1
    lab = _normalize_label(label)
    return int(label2id.get(lab, -1))


def _action_type_id_from_counts(action_counts: List[int]) -> int:
    """
    Map action counts to a single "mode" action id aligned to:
      0=post,1=retweet,2=reply,3=like,4=do_nothing
    Returns -1 if no actions observed.
    """
    if not action_counts or sum(int(x) for x in action_counts) <= 0:
        return -1
    try:
        return int(max(range(len(action_counts)), key=lambda i: int(action_counts[i])))
    except Exception:
        return -1


def _neighbor_recent_items(
    macro: Dict[str, Any],
    neighbors: List[str],
    t: int,
    k_recent: int,
) -> List[Dict[str, Any]]:
    """
    Collect neighbor stage-t tweet items (dict-like), sort by date desc, and return top-k.
    If k_recent<=0, return all items.
    """
    items: List[Tuple[Optional[datetime], Dict[str, Any]]] = []
    for nb in neighbors:
        nb_dict = macro.get(nb)
        if not isinstance(nb_dict, dict):
            continue
        stage = nb_dict.get(t) or []
        if not isinstance(stage, list) or not stage:
            continue
        for it in stage:
            d = _as_mapping(it)
            if not d:
                continue
            dt = _parse_datetime(d.get("date") or d.get("time") or d.get("current_time"))
            items.append((dt, d))
    # sort: dt present first, then by dt desc
    items.sort(key=lambda x: (x[0] is None, -(x[0].timestamp() if x[0] else 0.0)))
    out = [d for _dt, d in items]
    if k_recent and k_recent > 0:
        out = out[: int(k_recent)]
    return out

def _infer_action_type_from_text(text: Optional[str]) -> str:
    """
    Heuristic action type inference aligned with hisim_social_env.action_types:
      ["post","retweet","reply","like","do_nothing"]
    Macro data usually only contains tweet text, so "like" is rarely observable.
    """
    t = (text or "").strip()
    if not t:
        return "do_nothing"
    low = t.lower()
    if low.startswith("rt @") or " rt @" in low:
        return "retweet"
    if low.startswith("@"):
        return "reply"
    return "post"

def _action_type_counts_from_texts(texts: List[str]) -> Dict[str, int]:
    counter = Counter()
    for txt in texts or []:
        at = _infer_action_type_from_text(txt)
        counter[at] += 1
    # ensure fixed keys
    out = {k: int(counter.get(k, 0)) for k in ["post", "retweet", "reply", "like", "do_nothing"]}
    return out


def _parse_datetime(s: Any) -> Optional[datetime]:
    """
    解析常见的时间字符串，尽量返回 timezone-aware datetime（UTC）。
    支持：
    - 2017-10-15T23:19:07.000Z
    - 2017-10-15T21:39:49
    - 2017-10-15 23:19:07+00:00
    """
    if s is None:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    ss = str(s).strip()
    if not ss:
        return None
    try:
        # normalize Z
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        # fromisoformat supports 'YYYY-MM-DDTHH:MM:SS(.ffffff)(+HH:MM)'
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    # try common fallback formats
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ss, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def _extract_micro_text(item: Dict[str, Any]) -> str:
    """
    micro 数据：优先用 gt_text（真实响应），其次从 tweet_page 提取内容。
    """
    gt = item.get("gt_text")
    if gt:
        return str(gt).strip()
    tp = item.get("tweet_page")
    if tp:
        return str(tp).strip()
    return ""


def _load_micro_items(micro_path: str) -> List[Dict[str, Any]]:
    """
    统一加载 micro 数据（目前仓库里主要是 .pkl，顶层为 list[dict]）。
    """
    if not micro_path or not os.path.exists(micro_path):
        return []
    if micro_path.endswith(".pkl"):
        with open(micro_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            # 兼容 dict[user]->list 的情况
            out: List[Dict[str, Any]] = []
            for u, items in data.items():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if isinstance(it, dict):
                        it2 = dict(it)
                        it2.setdefault("user", u)
                        out.append(it2)
            return out
        return []
    if micro_path.endswith(".json"):
        try:
            with open(micro_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            return []
        except Exception:
            return []
    return []


def _build_stage_time_windows_from_macro(
    macro: Dict[str, Any],
    sample_users: int = -1,
    max_items_per_user_stage: int = 2,
) -> Dict[int, Tuple[Optional[datetime], Optional[datetime]]]:
    """
    从 macro 数据估计每个 stage 的时间窗（min/max）。
    用于将 micro 的 current_time 对齐到 stage t。
    """
    users = list(macro.keys())
    if sample_users > 0:
        users = users[:sample_users]
    min_dt: Dict[int, Optional[datetime]] = {t: None for t in range(14)}
    max_dt: Dict[int, Optional[datetime]] = {t: None for t in range(14)}
    for u in users:
        u_dict = macro.get(u)
        if not isinstance(u_dict, dict):
            continue
        for t in range(14):
            stage = u_dict.get(t) or []
            if not isinstance(stage, list) or not stage:
                continue
            # sample a few items to reduce cost
            for it in stage[: max_items_per_user_stage if max_items_per_user_stage > 0 else len(stage)]:
                d = _as_mapping(it)
                if not d:
                    continue
                dt = _parse_datetime(d.get("date") or d.get("time"))
                if not dt:
                    continue
                if min_dt[t] is None or dt < min_dt[t]:
                    min_dt[t] = dt
                if max_dt[t] is None or dt > max_dt[t]:
                    max_dt[t] = dt
    return {t: (min_dt[t], max_dt[t]) for t in range(14)}


def _assign_micro_to_stages(
    micro_items: List[Dict[str, Any]],
    stage_windows: Dict[int, Tuple[Optional[datetime], Optional[datetime]]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    按 stage 时间窗把 micro items 分桶到 stage t。
    若某条 micro 没有时间或不落在任何窗，忽略（可在上层用随机 fallback 补足）。
    """
    buckets: Dict[int, List[Dict[str, Any]]] = {t: [] for t in range(14)}
    for it in micro_items:
        dt = _parse_datetime(it.get("current_time") or it.get("date") or it.get("time"))
        if not dt:
            continue
        for t, (lo, hi) in stage_windows.items():
            if lo is None or hi is None:
                continue
            if lo <= dt <= hi:
                buckets[t].append(it)
                break
    return buckets


def _extract_label(tweet: Dict[str, Any]) -> Optional[str]:
    for k in ("stance_label", "stance", "label", "content_label", "behavior"):
        v = tweet.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s != "":
            return _normalize_label(s)
    return None

def _neighbor_stage_label(macro, neighbors, t):
    labels = []
    for nb in neighbors:
        nb_stage = macro.get(nb, {}).get(t + 1, [])
        lab = _stage_label(nb_stage)
        if lab:
            labels.append(lab)
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def _neighbor_stage_label_dist(macro, neighbors, t):
    counter = Counter()
    for nb in neighbors:
        nb_stage = macro.get(nb, {}).get(t + 1, [])
        lab = _stage_label(nb_stage)
        if lab:
            counter[lab] += 1
    return dict(counter)
    

def _stage_label(stage_items: List[Any]) -> Optional[str]:
    """
    stage 内多个 tweet 的 stance_label 汇聚：
    - 优先多数投票
    - 若无可用标签，返回 None
    """
    labels: List[str] = []
    for it in stage_items:
        d = _as_mapping(it)
        if not d:
            continue
        lab = _extract_label(d)
        if lab:
            labels.append(_normalize_label(lab))
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def _stage_label_dist(stage_items: List[Any]) -> Dict[str, int]:
    """
    stage 内 stance_label 的分布（label -> count）。
    用于构造 target_distribution（比单一多数标签更稳定）。
    """
    counter: Counter = Counter()
    for it in stage_items:
        d = _as_mapping(it)
        if not d:
            continue
        lab = _extract_label(d)
        if lab:
            counter[_normalize_label(lab)] += 1
    return dict(counter)


def _stage_texts(stage_items: List[Any], max_tweets: int) -> List[str]:
    texts: List[str] = []
    for it in stage_items:
        d = _as_mapping(it)
        if not d:
            continue
        txt = _extract_text(d)
        if not txt:
            continue
        texts.append(txt)
        if max_tweets > 0 and len(texts) >= max_tweets:
            break
    return texts


def _find_macro_event_files(hisim_with_tweet_dir: str) -> Dict[str, Dict[str, str]]:
    """
    返回形如：
      { topic: { "e1": "/.../topic_macro_e1.pkl", "e2": "/.../topic_macro_e2.pkl", "p1": "/.../topic_macro_p1.pkl", ... } }
    """
    out: Dict[str, Dict[str, str]] = defaultdict(dict)
    # 支持多种宏观文件命名：*_macro_e1.pkl、*_macro_e2.pkl、*_macro_p1.pkl 等
    for path in glob.glob(os.path.join(hisim_with_tweet_dir, "*_macro_*.pkl")):
        base = os.path.basename(path)
        # e.g., metoo_macro_e1.pkl / blm_macro_p1.pkl
        if "_macro_" not in base:
            continue
        topic = base.split("_macro_")[0]
        event = base.split("_macro_")[1].split(".pkl")[0]  # "e1" / "e2" / "p1" ...
        if topic and event:
            out[topic][event] = path
    return dict(out)


@dataclass
class BuildArgs:
    topic: str
    event: str
    user: str
    t: int
    persona: str
    user_history: str
    self_texts: List[str]
    self_label_t: Optional[str]
    neighbor_texts: List[Tuple[str, str]]  # (neighbor_user, text)
    neighbor_label_counter: Dict[str, int]
    # action-type features aligned to ["post","retweet","reply","like","do_nothing"]
    self_action_counts: List[int]
    self_action_ratio: List[float]
    neighbor_action_counts: List[int]
    neighbor_action_ratio: List[float]
    population_texts: List[Tuple[str, str]]  # (user, text) aggregated from secondary users
    population_label_counter: Dict[str, int]
    label2id: Dict[str, int]
    is_core_user: bool
    # - neighbor_tp1: predict neighbors' stance at stage t+1 (belief about others)
    # - self_tp1: predict the user's OWN stance at stage t+1 (Scheme C for non-core users without network)
    target_mode: Literal["neighbor_tp1", "self_tp1"] = "neighbor_tp1"


def build_belief_input(args: BuildArgs, max_neighbor_lines: int = 80) -> str:
    """
    将 UserState/NeighborState 组装成 BeliefInput（作为 question）。
    目标：预测 user 在 t+1 的 stance_label（离散类别）。
    """
    label_lines = [f"{lab} -> {idx}" for lab, idx in sorted(args.label2id.items(), key=lambda x: x[1])]
    label_spec = "\n".join(label_lines)

    lines: List[str] = []
    lines.append("You are a belief-state predictor in a social media opinion dynamics setting.")
    lines.append(f"Topic: {args.topic}")
    lines.append(f"Event: {args.event}")
    lines.append(f"User: {args.user}")
    lines.append(f"Stage t: {args.t}")
    lines.append("")
    if args.persona:
        lines.append("User persona (profile):")
        lines.append(args.persona.strip())
        lines.append("")

    if args.user_history:
        lines.append("User historical posts / memory (observed):")
        lines.append(args.user_history.strip())
        lines.append("")

    if args.self_label_t:
        lines.append(f"User stance at stage t (observed): {args.self_label_t}")
        lines.append("")

    # Core user action-type summary at stage t (macro-inferred, aligned to 5-way action space)
    try:
        if args.self_action_counts:
            at_names = ["post", "retweet", "reply", "like", "do_nothing"]
            lines.append("User action-type distribution at stage t (observed, aggregated):")
            lines.append(", ".join([f"{n}:{int(args.self_action_counts[i])}" for i, n in enumerate(at_names)]))
            lines.append("")
    except Exception:
        pass

    if args.self_texts:
        lines.append("User posts at stage t (observed):")
        for i, txt in enumerate(args.self_texts[:20], 1):
            lines.append(f"- ({i}) {txt}")
        lines.append("")

    # Neighbor summary
    if args.neighbor_label_counter:
        top = sorted(args.neighbor_label_counter.items(), key=lambda x: -x[1])[:10]
        lines.append("Neighbor stance distribution at stage t (observed, aggregated):")
        lines.append(", ".join([f"{k}:{v}" for k, v in top]))
        lines.append("")

    try:
        if args.neighbor_action_counts:
            at_names = ["post", "retweet", "reply", "like", "do_nothing"]
            lines.append("Neighbor action-type distribution at stage t (observed, aggregated):")
            lines.append(", ".join([f"{n}:{int(args.neighbor_action_counts[i])}" for i, n in enumerate(at_names)]))
            lines.append("")
    except Exception:
        pass

    if args.neighbor_texts:
        lines.append("Neighbor posts at stage t (observed, aggregated):")
        # avoid too long contexts
        count = 0
        for nb, txt in args.neighbor_texts:
            lines.append(f"- [{nb}] {txt}")
            count += 1
            if max_neighbor_lines > 0 and count >= max_neighbor_lines:
                break
        lines.append("")

    # Population-level observation from secondary users (no per-user supervision)
    if args.population_label_counter:
        top = sorted(args.population_label_counter.items(), key=lambda x: -x[1])[:10]
        lines.append("Population-level stance distribution at stage t (secondary users, observed):")
        lines.append(", ".join([f"{k}:{v}" for k, v in top]))
        lines.append("")
    if args.population_texts:
        lines.append("Population-level posts at stage t (secondary users, observed):")
        for u, txt in args.population_texts[:max_neighbor_lines]:
            lines.append(f"- [{u}] {txt}")
        lines.append("")

    if getattr(args, "target_mode", "neighbor_tp1") == "self_tp1":
        lines.append("Task: Predict the user's OWN MOST LIKELY stance label at stage t+1.")
    else:
        lines.append("Task: Predict the MOST LIKELY stance label of the user's social neighbors at stage t+1. This represents the user's belief about how others will shift their stance.")
    lines.append("Valid labels:")
    lines.append(label_spec)
    lines.append("")
    lines.append("Return ONLY the label id in the format: \\boxed{<id>}")
    return "\n".join(lines)


def build_action_imitation_input(args: BuildArgs, max_neighbor_lines: int = 80) -> str:
    """
    Build an offline imitation-learning prompt for CORE-user action_type prediction.

    Target: predict the user's NEXT-stage (t+1) action_type id in {0..4}:
      0=post, 1=retweet, 2=reply, 3=like, 4=do_nothing
    """
    action_spec = "\n".join(
        [
            "post -> 0",
            "retweet -> 1",
            "reply -> 2",
            "like -> 3",
            "do_nothing -> 4",
        ]
    )

    lines: List[str] = []
    lines.append("You are an action-policy predictor in a social media user simulation setting.")
    lines.append(f"Topic: {args.topic}")
    lines.append(f"Event: {args.event}")
    lines.append(f"User: {args.user}")
    lines.append(f"Stage t: {args.t}")
    lines.append("")
    if args.persona:
        lines.append("User persona (profile):")
        lines.append(args.persona.strip())
        lines.append("")

    if args.user_history:
        lines.append("User historical posts / memory (observed):")
        lines.append(args.user_history.strip())
        lines.append("")

    if args.self_label_t:
        lines.append(f"User stance at stage t (observed): {args.self_label_t}")
        lines.append("")

    # User action distribution at stage t (observed)
    try:
        if args.self_action_counts:
            at_names = ["post", "retweet", "reply", "like", "do_nothing"]
            lines.append("User action-type distribution at stage t (observed, aggregated):")
            lines.append(", ".join([f"{n}:{int(args.self_action_counts[i])}" for i, n in enumerate(at_names)]))
            lines.append("")
    except Exception:
        pass

    if args.self_texts:
        lines.append("User posts at stage t (observed):")
        for i, txt in enumerate(args.self_texts[:20], 1):
            lines.append(f"- ({i}) {txt}")
        lines.append("")

    # Neighbor summary
    if args.neighbor_label_counter:
        top = sorted(args.neighbor_label_counter.items(), key=lambda x: -x[1])[:10]
        lines.append("Neighbor stance distribution at stage t (observed, aggregated):")
        lines.append(", ".join([f"{k}:{v}" for k, v in top]))
        lines.append("")

    try:
        if args.neighbor_action_counts:
            at_names = ["post", "retweet", "reply", "like", "do_nothing"]
            lines.append("Neighbor action-type distribution at stage t (observed, aggregated):")
            lines.append(", ".join([f"{n}:{int(args.neighbor_action_counts[i])}" for i, n in enumerate(at_names)]))
            lines.append("")
    except Exception:
        pass

    if args.neighbor_texts:
        lines.append("Neighbor posts at stage t (observed, aggregated):")
        count = 0
        for nb, txt in args.neighbor_texts:
            lines.append(f"- [{nb}] {txt}")
            count += 1
            if max_neighbor_lines > 0 and count >= max_neighbor_lines:
                break
        lines.append("")

    # Population-level observation (optional)
    if args.population_label_counter:
        top = sorted(args.population_label_counter.items(), key=lambda x: -x[1])[:10]
        lines.append("Population-level stance distribution at stage t (secondary users, observed):")
        lines.append(", ".join([f"{k}:{v}" for k, v in top]))
        lines.append("")
    if args.population_texts:
        lines.append("Population-level posts at stage t (secondary users, observed):")
        for u, txt in args.population_texts[:max_neighbor_lines]:
            lines.append(f"- [{u}] {txt}")
        lines.append("")

    lines.append("Task: Predict the user's MOST LIKELY action_type at stage t+1.")
    lines.append("Valid action types:")
    lines.append(action_spec)
    lines.append("")
    lines.append("Return ONLY the action type id in the format: \\boxed{<id>}")
    return "\n".join(lines)


def _build_example_from_states(
    build_args: BuildArgs,
    target_label: str,
    target_id: int,
    target_distribution: Dict[int, int],
) -> Dict[str, Any]:
    q = build_belief_input(build_args)
    # NOTE:
    # HuggingFace Datasets / pyarrow 不允许 map/dict 的 key 是 int（必须是 str/bytes）。
    # 这里把分布的 key 转成 str，同时额外提供 list 版本，方便后续训练代码使用。
    target_dist_str: Dict[str, int] = {str(k): int(v) for k, v in (target_distribution or {}).items()}
    target_dist_ids = sorted([int(k) for k in target_distribution.keys()]) if target_distribution else []
    target_dist_counts = [int(target_distribution[i]) for i in target_dist_ids] if target_distribution else []
    total = sum(target_distribution.values())
    target_distribution_prob = {
        str(k): v / total for k, v in target_distribution.items()
    } 
    return {
        "question": q,
        "answer": f"\\boxed{{{target_id}}}",
        "target_label": target_label,
        "target_id": int(target_id),
        # safe for HF datasets
        "target_distribution": target_dist_str,
        # easy-to-consume numeric form
        "target_distribution_ids": target_dist_ids,
        "target_distribution_counts": target_dist_counts,
        "topic": build_args.topic,
        "event": build_args.event,
        "user": build_args.user,
        "t": int(build_args.t),
        "self_label_t": build_args.self_label_t if build_args.self_label_t is not None else "",
        "is_core_user": bool(build_args.is_core_user),
        "target_distribution_prob": target_distribution_prob,
        # === structured action features (macro-inferred) ===
        # aligned to ["post","retweet","reply","like","do_nothing"]
        "self_action_counts_t": [int(x) for x in (build_args.self_action_counts or [0, 0, 0, 0, 0])],
        "self_action_ratio_t": [float(x) for x in (build_args.self_action_ratio or [0.0, 0.0, 0.0, 0.0, 0.0])],
        "neighbor_action_counts_t": [int(x) for x in (build_args.neighbor_action_counts or [0, 0, 0, 0, 0])],
        "neighbor_action_ratio_t": [float(x) for x in (build_args.neighbor_action_ratio or [0.0, 0.0, 0.0, 0.0, 0.0])],
    }


def build_label2id_from_macros(macro_paths: List[str]) -> Dict[str, int]:
    labels: Set[str] = set()
    for path in macro_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            continue
        for _, user_dict in data.items():
            if not isinstance(user_dict, dict):
                continue
            for _, stage_items in user_dict.items():
                if not isinstance(stage_items, list):
                    continue
                for it in stage_items:
                    d = _as_mapping(it)
                    if not d:
                        continue
                    lab = _extract_label(d)
                    if lab:
                        labels.add(_normalize_label(lab))
    if not labels:
        raise RuntimeError("未能从 macro 数据中抽取任何标签字段（stance_label 等）。")
    # 为了与 hisim_social_env 的实现对齐，这里默认强制 K=3，并优先使用 canonical 映射
    # Neutral=0, Oppose=1, Support=2
    canonical = {"Neutral": 0, "Oppose": 1, "Support": 2}
    if set(canonical.keys()).issubset(labels):
        return dict(canonical)

    # fallback: 稳定排序取前三类；不足则用 canonical 补齐
    picked = sorted(list(labels))[:3]
    if len(picked) < 3:
        picked = (picked + ["Neutral", "Oppose", "Support"])[:3]
    return {lab: i for i, lab in enumerate(picked)}


def convert_hisim_macro_to_belief_hf_dataset(
    hisim_data_root: str,
    output_dir: str,
    topics: Optional[List[str]] = None,
    events: Optional[List[str]] = None,
    neighbor_mode: str = "followers",
    user_scope: str = "all",
    max_users: int = -1,
    max_self_tweets: int = 8,
    max_neighbor_users: int = 50,
    max_neighbor_tweets_total: int = 120,
    neighbor_k_recent_tweets: int = 0,
    include_user_history: bool = True,
    max_user_history_chars: int = 2000,
    use_population_observation: bool = True,
    population_scope: str = "secondary",
    population_text_source: str = "macro",
    population_micro_user_scope: str = "all",
    population_micro_sampling: str = "time",
    stage_window_sample_users: int = -1,
    stage_window_max_items_per_user_stage: int = 2,
    max_population_tweets_total: int = 200,
    export_micro_user_sequences: bool = False,
    split_by_user: bool = True,
    force_k: int = 3,
    export_z_transition_dataset: bool = True,
    z_transition_out_dir: str = "",
    # z_transition population belief representation:
    # - "scalar": K=1 scalar z ∈ [-1,1] (Oppose=-1, Neutral=0, Support=+1) averaged over secondary users
    # - "dist":   K=3 distribution z ∈ Δ^3 aligned to label2id ids (Neutral/Oppose/Support)
    z_transition_population_mode: str = "scalar",
    noncore_target_mode: str = "self",
    # Core-user target definition:
    # - "self": predict the user's OWN stance at t+1 (label from user's stage_tp1)
    # - "neighbor": predict neighbors' stance at t+1 (label from neighbors' stage_tp1 majority)
    core_target_mode: str = "neighbor",
    # Neighbor list handling
    shuffle_neighbors_before_truncation: bool = True,
    neighbor_shuffle_seed: int = 0,
    # z-transition dataset conditioning:
    # - "population_only": current behavior (population texts only; no per-core conditioning)
    # - "core_user": build a z-transition sample per (core_user, t) using build_belief_input as question,
    #               and label with secondary-population z_t/z_{t+1}.
    z_transition_conditioning: str = "population_only",
    # === Stage4 (offline) action policy imitation dataset (core users only) ===
    export_action_imitation_dataset: bool = False,
    action_imitation_out_dir: str = "",
    # === Preview (auto-generated after export) ===
    export_preview: bool = True,
    preview_num_per_split: int = 3,
    preview_seed: int = 42,
    preview_max_chars: int = 600,
    preview_filename: str = "preview.jsonl",
) -> None:
    """
    主转换入口：扫描 hisim_data_root 下的宏观数据，生成信念网络训练样本，并保存为 HF dataset。
    """
    hisim_with_tweet_dir = os.path.join(hisim_data_root, "hisim_with_tweet")
    user_data_dir = os.path.join(hisim_data_root, "user_data")
    if not os.path.isdir(hisim_with_tweet_dir):
        raise FileNotFoundError(f"目录不存在: {hisim_with_tweet_dir}")
    if not os.path.isdir(user_data_dir):
        raise FileNotFoundError(f"目录不存在: {user_data_dir}")
    # 可能会在处理中途导出一些辅助文件（如 micro_user_sequences），这里提前创建输出目录
    _ensure_dir(output_dir)

    macro_map = _find_macro_event_files(hisim_with_tweet_dir)
    if not macro_map:
        raise RuntimeError(f"在 {hisim_with_tweet_dir} 下没有找到任何 *_macro_*.pkl")

    if topics:
        macro_map = {k: v for k, v in macro_map.items() if k in set(topics)}
        if not macro_map:
            raise RuntimeError(f"过滤 topics={topics} 后没有匹配到任何 *_macro_*.pkl")

    # 事件过滤：如果指定了 events，则只处理这些事件；否则使用该 topic 下存在的全部事件
    filtered: Dict[str, Dict[str, str]] = {}
    for topic, ev in macro_map.items():
        if not isinstance(ev, dict) or not ev:
            continue
        if events:
            keep = {e: p for e, p in ev.items() if e in set(events)}
        else:
            keep = dict(ev)
        if keep:
            filtered[topic] = keep
    if not filtered:
        raise RuntimeError(
            "未找到符合条件的 macro 数据。"
            "请检查 topic/event 是否存在，例如 metoo_macro_e1/e2.pkl 或 blm_macro_p1.pkl。"
        )

    all_macro_paths: List[str] = []
    for _, ev in filtered.items():
        all_macro_paths.extend(list(ev.values()))

    print("收集标签集合并构建 label2id...")
    label2id = build_label2id_from_macros(all_macro_paths)
    # 兼容参数：当前程序的社交环境强制 K=3，这里也保持一致
    if int(force_k) != 3:
        print(f"[WARN] force_k={force_k} 目前未实现可变 K；将继续使用 K=3 以对齐 hisim_social_env。")
    print(f"标签数(K=3): {len(label2id)} | labels: {list(label2id.keys())}")

    # split buckets (belief dataset)
    split_names = ("train", "validation", "test")
    split_examples: Dict[str, List[Dict[str, Any]]] = {k: [] for k in split_names}

    # split buckets (z-transition dataset)
    z_split_examples: Dict[str, List[Dict[str, Any]]] = {k: [] for k in split_names}

    # split buckets (action imitation dataset; core users only)
    a_split_examples: Dict[str, List[Dict[str, Any]]] = {k: [] for k in split_names}

    def _truncate_text(s: Any, n: int) -> str:
        try:
            ss = str(s) if s is not None else ""
        except Exception:
            ss = ""
        n = int(n)
        if n <= 0:
            return ss
        if len(ss) <= n:
            return ss
        return ss[:n] + " ..."

    def _write_preview_jsonl(out_dir: str, split_to_items: Dict[str, List[Dict[str, Any]]], *, tag: str) -> None:
        """
        Write a single preview file into out_dir with a few sampled items per split.
        Intended for quick human inspection of the exact data fed to the model.
        """
        if not bool(export_preview):
            return
        try:
            _ensure_dir(out_dir)
            fn = str(preview_filename or "preview.jsonl").strip() or "preview.jsonl"
            path = os.path.join(out_dir, fn)
            rng = random.Random(int(preview_seed))
            nps = max(1, int(preview_num_per_split))
            maxc = max(0, int(preview_max_chars))

            with open(path, "w", encoding="utf-8") as f:
                for split in ("train", "validation", "test"):
                    items = split_to_items.get(split) or []
                    if not isinstance(items, list) or len(items) == 0:
                        continue
                    k = min(nps, len(items))
                    idxs = list(range(len(items)))
                    rng.shuffle(idxs)
                    for i in idxs[:k]:
                        it = items[i]
                        if not isinstance(it, dict):
                            continue
                        rec = {
                            "_tag": str(tag),
                            "_split": str(split),
                            "_idx": int(i),
                            "topic": it.get("topic"),
                            "event": it.get("event"),
                            "user": it.get("user"),
                            "t": it.get("t", it.get("stage_t")),
                            "is_core_user": it.get("is_core_user"),
                            "answer": it.get("answer"),
                            "target_id": it.get("target_id", it.get("target_action_type_id")),
                            "target_label": it.get("target_label"),
                            "z_t": it.get("z_t"),
                            "z_target": it.get("z_target"),
                            "z_mask": it.get("z_mask"),
                            "question_preview": _truncate_text(it.get("question", ""), maxc),
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[preview] wrote {path}")
        except Exception as e:
            print(f"[preview] skipped due to error: {e}")

    stats = {
        "topics": list(filtered.keys()),
        "label2id_size": len(label2id),
        "num_examples": 0,
        "skipped_missing_stage": 0,
        "skipped_missing_target": 0,
        "skipped_missing_label": 0,
        # 更细粒度的跳过原因（便于定位是不是“邻居网络覆盖/方向”问题）
        "skipped_no_neighbors": 0,
        "skipped_neighbors_not_in_macro": 0,
        "skipped_neighbors_no_posts_tp1": 0,
        "skipped_neighbors_no_labels_tp1": 0,
        # core vs non-core
        "num_candidates_core": 0,
        "num_candidates_noncore": 0,
        "num_examples_core": 0,
        "num_examples_noncore": 0,
        "skipped_user_scope": 0,
        "core_user_count_role_desc": 0,
        "core_user_count_history": 0,
        "core_user_count_intersection": 0,
        "core_user_definition": "",
        "core_user_history_nonempty_files": 0,
        # population observation stats
        "population_scope": population_scope,
        "use_population_observation": bool(use_population_observation),
        "population_text_source": population_text_source,
        "population_micro_user_scope": population_micro_user_scope,
        "population_micro_sampling": population_micro_sampling,
        "max_population_tweets_total": int(max_population_tweets_total),
        "export_z_transition_dataset": bool(export_z_transition_dataset),
        "noncore_target_mode": str(noncore_target_mode),
        "core_target_mode": str(core_target_mode),
        # neighbor sampling options
        "shuffle_neighbors_before_truncation": bool(shuffle_neighbors_before_truncation),
        "neighbor_shuffle_seed": int(neighbor_shuffle_seed),
        # target-id histogram (helps detect missing classes like Oppose=1)
        "target_id_counts": {0: 0, 1: 0, 2: 0},
        # structured neighbor feature window
        "neighbor_k_recent_tweets": int(neighbor_k_recent_tweets),
    }

    z_stats = {
        "topics": list(filtered.keys()),
        "label2id_size": len(label2id),
        "num_examples": 0,
        "export_z_transition_dataset": bool(export_z_transition_dataset),
        "z_transition_population_mode": str(z_transition_population_mode),
        "z_transition_definition": "scalar: z in [-1,1] or dist: z in Δ^3; computed from SECONDARY users at each stage (per-user stage-majority label), aligned to label2id",
        "z_transition_conditioning": str(z_transition_conditioning),
    }

    a_stats = {
        "topics": list(filtered.keys()),
        "num_examples": 0,
        "num_examples_core": 0,
        "action_id_counts": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        "definition": "Predict CORE user's next-stage (t+1) action_type id from stage-t context; action_type in [post,retweet,reply,like,do_nothing].",
    }

    def _load_user_history_snippet(topic: str, user: str) -> str:
        """
        核心用户的个人历史（memory）来自 user_data/<topic>/<topic>_v2/<user>.txt
        该文件通常是按行 JSON（每行一个 tweet），这里抽取 rawContent 拼成一段简短文本。
        """
        if not include_user_history or max_user_history_chars <= 0:
            return ""
        key = f"{topic}:{user}"
        if key in history_cache:
            return history_cache[key]
        p = os.path.join(user_data_dir, topic, f"{topic}_v2", f"{user}.txt")
        if not os.path.exists(p):
            history_cache[key] = ""
            return ""
        texts: List[str] = []
        try:
            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 200:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    # each line is often JSON
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            rc = obj.get("rawContent") or obj.get("content") or obj.get("text")
                            if rc:
                                texts.append(str(rc).strip())
                        else:
                            texts.append(str(obj).strip())
                    except Exception:
                        texts.append(line)
        except Exception:
            history_cache[key] = ""
            return ""
        # take the last few as “recent memory”
        if len(texts) > 20:
            texts = texts[-20:]
        out = "\n".join([f"- {t}" for t in texts if t])
        if len(out) > max_user_history_chars:
            out = out[: max_user_history_chars].rstrip() + "..."
        history_cache[key] = out
        return out

    for topic, ev_paths in filtered.items():
        # load user data for this topic (if exists)
        role_desc_path = os.path.join(user_data_dir, topic, "role_desc_v2_clean.json")
        follower_dict_path = os.path.join(user_data_dir, topic, "follower_dict.json")
        role_desc: Dict[str, str] = {}
        follower_dict: Dict[str, List[str]] = {}
        if os.path.exists(role_desc_path):
            try:
                role_desc = _load_json(role_desc_path)
            except Exception as e:
                print(f"[WARN] 无法读取 role_desc: {role_desc_path} | {e}")
        if os.path.exists(follower_dict_path):
            try:
                follower_dict = _load_json(follower_dict_path)
            except Exception as e:
                print(f"[WARN] 无法读取 follower_dict: {follower_dict_path} | {e}")

        # === 核心用户判定（验证“主要/次要”标签）===
        # 经验上：核心用户 = 同时存在 persona(role_desc) 与个人历史(<topic>_v2/*.txt) 的用户集合
        hist_dir = os.path.join(user_data_dir, topic, f"{topic}_v2")
        hist_users: Set[str] = set()
        if os.path.isdir(hist_dir):
            for fp in glob.glob(os.path.join(hist_dir, "*.txt")):
                hist_users.add(os.path.splitext(os.path.basename(fp))[0])
        role_users = set(role_desc.keys())
        core_users = role_users.intersection(hist_users) if role_users and hist_users else set()
        # 记录验证统计（只记一次）
        stats["core_user_count_role_desc"] = max(stats["core_user_count_role_desc"], len(role_users))
        stats["core_user_count_history"] = max(stats["core_user_count_history"], len(hist_users))
        stats["core_user_count_intersection"] = max(stats["core_user_count_intersection"], len(core_users))
        # 更稳健的核心用户定义：优先使用 role_desc 覆盖的用户（通常就是核心 LLM 用户）
        # 如果 role_desc 缺失，则退化使用 history 文件名集合
        if role_users:
            core_users = set(role_users)
            stats["core_user_definition"] = "role_desc_keys"
        elif hist_users:
            core_users = set(hist_users)
            stats["core_user_definition"] = "history_file_stems"
        else:
            core_users = set()
            stats["core_user_definition"] = "none"
        # history 非空文件数量（用于判断“核心用户更全面”在该 topic 是否成立）
        if os.path.isdir(hist_dir):
            nonempty = 0
            for fp in glob.glob(os.path.join(hist_dir, "*.txt")):
                try:
                    if os.path.getsize(fp) > 0:
                        nonempty += 1
                except Exception:
                    pass
            stats["core_user_history_nonempty_files"] = max(stats.get("core_user_history_nonempty_files", 0), nonempty)

        # if need following mode, invert follower_dict
        following_dict: Dict[str, List[str]] = defaultdict(list)
        if neighbor_mode == "following" and follower_dict:
            for u, followers in follower_dict.items():
                if not isinstance(followers, list):
                    continue
                for f_u in followers:
                    if f_u:
                        following_dict[str(f_u)].append(str(u))

        # user history cache per topic (avoid repeated file IO)
        history_cache: Dict[str, str] = {}

        # 按该 topic 实际具备的事件列表迭代（例如 metoo: e1/e2；blm: p1）
        for event in sorted(ev_paths.keys()):
            macro_path = ev_paths[event]
            print(f"读取 macro: topic={topic} event={event} path={macro_path}")
            with open(macro_path, "rb") as f:
                macro = pickle.load(f)
            if not isinstance(macro, dict):
                print(f"[WARN] macro 顶层不是 dict，跳过: {macro_path}")
                continue

            # 加载 micro（用于增强 population_texts）
            micro_items: List[Dict[str, Any]] = []
            micro_path = os.path.join(hisim_with_tweet_dir, f"{topic}_micro.pkl")
            micro_path_json = os.path.join(hisim_with_tweet_dir, f"{topic}_micro.json")
            if population_text_source == "micro":
                micro_items = _load_micro_items(micro_path) or _load_micro_items(micro_path_json)
                # 过滤 micro 用户范围（注意：micro 数据不一定覆盖次要用户；默认 all 以提升多样性）
                if population_micro_user_scope != "all":
                    allowed: Set[str]
                    if population_micro_user_scope == "core":
                        allowed = set(core_users)
                    else:
                        allowed = set([u for u in macro.keys() if str(u) not in core_users])
                    micro_items = [it for it in micro_items if str(it.get("user", "")) in allowed]

                # 可选：导出 micro 用户序列，供后续 latent z 模型训练
                if export_micro_user_sequences and micro_items:
                    out_path = os.path.join(output_dir, f"{topic}_{event}_micro_user_sequences.jsonl")
                    by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                    for it in micro_items:
                        u = str(it.get("user", "unknown"))
                        by_user[u].append(it)
                    with open(out_path, "w", encoding="utf-8") as f:
                        for u, items in by_user.items():
                            rows = []
                            for x in items:
                                dt = _parse_datetime(x.get("current_time"))
                                rows.append(
                                    {
                                        "current_time": x.get("current_time"),
                                        "current_time_epoch": int(dt.timestamp()) if dt else None,
                                        "trigger_news": x.get("trigger_news", ""),
                                        "tweet_page": x.get("tweet_page", ""),
                                        "gt_text": x.get("gt_text", ""),
                                        "gt_msg_type": x.get("gt_msg_type", ""),
                                    }
                                )
                            # sort by time if possible
                            rows.sort(key=lambda r: (r["current_time_epoch"] is None, r["current_time_epoch"] or 0))
                            f.write(json.dumps({"user": u, "items": rows}, ensure_ascii=False) + "\n")

            # === 预计算 population-level observation（按 stage t 聚合次要用户）===
            # 只生成“群体观测”，不为次要用户生成监督样本。
            population_cache: Dict[int, Dict[str, Any]] = {}
            if use_population_observation:
                # stage windows (for micro time alignment)
                stage_windows = _build_stage_time_windows_from_macro(
                    macro,
                    sample_users=stage_window_sample_users,
                    max_items_per_user_stage=stage_window_max_items_per_user_stage,
                )
                micro_buckets = _assign_micro_to_stages(micro_items, stage_windows) if (population_text_source == "micro") else {}

                population_users: List[str]
                if population_scope == "all":
                    population_users = list(macro.keys())
                else:
                    # secondary users = not in core_users
                    population_users = [u for u in macro.keys() if str(u) not in core_users]

                for t in range(0, 14):
                    pop_label_counter: Counter = Counter()
                    pop_texts: List[Tuple[str, str]] = []
                    for u in population_users:
                        u_dict = macro.get(u)
                        if not isinstance(u_dict, dict):
                            continue
                        stage = u_dict.get(t) or []
                        if not isinstance(stage, list) or not stage:
                            continue
                        # labels: count all labeled tweets in this stage
                        for it in stage:
                            d = _as_mapping(it)
                            if not d:
                                continue
                            lab = _extract_label(d)
                            if lab:
                                pop_label_counter[lab] += 1
                            # texts: macro 文本抽样（仅当 population_text_source=macro）
                            if population_text_source == "macro":
                                if max_population_tweets_total > 0 and len(pop_texts) >= max_population_tweets_total:
                                    continue
                                txt = _extract_text(d)
                                if txt:
                                    pop_texts.append((str(u), txt))

                    # texts: micro 文本抽样（population_text_source=micro）
                    if population_text_source == "micro" and micro_items:
                        cand = micro_buckets.get(t, []) if population_micro_sampling == "time" else micro_items
                        # 打乱后截断，避免同一用户集中
                        if cand:
                            import random

                            random.shuffle(cand)
                            for it in cand:
                                if max_population_tweets_total > 0 and len(pop_texts) >= max_population_tweets_total:
                                    break
                                txt = _extract_micro_text(it)
                                if not txt:
                                    continue
                                pop_texts.append((str(it.get("user", "unknown")), txt))

                    population_cache[t] = {
                        "label_counter": dict(pop_label_counter),
                        "texts": pop_texts[: max_population_tweets_total if max_population_tweets_total > 0 else None],
                    }

            # === 可选：导出 z(t)->z(t+1) transition 数据集（用于训练 BeliefEncoder.population_update_head）===
            # 支持两种 z 表示：
            # - scalar: K=1 标量 z∈[-1,1]（Oppose=-1, Neutral=0, Support=+1），取 secondary users 的 stage-majority label 均值
            # - dist:   K=3 分布 z∈Δ^3（按 label2id 的 id 顺序），取 secondary users 的 stage-majority label 分布
            if export_z_transition_dataset:
                z_pop_mode = str(z_transition_population_mode or "scalar").strip().lower()
                if z_pop_mode not in ("scalar", "dist"):
                    z_pop_mode = "scalar"

                def _stage_scalar_over_users(users_list: List[str], stage_t: int) -> Tuple[float, int]:
                    # label -> scalar mapping
                    v_map = {"Oppose": -1.0, "Neutral": 0.0, "Support": 1.0}
                    total = 0
                    acc = 0.0
                    for uu in users_list:
                        ud = macro.get(uu)
                        if not isinstance(ud, dict):
                            continue
                        st_items = ud.get(stage_t) or []
                        if not isinstance(st_items, list) or not st_items:
                            continue
                        lab = _stage_label(st_items)
                        lab = _normalize_label(lab)
                        if not lab:
                            continue
                        total += 1
                        acc += float(v_map.get(str(lab), 0.0))
                    if total <= 0:
                        return 0.0, 0
                    z = acc / float(total)
                    return float(max(-1.0, min(1.0, z))), int(total)

                def _stage_dist_over_users(users_list: List[str], stage_t: int) -> Tuple[List[float], int]:
                    """
                    Compute a 3-way stance distribution over secondary users at stage_t.
                    We take each user's stage-majority label, map to label2id, then normalize counts.
                    """
                    K = 3
                    counts = [0 for _ in range(K)]
                    total = 0
                    for uu in users_list:
                        ud = macro.get(uu)
                        if not isinstance(ud, dict):
                            continue
                        st_items = ud.get(stage_t) or []
                        if not isinstance(st_items, list) or not st_items:
                            continue
                        lab = _stage_label(st_items)
                        lab = _normalize_label(lab)
                        if not lab:
                            continue
                        if lab not in label2id:
                            continue
                        try:
                            idx = int(label2id[lab])
                        except Exception:
                            continue
                        if 0 <= idx < K:
                            counts[idx] += 1
                            total += 1
                    if total <= 0:
                        # uniform but will be masked out by labeled_count=0
                        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 0
                    return [c / float(total) for c in counts], int(total)

                # secondary users = not in core_users
                secondary_users = [u for u in macro.keys() if str(u) not in core_users]
                z_cond = str(z_transition_conditioning or "population_only").strip().lower()
                for t in range(0, 13):
                    if z_pop_mode == "dist":
                        z_t, labeled_t = _stage_dist_over_users(secondary_users, t)
                        z_tp1, labeled_tp1 = _stage_dist_over_users(secondary_users, t + 1)
                    else:
                        z_t, labeled_t = _stage_scalar_over_users(secondary_users, t)
                        z_tp1, labeled_tp1 = _stage_scalar_over_users(secondary_users, t + 1)

                    if z_cond in ("core_user", "core", "conditioned", "cond"):
                        # Build a conditioned sample for EACH core user at stage t, but label by secondary population z(t+1).
                        for user in list(core_users):
                            ud = macro.get(user)
                            if not isinstance(ud, dict):
                                continue
                            stage_ut = ud.get(t) or []
                            stage_utp1 = ud.get(t + 1) or []
                            if not isinstance(stage_ut, list) or not isinstance(stage_utp1, list):
                                continue

                            persona = role_desc.get(str(user), "")
                            user_history = _load_user_history_snippet(topic, str(user))
                            self_label_t = _stage_label(stage_ut)
                            self_texts = _stage_texts(stage_ut, max_tweets=max_self_tweets)
                            # neighbor slice consistent with belief dataset
                            neighbors = list(follower_dict.get(str(user), [])) if follower_dict else []
                            neighbors = [nb for nb in neighbors if nb in macro]
                            if max_neighbor_users > 0 and len(neighbors) > max_neighbor_users:
                                neighbors = neighbors[:max_neighbor_users]
                            neighbor_texts: List[Tuple[str, str]] = []
                            neighbor_label_counter: Counter = Counter()
                            for nb in neighbors:
                                nb_dict = macro.get(nb)
                                if not isinstance(nb_dict, dict):
                                    continue
                                nb_stage = nb_dict.get(t) or []
                                if not isinstance(nb_stage, list) or not nb_stage:
                                    continue
                                nb_lab = _stage_label(nb_stage)
                                if nb_lab:
                                    neighbor_label_counter[nb_lab] += 1
                                nb_texts = _stage_texts(nb_stage, max_tweets=2)
                                for txt in nb_texts:
                                    if txt:
                                        neighbor_texts.append((str(nb), txt))
                                if max_neighbor_tweets_total > 0 and len(neighbor_texts) >= max_neighbor_tweets_total:
                                    break

                            pop = population_cache.get(t, {}) if use_population_observation else {}
                            pop_label_counter = pop.get("label_counter", {}) if isinstance(pop, dict) else {}
                            pop_texts = pop.get("texts", []) if isinstance(pop, dict) else []

                            bargs = BuildArgs(
                                topic=str(topic),
                                event=str(event),
                                user=str(user),
                                t=int(t),
                                persona=str(persona),
                                user_history=str(user_history),
                                self_texts=self_texts,
                                self_label_t=self_label_t,
                                neighbor_texts=neighbor_texts,
                                neighbor_label_counter=dict(neighbor_label_counter),
                                population_texts=pop_texts,
                                population_label_counter=pop_label_counter,
                                label2id=label2id,
                                is_core_user=True,
                                target_mode="neighbor_tp1",  # doesn't matter for z_transition; kept for prompt consistency
                            )
                            q = build_belief_input(bargs)

                            # structured conditioning fields
                            core_stance_id_t = int(label2id.get(str(self_label_t), -1)) if self_label_t else -1
                            core_action_type_t = _infer_action_type_from_text(self_texts[0] if self_texts else "")
                            action2id = {"post": 0, "retweet": 1, "reply": 2, "like": 3, "do_nothing": 4}
                            core_action_type_id_t = int(action2id.get(core_action_type_t, 4))
                            neigh_action_counts = _action_type_counts_from_texts([txt for _, txt in neighbor_texts][: max(0, int(max_neighbor_tweets_total))])

                            ex = {
                                "question": q,
                                "answer": "\\boxed{0}",
                                "topic": str(topic),
                                "event": str(event),
                                "t": int(t),
                                "stage_t": int(t),
                                "is_core_user": True,
                                # transition supervision (scalar K=1 or dist K=3)
                                "z_t": z_t if z_pop_mode == "dist" else float(z_t),
                                "z_target": z_tp1 if z_pop_mode == "dist" else float(z_tp1),
                                "z_mask": 1.0 if int(labeled_tp1) > 0 else 0.0,
                                "labeled_secondary_users_t": int(labeled_t),
                                "labeled_secondary_users_tp1": int(labeled_tp1),
                                # extra conditioning (optional)
                                "core_stance_id_t": int(core_stance_id_t),
                                "core_action_type_id_t": int(core_action_type_id_t),
                                "has_user_history": 1 if bool(user_history) else 0,
                                "has_neighbors": 1 if bool(neighbors) else 0,
                                "neighbor_action_type_counts_t": neigh_action_counts,
                            }
                            # Split by stage to avoid leakage
                            split = "train" if t <= 9 else ("validation" if t == 10 else "test")
                            z_split_examples[split].append(ex)
                            z_stats["num_examples"] += 1
                    else:
                        # population-only (current behavior)
                        pop = population_cache.get(t, {}) if use_population_observation else {}
                        pop_texts = pop.get("texts", []) if isinstance(pop, dict) else []
                        rendered_texts: List[str] = []
                        for u_txt in (pop_texts or [])[: min(120, max(0, int(max_population_tweets_total)) if max_population_tweets_total > 0 else 120)]:
                            try:
                                u0, txt0 = u_txt
                                if txt0:
                                    rendered_texts.append(f"- [{u0}] {txt0}")
                            except Exception:
                                continue
                        q_lines: List[str] = []
                        q_lines.append("You are predicting how the SECONDARY-user population stance evolves over stages.")
                        q_lines.append(f"Topic: {topic}")
                        q_lines.append(f"Event: {event}")
                        q_lines.append(f"Stage t: {t}")
                        q_lines.append("")
                        q_lines.append("Population-level observed texts at stage t (secondary users):")
                        if rendered_texts:
                            q_lines.extend(rendered_texts[:200])
                        else:
                            q_lines.append("(no population texts available)")
                        q_lines.append("")
                        q_lines.append("Task: Predict the NEXT-stage population stance scalar z(t+1) in [-1, 1].")
                        q_lines.append("Interpretation: Oppose=-1, Neutral=0, Support=+1; z is the mean over secondary users at that stage.")
                        q = "\n".join(q_lines)
                        ex = {
                            "question": q,
                            "answer": "\\boxed{0}",
                            "topic": str(topic),
                            "event": str(event),
                            "t": int(t),
                            "stage_t": int(t),
                            "is_core_user": False,
                            "z_t": z_t if z_pop_mode == "dist" else float(z_t),
                            "z_target": z_tp1 if z_pop_mode == "dist" else float(z_tp1),
                            "z_mask": 1.0 if int(labeled_tp1) > 0 else 0.0,
                            "labeled_secondary_users_t": int(labeled_t),
                            "labeled_secondary_users_tp1": int(labeled_tp1),
                        }
                        split = "train" if t <= 9 else ("validation" if t == 10 else "test")
                        z_split_examples[split].append(ex)
                        z_stats["num_examples"] += 1

            users = list(macro.keys())
            if max_users > 0:
                users = users[: max_users]

            for user in users:
                user_dict = macro.get(user)
                if not isinstance(user_dict, dict):
                    continue
                is_core_user = str(user) in core_users
                if user_scope == "core" and not is_core_user:
                    stats["skipped_user_scope"] += 1
                    continue
                if user_scope == "noncore" and is_core_user:
                    stats["skipped_user_scope"] += 1
                    continue
                if is_core_user:
                    stats["num_candidates_core"] += 1
                else:
                    stats["num_candidates_noncore"] += 1
                # 允许为次要用户也生成监督样本（用于训练“次要用户信念网络”）。
                # 通过 `--user-scope core/noncore/all` 控制生成范围；样本内用 is_core_user 字段区分。
                # stages expected 0..13
                # iterate t in [0,12], require t+1 exists and non-empty
                for t in range(0, 13):
                    if t not in user_dict or (t + 1) not in user_dict:
                        stats["skipped_missing_stage"] += 1
                        continue
                    stage_t = user_dict.get(t) or []
                    stage_tp1 = user_dict.get(t + 1) or []
                    if not isinstance(stage_t, list) or not isinstance(stage_tp1, list):
                        stats["skipped_missing_stage"] += 1
                        continue
                    if len(stage_tp1) == 0:
                        stats["skipped_missing_target"] += 1
                        continue

                    # === target definition ===
                    # Choose supervision target for this sample.
                    # - core_target_mode controls CORE users (default historical behavior was neighbor_tp1).
                    # - noncore_target_mode controls NONCORE users (default self_tp1 due to sparse neighbor graph).
                    noncore_tm = str(noncore_target_mode or "self").strip().lower()
                    core_tm = str(core_target_mode or "neighbor").strip().lower()
                    if is_core_user:
                        target_mode = "self_tp1" if core_tm in ("self", "self_tp1", "own", "user") else "neighbor_tp1"
                    else:
                        target_mode = "self_tp1" if noncore_tm in ("self", "self_tp1", "own", "user") else "neighbor_tp1"

                    neighbors: List[str] = []
                    target_counter: Dict[int, int] = {}
                    target_label: Optional[str] = None

                    if target_mode == "self_tp1":
                        # Predict user's own stance at t+1 (supervised by stage_tp1 label distribution)
                        target_counter_raw = _stage_label_dist(stage_tp1)
                        if not target_counter_raw:
                            stats["skipped_missing_label"] += 1
                            continue
                        target_counter = {label2id[lab]: int(cnt) for lab, cnt in target_counter_raw.items() if lab in label2id}
                        if not target_counter:
                            stats["skipped_missing_label"] += 1
                            continue
                        target_label = _stage_label(stage_tp1)
                        if not target_label:
                            stats["skipped_missing_label"] += 1
                            continue
                        if target_label not in label2id:
                            stats["skipped_missing_label"] += 1
                            continue
                        target_id = label2id[target_label]
                    else:
                        # neighbors
                        if neighbor_mode == "following":
                            neighbors = list(following_dict.get(str(user), []))
                        else:
                            neighbors = list(follower_dict.get(str(user), [])) if follower_dict else []

                        # limit and filter to users existing in macro
                        if shuffle_neighbors_before_truncation and len(neighbors) > 1:
                            try:
                                import random
                                # deterministic per-sample shuffle so dataset build is reproducible
                                seed = int(neighbor_shuffle_seed) ^ int(_stable_hash_to_bucket(f"{topic}:{event}:{user}:{t}", buckets=2**31-1))
                                rnd = random.Random(seed)
                                rnd.shuffle(neighbors)
                            except Exception:
                                pass
                        if max_neighbor_users > 0 and len(neighbors) > max_neighbor_users:
                            neighbors = neighbors[:max_neighbor_users]
                        neighbors = [nb for nb in neighbors if nb in macro]

                        if not neighbors:
                            stats["skipped_missing_target"] += 1
                            stats["skipped_no_neighbors"] += 1
                            continue

                        # 在当前结构下，简单用一个近似：如果 neighbors 非空但全都没能提供 tp1 label，
                        # 通常是“t+1 没发帖”或“没有标签”。下面会分别计数。
                        target_counter_raw = _neighbor_stage_label_dist(macro, neighbors, t)
                        if not target_counter_raw:
                            # 判断是“没发帖”还是“发帖但无标签”
                            any_posts_tp1 = False
                            for nb in neighbors:
                                nb_dict = macro.get(nb)
                                if not isinstance(nb_dict, dict):
                                    continue
                                nb_tp1 = nb_dict.get(t + 1) or []
                                if isinstance(nb_tp1, list) and len(nb_tp1) > 0:
                                    any_posts_tp1 = True
                                    break
                            if not any_posts_tp1:
                                stats["skipped_neighbors_no_posts_tp1"] += 1
                            else:
                                stats["skipped_neighbors_no_labels_tp1"] += 1
                            stats["skipped_missing_label"] += 1
                            continue

                        # 映射到 label id 空间
                        target_counter = {
                            label2id[lab]: cnt
                            for lab, cnt in target_counter_raw.items()
                            if lab in label2id
                        }

                        if not target_counter:
                            stats["skipped_missing_label"] += 1
                            stats["skipped_neighbors_no_labels_tp1"] += 1
                            continue

                        target_label = _neighbor_stage_label(macro, neighbors, t)
                        if not target_label:
                            stats["skipped_missing_label"] += 1
                            continue
                        if target_label not in label2id:
                            stats["skipped_missing_label"] += 1
                            continue
                        target_id = label2id[target_label]

                    persona = role_desc.get(str(user), "")
                    user_history = _load_user_history_snippet(topic, str(user)) if is_core_user else ""
                    self_label_t = _stage_label(stage_t)
                    self_texts = _stage_texts(stage_t, max_tweets=max_self_tweets)
                    # macro-inferred action distribution for the user at stage t
                    self_action_counts = _action_counts_from_stage(stage_t)
                    self_action_ratio = _counts_to_ratio(self_action_counts)

                    

                    neighbor_texts: List[Tuple[str, str]] = []
                    neighbor_label_counter: Counter = Counter()
                    # macro-inferred action distribution for neighbors at stage t (aggregated across neighbors)
                    neighbor_action_counts = [0, 0, 0, 0, 0]
                    if neighbors:
                        for nb in neighbors:
                            nb_dict = macro.get(nb)
                            if not isinstance(nb_dict, dict):
                                continue
                            nb_stage = nb_dict.get(t) or []
                            if not isinstance(nb_stage, list) or not nb_stage:
                                continue
                            nb_lab = _stage_label(nb_stage)
                            if nb_lab:
                                neighbor_label_counter[nb_lab] += 1
                            # action counts: sum over neighbor stages
                            try:
                                c = _action_counts_from_stage(nb_stage)
                                for i in range(5):
                                    neighbor_action_counts[i] += int(c[i])
                            except Exception:
                                pass
                            # take at most 1-2 texts per neighbor to control length
                            nb_texts = _stage_texts(nb_stage, max_tweets=2)
                            for txt in nb_texts:
                                if txt:
                                    neighbor_texts.append((str(nb), txt))
                            if max_neighbor_tweets_total > 0 and len(neighbor_texts) >= max_neighbor_tweets_total:
                                break
                    neighbor_action_ratio = _counts_to_ratio(neighbor_action_counts)

                    pop = population_cache.get(t, {}) if use_population_observation else {}
                    pop_label_counter = pop.get("label_counter", {}) if isinstance(pop, dict) else {}
                    pop_texts = pop.get("texts", []) if isinstance(pop, dict) else []

                    bargs = BuildArgs(
                        topic=str(topic),
                        event=str(event),
                        user=str(user),
                        t=int(t),
                        persona=str(persona),
                        user_history=str(user_history),
                        self_texts=self_texts,
                        self_label_t=self_label_t,
                        neighbor_texts=neighbor_texts,
                        neighbor_label_counter=dict(neighbor_label_counter),
                        self_action_counts=self_action_counts,
                        self_action_ratio=self_action_ratio,
                        neighbor_action_counts=neighbor_action_counts,
                        neighbor_action_ratio=neighbor_action_ratio,
                        population_texts=pop_texts,
                        population_label_counter=pop_label_counter,
                        label2id=label2id,
                        is_core_user=is_core_user,
                        target_mode=target_mode,
                    )

                    ex = _build_example_from_states(bargs, target_label=target_label, target_id=target_id, target_distribution=target_counter)
                    try:
                        # update histogram for quick sanity checks
                        tid = int(target_id)
                        if tid in stats["target_id_counts"]:
                            stats["target_id_counts"][tid] += 1
                    except Exception:
                        pass

                    if split_by_user:
                        bucket = _stable_hash_to_bucket(str(user), buckets=10)
                        split = "train" if bucket <= 7 else ("validation" if bucket == 8 else "test")
                    else:
                        split = "train"

                    split_examples[split].append(ex)
                    stats["num_examples"] += 1
                    if is_core_user:
                        stats["num_examples_core"] += 1
                    else:
                        stats["num_examples_noncore"] += 1

                    # === Optional: Stage4 action imitation dataset (core users only) ===
                    if export_action_imitation_dataset and bool(is_core_user):
                        try:
                            # Target action_type is inferred from stage t+1 macro items (mode action id).
                            # If no actions observed, treat as do_nothing (4).
                            action_counts_tp1 = _action_counts_from_stage(stage_tp1)
                            aid = _action_type_id_from_counts(action_counts_tp1)
                            if aid is None or int(aid) < 0:
                                aid = 4
                            aid = int(max(0, min(4, int(aid))))
                            q_action = build_action_imitation_input(bargs)
                            a_ex = {
                                "question": q_action,
                                "answer": f"\\boxed{{{aid}}}",
                                "target_action_type_id": int(aid),
                                "target_action_counts_tp1": [int(x) for x in (action_counts_tp1 or [0, 0, 0, 0, 0])],
                                "topic": bargs.topic,
                                "event": bargs.event,
                                "user": bargs.user,
                                "t": int(bargs.t),
                                "is_core_user": True,
                            }
                            a_split_examples[split].append(a_ex)
                            a_stats["num_examples"] += 1
                            a_stats["num_examples_core"] += 1
                            try:
                                a_stats["action_id_counts"][int(aid)] = int(a_stats["action_id_counts"].get(int(aid), 0)) + 1
                            except Exception:
                                pass
                        except Exception as e:
                            # keep belief dataset generation robust (do not fail the whole conversion)
                            try:
                                stats["skipped_action_imitation"] = int(stats.get("skipped_action_imitation", 0)) + 1
                            except Exception:
                                pass

    if stats["num_examples"] == 0:
        raise RuntimeError("没有构造出任何样本，请检查数据路径/字段或参数。")

    _ensure_dir(output_dir)
    # save dataset
    if _HAS_DATASETS:
        ds_dict = DatasetDict({k: Dataset.from_list(v) for k, v in split_examples.items() if v})  # type: ignore[misc]
        print(f"保存 HuggingFace 数据集到: {output_dir}")
        ds_dict.save_to_disk(output_dir)  # type: ignore[union-attr]
    else:
        print("[WARN] 当前环境未安装 `datasets`，将改为输出 JSONL（你可 `pip install datasets` 后再生成 HF 数据集）。")
        for split, items in split_examples.items():
            if not items:
                continue
            jsonl_path = os.path.join(output_dir, f"{split}.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"已输出 JSONL 到: {output_dir}")

    # save metadata
    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    # preview (belief dataset)
    _write_preview_jsonl(str(output_dir), split_examples, tag="belief")

    # === save z-transition dataset (optional) ===
    if export_z_transition_dataset and z_stats.get("num_examples", 0) > 0:
        z_out = str(z_transition_out_dir or "").strip()
        if not z_out:
            z_out = output_dir.rstrip("/") + "_z_transition"
        _ensure_dir(z_out)
        if _HAS_DATASETS:
            z_ds = DatasetDict({k: Dataset.from_list(v) for k, v in z_split_examples.items() if v})  # type: ignore[misc]
            print(f"保存 z_transition HuggingFace 数据集到: {z_out}")
            z_ds.save_to_disk(z_out)  # type: ignore[union-attr]
        else:
            print("[WARN] 当前环境未安装 `datasets`，将改为输出 z_transition JSONL。")
            for split, items in z_split_examples.items():
                if not items:
                    continue
                jsonl_path = os.path.join(z_out, f"{split}.jsonl")
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for it in items:
                        f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"已输出 z_transition JSONL 到: {z_out}")
        with open(os.path.join(z_out, "label2id.json"), "w", encoding="utf-8") as f:
            json.dump(label2id, f, ensure_ascii=False, indent=2)
        with open(os.path.join(z_out, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(z_stats, f, ensure_ascii=False, indent=2)
        # preview (z_transition dataset)
        _write_preview_jsonl(str(z_out), z_split_examples, tag="z_transition")

    # === save action imitation dataset (optional; core users only) ===
    if export_action_imitation_dataset and a_stats.get("num_examples", 0) > 0:
        a_out = str(action_imitation_out_dir or "").strip()
        if not a_out:
            a_out = output_dir.rstrip("/") + "_action_imitation_core"
        _ensure_dir(a_out)
        if _HAS_DATASETS:
            a_ds = DatasetDict({k: Dataset.from_list(v) for k, v in a_split_examples.items() if v})  # type: ignore[misc]
            print(f"保存 action_imitation(HF) 数据集到: {a_out}")
            a_ds.save_to_disk(a_out)  # type: ignore[union-attr]
        else:
            print("[WARN] 当前环境未安装 `datasets`，将改为输出 action_imitation JSONL。")
            for split, items in a_split_examples.items():
                if not items:
                    continue
                jsonl_path = os.path.join(a_out, f"{split}.jsonl")
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for it in items:
                        f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"已输出 action_imitation JSONL 到: {a_out}")
        with open(os.path.join(a_out, "action2id.json"), "w", encoding="utf-8") as f:
            json.dump({"post": 0, "retweet": 1, "reply": 2, "like": 3, "do_nothing": 4}, f, ensure_ascii=False, indent=2)
        with open(os.path.join(a_out, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(a_stats, f, ensure_ascii=False, indent=2)
        # preview (action imitation dataset)
        _write_preview_jsonl(str(a_out), a_split_examples, tag="action_imitation")

    print("完成。")


def main():
    """
    用法示例（生成 metoo 的 belief 数据集）：

    python ECON/convert_hisim_to_econ_dataset.py \\
      --hisim-data-root /home/zhuyinzhou/MAS/HiSim/data \\
      --out-dir /home/zhuyinzhou/MAS/ECON/data/hisim_belief_dataset \\
      --topics metoo \\
      --neighbor-mode followers

    BLM 只有一个宏观事件文件 `blm_macro_p1.pkl`，因此示例：
    python ECON/convert_hisim_to_econ_dataset.py \\
      --hisim-data-root /home/zhuyinzhou/MAS/HiSim/data \\
      --out-dir /home/zhuyinzhou/MAS/ECON/data/hisim_belief_dataset_blm \\
      --topics blm \\
      --events p1
    """
    import argparse

    parser = argparse.ArgumentParser(description="Convert HiSim macro data into belief-network HF dataset.")
    parser.add_argument(
        "--hisim-data-root",
        type=str,
        default="/data/zhuyinzhou/HiSim/data",
        help="HiSim/data 根目录（包含 hisim_with_tweet/ 与 user_data/）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/zhuyinzhou/MAS/ECON/data/hisim_belief_dataset",
        help="输出 HuggingFace 数据集目录（save_to_disk）",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="",
        help="要处理的 topic，逗号分隔（例如 metoo）。留空表示自动检测所有 topic（但仅处理同时有 e1/e2 的）。",
    )
    parser.add_argument(
        "--events",
        type=str,
        default="",
        help="要处理的 event，逗号分隔（例如 e1,e2 或 p1）。留空表示使用该 topic 下存在的全部 event。",
    )
    parser.add_argument(
        "--neighbor-mode",
        type=str,
        default="followers",
        choices=["followers", "following"],
        help="邻居方向：followers=使用 follower_dict[user]；following=将 follower_dict 反转得到关注列表",
    )
    parser.add_argument(
        "--user-scope",
        type=str,
        default="all",
        choices=["all", "core", "noncore"],
        help="用户范围：all=全部；core=仅核心用户（role_desc+history齐全）；noncore=仅次要/普通用户",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=-1,
        help="最多处理多少个用户（-1 表示全部）",
    )
    parser.add_argument(
        "--max-self-tweets",
        type=int,
        default=8,
        help="UserState 中最多保留多少条用户在 stage t 的推文文本",
    )
    parser.add_argument(
        "--max-neighbor-users",
        type=int,
        default=50,
        help="NeighborState 中最多聚合多少个邻居用户",
    )
    parser.add_argument(
        "--max-neighbor-tweets-total",
        type=int,
        default=120,
        help="NeighborState 中最多聚合多少条邻居推文（全体邻居合计）",
    )
    parser.add_argument(
        "--neighbor-k-recent-tweets",
        type=int,
        default=0,
        help="用于结构化邻居特征的最近 k 条邻居推文（按 date 降序）。0 表示使用全部可用邻居推文。",
    )
    parser.add_argument(
        "--no-split-by-user",
        action="store_true",
        help="默认按 user 做 train/val/test 划分（避免同一用户泄漏）。传入此参数则全部放入 train。",
    )
    parser.add_argument(
        "--no-user-history",
        action="store_true",
        help="默认会把核心用户的历史推文（user_data/<topic>/<topic>_v2/<user>.txt）拼进 question；传入此参数则不加入。",
    )
    parser.add_argument(
        "--max-user-history-chars",
        type=int,
        default=2000,
        help="核心用户历史片段最多保留多少字符（避免 prompt 过长）",
    )
    parser.add_argument(
        "--no-population-observation",
        action="store_true",
        help="默认会把次要用户在每个 stage 的 population-level observation 注入 question；传入此参数则不加入。",
    )
    parser.add_argument(
        "--population-scope",
        type=str,
        default="secondary",
        choices=["secondary", "all"],
        help="population observation 的用户范围：secondary=仅次要用户；all=全体用户",
    )
    parser.add_argument(
        "--population-text-source",
        type=str,
        default="macro",
        choices=["macro", "micro", "none"],
        help="population_texts 的来源：macro=从 macro stage 抽文本；micro=从 micro 抽真实 gt_text；none=不注入 population 文本（仍可保留分布）",
    )
    parser.add_argument(
        "--population-micro-user-scope",
        type=str,
        default="all",
        choices=["all", "core", "secondary"],
        help="当 population-text-source=micro 时，micro 抽样的用户范围：all/core/secondary",
    )
    parser.add_argument(
        "--population-micro-sampling",
        type=str,
        default="time",
        choices=["time", "random"],
        help="当 population-text-source=micro 时，抽样方式：time=按 macro stage 时间窗对齐；random=全局随机",
    )
    parser.add_argument(
        "--stage-window-sample-users",
        type=int,
        default=-1,
        help="估计 stage 时间窗时最多采样多少用户（-1 全量；为了速度可设 200/300）",
    )
    parser.add_argument(
        "--stage-window-max-items-per-user-stage",
        type=int,
        default=2,
        help="估计 stage 时间窗时，每个 user-stage 最多取多少条 tweet 解析时间（越大越准但越慢）",
    )
    parser.add_argument(
        "--max-population-tweets-total",
        type=int,
        default=200,
        help="population observation 中最多保留多少条次要用户推文（每个 stage）",
    )
    parser.add_argument(
        "--export-micro-user-sequences",
        action="store_true",
        help="当 population-text-source=micro 时，额外导出每个用户的 micro 序列（jsonl），便于后续训练 latent z/attention/RNN",
    )
    parser.add_argument(
        "--force-k",
        type=int,
        default=3,
        help="强制标签空间大小 K。你的社交模拟训练链路目前固定 K=3（Neutral/Oppose/Support）。",
    )
    parser.add_argument(
        "--export-z-transition-dataset",
        action="store_true",
        help="同时导出 z(t)->z(t+1) transition 数据集（用于训练 BeliefEncoder.population_update_head）。",
    )
    parser.add_argument(
        "--z-transition-out-dir",
        type=str,
        default="",
        help="z_transition 数据集输出目录（留空则不单独输出；你也可以直接用 --out-dir 的目录）。",
    )
    parser.add_argument(
        "--export-action-imitation-dataset",
        action="store_true",
        help="同时导出 Stage4 action imitation 数据集（core users only，监督 action_type id in [0..4]）。",
    )
    parser.add_argument(
        "--action-imitation-out-dir",
        type=str,
        default="",
        help="action imitation 数据集输出目录（留空则使用 <out-dir>_action_imitation_core）。",
    )
    parser.add_argument(
        "--noncore-target-mode",
        type=str,
        default="self",
        choices=["self", "neighbor"],
        help="noncore 用户的监督目标：self=预测用户自身在 t+1 的 stance（推荐）；neighbor=预测邻居在 t+1 的 stance（需要 noncore 有邻居图）。",
    )
    parser.add_argument(
        "--core-target-mode",
        type=str,
        default="neighbor",
        choices=["self", "neighbor"],
        help="core 用户的监督目标：self=预测用户自身在 t+1 的 stance；neighbor=预测邻居在 t+1 的 stance（旧默认）。",
    )
    parser.add_argument(
        "--no-shuffle-neighbors",
        action="store_true",
        help="默认会在截断 max-neighbor-users 之前对邻居列表做确定性打乱（避免顺序偏置）。传入此参数则不打乱。",
    )
    parser.add_argument(
        "--neighbor-shuffle-seed",
        type=int,
        default=0,
        help="邻居打乱的全局种子（会与 topic/event/user/t 的 hash 组合，确保每个样本可复现）。",
    )
    parser.add_argument(
        "--z-transition-conditioning",
        type=str,
        default="population_only",
        choices=["population_only", "core_user"],
        help="z_transition 数据集的条件输入：population_only=仅次要用户群体观测；core_user=以 core 用户视角构造 question，并监督 secondary population 的 z(t)->z(t+1)。",
    )
    parser.add_argument(
        "--z-transition-population-mode",
        type=str,
        default="scalar",
        choices=["scalar", "dist"],
        help="z_transition 的 z 表示：scalar=z∈[-1,1]；dist=z∈Δ^3（由 secondary 用户 stance 分布归一化得到，形如 [p_neu,p_opp,p_sup]，按 label2id 的 id 顺序）。",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="默认会在每个输出目录生成 preview.jsonl（抽样若干条数据用于人工检查）。传入此参数则关闭。",
    )
    parser.add_argument(
        "--preview-num-per-split",
        type=int,
        default=3,
        help="preview.jsonl 中每个 split 抽样多少条样本（默认 3）。",
    )
    parser.add_argument(
        "--preview-seed",
        type=int,
        default=42,
        help="preview 抽样随机种子（保证可复现）。",
    )
    parser.add_argument(
        "--preview-max-chars",
        type=int,
        default=600,
        help="preview.jsonl 里 question 的截断长度（避免文件过大）。0 表示不截断。",
    )
    parser.add_argument(
        "--preview-filename",
        type=str,
        default="preview.jsonl",
        help="预览文件名（默认 preview.jsonl）。",
    )

    args = parser.parse_args()

    topics = [t.strip() for t in args.topics.split(",") if t.strip()] if args.topics else None
    events = [e.strip() for e in args.events.split(",") if e.strip()] if args.events else None

    convert_hisim_macro_to_belief_hf_dataset(
        hisim_data_root=args.hisim_data_root,
        output_dir=args.out_dir,
        topics=topics,
        events=events,
        neighbor_mode=args.neighbor_mode,
        user_scope=args.user_scope,
        max_users=args.max_users,
        max_self_tweets=args.max_self_tweets,
        max_neighbor_users=args.max_neighbor_users,
        max_neighbor_tweets_total=args.max_neighbor_tweets_total,
        neighbor_k_recent_tweets=int(args.neighbor_k_recent_tweets),
        include_user_history=(not args.no_user_history),
        max_user_history_chars=args.max_user_history_chars,
        use_population_observation=(not args.no_population_observation),
        population_scope=args.population_scope,
        population_text_source=args.population_text_source,
        population_micro_user_scope=args.population_micro_user_scope,
        population_micro_sampling=args.population_micro_sampling,
        stage_window_sample_users=args.stage_window_sample_users,
        stage_window_max_items_per_user_stage=args.stage_window_max_items_per_user_stage,
        max_population_tweets_total=args.max_population_tweets_total,
        export_micro_user_sequences=args.export_micro_user_sequences,
        split_by_user=(not args.no_split_by_user),
        force_k=int(args.force_k),
        export_z_transition_dataset=bool(args.export_z_transition_dataset),
        z_transition_out_dir=str(args.z_transition_out_dir or ""),
        z_transition_population_mode=str(getattr(args, "z_transition_population_mode", "scalar") or "scalar"),
        export_action_imitation_dataset=bool(getattr(args, "export_action_imitation_dataset", False)),
        action_imitation_out_dir=str(getattr(args, "action_imitation_out_dir", "") or ""),
        export_preview=(not bool(getattr(args, "no_preview", False))),
        preview_num_per_split=int(getattr(args, "preview_num_per_split", 3)),
        preview_seed=int(getattr(args, "preview_seed", 42)),
        preview_max_chars=int(getattr(args, "preview_max_chars", 600)),
        preview_filename=str(getattr(args, "preview_filename", "preview.jsonl") or "preview.jsonl"),
        noncore_target_mode=str(args.noncore_target_mode or "self"),
        core_target_mode=str(args.core_target_mode or "neighbor"),
        shuffle_neighbors_before_truncation=(not bool(args.no_shuffle_neighbors)),
        neighbor_shuffle_seed=int(args.neighbor_shuffle_seed),
        z_transition_conditioning=str(args.z_transition_conditioning or "population_only"),
    )


if __name__ == "__main__":
    main()






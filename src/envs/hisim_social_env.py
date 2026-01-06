import os
import json
import pickle
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import gym
from gym import spaces
from loguru import logger
import torch


def _normalize_label(s: Any) -> str:
    """把可能形态的 stance label 归一化到稳定字符串。"""
    if s is None:
        return ""
    ss = str(s).strip()
    if not ss:
        return ""
    # 常见大小写/空白差异
    ss2 = re.sub(r"\s+", " ", ss).strip()
    # 兼容一些变体
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


class PopulationZUpdater:
    """
    population_z 更新器接口。

    目标：把环境里的 z 更新规则接口化，便于替换为更复杂/可学习模块。
    """

    def reset(self, num_labels: int) -> List[float]:
        k = max(1, int(num_labels))
        return [1.0 / k for _ in range(k)]

    def init_aggregator(self, num_labels: int) -> Dict[str, Any]:
        """创建一个可在 stage 内逐步聚合的输入容器。"""
        k = max(1, int(num_labels))
        return {
            "k": k,
            "n": 0,
            "counts": [0 for _ in range(k)],
            "users": [],
            "post_texts": [],
        }

    def accumulate(
        self,
        agg: Dict[str, Any],
        pred_sid: int,
        *,
        t: int,
        user: str,
        post_text: str,
    ) -> Dict[str, Any]:
        """将单步信息聚合进 agg（默认行为：计数 pred_sid）。"""
        if not isinstance(agg, dict):
            agg = self.init_aggregator(3)
        k = int(agg.get("k", 3))
        sid = int(pred_sid) if pred_sid is not None else 0
        sid = max(0, min(k - 1, sid))
        counts = agg.get("counts")
        if not isinstance(counts, list) or len(counts) != k:
            counts = [0 for _ in range(k)]
        counts[sid] += 1
        agg["counts"] = counts
        agg["n"] = int(agg.get("n", 0)) + 1
        try:
            agg_users = agg.get("users")
            if isinstance(agg_users, list):
                agg_users.append(str(user))
        except Exception:
            pass
        try:
            agg_texts = agg.get("post_texts")
            if isinstance(agg_texts, list):
                txt = str(post_text or "")
                if txt:
                    agg_texts.append(txt[:512])
        except Exception:
            pass
        return agg

    def update(
        self,
        z: List[float],
        agg: Dict[str, Any],
        *,
        t: int,
        stage_end: bool,
    ) -> List[float]:
        raise NotImplementedError


class ABMDecayUpdater(PopulationZUpdater):
    """默认 baseline：z <- decay*z + alpha*onehot(pred_sid)，再归一化到单纯形。"""

    def __init__(self, alpha: float = 0.03, decay: float = 0.995):
        self.alpha = float(alpha)
        self.decay = float(decay)

    def update(
        self,
        z: List[float],
        agg: Dict[str, Any],
        *,
        t: int,
        stage_end: bool,
    ) -> List[float]:
        # 逻辑上延迟到 stage end 才更新
        if not stage_end:
            return z
        if not isinstance(z, list) or not z:
            z = self.reset(3)
        k = len(z)
        counts = agg.get("counts") if isinstance(agg, dict) else None
        if not isinstance(counts, list) or len(counts) != k:
            counts = [0 for _ in range(k)]
        total = int(agg.get("n", sum(int(c) for c in counts))) if isinstance(agg, dict) else int(sum(int(c) for c in counts))
        total = max(0, total)

        # 近似等价于：对每个 core post 做一次 decay + alpha*onehot，再归一化
        # 这里按 stage 聚合：z <- (decay^N)*z + alpha*counts，再归一化
        zz = [float(v) * (self.decay ** float(total)) for v in z]
        for i in range(k):
            try:
                zz[i] += self.alpha * float(counts[i])
            except Exception:
                pass
        ssum = float(sum(zz))
        return [v / ssum for v in zz] if ssum > 0 else [1.0 / k for _ in range(k)]


class ScalarABMDecayUpdater:
    """
    标量版 baseline updater：z ∈ [-1,1]
    - 先从 stage 内聚合的 stance 计数推一个 z_hat（按 label 名映射到 -1/0/1 的加权平均）
    - 再做平滑更新：z <- decay*z + alpha*z_hat
    """

    def __init__(self, alpha: float = 0.03, decay: float = 0.995):
        self.alpha = float(alpha)
        self.decay = float(decay)

    def reset(self) -> float:
        return 0.0

    def init_aggregator(self) -> Dict[str, Any]:
        # 仍用 K=3 的计数聚合（对齐核心 stance labels）
        return {"k": 3, "n": 0, "counts": [0, 0, 0], "users": [], "post_texts": []}

    def update(
        self,
        z: float,
        agg: Dict[str, Any],
        *,
        id2label: Dict[int, str],
        stage_end: bool,
    ) -> float:
        if not stage_end:
            return float(z)
        prev = float(z)
        try:
            counts = (agg or {}).get("counts") or [0, 0, 0]
            counts = [int(x) for x in list(counts)[:3]]
            total = sum(max(0, c) for c in counts)
            if total <= 0:
                z_hat = 0.0
            else:
                v_map = {"Oppose": -1.0, "Neutral": 0.0, "Support": 1.0}
                z_hat = 0.0
                for i in range(3):
                    lab = str(id2label.get(i, ""))
                    z_hat += float(max(0, counts[i])) * float(v_map.get(lab, 0.0))
                z_hat = z_hat / float(total)
            out = (self.decay * prev) + (self.alpha * float(z_hat))
            return float(max(-1.0, min(1.0, out)))
        except Exception:
            return float(max(-1.0, min(1.0, prev)))


class ScalarNoopUpdater:
    """标量版 noop：stage end 时保持不变（并夹紧到[-1,1]）。"""

    def reset(self) -> float:
        return 0.0

    def init_aggregator(self) -> Dict[str, Any]:
        return {"k": 3, "n": 0, "counts": [0, 0, 0], "users": [], "post_texts": []}

    def update(
        self,
        z: float,
        agg: Dict[str, Any],
        *,
        id2label: Dict[int, str],
        stage_end: bool,
    ) -> float:
        if not stage_end:
            return float(z)
        try:
            return float(max(-1.0, min(1.0, float(z))))
        except Exception:
            return 0.0


class NoopUpdater(PopulationZUpdater):
    """不更新 z（用于 ablation/debug）。"""

    def update(
        self,
        z: List[float],
        agg: Dict[str, Any],
        *,
        t: int,
        stage_end: bool,
    ) -> List[float]:
        if not stage_end:
            return z
        if not isinstance(z, list) or not z:
            return self.reset(3)
        ssum = float(sum(float(v) for v in z))
        return [float(v) / ssum for v in z] if ssum > 0 else self.reset(len(z))


def _as_mapping(obj: Any) -> Optional[Dict[str, Any]]:
    """macro pkl 的 tweet item 常见为 pandas.Series；尝试转成 dict。"""
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


def _extract_text(tweet: Dict[str, Any]) -> str:
    for k in ("rawContent", "content", "text", "full_text", "message"):
        v = tweet.get(k)
        if v:
            return str(v).strip()
    return ""


def _extract_label(tweet: Dict[str, Any]) -> Optional[str]:
    for k in ("stance_label", "stance", "label", "content_label", "behavior"):
        v = tweet.get(k)
        if v is None:
            continue
        s = _normalize_label(v)
        if s != "":
            return s
    return None


def _stage_label(stage_items: List[Any]) -> Optional[str]:
    labels: List[str] = []
    for it in stage_items:
        d = _as_mapping(it)
        if not d:
            continue
        lab = _extract_label(d)
        if lab:
            labels.append(lab)
    if not labels:
        return None
    # majority vote
    from collections import Counter

    return Counter(labels).most_common(1)[0][0]


def _stage_texts(stage_items: List[Any], max_tweets: int = 8) -> List[str]:
    out: List[str] = []
    for it in stage_items:
        d = _as_mapping(it)
        if not d:
            continue
        txt = _extract_text(d)
        if txt:
            out.append(txt)
        if max_tweets > 0 and len(out) >= max_tweets:
            break
    return out


def _parse_datetime(s: Any) -> Optional[datetime]:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    ss = str(s).strip()
    if not ss:
        return None
    try:
        if ss.endswith("Z"):
            ss = ss[:-1] + "+00:00"
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _build_stage_time_windows_from_macro(
    macro: Dict[str, Any],
    sample_users: int = 200,
    max_items_per_user_stage: int = 2,
) -> Dict[int, Tuple[Optional[datetime], Optional[datetime]]]:
    users = list(macro.keys())[: max(0, sample_users) if sample_users > 0 else len(macro)]
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
            for it in stage[: max_items_per_user_stage]:
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


def _load_micro_items(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    return []


def _extract_micro_text(item: Dict[str, Any]) -> str:
    gt = item.get("gt_text")
    if gt:
        return str(gt).strip()
    tp = item.get("tweet_page")
    if tp:
        return str(tp).strip()
    return ""


def _assign_micro_to_stages(
    micro_items: List[Dict[str, Any]],
    stage_windows: Dict[int, Tuple[Optional[datetime], Optional[datetime]]],
) -> Dict[int, List[str]]:
    buckets: Dict[int, List[str]] = {t: [] for t in range(14)}
    for it in micro_items:
        dt = _parse_datetime(it.get("current_time") or it.get("date") or it.get("time"))
        if not dt:
            continue
        txt = _extract_micro_text(it)
        if not txt:
            continue
        for t, (lo, hi) in stage_windows.items():
            if lo is None or hi is None:
                continue
            if lo <= dt <= hi:
                buckets[t].append(txt)
                break
    return buckets


def _jaccard_sim(a: str, b: str) -> float:
    wa = set(re.findall(r"[A-Za-z0-9']+", (a or "").lower()))
    wb = set(re.findall(r"[A-Za-z0-9']+", (b or "").lower()))
    if not wa or not wb:
        return 0.0
    return float(len(wa & wb) / max(1, len(wa | wb)))


class HiSimSocialEnv(gym.Env):
    """
    社交媒体交流过程仿真环境（用于 ECON 训练前的结构改造与后续 RL 训练）。

    设计目标（与你的设定对齐）：
    - 300 核心用户（有 persona/history/network），700 边缘用户（用 population-level latent z/ABM 表示）
    - 13 个 stage，每个 stage 内让 300 核心用户都发一次
    - 动作包含：action_type（离散5类） + (可选) stance_id + (可选) post_text
    - 边缘用户通过 latent z 的更新规则参与状态转移（当前实现为可配置的 ABM baseline，后续可替换为可学习模块）
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, **kwargs):
        super().__init__()

        self.hisim_data_root = kwargs.get("hisim_data_root", "/home/zhuyinzhou/MAS/HiSim/data")
        self.topic = kwargs.get("topic", "metoo")
        self.event = kwargs.get("event", "e1")

        self.max_question_length = int(kwargs.get("max_question_length", 1024))
        self.max_answer_length = int(kwargs.get("max_answer_length", 512))

        # rollout settings
        self.n_stages = int(kwargs.get("n_stages", 13))  # use 0..12 by default
        self.max_neighbor_posts = int(kwargs.get("max_neighbor_posts", 8))
        self.max_population_texts = int(kwargs.get("max_population_texts", 20))
        # A: supervision masking for missing next-step labels/text
        self.mask_missing_gt = bool(kwargs.get("mask_missing_gt", True))
        # C: gate z supervision by minimum labeled edge users at target stage
        self.min_edge_labels_for_z_target = int(kwargs.get("min_edge_labels_for_z_target", 0))
        # secondary users (edge) simulation via belief network (innovation hook)
        self.use_secondary_belief_sim = bool(kwargs.get("use_secondary_belief_sim", False))
        self.secondary_sim_max_users = int(kwargs.get("secondary_sim_max_users", 200))
        self.secondary_sim_use_micro_texts = bool(kwargs.get("secondary_sim_use_micro_texts", True))
        # core user profile/memory prompt controls
        self.max_user_history_lines = int(kwargs.get("max_user_history_lines", 40))
        self.max_recent_self_posts = int(kwargs.get("max_recent_self_posts", 6))

        # ABM/latent z update (baseline; can be swapped later)
        # population_belief_mode (new preferred name; keep backward compatibility with population_z_mode):
        # - "categorical3": legacy, z is a prob vector over K=3 labels
        # - "scalar"/"continuous": z is a scalar in [-1,1] representing direction & intensity
        _pbm = kwargs.get("population_belief_mode", None)
        if _pbm is None:
            _pbm = kwargs.get("population_z_mode", "categorical3")
        _pbm = str(_pbm).strip().lower()
        if _pbm in ("scalar", "continuous", "cont", "float", "regression"):
            self.population_z_mode = "continuous"
        else:
            self.population_z_mode = "categorical3"
        self.z_alpha = float(kwargs.get("z_alpha", 0.03))  # update rate
        self.z_decay = float(kwargs.get("z_decay", 0.995))  # slow forgetting
        self.stance_k = int(kwargs.get("stance_k", 3))  # 核心用户 stance label 仍保持 K=3
        self.label2id_path = kwargs.get("label2id_path", "")
        self.population_z_updater_name = str(kwargs.get("population_z_updater", "abm_decay")).strip().lower()
        self.z_agg_max_texts = int(kwargs.get("z_agg_max_texts", 0))  # 0=不存 text；>0 存储部分 post 文本供 NN updater 用

        # reward weights
        self.reward_w_action_type = float(kwargs.get("reward_w_action_type", 0.2))
        self.reward_w_stance = float(kwargs.get("reward_w_stance", 0.7))
        self.reward_w_text = float(kwargs.get("reward_w_text", 0.3))
        # normalize if user provided weird weights (keep backward compatibility if action_type weight is 0)
        sw = max(0.0, self.reward_w_action_type) + max(0.0, self.reward_w_stance) + max(0.0, self.reward_w_text)
        if sw > 0:
            self.reward_w_action_type /= sw
            self.reward_w_stance /= sw
            self.reward_w_text /= sw

        # action types (core user behavior)
        self.action_types: List[str] = ["post", "retweet", "reply", "like", "do_nothing"]
        self.action2id: Dict[str, int] = {a: i for i, a in enumerate(self.action_types)}

        # Load data
        macro_path = os.path.join(self.hisim_data_root, "hisim_with_tweet", f"{self.topic}_macro_{self.event}.pkl")
        if not os.path.exists(macro_path):
            raise FileNotFoundError(f"macro 文件不存在: {macro_path}")
        with open(macro_path, "rb") as f:
            self.macro = pickle.load(f)
        if not isinstance(self.macro, dict):
            raise TypeError(f"macro 顶层结构不是 dict: {type(self.macro)}")

        role_desc_path = os.path.join(self.hisim_data_root, "user_data", self.topic, "role_desc_v2_clean.json")
        follower_dict_path = os.path.join(self.hisim_data_root, "user_data", self.topic, "follower_dict.json")
        # user historical posts (observed memory)
        # default path aligns with convert_hisim_to_econ_dataset.py: user_data/<topic>/<topic>_v2/<user>.txt
        self.user_history_dir = kwargs.get(
            "user_history_dir",
            os.path.join(self.hisim_data_root, "user_data", self.topic, f"{self.topic}_v2"),
        )
        self.role_desc = {}
        self.follower_dict = {}
        self.user_histories: Dict[str, str] = {}
        if os.path.exists(role_desc_path):
            with open(role_desc_path, "r", encoding="utf-8") as f:
                self.role_desc = json.load(f)
        if os.path.exists(follower_dict_path):
            with open(follower_dict_path, "r", encoding="utf-8") as f:
                self.follower_dict = json.load(f)

        self.core_users = sorted(list(self.role_desc.keys()))
        self.all_users = sorted(list(self.macro.keys()))
        self.edge_users = [u for u in self.all_users if u not in set(self.core_users)]
        logger.info(f"[HiSimSocialEnv] topic={self.topic} event={self.event} macro_users={len(self.all_users)} core={len(self.core_users)} edge={len(self.edge_users)}")

        # optional: validate user scale
        expected_core = int(kwargs.get("expected_core_users", -1))
        expected_edge = int(kwargs.get("expected_edge_users", -1))
        strict_counts = bool(kwargs.get("strict_user_counts", False))
        if expected_core > 0 and len(self.core_users) != expected_core:
            msg = f"[HiSimSocialEnv] 核心用户数量不符合预期: got={len(self.core_users)} expected={expected_core}"
            if strict_counts:
                raise ValueError(msg)
            logger.warning(msg)
        if expected_edge > 0 and len(self.edge_users) != expected_edge:
            msg = f"[HiSimSocialEnv] 次要用户数量不符合预期: got={len(self.edge_users)} expected={expected_edge}"
            if strict_counts:
                raise ValueError(msg)
            logger.warning(msg)

        # load observed historical posts for core users if present
        try:
            if isinstance(self.user_history_dir, str) and self.user_history_dir and os.path.isdir(self.user_history_dir):
                for u in self.core_users:
                    p = os.path.join(self.user_history_dir, f"{u}.txt")
                    if not os.path.exists(p):
                        continue
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            txt = f.read()
                        self.user_histories[str(u)] = str(txt or "").strip()
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"[HiSimSocialEnv] 读取 user_history_dir 失败: {self.user_history_dir} err={e}")

        # label2id: 强制 K=3（与 learner/runner 侧的 vshape(3,) 对齐）
        loaded_label2id: Optional[Dict[str, int]] = None
        if self.label2id_path and os.path.exists(self.label2id_path):
            try:
                with open(self.label2id_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and all(isinstance(v, int) for v in raw.values()):
                    loaded_label2id = {str(k): int(v) for k, v in raw.items()}
            except Exception as e:
                logger.warning(f"[HiSimSocialEnv] 读取 label2id_path 失败: {self.label2id_path} err={e}")

        if loaded_label2id:
            # normalize keys
            tmp = {_normalize_label(k): int(v) for k, v in loaded_label2id.items()}
            # 只保留 0..K-1
            tmp = {k: v for k, v in tmp.items() if 0 <= int(v) < self.stance_k}
            # 补全缺失（尽量用常见三类）
            if self.stance_k == 3:
                canonical = {"Neutral": 0, "Oppose": 1, "Support": 2}
                for k, v in canonical.items():
                    tmp.setdefault(k, v)
            self.label2id = {k: int(v) for k, v in sorted(tmp.items(), key=lambda x: x[1])[: self.stance_k]}
        else:
            # 从 macro 扫 label（采样即可），然后尽量映射到三类
            labels = set()
            for u in self.all_users[:200]:
                ud = self.macro.get(u)
                if not isinstance(ud, dict):
                    continue
                for t in range(14):
                    st = ud.get(t) or []
                    if not isinstance(st, list):
                        continue
                    for it in st:
                        d = _as_mapping(it)
                        if not d:
                            continue
                        lab = _extract_label(d)
                        if lab:
                            labels.add(_normalize_label(lab))

            if self.stance_k == 3:
                canonical = {"Neutral": 0, "Oppose": 1, "Support": 2}
                # 如果 canonical 都出现过，直接用 canonical；否则退化为取前三个并告警
                if set(canonical.keys()).issubset(labels):
                    self.label2id = dict(canonical)
                else:
                    picked = sorted(list(labels))[:3]
                    if len(picked) < 3:
                        # 极端情况下不足 3，补 Neutral/Oppose/Support
                        picked = (picked + ["Neutral", "Oppose", "Support"])[:3]
                    self.label2id = {lab: i for i, lab in enumerate(picked)}
                    logger.warning(f"[HiSimSocialEnv] macro labels!=canonical，已退化选取前三类作为 K=3: {self.label2id}")
            else:
                self.label2id = {lab: i for i, lab in enumerate(sorted(labels))}

        # 最终固定维度
        self.id2label = {i: lab for lab, i in self.label2id.items()}
        self.num_labels = int(self.stance_k)
        logger.info(f"[HiSimSocialEnv] labels(K={self.num_labels})={self.label2id}")

        # population_z updater
        # NOTE:
        # - categorical3: use list-prob updater (PopulationZUpdater)
        # - continuous/scalar: use scalar updater (z in [-1,1]) but still aggregate counts over K=3 labels
        self._z_accumulator = PopulationZUpdater()  # only used for stage-wise count aggregation
        self.z_scalar_updater: Optional[Any] = None
        if self.population_z_mode == "continuous":
            if self.population_z_updater_name in ("abm_decay", "abm", "decay"):
                self.z_scalar_updater = ScalarABMDecayUpdater(alpha=self.z_alpha, decay=self.z_decay)
            elif self.population_z_updater_name in ("none", "noop", "no_op", "no-op"):
                self.z_scalar_updater = ScalarNoopUpdater()
            else:
                logger.warning(f"[HiSimSocialEnv] 未知 population_z_updater={self.population_z_updater_name}（continuous 模式），回退 scalar abm_decay")
                self.z_scalar_updater = ScalarABMDecayUpdater(alpha=self.z_alpha, decay=self.z_decay)
            # keep a categorical updater instance for compatibility (should not be used in continuous mode)
            self.z_updater = ABMDecayUpdater(alpha=self.z_alpha, decay=self.z_decay)
        else:
            if self.population_z_updater_name in ("abm_decay", "abm", "decay"):
                self.z_updater = ABMDecayUpdater(alpha=self.z_alpha, decay=self.z_decay)
            elif self.population_z_updater_name in ("none", "noop", "no_op", "no-op"):
                self.z_updater = NoopUpdater()
            else:
                logger.warning(f"[HiSimSocialEnv] 未知 population_z_updater={self.population_z_updater_name}，回退 abm_decay")
                self.z_updater = ABMDecayUpdater(alpha=self.z_alpha, decay=self.z_decay)

        # === supervision target for latent z (secondary users) ===
        # - categorical3: z ∈ Δ^K (K=num_labels=3)
        # - continuous: z ∈ [-1,1] scalar (mean stance value over secondary users)
        self.edge_dist_by_stage: Dict[int, List[float]] = {}
        self.edge_z_scalar_by_stage: Dict[int, float] = {}
        self.edge_label_count_by_stage: Dict[int, int] = {}
        for t in range(14):
            counts = [0 for _ in range(self.num_labels)]
            total = 0
            for u in self.edge_users:
                ud = self.macro.get(u)
                if not isinstance(ud, dict):
                    continue
                stage = ud.get(t) or []
                if not isinstance(stage, list) or not stage:
                    continue
                lab = _normalize_label(_stage_label(stage))
                if not lab or lab not in self.label2id:
                    continue
                counts[self.label2id[lab]] += 1
                total += 1
            self.edge_label_count_by_stage[t] = int(total)
            if total <= 0:
                self.edge_dist_by_stage[t] = [1.0 / self.num_labels for _ in range(self.num_labels)]
            else:
                self.edge_dist_by_stage[t] = [c / total for c in counts]
            # continuous scalar: map (Oppose, Neutral, Support) -> (-1,0,+1) then take expectation
            try:
                p = self.edge_dist_by_stage[t]
                # indices are aligned to label2id; in canonical mapping we expect Neutral=0, Oppose=1, Support=2
                # but we keep it robust: compute by label name if possible
                v_map = {"Oppose": -1.0, "Neutral": 0.0, "Support": 1.0}
                z_val = 0.0
                for i in range(self.num_labels):
                    lab = self.id2label.get(i, "")
                    z_val += float(p[i]) * float(v_map.get(str(lab), 0.0))
                self.edge_z_scalar_by_stage[t] = float(max(-1.0, min(1.0, z_val)))
            except Exception:
                self.edge_z_scalar_by_stage[t] = 0.0

        # precompute micro texts bucketed by stage (for population observation)
        micro_path = os.path.join(self.hisim_data_root, "hisim_with_tweet", f"{self.topic}_micro.pkl")
        micro_items = _load_micro_items(micro_path)
        if micro_items:
            stage_windows = _build_stage_time_windows_from_macro(self.macro, sample_users=200, max_items_per_user_stage=2)
            self.micro_texts_by_stage = _assign_micro_to_stages(micro_items, stage_windows)
        else:
            self.micro_texts_by_stage = {t: [] for t in range(14)}

        # Spaces
        self.action_space = spaces.Text(max_length=self.max_answer_length)
        self.observation_space = spaces.Text(max_length=self.max_question_length)

        # Episode state
        self.stage_t = 0
        self.core_idx = 0
        self.episode_steps = 0
        self.episode_limit = self.n_stages * len(self.core_users)

        # social state
        # (user, stage)-> {"action_type":str, "stance_id":Optional[int], "text":str}
        self.core_posts: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # (edge_user, stage)-> {"action_type":str, "stance_id":Optional[int], "text":str}
        # simulated from belief network (optional)
        self.secondary_posts: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # population latent z (either prob vec over labels or scalar)
        if self.population_z_mode == "continuous":
            # scalar z in [-1,1]
            try:
                self.population_z = float(self.z_scalar_updater.reset()) if self.z_scalar_updater is not None else 0.0
            except Exception:
                self.population_z = 0.0
        else:
            self.population_z = self.z_updater.reset(self.num_labels)
        # stage-level aggregation for delayed z update
        if self.population_z_mode == "continuous":
            self._z_agg = self.z_scalar_updater.init_aggregator() if self.z_scalar_updater is not None else {"k": 3, "n": 0, "counts": [0, 0, 0], "users": [], "post_texts": []}
        else:
            self._z_agg = self.z_updater.init_aggregator(self.num_labels)

        self.current_obs: Optional[str] = None
        self.current_info: Dict[str, Any] = {}

    def get_belief_tensor(self, belief_inputs: Optional[Dict[str, Any]], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        把 belief_inputs（dict）转换为 learner/NN 直接可用的张量字典。
        仅包含数值特征；文本字段保持在 info 里（若未来需要 tokenization，可在上层处理）。
        """
        if device is None:
            device = torch.device("cpu")
        if not isinstance(belief_inputs, dict):
            belief_inputs = {}

        # K for core stance labels remains 3; population_z may be scalar when population_z_mode="continuous".
        k = int(belief_inputs.get("k") or self.num_labels or 3)
        k = max(1, k)

        nb = belief_inputs.get("neighbor_stance_counts") or [0 for _ in range(k)]
        if not isinstance(nb, list) or len(nb) != k:
            nb = (list(nb) if isinstance(nb, (tuple, list)) else [])[:k]
            nb = (nb + [0 for _ in range(k)])[:k]

        pz_raw = belief_inputs.get("population_z")
        if self.population_z_mode == "continuous":
            # scalar z in [-1,1] -> tensor shape (1,)
            try:
                z = float(pz_raw) if pz_raw is not None else float(self.population_z)
            except Exception:
                z = 0.0
            z = max(-1.0, min(1.0, z))
            pz = [z]
        else:
            pz = pz_raw or [1.0 / k for _ in range(k)]
            if not isinstance(pz, list) or len(pz) != k:
                pz = (list(pz) if isinstance(pz, (tuple, list)) else [])[:k]
                pz = (pz + [1.0 / k for _ in range(k)])[:k]

        t = int(belief_inputs.get("t", 0))
        is_core = bool(belief_inputs.get("is_core_user", True))

        return {
            "t": torch.tensor([t], dtype=torch.int64, device=device),  # (1,)
            "is_core_user": torch.tensor([1 if is_core else 0], dtype=torch.int64, device=device),  # (1,)
            "neighbor_stance_counts": torch.tensor(nb, dtype=torch.float32, device=device),  # (K,)
            "population_z": torch.tensor(pz, dtype=torch.float32, device=device),  # (K,) or (1,)
        }

    def _collect_belief_inputs(self, user: str, t: int) -> Dict[str, Any]:
        """
        显式收集 belief 相关输入（原来只隐式写入 observation 文本）。

        注意：此函数读取的是“当前环境状态下”的信息，因此在 step 里要在更新 population_z 之前调用，
        才能与用于决策的 observation 一致。
        """
        persona = self.role_desc.get(user, "")
        user_history = self.user_histories.get(str(user), "")
        neighbor_counter, neighbor_texts = self._neighbor_context(user, t)
        pop_dist, pop_texts = self._population_obs(t)

        # 计数向量化（按 label id 对齐）
        nb_counts = [0 for _ in range(self.num_labels)]
        for lab, c in (neighbor_counter or {}).items():
            labn = _normalize_label(lab)
            if labn in self.label2id:
                nb_counts[self.label2id[labn]] += int(c)

        out = {
            "user": str(user),
            "t": int(t),
            "k": int(self.num_labels),
            "is_core_user": True,
            "persona": str(persona) if persona is not None else "",
            "user_history": str(user_history) if user_history is not None else "",
            "neighbor_stance_counts": nb_counts,  # List[int], len=K
            "neighbor_texts": [str(x[1]) for x in (neighbor_texts or [])],
            # population_z: either List[float] (categorical3) or float scalar (continuous)
            "population_z": self.population_z if self.population_z_mode == "continuous" else [float(v) for v in (self.population_z or [])],
            "population_dist": {str(k): float(v) for k, v in (pop_dist or {}).items()},
            "population_texts": [str(x) for x in (pop_texts or [])],
        }
        return out

    def get_env_info(self) -> Dict[str, Any]:
        return {
            "episode_limit": self.episode_limit,
            # NOTE: env.step 实际消费的是 LLM 输出的 JSON 文本；这里的 n_actions 主要用于 runner/buffer 的占位形状
            "n_actions": len(self.action_types),
            "obs_shape": (self.max_question_length,),
            "state_shape": (1,),
        }

    def _current_user(self) -> str:
        return self.core_users[self.core_idx]

    def _gt_for(self, user: str, t: int) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """返回 (gt_label_id, gt_label_str, gt_text_sample)"""
        ud = self.macro.get(user)
        if not isinstance(ud, dict):
            return None, None, None
        stage = ud.get(t) or []
        if not isinstance(stage, list) or not stage:
            return None, None, None
        lab = _normalize_label(_stage_label(stage))
        if not lab or lab not in self.label2id:
            return None, lab, None
        gt_texts = _stage_texts(stage, max_tweets=6)
        gt_text = random.choice(gt_texts) if gt_texts else None
        return self.label2id[lab], lab, gt_text

    def _neighbor_context(self, user: str, t: int) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
        """基于已模拟生成的 core_posts 聚合邻居状态（不直接读取 macro，避免泄漏）。"""
        neighbors = self.follower_dict.get(user, [])
        if not isinstance(neighbors, list):
            neighbors = []
        counter: Dict[str, int] = {}
        texts: List[Tuple[str, str]] = []
        for nb in neighbors:
            key = (str(nb), t)
            post = self.core_posts.get(key)
            if not post and self.use_secondary_belief_sim:
                post = self.secondary_posts.get(key)
            if not post:
                continue
            # only count stance if this action actually expresses stance
            at = str(post.get("action_type") or "").strip().lower()
            sid = post.get("stance_id")
            if at in ("post", "retweet", "reply") and isinstance(sid, int) and sid in self.id2label:
                lab = self.id2label[sid]
                counter[lab] = counter.get(lab, 0) + 1
            txt = post.get("text")
            if txt:
                texts.append((str(nb), str(txt)))
            if self.max_neighbor_posts > 0 and len(texts) >= self.max_neighbor_posts:
                break
        return counter, texts

    def _infer_action_type_from_text(self, text: Optional[str]) -> str:
        """
        从真实 tweet 文本粗略推断动作类型（用于 imitation-style reward 的 action_type 部分）。
        注意：HiSim 的 macro 数据一般只有 tweet 文本，like / do_nothing 通常不可观测。
        """
        s = str(text or "").strip()
        if not s:
            return "do_nothing"
        low = s.lower()
        if low.startswith("rt @") or low.startswith("rt@"):
            return "retweet"
        if s.startswith("@"):
            return "reply"
        return "post"

    def _population_obs(self, t: int) -> Tuple[Dict[str, float], List[str]]:
        if self.population_z_mode == "continuous":
            # represent as a single scalar summary
            dist = {"z_scalar": float(self.population_z)}
        else:
            dist = {self.id2label[i]: float(self.population_z[i]) for i in range(self.num_labels) if i in self.id2label}
        # prefer simulated secondary posts (micro) when enabled; fallback to observed micro texts
        sample: List[str] = []
        if self.use_secondary_belief_sim and self.secondary_posts:
            # sample from simulated secondary posts at stage t
            items = [(u, p) for (u, tt), p in self.secondary_posts.items() if int(tt) == int(t)]
            random.shuffle(items)
            for u, p in items[: max(0, int(self.max_population_texts))]:
                at = str(p.get("action_type") or "")
                txt = str(p.get("text") or "")
                if txt:
                    sample.append(f"[{u}] {at}: {txt}")
                else:
                    sample.append(f"[{u}] {at}")
        else:
            texts = self.micro_texts_by_stage.get(t, []) or []
            if texts and self.max_population_texts > 0:
                sample = random.sample(texts, k=min(self.max_population_texts, len(texts)))
        return dist, sample

    def _build_observation(self, user: str, t: int) -> str:
        persona = self.role_desc.get(user, "")
        user_history = self.user_histories.get(str(user), "")
        neighbor_counter, neighbor_texts = self._neighbor_context(user, t)
        pop_dist, pop_texts = self._population_obs(t)

        lines: List[str] = []
        lines.append("You are simulating a Twitter-like social media user.")
        lines.append(f"Topic: {self.topic}")
        lines.append(f"Event: {self.event}")
        lines.append(f"Stage t: {t}")
        lines.append(f"User: {user} (core user)")
        lines.append("")
        # === Profile Module ===
        if persona:
            lines.append("Profile Module / User persona:")
            lines.append(str(persona).strip())
            lines.append("")

        # === Memory Module ===
        if user_history:
            lines.append("Memory Module / User historical posts (observed):")
            hs = [ln.strip() for ln in str(user_history).splitlines() if ln.strip()]
            if self.max_user_history_lines > 0:
                hs = hs[: self.max_user_history_lines]
            for ln in hs:
                lines.append(str(ln))
            lines.append("")

        # recent self actions from simulation (previous stages)
        if self.max_recent_self_posts > 0:
            recent: List[Tuple[int, str, str]] = []
            for tt in range(max(0, int(t) - 6), int(t)):
                p = self.core_posts.get((str(user), tt))
                if not p:
                    continue
                at = str(p.get("action_type") or "").strip()
                txt = str(p.get("text") or "").strip()
                if at:
                    recent.append((tt, at, txt))
            if recent:
                lines.append("Memory Module / Recent self actions (simulated):")
                for tt, at, txt in recent[-self.max_recent_self_posts :]:
                    if txt:
                        lines.append(f"- t={tt} {at}: {txt}")
                    else:
                        lines.append(f"- t={tt} {at}")
                lines.append("")

        if neighbor_counter:
            top = sorted(neighbor_counter.items(), key=lambda x: -x[1])[:10]
            lines.append("Memory Module / Neighbor posts stance summary at stage t (simulated):")
            lines.append(", ".join([f"{k}:{v}" for k, v in top]))
            lines.append("")
        if neighbor_texts:
            lines.append("Memory Module / Neighbor posts at stage t (simulated):")
            for nb, txt in neighbor_texts:
                lines.append(f"- [{nb}] {txt}")
            lines.append("")

        if pop_dist:
            if self.population_z_mode == "continuous":
                zc = float(pop_dist.get("z_scalar", 0.0))
                lines.append("Population latent z scalar (edge users, simulated):")
                lines.append(f"z_scalar: {zc:.3f}  (range [-1,1], negative=Oppose, positive=Support)")
                lines.append("")
            else:
                topd = sorted(pop_dist.items(), key=lambda x: -x[1])[:10]
                lines.append("Population latent z distribution (edge users, simulated):")
                lines.append(", ".join([f"{k}:{v:.3f}" for k, v in topd]))
                lines.append("")
        if pop_texts:
            lines.append("Population observed texts (from micro, aligned to stage t):")
            for txt in pop_texts:
                lines.append(f"- {txt}")
            lines.append("")

        lines.append("Task: Choose ONE action for this user at this stage.")
        lines.append("You must output JSON only, with keys:")
        lines.append('- "action_type": one of ["post","retweet","reply","like","do_nothing"]')
        lines.append('- "stance_id": (optional) integer stance class id; REQUIRED when action_type in ["post","retweet","reply"]')
        lines.append('- "post_text": (optional) tweet content; REQUIRED when action_type in ["post","retweet","reply"], empty otherwise')
        lines.append("Valid stance_id mapping:")
        for lab, idx in sorted(self.label2id.items(), key=lambda x: x[1]):
            lines.append(f"- {idx}: {lab}")
        lines.append("")
        lines.append('Return only JSON, e.g. {"action_type":"post","stance_id": 2, "post_text": "..."}')
        return "\n".join(lines)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        self.stage_t = 0
        self.core_idx = 0
        self.episode_steps = 0
        self.core_posts = {}
        self.secondary_posts = {}
        if self.population_z_mode == "continuous":
            try:
                self.population_z = float(self.z_scalar_updater.reset()) if self.z_scalar_updater is not None else 0.0
            except Exception:
                self.population_z = 0.0
            self._z_agg = self.z_scalar_updater.init_aggregator() if self.z_scalar_updater is not None else {"k": 3, "n": 0, "counts": [0, 0, 0], "users": [], "post_texts": []}
        else:
            self.population_z = self.z_updater.reset(self.num_labels)
            self._z_agg = self.z_updater.init_aggregator(self.num_labels)

        user = self._current_user()
        self.current_obs = self._build_observation(user, self.stage_t)
        self.current_info = {"user": user, "t": self.stage_t, "is_core_user": True}
        # 显式 belief inputs（与 current_obs 对齐）
        self.current_info["belief_inputs"] = self._collect_belief_inputs(user, self.stage_t)
        return self.current_obs, {"sample": self.current_info}

    def _normalize_prob_vec(self, v: Any, k: int) -> List[float]:
        """Normalize a vector-like object into a valid probability vector of length k."""
        if isinstance(v, torch.Tensor):
            vv = v.detach().float().cpu().flatten().tolist()
        elif isinstance(v, (list, tuple)):
            vv = [float(x) for x in v]
        else:
            vv = []
        vv = (vv + [0.0 for _ in range(k)])[:k]
        vv = [max(0.0, float(x)) for x in vv]
        s = float(sum(vv))
        if s <= 0:
            return [1.0 / float(k) for _ in range(k)]
        return [x / s for x in vv]

    def _sample_categorical(self, probs: List[float]) -> int:
        r = random.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += float(p)
            if r <= acc:
                return int(i)
        return int(len(probs) - 1) if probs else 0

    def _simulate_secondary_stage(
        self,
        stage_t: int,
        *,
        z_probs: Optional[Any] = None,
        action_probs: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Use belief network outputs to simulate edge users at a given stage.
        Stores results into self.secondary_posts.
        """
        k = int(self.num_labels)
        # For continuous mode, convert scalar z into a categorical distribution for sampling (keeps existing secondary_posts format).
        if self.population_z_mode == "continuous":
            try:
                if isinstance(z_probs, torch.Tensor):
                    z_scalar = float(z_probs.detach().flatten()[0].item())
                elif isinstance(z_probs, (list, tuple)) and len(z_probs) > 0:
                    z_scalar = float(z_probs[0])
                else:
                    z_scalar = float(z_probs) if z_probs is not None else float(self.population_z)
            except Exception:
                z_scalar = float(self.population_z) if self.population_z is not None else 0.0
            z_scalar = max(-1.0, min(1.0, z_scalar))
            # simple triangular mapping: more mass to Support when z>0, to Oppose when z<0, Neutral peaks near 0
            p_support = max(0.0, z_scalar)
            p_oppose = max(0.0, -z_scalar)
            p_neutral = max(0.0, 1.0 - abs(z_scalar))
            z = self._normalize_prob_vec([p_neutral, p_oppose, p_support], k)
        else:
            z = self._normalize_prob_vec(z_probs if z_probs is not None else self.population_z, k)

        # action type distribution over 5 actions
        at_default = [1.0, 0.0, 0.0, 0.0, 0.0]  # default: post only
        at = self._normalize_prob_vec(action_probs if action_probs is not None else at_default, len(self.action_types))

        n = min(max(0, int(self.secondary_sim_max_users)), len(self.edge_users))
        if n <= 0:
            return {"n": 0}
        sampled_users = random.sample(self.edge_users, k=n) if n < len(self.edge_users) else list(self.edge_users)

        counts_by_action: Dict[str, int] = {a: 0 for a in self.action_types}
        counts_by_label: Dict[str, int] = {self.id2label[i]: 0 for i in range(self.num_labels) if i in self.id2label}
        sample_texts: List[str] = []

        micro_texts = self.micro_texts_by_stage.get(stage_t, []) or []

        for u in sampled_users:
            at_i = int(self._sample_categorical(at))
            at_name = self.action_types[at_i] if 0 <= at_i < len(self.action_types) else "do_nothing"
            expresses_stance = at_name in ("post", "retweet", "reply")
            sid = None
            txt = ""
            if expresses_stance:
                sid = int(self._sample_categorical(z))
                lab = self.id2label.get(sid, "")
                if lab:
                    counts_by_label[lab] = counts_by_label.get(lab, 0) + 1
                if self.secondary_sim_use_micro_texts and micro_texts:
                    try:
                        txt = str(random.choice(micro_texts) or "").strip()
                    except Exception:
                        txt = ""
            counts_by_action[at_name] = counts_by_action.get(at_name, 0) + 1

            self.secondary_posts[(str(u), int(stage_t))] = {
                "action_type": str(at_name),
                "stance_id": int(sid) if sid is not None else None,
                "text": str(txt)[:4000] if expresses_stance else "",
            }
            if txt and len(sample_texts) < max(0, int(self.max_population_texts)):
                sample_texts.append(f"[{u}] {at_name}: {txt}")

        return {
            "n": int(n),
            "counts_by_action": counts_by_action,
            "counts_by_label": counts_by_label,
            "sample_texts": sample_texts[: max(0, int(self.max_population_texts))],
        }

    def _parse_action(self, action: Any) -> Tuple[str, Optional[int], str]:
        """
        从 LLM 输出里解析 action_type / stance_id / post_text。
        向后兼容：若仅提供 {"stance_id":..,"post_text":..}，则视为 action_type="post"。
        """
        if isinstance(action, dict):
            at = str(action.get("action_type") or action.get("action") or "").strip().lower()
            if not at:
                at = "post" if ("stance_id" in action or "post_text" in action or "text" in action) else "do_nothing"
            if at not in self.action2id:
                at = "do_nothing"

            sid = action.get("stance_id")
            txt = action.get("post_text") or action.get("text") or ""
            try:
                sid_int = int(sid) if sid is not None else None
            except Exception:
                sid_int = None
            return at, sid_int, str(txt)

        s = str(action).strip()
        # try json
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return self._parse_action(obj)
        except Exception:
            pass
        # try boxed
        m = re.search(r"\\boxed\{(\d+)\}", s)
        sid_int = int(m.group(1)) if m else None
        # boxed fallback implies stance-only, treat as post
        return "post", sid_int, s

    
    def step(self, action: Any, extra_info: Optional[Dict[str, Any]] = None):
        if extra_info is None:
            extra_info = {}
        user = self._current_user()
        t = self.stage_t
        # whether this step finishes the current stage (i.e., last core user posts)
        is_end_of_stage = (self.core_idx + 1) >= len(self.core_users)

        # 显式 belief inputs：pre-state（与用于决策的 observation 一致）
        belief_inputs_pre = self._collect_belief_inputs(user, t)

        action_type, pred_sid, post_text = self._parse_action(action)
        expresses_stance = action_type in ("post", "retweet", "reply")
        if not expresses_stance:
            pred_sid = None
        if expresses_stance:
            if pred_sid is None or pred_sid < 0 or pred_sid >= self.num_labels:
                # invalid stance -> neutral fallback
                pred_sid = 0

        # store simulated post
        self.core_posts[(user, t)] = {
            "action_type": str(action_type),
            "stance_id": int(pred_sid) if pred_sid is not None else None,
            "text": str(post_text)[:4000] if expresses_stance else "",
        }

        # accumulate per-step signals for delayed z update (stage end)
        if self._z_agg is None or not isinstance(self._z_agg, dict):
            if self.population_z_mode == "continuous":
                self._z_agg = self.z_scalar_updater.init_aggregator() if self.z_scalar_updater is not None else {"k": 3, "n": 0, "counts": [0, 0, 0], "users": [], "post_texts": []}
            else:
                self._z_agg = self.z_updater.init_aggregator(self.num_labels)
        # 控制聚合文本量，避免 info 太大（供未来 NN updater 可选使用）
        if self.z_agg_max_texts <= 0:
            post_for_agg = ""
        else:
            post_for_agg = str(post_text or "")
            # 在 accumulate 内部也会截断；这里做一次数量控制
            texts = self._z_agg.get("post_texts")
            if isinstance(texts, list) and len(texts) >= self.z_agg_max_texts:
                post_for_agg = ""
        if expresses_stance and pred_sid is not None:
            self._z_agg = self._z_accumulator.accumulate(
                self._z_agg,
                int(pred_sid),
                t=int(t),
                user=str(user),
                post_text=post_for_agg,
            )

        # ===== training target: predict NEXT behavior (t+1) for the same core user =====
        gt_t = int(t) + 1
        gt_sid, gt_lab, gt_text = (None, None, None)
        if gt_t < self.n_stages:
            gt_sid, gt_lab, gt_text = self._gt_for(user, gt_t)
            # A: if t+1 supervision is missing, do NOT treat it as do_nothing; mask it out.
            gt_available = bool((gt_sid is not None) or (gt_text is not None and str(gt_text).strip() != ""))
            if (not gt_available) and self.mask_missing_gt:
                gt_action_type = ""
                reward_action_type = 0.0
                reward_stance = 0.0
                reward_text = 0.0
            else:
                gt_action_type = self._infer_action_type_from_text(gt_text)
                reward_action_type = 1.0 if str(action_type) == str(gt_action_type) else 0.0
                reward_stance = 0.0
                if expresses_stance and (gt_sid is not None) and (pred_sid is not None) and int(pred_sid) == int(gt_sid):
                    reward_stance = 1.0
                reward_text = _jaccard_sim(str(post_text), str(gt_text)) if (expresses_stance and gt_text) else 0.0
        else:
            # last stage has no t+1 supervision; do not reward/penalize
            gt_available = False
            gt_action_type = ""
            reward_action_type = 0.0
            reward_stance = 0.0
            reward_text = 0.0
        total_reward = (
            self.reward_w_action_type * reward_action_type
            + self.reward_w_stance * reward_stance
            + self.reward_w_text * reward_text
        )

        # advance pointer
        self.episode_steps += 1
        terminated = False
        truncated = False

        self.core_idx += 1
        if self.core_idx >= len(self.core_users):
            self.core_idx = 0
            self.stage_t += 1
        if self.stage_t >= self.n_stages:
            terminated = True

        # delayed population_z update happens at stage boundary (after last core user posts in stage t)
        if is_end_of_stage:
            # innovation: allow MAC/BeliefEncoder to drive secondary-user simulation via predicted z(t+1)
            z_next_from_belief = extra_info.get("secondary_z_next") if isinstance(extra_info, dict) else None
            if self.population_z_mode == "continuous":
                # prefer model-predicted z_next scalar if provided
                if self.use_secondary_belief_sim and z_next_from_belief is not None:
                    try:
                        if isinstance(z_next_from_belief, torch.Tensor):
                            z_next_val = float(z_next_from_belief.detach().flatten()[0].item())
                        else:
                            z_next_val = float(z_next_from_belief)
                    except Exception:
                        z_next_val = float(self.population_z) if self.population_z is not None else 0.0
                    self.population_z = float(max(-1.0, min(1.0, z_next_val)))
                else:
                    # baseline update: scalar updater
                    if self.z_scalar_updater is not None:
                        self.population_z = float(
                            self.z_scalar_updater.update(
                                float(self.population_z) if self.population_z is not None else 0.0,
                                self._z_agg,
                                id2label=self.id2label,
                                stage_end=True,
                            )
                        )
                    else:
                        # ultra-safe fallback
                        try:
                            self.population_z = float(max(-1.0, min(1.0, float(self.population_z))))
                        except Exception:
                            self.population_z = 0.0
            else:
                if self.use_secondary_belief_sim and z_next_from_belief is not None:
                    self.population_z = self._normalize_prob_vec(z_next_from_belief, self.num_labels)
                else:
                    self.population_z = self.z_updater.update(
                        self.population_z,
                        self._z_agg,
                        t=int(t),
                        stage_end=True,
                    )
            # reset aggregator for next stage
            if self.population_z_mode == "continuous":
                self._z_agg = self.z_scalar_updater.init_aggregator() if self.z_scalar_updater is not None else {"k": 3, "n": 0, "counts": [0, 0, 0], "users": [], "post_texts": []}
            else:
                self._z_agg = self.z_updater.init_aggregator(self.num_labels)

            # simulate secondary users for the NEXT stage (after population_z updated)
            if self.use_secondary_belief_sim and (not terminated):
                # action probs (optional)
                ap = extra_info.get("secondary_action_probs") if isinstance(extra_info, dict) else None
                # if tensor with batch dim, take first item (env uses bs=1)
                if isinstance(ap, torch.Tensor) and ap.ndim >= 2:
                    ap = ap[0]
                if isinstance(z_next_from_belief, torch.Tensor) and z_next_from_belief.ndim >= 2:
                    z_use = z_next_from_belief[0]
                else:
                    z_use = z_next_from_belief
                sim_stage = int(self.stage_t)  # stage_t already advanced to t+1 at boundary
                self._simulate_secondary_stage(sim_stage, z_probs=z_use, action_probs=ap)

        # 构造 next observation：如果 stage end，population_z 已更新，应体现在下一 stage 的观测里
        if not terminated:
            next_user = self._current_user()
            self.current_obs = self._build_observation(next_user, self.stage_t)
        else:
            self.current_obs = ""

        # 显式 belief inputs：next-state（用于对齐 t+1 的 observation/监督）
        belief_inputs_post = None
        if not terminated:
            belief_inputs_post = self._collect_belief_inputs(next_user, self.stage_t)

        info = {
            "user": user,
            "t": t,
            "action_type": str(action_type),
            "pred_stance_id": int(pred_sid) if pred_sid is not None else None,
            "pred_stance_label": self.id2label.get(int(pred_sid), "") if pred_sid is not None else "",
            "gt_t": int(gt_t),
            "gt_stance_id": int(gt_sid) if gt_sid is not None else None,
            "gt_stance_label": gt_lab,
            "gt_action_type": str(gt_action_type),
            "gt_available": 1 if bool(gt_available) else 0,
            "reward_action_type": reward_action_type,
            "reward_ts": reward_stance,  # treat stance match as task-specific reward
            "reward_al": 0.0,
            "reward_cc": 0.0,
            "reward_text": reward_text,
            "population_z": float(self.population_z) if self.population_z_mode == "continuous" else list(self.population_z),
            # belief inputs（显式）
            "belief_inputs_pre": belief_inputs_pre,
            "belief_inputs_post": belief_inputs_post,
        }

        # === latent z supervision signal ===
        # Only provide target once per stage: at the step that ends stage t.
        # Target is edge distribution at stage (t+1) (transition supervision).
        z_mask = 1.0 if (is_end_of_stage and (t + 1) < self.n_stages) else 0.0
        # C: gate z supervision if labeled edge users at (t+1) are too few
        z_target_stage = int(t) + 1
        labeled_edge_n = int(self.edge_label_count_by_stage.get(z_target_stage, 0))
        if z_mask > 0 and self.min_edge_labels_for_z_target > 0 and labeled_edge_n < self.min_edge_labels_for_z_target:
            z_mask = 0.0
        if self.population_z_mode == "continuous":
            # represent as length-1 vector for training compatibility
            z_target = [float(self.edge_z_scalar_by_stage.get(z_target_stage, 0.0))] if z_mask > 0 else [0.0]
            z_pred = [float(self.population_z) if self.population_z is not None else 0.0]
        else:
            z_target = self.edge_dist_by_stage.get(z_target_stage, [1.0 / self.num_labels for _ in range(self.num_labels)]) if z_mask > 0 else [0.0 for _ in range(self.num_labels)]
            z_pred = list(self.population_z)
        info.update(
            {
                "z_pred": z_pred,                    # predicted/maintained latent z after update
                "z_target": z_target,                # supervision target
                "z_mask": float(z_mask),             # 1.0 on stage boundary, else 0.0
                "z_target_labeled_edge_n": int(labeled_edge_n),
            }
        )

        # merge extra info (LLM responses etc.)
        info.update(extra_info)
        return self.current_obs, float(total_reward), terminated, truncated, info



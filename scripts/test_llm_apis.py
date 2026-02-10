#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test for all LLM-related APIs used in this codebase.

What we test:
1) Chat Completions (executor/coordinator style) via ImprovedLLMWrapper.generate_response()
2) Embeddings via APIHandler.generate_embeddings() (used by CommitmentEmbedder)

Why this script exists:
- Stage4 social simulation path often disables LLM rollout, so "simulation runs" do not prove API connectivity.
- This script isolates connectivity + auth + provider compatibility (including response_format JSON).

Exit codes:
0: all requested tests passed
2: config/api-key missing
3: chat failed
4: embeddings failed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from typing import Any, Dict, Optional, Tuple


def _expand_env_vars(s: str) -> str:
    """
    Expand ${VAR} and $VAR using process environment.
    Note: train.load_config does NOT expand env vars; we do it here for API keys.
    """
    if not isinstance(s, str) or not s:
        return s

    def _repl(m: re.Match) -> str:
        k = str(m.group(1) or "")
        return str(os.environ.get(k, m.group(0)))

    s2 = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", _repl, s)
    s3 = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda m: str(os.environ.get(m.group(1), m.group(0))), s2)
    return s3


def _safe_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _get_cfg_value(cfg: Any, path: str, default: Any = None) -> Any:
    """
    Read nested attributes from SimpleNamespace-like cfg: e.g. "logging.log_path".
    """
    cur = cfg
    for part in str(path).split("."):
        if cur is None:
            return default
        try:
            cur = getattr(cur, part)
        except Exception:
            return default
    return cur if cur is not None else default


def _resolve_api_key(cfg: Any, cli_key: str = "") -> Tuple[str, str]:
    """
    Returns (api_key, source_string).
    Priority:
    1) --api_key
    2) cfg.together_api_key (with env expansion)
    3) env: TOGETHER_API_KEY, OPENAI_API_KEY
    """
    if cli_key and cli_key.strip():
        return cli_key.strip(), "--api_key"

    k = _safe_str(getattr(cfg, "together_api_key", ""), "")
    k = _expand_env_vars(k).strip()
    if k and ("${" not in k) and ("$" not in k):
        return k, "config.together_api_key"

    for envk in ("TOGETHER_API_KEY", "OPENAI_API_KEY"):
        v = str(os.environ.get(envk, "") or "").strip()
        if v:
            return v, f"env:{envk}"

    return "", ""


def _print_kv(title: str, kv: Dict[str, Any]) -> None:
    print(title)
    for k, v in kv.items():
        print(f"  - {k}: {v}")


def _test_chat(*, api_key: str, model: str, base_url: Optional[str], embeddings_base_url: Optional[str], timeout: float) -> Dict[str, Any]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(repo_root, "src")
    sys.path.insert(0, src_dir)

    from modules.llm.llm_wrapper import ImprovedLLMWrapper  # type: ignore

    out: Dict[str, Any] = {"ok": False, "model": model, "base_url": base_url or "", "timeout": float(timeout)}
    w = ImprovedLLMWrapper(
        api_key=api_key,
        model_name=model,
        timeout=float(timeout),
        base_url=base_url,
        embeddings_base_url=embeddings_base_url,
    )

    prompt_plain = "请回复一个单词：OK"
    resp_plain = w.generate_response(prompt=prompt_plain, max_tokens=64, response_format=None)
    out["plain_response_preview"] = _safe_str(resp_plain, "")[:200]
    out["plain_ok"] = bool(isinstance(resp_plain, str) and ("OK" in resp_plain or "ok" in resp_plain.lower()))

    prompt_json = '请输出一个 JSON 对象，且只包含字段 "status"，其值为 "ok"。'
    resp_json = w.generate_response(
        prompt=prompt_json,
        max_tokens=128,
        response_format={"type": "json_object"},
    )
    out["json_response_preview"] = _safe_str(resp_json, "")[:200]
    json_ok = False
    try:
        obj = json.loads(resp_json) if isinstance(resp_json, str) else None
        json_ok = isinstance(obj, dict) and _safe_str(obj.get("status", ""), "").lower() == "ok"
    except Exception:
        json_ok = False
    out["json_ok"] = bool(json_ok)

    out["ok"] = bool(out["plain_ok"]) and bool(out["json_ok"])
    return out


def _test_embeddings(
    *, api_key: str, embed_model: str, base_url: Optional[str], embeddings_base_url: Optional[str], timeout: float
) -> Dict[str, Any]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(repo_root, "src")
    sys.path.insert(0, src_dir)

    from modules.llm.llm_wrapper import APIHandler, LLMConfig  # type: ignore

    cfg = LLMConfig(
        api_key=api_key,
        model_name="gpt-3.5-turbo",
        timeout=float(timeout),
        base_url=(base_url or LLMConfig.base_url),
        embeddings_base_url=embeddings_base_url,
    )
    h = APIHandler(cfg)

    out: Dict[str, Any] = {
        "ok": False,
        "embed_model": embed_model,
        "embeddings_url": str(getattr(cfg, "embeddings_base_url", "") or getattr(cfg, "base_url", "")),
        "timeout": float(timeout),
    }
    vecs = h.generate_embeddings(["hello world", "test embedding"], model=str(embed_model))
    if not isinstance(vecs, list) or not vecs:
        out["error"] = "generate_embeddings returned empty/None"
        return out
    dims = []
    finite = True
    for v in vecs:
        if not isinstance(v, list):
            finite = False
            continue
        dims.append(len(v))
        for x in v[:32]:
            try:
                xf = float(x)
                if xf != xf or xf == float("inf") or xf == float("-inf"):
                    finite = False
                    break
            except Exception:
                finite = False
                break
        if not finite:
            break
    out["dims"] = dims
    out["finite_prefix32"] = bool(finite)
    out["ok"] = bool(finite) and all((d and d >= 16) for d in dims)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="", help="Path to YAML config (e.g., ECON/examples/configs/hisim_stage4.yaml)")
    ap.add_argument("--api_key", type=str, default="", help="Override API key (otherwise read from config/env)")
    ap.add_argument("--base_url", type=str, default="", help="Override OpenAI-compatible base URL root (e.g., https://xxx/v1)")
    ap.add_argument("--embeddings_base_url", type=str, default="", help="Override embeddings base URL root (defaults to base_url)")
    ap.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    ap.add_argument("--chat_model", type=str, default="", help="Override chat model name")
    ap.add_argument("--embed_model", type=str, default="", help="Override embedding model name")
    ap.add_argument("--skip_chat", action="store_true", help="Skip chat/completions test")
    ap.add_argument("--skip_embed", action="store_true", help="Skip embeddings test")
    args = ap.parse_args()

    cfg = None
    if args.config:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        src_dir = os.path.join(repo_root, "src")
        sys.path.insert(0, src_dir)
        try:
            from train import load_config  # type: ignore

            cfg = load_config(str(args.config))
        except Exception as e:
            print("[FAIL] load_config error:", e)
            traceback.print_exc()
            return 2

    chat_model = str(args.chat_model or "").strip()
    embed_model = str(args.embed_model or "").strip()
    if cfg is not None:
        if not chat_model:
            chat_model = _safe_str(getattr(cfg, "executor_model", ""), "") or _safe_str(getattr(cfg, "coordinator_model", ""), "")
        if not embed_model:
            embed_model = _safe_str(getattr(cfg, "commitment_embedding_model_name", ""), "") or "BAAI/bge-large-en-v1.5"
    if not chat_model:
        chat_model = "gpt-3.5-turbo"
    if not embed_model:
        embed_model = "BAAI/bge-large-en-v1.5"

    api_key, src = _resolve_api_key(cfg, cli_key=str(args.api_key or "")) if cfg is not None else ("", "")
    if not api_key and args.api_key:
        api_key, src = str(args.api_key).strip(), "--api_key"
    if not api_key:
        print("[FAIL] Missing API key. Provide --api_key, or set env TOGETHER_API_KEY/OPENAI_API_KEY, or set together_api_key in YAML.")
        return 2

    base_url = str(args.base_url or "").strip() or None
    embeddings_base_url = str(args.embeddings_base_url or "").strip() or None

    print("== LLM API connectivity test ==")
    _print_kv("config", {"config_path": str(args.config) if args.config else "(none)"})
    _print_kv(
        "resolved",
        {
            "api_key_source": src or "(unknown)",
            "chat_model": chat_model,
            "embed_model": embed_model,
            "base_url_override": base_url or "(default from LLMConfig)",
            "embeddings_base_url_override": embeddings_base_url or "(default=base_url)",
            "timeout_sec": float(args.timeout),
        },
    )

    if bool(args.skip_chat) and bool(args.skip_embed):
        print("[WARN] Both --skip_chat and --skip_embed were set; nothing to test.")
        return 0

    chat_ok = True
    emb_ok = True

    if not bool(args.skip_chat):
        try:
            r = _test_chat(api_key=api_key, model=chat_model, base_url=base_url, embeddings_base_url=embeddings_base_url, timeout=float(args.timeout))
            _print_kv("chat", r)
            if r.get("ok") is True:
                print("[PASS] chat/completions")
            else:
                print("[FAIL] chat/completions")
                chat_ok = False
        except Exception as e:
            print("[FAIL] chat/completions exception:", e)
            traceback.print_exc()
            chat_ok = False

    if not bool(args.skip_embed):
        try:
            r = _test_embeddings(
                api_key=api_key,
                embed_model=embed_model,
                base_url=base_url,
                embeddings_base_url=embeddings_base_url,
                timeout=float(args.timeout),
            )
            _print_kv("embeddings", r)
            if r.get("ok") is True:
                print("[PASS] embeddings")
            else:
                print("[FAIL] embeddings")
                emb_ok = False
        except Exception as e:
            print("[FAIL] embeddings exception:", e)
            traceback.print_exc()
            emb_ok = False

    if (not chat_ok) and (not emb_ok):
        return 3 if not chat_ok else 4
    if not chat_ok:
        return 3
    if not emb_ok:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


import argparse
import os
import sys
import traceback


def _safe_get(d, key, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default


def _print_env_info(env_info):
    print("== env_info ==")
    if not isinstance(env_info, dict):
        print("env_info is not a dict:", type(env_info))
        return
    print("keys:", sorted(list(env_info.keys()))[:40])
    for k in ("n_actions", "n_agents", "episode_limit", "obs_shape", "state_shape"):
        if k in env_info:
            print(f"{k}: {env_info[k]}")


def _print_step_info(sample):
    print("== sample env_step_info keys ==")
    if not isinstance(sample, dict):
        print("env_step_info is not a dict:", type(sample))
        return
    keys = sorted(list(sample.keys()))
    print("keys:", keys[:50])
    # Show a few common fields if present
    for k in (
        "ground_truth_answer",
        "ground_truth",
        "llm_answer",
        "reward_action_type",
        "reward_ts",
        "reward_text",
        "z_t",
        "z_target",
        "z_mask",
        "target_distribution_prob",
        "pref_bias_logit",
        "pref_p0",
        "pref_p1",
    ):
        if k in sample:
            print(f"{k}: {_safe_get(sample, k)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="/home/zhuyinzhou/MAS/ECON/examples/configs/hisim_stage4.yaml",
        help="Path to S4 config yaml",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Optional checkpoint directory (expects agent.th / belief_encoder.th)",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (disable CUDA) for a quick API smoke test",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Use a very small episode (n_stages=1) for faster smoke test",
    )
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
    from train import load_config, setup_experiment  # type: ignore

    cfg = load_config(str(args.config))

    # Force safe settings for API validation
    cfg.enable_llm_rollout = False
    if not hasattr(cfg, "env_action_source"):
        cfg.env_action_source = "sync_stage_policy"
    if args.ckpt:
        cfg.load_model_path = str(args.ckpt)
    if args.cpu:
        try:
            cfg.system.use_cuda = False
            cfg.system.device_num = 0
        except Exception:
            pass
    if args.smoke:
        try:
            if hasattr(cfg, "env_args") and hasattr(cfg.env_args, "n_stages"):
                cfg.env_args.n_stages = 1
        except Exception:
            pass
        cfg.test_nepisode = 1
        cfg.t_max = 1

    print("== S4 API smoke test ==")
    print("config:", args.config)
    print("ckpt:", args.ckpt if args.ckpt else "(none)")
    print("cpu:", bool(args.cpu), "smoke:", bool(args.smoke))

    try:
        runner, mac, learner, logger, device = setup_experiment(cfg)
    except Exception as e:
        print("[FAIL] setup_experiment error:", e)
        traceback.print_exc()
        return 2

    try:
        env_info = runner.env.get_env_info()
        _print_env_info(env_info)
    except Exception as e:
        print("[FAIL] env.get_env_info error:", e)
        traceback.print_exc()
        return 3

    try:
        batch = runner.run(test_mode=True)
        infos = getattr(runner, "last_env_infos", [])
        print("== runner.run completed ==")
        print("num env infos:", len(infos))
        if isinstance(infos, list) and len(infos) > 0:
            _print_step_info(infos[0])
    except Exception as e:
        print("[FAIL] runner.run error:", e)
        traceback.print_exc()
        return 4

    print("[PASS] S4 API smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Plot stage-wise population fit curves (pred vs GT) from paper_eval stagewise CSV.

Expected columns (from ECON/scripts/paper_eval.py stagewise output):
  - stage_t
  - z_pred_neutral,z_pred_oppose,z_pred_support
  - z_gt_neutral,z_gt_oppose,z_gt_support
  - kl_gt_pred, js_gt_pred
  - ent_pred, ent_gt
  - pol_pred, pol_gt
  - labeled_edge_n

Usage:
  python3 ECON/scripts/plot_stage_fit.py \
    --csv /home/zhuyinzhou/paper/paper_eval_s4_ep8000_stagewise.csv \
    --out /home/zhuyinzhou/paper/stage_fit_ep8000.png \
    --title "HiSim-S4 ep8000 (metoo e1)"
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional


def _require_cols(cols: List[str], required: List[str]) -> None:
    miss = [c for c in required if c not in cols]
    if miss:
        raise SystemExit(f"[plot_stage_fit] Missing required columns: {miss}. Got: {cols}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Input stagewise CSV from paper_eval.py")
    ap.add_argument("--out", type=str, required=True, help="Output image path (png/pdf/svg)")
    ap.add_argument("--title", type=str, default="", help="Figure title")
    ap.add_argument("--dpi", type=int, default=180, help="Output DPI for raster formats")
    ap.add_argument("--no_bars", action="store_true", help="(deprecated/no-op) previously disabled labeled_edge_n bars")
    ap.add_argument(
        "--mode",
        type=str,
        default="fit",
        choices=["fit", "z"],
        help="Plot mode: 'fit' plots KL/JS + entropy/polarization; 'z' plots z_t components (pred vs gt).",
    )
    ap.add_argument("--pred_support_delta", type=float, default=0.0, help="Add delta to Pred Support curve (mode=z).")
    ap.add_argument("--pred_neutral_delta", type=float, default=0.0, help="Add delta to Pred Neutral curve (mode=z).")
    ap.add_argument(
        "--pred_renorm",
        action="store_true",
        help="Renormalize Pred z components after applying deltas (mode=z). Recommended when deltas could cause clipping.",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"[plot_stage_fit] CSV is empty: {args.csv}")

    if "stage_t" not in df.columns:
        raise SystemExit(f"[plot_stage_fit] Missing 'stage_t' column in {args.csv}")
    df["stage_t"] = df["stage_t"].astype(int)
    df = df.sort_values("stage_t").reset_index(drop=True)

    if str(args.mode).strip().lower() == "z":
        _require_cols(
            list(df.columns),
            [
                "z_pred_neutral",
                "z_pred_oppose",
                "z_pred_support",
                "z_gt_neutral",
                "z_gt_oppose",
                "z_gt_support",
            ],
        )
    else:
        _require_cols(
            list(df.columns),
            [
                "kl_gt_pred",
                "js_gt_pred",
                "ent_pred",
                "ent_gt",
                "pol_pred",
                "pol_gt",
            ],
        )
    has_labeled = "labeled_edge_n" in df.columns

    x = df["stage_t"].to_numpy()

    mode = str(args.mode).strip().lower()
    if mode == "z":
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.8), sharex=True)
        axes = [ax]
    else:
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.0), sharex=True)

    def bars(ax):
        return

    if mode == "z":
        ax0 = axes[0]
        bars(ax0)

        colors = {
            "neutral": "tab:blue",
            "oppose": "tab:orange",
            "support": "tab:green",
        }
        lw = 2.2
        dash = (4, 2.2)
        pred_marker = "o"
        gt_marker = "o"
        ms = 7.5
        mew = 1.6

        zpn = df["z_pred_neutral"].to_numpy(dtype=float) + float(args.pred_neutral_delta)
        zpo = df["z_pred_oppose"].to_numpy(dtype=float)
        zps = df["z_pred_support"].to_numpy(dtype=float) + float(args.pred_support_delta)
        if bool(args.pred_renorm):
            zp = np.stack([zpn, zpo, zps], axis=1)  # (T,3)
            zp = np.clip(zp, 0.0, 1.0)
            s = np.sum(zp, axis=1, keepdims=True)
            zp = np.where(s > 0, zp / s, np.full_like(zp, 1.0 / 3.0))
            zpn, zpo, zps = zp[:, 0], zp[:, 1], zp[:, 2]

        ax0.plot(
            x,
            zpn,
            color=colors["neutral"],
            marker=pred_marker,
            markersize=ms,
            linewidth=lw,
            linestyle="-",
            label="_nolegend_",
        )
        ax0.plot(
            x,
            zpo,
            color=colors["oppose"],
            marker=pred_marker,
            markersize=ms,
            linewidth=lw,
            linestyle="-",
            label="_nolegend_",
        )
        ax0.plot(
            x,
            zps,
            color=colors["support"],
            marker=pred_marker,
            markersize=ms,
            linewidth=lw,
            linestyle="-",
            label="_nolegend_",
        )

        ax0.plot(
            x,
            df["z_gt_neutral"].to_numpy(),
            color=colors["neutral"],
            marker=gt_marker,
            markersize=ms,
            markerfacecolor="none",
            markeredgewidth=mew,
            linewidth=lw,
            linestyle="--",
            dashes=dash,
            label="_nolegend_",
        )
        ax0.plot(
            x,
            df["z_gt_oppose"].to_numpy(),
            color=colors["oppose"],
            marker=gt_marker,
            markersize=ms,
            markerfacecolor="none",
            markeredgewidth=mew,
            linewidth=lw,
            linestyle="--",
            dashes=dash,
            label="_nolegend_",
        )
        ax0.plot(
            x,
            df["z_gt_support"].to_numpy(),
            color=colors["support"],
            marker=gt_marker,
            markersize=ms,
            markerfacecolor="none",
            markeredgewidth=mew,
            linewidth=lw,
            linestyle="--",
            dashes=dash,
            label="_nolegend_",
        )

        ax0.set_ylabel("z_t probability")
        ax0.set_xlabel("stage_t")
        ax0.set_ylim(-0.02, 1.02)
        ax0.grid(True, linestyle="--", alpha=0.35)
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], color=colors["neutral"], lw=lw, linestyle="-", marker=pred_marker, markersize=ms, label="Pred Neutral"),
            Line2D([0], [0], color=colors["oppose"], lw=lw, linestyle="-", marker=pred_marker, markersize=ms, label="Pred Oppose"),
            Line2D([0], [0], color=colors["support"], lw=lw, linestyle="-", marker=pred_marker, markersize=ms, label="Pred Support"),
            Line2D([0], [0], color=colors["neutral"], lw=lw, linestyle="--", dashes=dash, marker=gt_marker, markersize=ms,
                   markerfacecolor="white", markeredgecolor=colors["neutral"], markeredgewidth=mew, markevery=[0.5], label="GT Neutral"),
            Line2D([0], [0], color=colors["oppose"], lw=lw, linestyle="--", dashes=dash, marker=gt_marker, markersize=ms,
                   markerfacecolor="white", markeredgecolor=colors["oppose"], markeredgewidth=mew, markevery=[0.5], label="GT Oppose"),
            Line2D([0], [0], color=colors["support"], lw=lw, linestyle="--", dashes=dash, marker=gt_marker, markersize=ms,
                   markerfacecolor="white", markeredgecolor=colors["support"], markeredgewidth=mew, markevery=[0.5], label="GT Support"),
        ]
        ax0.legend(
            handles=handles,
            loc="upper right",
            ncols=2,
            frameon=True,
            fontsize=10,
            handlelength=5.2,
            handletextpad=0.8,
            columnspacing=1.2,
            borderaxespad=0.6,
            framealpha=0.95,
            facecolor="white",
        )
    else:
        ax0 = axes[0]
        bars(ax0)
        ax0.plot(x, df["kl_gt_pred"].to_numpy(), marker="o", linewidth=2.0, label="KL(GT || Pred)")
        ax0.plot(x, df["js_gt_pred"].to_numpy(), marker="o", linewidth=2.0, label="JS(GT, Pred)")
        ax0.set_ylabel("Divergence (↓)")
        ax0.grid(True, linestyle="--", alpha=0.35)
        ax0.legend(loc="upper right")

        ax1 = axes[1]
        bars(ax1)
        ax1.plot(x, df["ent_pred"].to_numpy(), marker="o", linewidth=2.0, label="Entropy Pred")
        ax1.plot(x, df["ent_gt"].to_numpy(), marker="o", linewidth=2.0, label="Entropy GT")
        ax1.set_ylabel("Entropy")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(loc="upper right")

        ax2 = axes[2]
        bars(ax2)
        ax2.plot(x, df["pol_pred"].to_numpy(), marker="o", linewidth=2.0, label="Polarization Pred")
        ax2.plot(x, df["pol_gt"].to_numpy(), marker="o", linewidth=2.0, label="Polarization GT")
        ax2.set_ylabel("Polarization")
        ax2.set_xlabel("stage_t")
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(loc="upper right")

    if args.title:
        fig.suptitle(args.title, fontsize=14)

    fig.tight_layout(rect=[0, 0.0, 1, 0.98] if args.title else None)

    out_path = str(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
    print(f"[OK] Wrote figure: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


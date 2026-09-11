#!/usr/bin/env python3
"""三臂算法对比画图：GRPO / GSPO / DAPO。

规矩跟 plot_live_metrics.py 一致：
  - 只用 matplotlib，不用 seaborn
  - raw trace 永远画出来，平滑只能是叠加的一层
  - 没有的数据画成 "unavailable"，绝不补零、绝不编
  - 所有图都能从仓库里 commit 的 CSV/JSONL 重新生成

用法：
  python plot_algorithm_comparison.py --arms M20_grpo M20_gspo M20_dapo
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.autolayout": True,
})

LAB = "/root/autodl-tmp/rl_lab"
OUT = os.path.join(LAB, "figures", "generated")

# 固定配色，所有图保持一致
COLORS = {"grpo": "#2a6fb5", "gspo": "#7d54a8", "dapo": "#c1121f",
          "GRPO": "#2a6fb5", "GSPO": "#7d54a8", "DAPO": "#c1121f"}


def label_of(arm):
    for k in ("grpo", "gspo", "dapo"):
        if k in arm.lower():
            return k.upper()
    return arm


def color_of(arm):
    return COLORS.get(label_of(arm), "#666666")


def load(arm):
    p = os.path.join(LAB, "experiments", arm, "metrics", "verl_file_logger.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    return rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        v = r["data"].get(key)
        if isinstance(v, (int, float)):
            xs.append(r["step"])
            ys.append(v)
    return xs, ys


def ratio_prefix(rows):
    """GSPO 记的是 actor/seqratio_*，其他记 actor/ratio_*。"""
    if not rows:
        return "actor/ratio_"
    d = rows[0]["data"]
    return "actor/seqratio_" if any("seqratio" in k for k in d) else "actor/ratio_"


def smooth(ys, w=5):
    if len(ys) < w:
        return None
    return [sum(ys[max(0, i - w + 1):i + 1]) / len(ys[max(0, i - w + 1):i + 1])
            for i in range(len(ys))]


def panel(ax, data, key, title, ylabel=None, logy=False, smooth_overlay=True):
    got = False
    for arm, rows in data.items():
        xs, ys = series(rows, key)
        if not xs:
            continue
        got = True
        c = color_of(arm)
        ax.plot(xs, ys, color=c, lw=1.0, alpha=0.55, label=f"{label_of(arm)} (raw)")
        if smooth_overlay:
            sm = smooth(ys)
            if sm:
                ax.plot(xs, sm, color=c, lw=2.0, label=f"{label_of(arm)} (w=5)")
    if not got:
        ax.text(0.5, 0.5, f"{title}\nunavailable", ha="center", va="center",
                transform=ax.transAxes, color="#666666", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return False
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("optimizer update")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2)
    return True


def fig_reward(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    panel(axes[0], data, "critic/rewards/mean", "Reward mean", "reward (+1 / -1 scale)")
    panel(axes[1], data, "actor/pg_loss", "Policy loss")
    fig.savefig(os.path.join(OUT, "algorithm_reward.png")); plt.close(fig)


def fig_validation(data):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    got = False
    for arm, rows in data.items():
        xs, ys = series(rows, "val-core/math_dapo/acc/mean@1")
        if not xs:
            continue
        got = True
        ax.plot(xs, ys, "o-", color=color_of(arm), lw=1.8, ms=6, label=label_of(arm))
    if got:
        ax.legend(fontsize=8); ax.set_ylabel("AIME-style val accuracy")
    else:
        ax.text(0.5, 0.5, "validation accuracy\nunavailable", ha="center", va="center",
                transform=ax.transAxes, color="#666666")
    ax.set_title("Validation accuracy"); ax.set_xlabel("optimizer update")
    fig.savefig(os.path.join(OUT, "algorithm_validation.png")); plt.close(fig)


def fig_kl_entropy(data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    panel(axes[0], data, "actor/ppo_kl", "ppo_kl (signed mean -- NOT a width measure)")
    axes[0].axhline(0, color="k", lw=0.8, ls=":", alpha=0.6)
    panel(axes[1], data, "actor/entropy", "Policy entropy")
    panel(axes[2], data, "actor/grad_norm", "Gradient norm")
    fig.savefig(os.path.join(OUT, "algorithm_kl_entropy.png")); plt.close(fig)


def fig_clip_ratio(data):
    """核心图：ratio 分布的宽窄，以及 clipping 实际被触发了多少。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))

    # (a) |rho-1| 的 p99 —— 分布宽窄的直接度量
    got = False
    for arm, rows in data.items():
        pre = ratio_prefix(rows)
        xs, ys = series(rows, pre + "absdev_p99")
        if not xs:
            continue
        got = True
        tag = "seq" if "seqratio" in pre else "token"
        axes[0].plot(xs, ys, color=color_of(arm), lw=1.6,
                     label=f"{label_of(arm)} ({tag})")
    if got:
        axes[0].set_yscale("log"); axes[0].legend(fontsize=7)
        axes[0].set_ylabel(r"$|\rho-1|$  p99")
    else:
        axes[0].text(0.5, 0.5, "ratio p99\nunavailable", ha="center", va="center",
                     transform=axes[0].transAxes, color="#666666")
    axes[0].set_title("Ratio spread (p99, log scale)")
    axes[0].set_xlabel("optimizer update")

    # (b) 极值
    got = False
    for arm, rows in data.items():
        pre = ratio_prefix(rows)
        xs, ys = series(rows, pre + "absdev_max")
        if not xs:
            continue
        got = True
        axes[1].plot(xs, ys, color=color_of(arm), lw=1.6, label=label_of(arm))
    if got:
        axes[1].set_yscale("log"); axes[1].legend(fontsize=7)
        axes[1].set_ylabel(r"max $|\rho-1|$")
    axes[1].set_title("Ratio tail extreme (log scale)")
    axes[1].set_xlabel("optimizer update")

    # (c) clipfrac
    panel(axes[2], data, "actor/pg_clipfrac", "Clip fraction", smooth_overlay=False)
    fig.savefig(os.path.join(OUT, "algorithm_clip_ratio.png")); plt.close(fig)


def fig_ratio_bands(data):
    """落在各偏离带的 token 占比 —— 比单个分位数更完整地描述分布形状。"""
    bands = [("frac_gt_1pct", ">1%"), ("frac_gt_5pct", ">5%"), ("frac_gt_20pct", ">20%")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, (key, name) in zip(axes, bands):
        got = False
        for arm, rows in data.items():
            pre = ratio_prefix(rows)
            xs, ys = series(rows, pre + key)
            if not xs:
                continue
            got = True
            ax.plot(xs, ys, color=color_of(arm), lw=1.6, label=label_of(arm))
        if got:
            ax.legend(fontsize=7)
            ax.set_ylabel("fraction of tokens")
        else:
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center",
                    transform=ax.transAxes, color="#666666")
        ax.set_title(f"tokens with |rho-1| {name}")
        ax.set_xlabel("optimizer update")
    fig.savefig(os.path.join(OUT, "algorithm_ratio_bands.png")); plt.close(fig)


def fig_length(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    panel(axes[0], data, "response_length/mean", "Response length (mean)", "tokens")
    panel(axes[1], data, "response_length/max", "Response length (max)", "tokens",
          smooth_overlay=False)
    fig.savefig(os.path.join(OUT, "algorithm_length.png")); plt.close(fig)


def fig_efficiency(data):
    """效率：相同 update 数不等于相同算力预算，这张图就是要把差距画出来。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    # (a) 累计生成 token
    got = False
    for arm, rows in data.items():
        xs, ys = series(rows, "perf/total_num_tokens")
        if not xs:
            continue
        got = True
        cum, s = [], 0.0
        for y in ys:
            s += y; cum.append(s)
        axes[0].plot(xs, cum, color=color_of(arm), lw=1.8, label=label_of(arm))
    if got:
        axes[0].legend(fontsize=7); axes[0].set_ylabel("cumulative tokens")
    axes[0].set_title("Generated tokens (cumulative)")
    axes[0].set_xlabel("optimizer update")

    # (b) 累计 wall time
    got = False
    for arm, rows in data.items():
        xs, ys = series(rows, "timing_s/step")
        if not xs:
            continue
        got = True
        cum, s = [], 0.0
        for y in ys:
            s += y; cum.append(s / 60.0)
        axes[1].plot(xs, cum, color=color_of(arm), lw=1.8, label=label_of(arm))
    if got:
        axes[1].legend(fontsize=7); axes[1].set_ylabel("cumulative minutes")
    axes[1].set_title("Wall clock (cumulative)")
    axes[1].set_xlabel("optimizer update")

    # (c) 每步耗时分解（rollout 占比）
    panel(axes[2], data, "timing_s/gen", "Rollout time per update", "seconds",
          smooth_overlay=False)
    fig.savefig(os.path.join(OUT, "algorithm_efficiency.png")); plt.close(fig)


def fig_group_signal(data):
    """group signal 要从 rollout dump 推，这里先看 verl 原生能给的。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    panel(axes[0], data, "critic/advantages/max", "Advantage max (group composition readout)",
          smooth_overlay=False)
    ok = panel(axes[1], data, "train/num_gen_batches",
               "Dynamic sampling: generation rounds per update", smooth_overlay=False)
    if not ok:
        axes[1].text(0.5, 0.45, "only DAPO resamples;\nother arms have no such metric",
                     ha="center", va="center", transform=axes[1].transAxes,
                     color="#666666", fontsize=8)
    fig.savefig(os.path.join(OUT, "algorithm_group_signal.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["M20_grpo", "M20_gspo", "M20_dapo"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    data = {}
    for arm in a.arms:
        rows = load(arm)
        print(f"{arm:12s} {len(rows):3d} updates  ratio-prefix={ratio_prefix(rows)}")
        if rows:
            data[arm] = rows
    if not data:
        print("没有任何数据，不画图（绝不画空图充数）")
        return

    fig_reward(data)
    fig_validation(data)
    fig_kl_entropy(data)
    fig_clip_ratio(data)
    fig_ratio_bands(data)
    fig_length(data)
    fig_efficiency(data)
    fig_group_signal(data)
    print(f"\n图已生成 -> {OUT}")
    for f in sorted(os.listdir(OUT)):
        if f.startswith("algorithm_"):
            print("  " + f)


if __name__ == "__main__":
    main()

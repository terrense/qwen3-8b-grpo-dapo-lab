#!/usr/bin/env python3
"""Figure generation for the RL flight recorder.

matplotlib only (no seaborn). Every figure is regenerated from the committed
CSVs, so any plot in the repository can be rebuilt from the repository alone.

Rules:
  * Raw traces are always drawn; a smoothed curve is only ever an overlay.
    A smoothed-only plot hides exactly the spikes this project studies.
  * Restart boundaries are drawn as explicit vertical markers. Runs are never
    stitched into one continuous line across a restart.
  * A panel with no real data is annotated "unavailable" -- never faked, never
    silently zero-filled.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.autolayout": True,
})

C = {"a": "#2a6fb5", "b": "#c1121f", "c": "#2a9d5c", "d": "#e08214",
     "e": "#7d54a8", "f": "#666666"}


def load(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if v is None or v == "":
                rr[k] = None
            else:
                try:
                    rr[k] = float(v)
                except ValueError:
                    rr[k] = v
        out.append(rr)
    return out


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        v, s = r.get(key), r.get("global_step")
        if isinstance(v, (int, float)) and isinstance(s, (int, float)):
            xs.append(s)
            ys.append(v)
    return xs, ys


def restart_steps(rows):
    out, prev = [], None
    for r in rows:
        e = r.get("restart_epoch")
        if prev is not None and e != prev:
            out.append(r.get("global_step"))
        prev = e
    return [s for s in out if s is not None]


def mark_restarts(ax, rows):
    for s in restart_steps(rows):
        ax.axvline(s, color=C["b"], ls="--", lw=1.2, alpha=0.8)
        ax.text(s, ax.get_ylim()[1], " RESTART", color=C["b"], fontsize=7,
                va="top", rotation=90)


def smooth(ys, w=5):
    if len(ys) < w:
        return None
    return [sum(ys[max(0, i - w + 1):i + 1]) / len(ys[max(0, i - w + 1):i + 1])
            for i in range(len(ys))]


def panel(ax, rows, key, label, color, smooth_overlay=True):
    xs, ys = series(rows, key)
    if not xs:
        ax.text(0.5, 0.5, f"{label}\nunavailable", ha="center", va="center",
                transform=ax.transAxes, color=C["f"], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return False
    ax.plot(xs, ys, color=color, lw=1.0, alpha=0.85, label=f"{label} (raw)")
    if smooth_overlay:
        sm = smooth(ys)
        if sm:
            ax.plot(xs, sm, color=color, lw=2.0, alpha=0.55, ls="-",
                    label=f"{label} (smoothed, w=5)")
    ax.set_xlabel("optimizer update")
    ax.legend(fontsize=7)
    return True


def fig_reward(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    panel(axes[0], rows, "reward_mean", "reward mean", C["a"])
    axes[0].set_title("Reward vs optimizer update")
    xs, ys = series(rows, "reward_std")
    if xs:
        axes[0].fill_between(xs, [m - s for m, s in zip(series(rows, "reward_mean")[1], ys)],
                             [m + s for m, s in zip(series(rows, "reward_mean")[1], ys)],
                             color=C["a"], alpha=0.12, label="+/- reward std")
    ok = panel(axes[1], rows, "validation_accuracy", "validation accuracy", C["c"])
    axes[1].set_title("Validation accuracy" + ("" if ok else " (not evaluated yet)"))
    for a in axes:
        mark_restarts(a, rows)
    fig.savefig(out)
    plt.close(fig)


def fig_policy(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    panel(axes[0], rows, "kl", "KL", C["b"])
    axes[0].set_title("KL divergence")
    panel(axes[1], rows, "entropy", "entropy", C["e"])
    axes[1].set_title("Policy entropy")
    xs, ys = series(rows, "clip_fraction")
    if xs:
        axes[2].plot(xs, ys, color=C["d"], lw=1.2, label="clip fraction (total)")
    for k, c, lab in (("clip_fraction_low", C["a"], "lower clip"),
                      ("clip_fraction_high", C["b"], "upper clip")):
        x2, y2 = series(rows, k)
        if x2:
            axes[2].plot(x2, y2, color=c, lw=1.0, ls="--", alpha=0.8, label=lab)
    if not xs:
        axes[2].text(0.5, 0.5, "clip fraction\nunavailable", ha="center", va="center",
                     transform=axes[2].transAxes, color=C["f"])
    axes[2].set_title("Clipping")
    axes[2].set_xlabel("optimizer update")
    axes[2].legend(fontsize=7)
    for a in axes:
        mark_restarts(a, rows)
    fig.savefig(out)
    plt.close(fig)


def fig_ratio_grad(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    qs = [("importance_ratio_p01", "p01"), ("importance_ratio_p05", "p05"),
          ("importance_ratio_p50", "p50"), ("importance_ratio_p95", "p95"),
          ("importance_ratio_p99", "p99"), ("importance_ratio_max", "max")]
    any_q = False
    for (k, lab), c in zip(qs, [C["a"], C["c"], C["f"], C["d"], C["b"], C["e"]]):
        x, y = series(rows, k)
        if x:
            any_q = True
            axes[0].plot(x, y, lw=1.1, color=c, label=lab)
    if any_q:
        axes[0].axhline(1.0, color="k", lw=0.8, ls=":", alpha=0.6)
        axes[0].legend(fontsize=7, ncol=3)
    else:
        axes[0].text(0.5, 0.5, "importance-ratio quantiles\nunavailable\n"
                              "(requires in-trainer instrumentation)",
                     ha="center", va="center", transform=axes[0].transAxes, color=C["f"],
                     fontsize=8)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    axes[0].set_title("Importance ratio distribution")
    axes[0].set_xlabel("optimizer update")
    panel(axes[1], rows, "grad_norm", "grad norm", C["b"])
    axes[1].set_title("Gradient norm")
    for a in axes:
        mark_restarts(a, rows)
    fig.savefig(out)
    plt.close(fig)


def fig_group(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    got = False
    for k, c, lab in (("mixed_group_ratio", C["c"], "mixed"),
                      ("all_correct_group_ratio", C["a"], "all-correct"),
                      ("all_wrong_group_ratio", C["b"], "all-wrong"),
                      ("zero_std_group_ratio", C["d"], "zero-std")):
        x, y = series(rows, k)
        if x:
            got = True
            axes[0].plot(x, y, lw=1.3, color=c, label=lab)
    if got:
        axes[0].legend(fontsize=7)
        axes[0].set_ylim(-0.02, 1.02)
    else:
        axes[0].text(0.5, 0.5, "group composition\nunavailable\n(set rollout_data_dir)",
                     ha="center", va="center", transform=axes[0].transAxes, color=C["f"])
    axes[0].set_title("GRPO group composition")
    axes[0].set_xlabel("optimizer update")
    ok = panel(axes[1], rows, "effective_signal_fraction",
               "effective signal fraction", C["c"], smooth_overlay=False)
    if ok:
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].axhline(0.3, color=C["b"], ls=":", lw=1,
                        label="0.30 heuristic warning")
        axes[1].legend(fontsize=7)
    axes[1].set_title("Fraction of groups carrying gradient signal")
    for a in axes:
        mark_restarts(a, rows)
    fig.savefig(out)
    plt.close(fig)


def fig_length(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    got = False
    for k, c, lab in (("response_length_mean", C["a"], "mean"),
                      ("response_length_median", C["c"], "median"),
                      ("response_length_p95", C["d"], "p95"),
                      ("response_length_max", C["b"], "max")):
        x, y = series(rows, k)
        if x:
            got = True
            axes[0].plot(x, y, lw=1.2, color=c, label=lab)
    if got:
        axes[0].legend(fontsize=7)
    else:
        axes[0].text(0.5, 0.5, "response length\nunavailable", ha="center", va="center",
                     transform=axes[0].transAxes, color=C["f"])
    axes[0].set_title("Response length")
    axes[0].set_xlabel("optimizer update")
    axes[0].set_ylabel("tokens")
    ok = panel(axes[1], rows, "truncation_rate", "truncation rate", C["b"],
               smooth_overlay=False)
    if ok:
        axes[1].axhline(0.30, color=C["d"], ls=":", lw=1, label="0.30 heuristic warning")
        axes[1].legend(fontsize=7)
    axes[1].set_title("Truncation rate")
    for a in axes:
        mark_restarts(a, rows)
    fig.savefig(out)
    plt.close(fig)


def fig_timing(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    keys = [("t_rollout", "rollout", C["a"]), ("t_reward", "verifier", C["c"]),
            ("t_old_logprob", "old logprob", C["e"]), ("t_ref_logprob", "ref logprob", C["f"]),
            ("t_actor_update", "actor update", C["b"]), ("t_weight_sync", "weight sync", C["d"])]
    got = False
    for k, lab, c in keys:
        x, y = series(rows, k)
        if x:
            got = True
            axes[0].plot(x, y, lw=1.2, color=c, label=lab)
    if got:
        axes[0].legend(fontsize=7, ncol=2)
    else:
        axes[0].text(0.5, 0.5, "stage timings\nunavailable", ha="center", va="center",
                     transform=axes[0].transAxes, color=C["f"])
    axes[0].set_title("Update time by stage")
    axes[0].set_xlabel("optimizer update")
    axes[0].set_ylabel("seconds")

    pk = [("pct_rollout", "rollout", C["a"]), ("pct_reward", "verifier", C["c"]),
          ("pct_logprob", "logprob", C["e"]), ("pct_actor_update", "actor update", C["b"]),
          ("pct_weight_sync", "weight sync", C["d"]), ("pct_other", "other", C["f"])]
    xs = [r["global_step"] for r in rows if isinstance(r.get("pct_rollout"), (int, float))]
    if xs:
        bottom = [0.0] * len(xs)
        for k, lab, c in pk:
            vals = [r.get(k) or 0.0 for r in rows
                    if isinstance(r.get("pct_rollout"), (int, float))]
            axes[1].bar(xs, vals, bottom=bottom, color=c, label=lab, width=0.8)
            bottom = [b + v for b, v in zip(bottom, vals)]
        axes[1].legend(fontsize=7, ncol=2)
        axes[1].set_ylabel("% of update wall time")
    else:
        axes[1].text(0.5, 0.5, "time attribution\nunavailable", ha="center", va="center",
                     transform=axes[1].transAxes, color=C["f"])
    axes[1].set_title("Where update wall time goes")
    axes[1].set_xlabel("optimizer update")
    fig.savefig(out)
    plt.close(fig)


def fig_gpu(run_dir, out):
    import json
    p = os.path.join(run_dir, "telemetry", "system.jsonl")
    if not os.path.exists(p):
        return
    ts, util, mem = [], {i: [] for i in range(8)}, {i: [] for i in range(8)}
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "gpus" not in r:
                continue
            ts.append(r["ts"])
            for g in r["gpus"]:
                i = g.get("index")
                if i is None:
                    continue
                util[i].append(g.get("util_gpu_pct"))
                mem[i].append(g.get("mem_used_gib"))
    if not ts:
        return
    t0 = ts[0]
    rel = [(t - t0) / 60.0 for t in ts]
    ngpu = sum(1 for i in util if util[i])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for i in range(ngpu):
        axes[0].plot(rel, util[i], lw=1.0, label=f"GPU{i}")
        axes[1].plot(rel, mem[i], lw=1.0, label=f"GPU{i}")
    axes[0].set_title("GPU utilization")
    axes[0].set_ylabel("%")
    axes[1].set_title("GPU memory used")
    axes[1].set_ylabel("GiB")
    for a in axes:
        a.set_xlabel("minutes since run start")
        a.legend(fontsize=7, ncol=4)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    fd = os.path.join(a.run_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    rows = load(os.path.join(a.run_dir, "metrics", "training_metrics.csv"))
    if rows:
        fig_reward(rows, os.path.join(fd, "01_reward_validation.png"))
        fig_policy(rows, os.path.join(fd, "02_policy_dynamics.png"))
        fig_ratio_grad(rows, os.path.join(fd, "03_ratio_grad.png"))
        fig_group(rows, os.path.join(fd, "04_group_signal.png"))
        fig_length(rows, os.path.join(fd, "05_length_dynamics.png"))
        fig_timing(rows, os.path.join(fd, "06_system_timing.png"))
    fig_gpu(a.run_dir, os.path.join(fd, "07_gpu_memory_util.png"))
    print(f"figures -> {fd} ({len(rows)} update rows)")


if __name__ == "__main__":
    main()

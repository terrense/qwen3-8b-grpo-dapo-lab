#!/usr/bin/env python3
"""Dr.GRPO 论断的严格检验：同一道题内部配对比较答对/答错的长度。

为什么需要配对：
  直接把所有答对和所有答错的长度平均起来比，有两个混杂因素，
  两个都会伪造出「答错更长且越来越长」的假象：

  (1) 难度混杂 —— 难题本身推理就长，而且更容易答错。
      所以「答错更长」可能只是「难题更长」。
  (2) 组成漂移 —— 训练推进后模型答对的变多，剩下还答错的是越来越难的题。
      所以「答错长度在涨」可能只是「还没攻克的题越来越难」。

  配对分析同时消掉这两个：只在**同一道题**内部比较，
  且只用那些既有答对又有答错的题（混合组）。这样难度被完全控制住。

  剩下的差异才可能是优化偏差。

数据来自 rollout dump，零 GPU 成本。长度用字符数做代理（dump 里没有 token 数）。
"""

import glob
import json
import os
import statistics
import sys

LAB = "/root/autodl-tmp/rl_lab"


def load_step(dump_dir, step):
    p = os.path.join(dump_dir, f"{step}.jsonl")
    if not os.path.exists(p):
        return {}
    groups = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            s = r.get("score")
            if isinstance(s, dict):
                s = s.get("score")
            try:
                s = float(s)
            except (TypeError, ValueError):
                continue
            groups.setdefault(r.get("input"), []).append((s, len(str(r.get("output", "")))))
    return groups


def paired_step(groups):
    """对每个混合组，算 (该组答错平均长度 - 该组答对平均长度)。"""
    diffs, ratios, n_mixed = [], [], 0
    for _, samples in groups.items():
        cor = [L for s, L in samples if s > 0]
        wrg = [L for s, L in samples if s <= 0]
        if not cor or not wrg:      # 只用混合组
            continue
        n_mixed += 1
        mc, mw = statistics.mean(cor), statistics.mean(wrg)
        diffs.append(mw - mc)
        ratios.append(mw / mc)
    if not diffs:
        return None
    return dict(n_mixed=n_mixed, mean_diff=statistics.mean(diffs),
                median_diff=statistics.median(diffs),
                mean_ratio=statistics.mean(ratios),
                frac_wrong_longer=sum(1 for d in diffs if d > 0) / len(diffs))


def trend(v):
    if len(v) < 6:
        return None
    k = max(2, len(v) // 3)
    a, b = statistics.mean(v[:k]), statistics.mean(v[-k:])
    return a, b


def main():
    arms = sys.argv[1:] or ["M20_grpo", "M20_gspo", "M20_dapo"]
    print("同一道题内部配对：只用混合组（既有答对又有答错），难度被完全控制\n")
    summary = {}
    for arm in arms:
        cand = sorted(glob.glob(os.path.join(LAB, "logs", f"{arm}-*", "rollout_dump")))
        if not cand:
            print(f"{arm}: 无 dump"); continue
        dump = cand[-1]
        steps = sorted(int(os.path.basename(p).split(".")[0])
                       for p in glob.glob(os.path.join(dump, "*.jsonl"))
                       if os.path.basename(p).split(".")[0].isdigit())
        rows = []
        for st in steps:
            r = paired_step(load_step(dump, st))
            if r:
                r["step"] = st
                rows.append(r)
        if not rows:
            print(f"{arm}: 无混合组"); continue

        print(f"{'='*76}")
        print(f"  {arm}")
        print(f"{'='*76}")
        print(f"{'step':>5s}{'混合组数':>10s}{'错-对 字符差':>14s}{'错/对 比值':>12s}{'错更长占比':>12s}")
        for r in rows:
            if r["step"] in (rows[0]["step"], rows[-1]["step"]) or r["step"] % 5 == 0:
                print(f"{r['step']:>5d}{r['n_mixed']:>10d}{r['mean_diff']:>14.0f}"
                      f"{r['mean_ratio']:>12.3f}{r['frac_wrong_longer']:>12.3f}")
        td = trend([r["mean_diff"] for r in rows])
        tr = trend([r["mean_ratio"] for r in rows])
        tf = trend([r["frac_wrong_longer"] for r in rows])
        if td:
            print(f"\n  错-对 字符差 : {td[0]:+.0f} -> {td[1]:+.0f}")
            print(f"  错/对 比值   : {tr[0]:.3f} -> {tr[1]:.3f}")
            print(f"  错更长的组占比: {tf[0]:.3f} -> {tf[1]:.3f}")
            summary[arm] = (td, tr, tf)
        print()

    if summary:
        print(f"{'='*76}")
        print("  汇总 —— 难度控制之后，Dr.GRPO 的论断还成立吗")
        print(f"{'='*76}")
        print(f"{'arm':12s}{'比值 首':>10s}{'比值 末':>10s}{'变化':>10s}{'错更长占比 末':>15s}")
        for arm, (td, tr, tf) in summary.items():
            print(f"{arm:12s}{tr[0]:>10.3f}{tr[1]:>10.3f}"
                  f"{(tr[1]-tr[0])/tr[0]*100:>9.1f}%{tf[1]:>15.3f}")
        print()
        print("  判据：Dr.GRPO 成立的话，比值应 > 1 且随训练上升。")
        print("        比值 ~ 1.0 且不动 = 难度控制后偏差消失 = 原来那个 1.6 是混杂。")


if __name__ == "__main__":
    main()

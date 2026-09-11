#!/usr/bin/env python3
"""按答对/答错分开统计回答长度 —— 直接检验 Dr.GRPO 的核心论断。

Dr.GRPO (arXiv 2503.20783) 说 GRPO 有优化偏差，会让回答变长，**尤其是答错的回答**。
原因是 token-mean 聚合把每条序列的 loss 除以它自己的 token 数，
所以一条长的错误回答，每个 token 挨的罚是短回答的 1/|y| —— 把错误答案写长可以稀释梯度。

聚合后的平均长度看不出这个 —— 必须按答对/答错分开。

数据来源是 rollout dump（trainer.rollout_data_dir），零 GPU 成本。

注意：dump 里只有文本没有 token 数，所以用字符数做代理。
同一个模型同一种语言下字符数和 token 数基本单调相关，比较趋势是够的，
但绝对值不要跟 response_length/mean（那个是 token）混着看。
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
        return None
    out = []
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
            out.append((s, len(str(r.get("output", "")))))
    return out


def analyse(arm):
    cand = sorted(glob.glob(os.path.join(LAB, "logs", f"{arm}-*", "rollout_dump")))
    if not cand:
        cand = [os.path.join(LAB, "logs", f"{arm}_dump")]
    dump = cand[-1]
    steps = sorted(int(os.path.basename(p).split(".")[0])
                   for p in glob.glob(os.path.join(dump, "*.jsonl"))
                   if os.path.basename(p).split(".")[0].isdigit())
    if not steps:
        return None
    rows = []
    for st in steps:
        d = load_step(dump, st)
        if not d:
            continue
        cor = [L for s, L in d if s > 0]
        wrg = [L for s, L in d if s <= 0]
        if not cor or not wrg:
            continue
        rows.append(dict(step=st, n=len(d), n_cor=len(cor), n_wrg=len(wrg),
                         len_cor=statistics.mean(cor), len_wrg=statistics.mean(wrg),
                         med_cor=statistics.median(cor), med_wrg=statistics.median(wrg)))
    return rows


def trend(vals):
    """首尾各取 1/3 求均值，返回 (前, 后, 变化率)。比首末两点稳。"""
    if len(vals) < 6:
        return None
    k = max(2, len(vals) // 3)
    a, b = statistics.mean(vals[:k]), statistics.mean(vals[-k:])
    return a, b, (b - a) / a * 100.0


def main():
    arms = sys.argv[1:] or ["M20_grpo", "M20_gspo", "M20_dapo"]
    summary = {}
    for arm in arms:
        rows = analyse(arm)
        if not rows:
            print(f"{arm}: 没有可用 dump")
            continue
        print(f"\n{'='*78}")
        print(f"  {arm}   ({len(rows)} 个 step 有可用数据)")
        print(f"{'='*78}")
        print(f"{'step':>5s}{'n':>6s}{'答对数':>8s}{'答错数':>8s}"
              f"{'答对长度':>11s}{'答错长度':>11s}{'错/对':>9s}")
        for r in rows:
            if r["step"] in (rows[0]["step"], rows[-1]["step"]) or r["step"] % 5 == 0:
                print(f"{r['step']:>5d}{r['n']:>6d}{r['n_cor']:>8d}{r['n_wrg']:>8d}"
                      f"{r['len_cor']:>11.0f}{r['len_wrg']:>11.0f}"
                      f"{r['len_wrg']/r['len_cor']:>9.3f}")
        tc = trend([r["len_cor"] for r in rows])
        tw = trend([r["len_wrg"] for r in rows])
        if tc and tw:
            print(f"\n  答对回答长度: {tc[0]:.0f} -> {tc[1]:.0f}  ({tc[2]:+.1f}%)")
            print(f"  答错回答长度: {tw[0]:.0f} -> {tw[1]:.0f}  ({tw[2]:+.1f}%)")
            print(f"  Dr.GRPO 关注的是答错那一行是否显著上涨")
            summary[arm] = dict(cor=tc, wrg=tw,
                                ratio=[r["len_wrg"] / r["len_cor"] for r in rows])

    if summary:
        print(f"\n{'='*78}")
        print("  汇总：答错回答的长度变化（Dr.GRPO 论断的直接检验）")
        print(f"{'='*78}")
        print(f"{'arm':12s}{'答对变化':>12s}{'答错变化':>12s}{'错/对 首':>12s}{'错/对 末':>12s}")
        for arm, s in summary.items():
            r = s["ratio"]
            k = max(2, len(r) // 3)
            print(f"{arm:12s}{s['cor'][2]:>11.1f}%{s['wrg'][2]:>11.1f}%"
                  f"{statistics.mean(r[:k]):>12.3f}{statistics.mean(r[-k:]):>12.3f}")
        print()
        print("  读法：如果 Dr.GRPO 的论断在这里成立，应该看到")
        print("        答错变化 明显 > 答对变化，且 错/对 比值随训练上升。")


if __name__ == "__main__":
    main()

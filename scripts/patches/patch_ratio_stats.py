#!/usr/bin/env python3
"""给 verl 打一个纯观测的 importance-ratio 分布钩子。

为什么需要：
  verl 只发 actor/ppo_kl 和 actor/pg_clipfrac。
  - ppo_kl 是 (old_logprob - logprob) 的**带符号均值**，均值接近零既可能是分布很窄，
    也可能是正负偏差互相抵消 —— 推不出 rho ~= 1。
  - pg_clipfrac 只统计越过 clip 边界（±0.2）的那部分尾巴，说不了边界以内的形状。
  两个都回答不了「ratio 分布到底有多宽」，而这恰恰是 GSPO（token ratio vs sequence
  ratio）和 clip-higher 的全部论点所在。

这个补丁做什么：
  在 compute_policy_loss_vanilla 和 compute_policy_loss_gspo 的 metrics dict 里
  额外塞进 ratio 的分位数。**只读已有张量，在 no_grad 下算，绝不参与 loss。**
  整段包在 try/except 里，任何异常都只是少几个指标，不会让训练挂掉。

可逆：加 --revert，或者直接 git -C repos/verl checkout verl/trainer/ppo/core_algos.py
幂等：重复执行不会重复插入。
"""

import argparse
import os
import shutil
import sys

TARGET = "/root/autodl-tmp/rl_lab/repos/verl/verl/trainer/ppo/core_algos.py"
MARK = "_verl_lab_ratio_stats"

HELPER = '''

# ==================== LAB INSTRUMENTATION (observation only) ====================
# 加这个是因为 verl 原生指标回答不了「ratio 分布有多宽」：
#   actor/ppo_kl    是带符号均值，正负偏差会抵消 -> 均值接近 0 推不出分布窄
#   actor/pg_clipfrac 只数越过 clip 边界的尾巴 -> 说不了边界以内的形状
# 而 GSPO 和 clip-higher 的论点全在这个分布上。
# 下面这个函数只读已有张量、在 no_grad 下计算、结果只进 metrics dict，
# 不参与任何梯度或 loss。异常一律吞掉，最多少几个观测值。
def _verl_lab_ratio_stats(ratio, response_mask, prefix):
    out = {}
    try:
        with torch.no_grad():
            r = torch.masked_select(ratio.detach(), response_mask.bool()).float()
            n = r.numel()
            if n == 0:
                return out
            # torch.quantile 有元素数上限，超了就等距抽样（确定性，不用随机数）
            if n > 1000000:
                idx = torch.linspace(0, n - 1, 1000000, device=r.device).long()
                rq = r[idx]
            else:
                rq = r
            qs = torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], device=rq.device, dtype=rq.dtype)
            q = torch.quantile(rq, qs)
            dev = (rq - 1.0).abs()
            dqs = torch.tensor([0.50, 0.95, 0.99], device=rq.device, dtype=rq.dtype)
            dq = torch.quantile(dev, dqs)
            out = {
                prefix + "_p01": q[0].item(),
                prefix + "_p05": q[1].item(),
                prefix + "_p50": q[2].item(),
                prefix + "_p95": q[3].item(),
                prefix + "_p99": q[4].item(),
                prefix + "_min": r.min().item(),
                prefix + "_max": r.max().item(),
                prefix + "_mean": r.mean().item(),
                prefix + "_std": r.std().item(),
                # |rho - 1| 的分位数 —— 判断分布宽窄最直接的量
                prefix + "_absdev_p50": dq[0].item(),
                prefix + "_absdev_p95": dq[1].item(),
                prefix + "_absdev_p99": dq[2].item(),
                prefix + "_absdev_max": dev.max().item(),
                # 落在各个偏离带的 token 占比
                prefix + "_frac_gt_1pct": (dev > 0.01).float().mean().item(),
                prefix + "_frac_gt_5pct": (dev > 0.05).float().mean().item(),
                prefix + "_frac_gt_20pct": (dev > 0.20).float().mean().item(),
                prefix + "_n_tokens": float(n),
            }
    except Exception:
        pass
    return out
# ================== END LAB INSTRUMENTATION ==================
'''

# (函数名, ratio 变量名, 指标前缀)
SITES = [
    ("compute_policy_loss_vanilla", "ratio", "actor/ratio"),
    ("compute_policy_loss_gspo", "seq_importance_ratio", "actor/seqratio"),
]

RETURN_LINE = "    return pg_loss, pg_metrics"


def find_func_span(lines, fname):
    start = None
    for i, l in enumerate(lines):
        if l.startswith("def " + fname + "("):
            start = i
            break
    if start is None:
        return None, None
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("def ") or lines[j].startswith("@register_policy_loss"):
            return start, j
    return start, len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    bak = TARGET + ".lab_orig"
    if a.revert:
        if os.path.exists(bak):
            shutil.copy(bak, TARGET)
            print("已还原 core_algos.py")
        else:
            print("没有备份，用 git checkout 还原")
        return

    src = open(TARGET, encoding="utf-8").read()
    if MARK in src:
        print("补丁已存在，跳过（幂等）")
        return
    if not os.path.exists(bak):
        shutil.copy(TARGET, bak)
        print("原文件已备份 -> " + os.path.basename(bak))

    lines = src.split("\n")

    # 1) 插入 helper：放在第一个 policy loss 函数之前
    anchor = None
    for i, l in enumerate(lines):
        if l.startswith("def compute_policy_loss_vanilla("):
            anchor = i
            break
    if anchor is None:
        print("找不到 compute_policy_loss_vanilla，放弃")
        sys.exit(1)
    # 往回退到它的装饰器
    while anchor > 0 and lines[anchor - 1].lstrip().startswith("@"):
        anchor -= 1
    lines[anchor:anchor] = HELPER.split("\n")

    # 2) 在每个站点的 return 前插入 update
    patched = []
    for fname, rvar, prefix in SITES:
        s, e = find_func_span(lines, fname)
        if s is None:
            print("  跳过（找不到）: " + fname)
            continue
        hit = None
        for j in range(s, e):
            if lines[j] == RETURN_LINE:
                hit = j
                break
        if hit is None:
            print("  跳过（找不到 return）: " + fname)
            continue
        ins = [
            "    # LAB: 纯观测，不影响 loss。见 scripts/patches/patch_ratio_stats.py",
            '    pg_metrics.update(_verl_lab_ratio_stats(' + rvar + ', response_mask, "' + prefix + '"))',
        ]
        lines[hit:hit] = ins
        patched.append(fname + " -> " + prefix)

    open(TARGET, "w", encoding="utf-8").write("\n".join(lines))
    print("已打补丁：")
    for p in patched:
        print("  " + p)


if __name__ == "__main__":
    main()

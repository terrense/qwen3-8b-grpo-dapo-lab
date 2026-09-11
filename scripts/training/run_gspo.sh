#!/usr/bin/env bash
# GSPO smoke —— 第一次真正把 token ratio 和 sequence ratio 的分布放在一起比。
#
# 为什么现在能跑了：之前我判断「ρ≈1 所以 GSPO 测不出东西」，那是错的
# （见 analysis/ratio_distribution.md）。实测 |ρ−1| p99 = 0.055、极值 0.42、
# 8% 的 token 偏离 >1% —— 尖峰厚尾，正是 GSPO 论点针对的形状。
#
# GSPO 在 verl 里是原生实现：core_algos.py:1546 compute_policy_loss_gspo，
# 用 stop-gradient 恒等式让前向值是 sequence ratio、反向仍走 per-token 路径。
#
# 观测：补丁给 vanilla 打的前缀是 actor/ratio，给 gspo 打的是 actor/seqratio。
# 所以这次跑出来的 actor/seqratio_* 可以直接跟 R1/rdist 的 actor/ratio_* 对比。
#
# 配置跟 R1 对齐（同 model/数据/seed/batch/G/长度/LR/温度/TP），
# 只改两个：policy_loss_mode=gspo，loss_agg_mode=seq-mean-token-mean
# （后者是 verl 实现的 docstring 里推荐的、也是论文的做法）。
set -xeuo pipefail
LAB=${LAB:-/root/autodl-tmp/rl_lab}
exec bash "$LAB/scripts/training/run_r1_grpo_baseline.sh" \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    "$@"

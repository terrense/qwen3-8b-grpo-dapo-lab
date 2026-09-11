#!/usr/bin/env bash
# 四臂消融：分离 GSPO 里 clipping 和 loss 聚合各自的贡献 + 检验 Dr.GRPO
#
# 为什么要 C_aggonly 这个臂：
#   GSPO 同时改了两件事 —— sequence-level ratio/clipping，以及 loss_agg_mode
#   从 token-mean 换成 seq-mean-token-mean。而实测 GSPO 的 clipfrac 20 步全是 0，
#   说明它的 clipping 从未被触发。那 GSPO 的任何效果都可能只来自聚合方式。
#   C_aggonly = vanilla loss + GSPO 的聚合方式，正好把这两个变量分开。
#
#     B_gspo - C_aggonly  = sequence-level clipping 的净贡献（预期 ~0，因为没触发）
#     C_aggonly - A_grpo  = loss 聚合方式的净贡献
#
# Dr.GRPO 用「干净两变量」版本：只关掉 std 归一化和长度归一化，
# KL 保持和对照一致（官方 recipe 还会关 KL，那是第三个变量，会污染对比 ——
# 见 docs/algorithms/dr_grpo.md 里的 matched-comparison caveat）。
#
# 验证集这次是 1500 题（旧的 200 题是它的子集）。
# 200 题在 50% 附近的标准误是 3.54 个百分点，4 个点的差距只有 1.1 个标准误，
# 分辨不出来。1500 题的标准误是 1.29 个百分点，4 个点 = 3.1 个标准误，能分辨。
set -uo pipefail
LAB=${LAB:-/root/autodl-tmp/rl_lab}
source "$LAB/env.sh"; export PATH=/root/miniconda3/bin:$PATH; lab_proxy_on >/dev/null
STEPS=${STEPS:-20}

run_arm () {
  local RID=$1 ALGO=$2; shift 2
  echo "############################################################"
  echo "# ARM $RID ($ALGO)  steps=$STEPS  $(date +%H:%M:%S)"
  echo "#   overrides: $*"
  echo "############################################################"
  cd "$LAB"
  env LAB=$LAB ALGORITHM=$ALGO SEED=20260910 \
    HF_HOME=$HF_HOME TMPDIR=$TMPDIR VLLM_CACHE_ROOT=$LAB/cache/vllm \
    TRITON_CACHE_DIR=$TRITON_CACHE_DIR HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    RAY_TMPDIR=$RAY_TMPDIR VLLM_LOGGING_LEVEL=WARNING \
    TOTAL_TRAINING_STEPS=$STEPS TEST_FREQ=10 SAVE_FREQ=-1 \
    http_proxy=${http_proxy:-} https_proxy=${https_proxy:-} no_proxy=${no_proxy:-} \
    bash "$LAB/scripts/training/run_with_observer.sh" "$RID" \
         bash "$LAB/scripts/training/run_r1_grpo_baseline.sh" "$@" \
    > "$LAB/logs/${RID}_launcher.log" 2>&1
  echo "  exit=$?  $(date +%H:%M:%S)  updates=$(wc -l < $LAB/experiments/$RID/metrics/verl_file_logger.jsonl 2>/dev/null || echo 0)"
}

# A: 对照
run_arm A_grpo GRPO

# B: 完整 GSPO
run_arm B_gspo GSPO \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean

# C: 只换聚合方式，policy loss 还是 vanilla —— 分离变量的关键臂
run_arm C_aggonly GRPO_seqagg \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean

# D: Dr.GRPO 干净两变量版（KL 保持和对照一致）
run_arm D_drgrpo DrGRPO \
  algorithm.norm_adv_by_std_in_grpo=False \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
  actor_rollout_ref.actor.loss_scale_factor=8192

echo
echo "################ ABLATE4 结束 $(date +%H:%M:%S) ################"
python3 - <<'PY'
import json, os, statistics
L="/root/autodl-tmp/rl_lab/experiments/"
A=["A_grpo","B_gspo","C_aggonly","D_drgrpo"]
print(f"{'arm':12s}{'upd':>5s}{'rew前5':>9s}{'rew后5':>9s}{'val@10':>9s}{'val@20':>9s}"
      f"{'clipfrac':>11s}{'advmax':>9s}{'len末':>9s}{'wall分':>8s}")
print("-"*92)
for a in A:
    p=L+a+"/metrics/verl_file_logger.jsonl"
    if not os.path.exists(p): print(f"{a:12s} 无数据"); continue
    r=[json.loads(l) for l in open(p)]
    if not r: print(f"{a:12s} 空"); continue
    c=lambda k:[x["data"].get(k) for x in r if isinstance(x["data"].get(k),(int,float))]
    v=lambda s: next((x["data"].get("val-core/math_dapo/acc/mean@1") for x in r
                      if x["step"]==s and x["data"].get("val-core/math_dapo/acc/mean@1") is not None), None)
    rew=c("critic/rewards/mean")
    f=lambda x,w,p=4: f"{x:{w}.{p}g}" if isinstance(x,(int,float)) else f"{'--':>{w}}"
    print(f"{a:12s}{len(r):>5d}{f(statistics.mean(rew[:5]),9)}{f(statistics.mean(rew[-5:]),9)}"
          f"{f(v(10),9)}{f(v(20),9)}{f(statistics.mean(c('actor/pg_clipfrac')),11)}"
          f"{f(statistics.mean(c('critic/advantages/max')),9)}{f(c('response_length/mean')[-1],9,5)}"
          f"{f(sum(c('timing_s/step'))/60,8)}")
print()
print("解读要点：")
print("  B - C  = sequence-level clipping 的净贡献（GSPO clipfrac 为 0 的话应该接近 0）")
print("  C - A  = loss 聚合方式的净贡献")
print("  D 的 advmax 应明显不同 —— Dr.GRPO 关掉了 std 归一化，advantage 不再被 std 除")
print("ABLATE4_DONE")
PY

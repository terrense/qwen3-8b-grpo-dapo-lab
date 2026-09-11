#!/usr/bin/env bash
# 20-update matched comparison: GRPO(control) vs GSPO vs DAPO
#
# 对齐的：Qwen3-8B、去重数据集、seed 20260910、train_batch_size=16、rollout.n=8、
#         ppo_mini_batch_size=8、max_response_length=8192、关 thinking、lr=1e-6、
#         温度1/top-p1、TP=2、4xH20、20 个 optimizer update、每 10 步验证一次
#
# 各臂的差异（就是要测的东西）：
#   GRPO  vanilla policy loss, token-mean, 对称 clip 0.2/0.2, use_kl_loss=True(0.001)
#   GSPO  loss_mode=gspo, seq-mean-token-mean          <- sequence-level ratio
#   DAPO  clip 0.2/0.28 + dynamic sampling + token-mean + overlong shaping + 去掉 KL
#
# GRPO 这个臂重跑而不是复用 R1：R1 跑在 ratio 观测补丁之前，没有 actor/ratio_*，
# 而 ratio 分布是现在最关键的观测面（见 analysis/ratio_distribution.md）。
#
# 重要：matched budget 按 optimizer update 对齐，但 DAPO 的 dynamic sampling 会
# 重采样，每个 update 的生成 token 大约是别人的 2 倍。所以最后两个视角都要报：
# (1) 相同 update 数  (2) 实际生成 token 数 / wall-clock / GPU-hours。
set -uo pipefail
LAB=${LAB:-/root/autodl-tmp/rl_lab}
source "$LAB/env.sh"; export PATH=/root/miniconda3/bin:$PATH; lab_proxy_on >/dev/null
STEPS=${STEPS:-20}

run_arm () {
  local RID=$1 SCRIPT=$2 ALGO=$3
  echo "############################################################"
  echo "# ARM $RID   ($ALGO)   steps=$STEPS   $(date +%H:%M:%S)"
  echo "############################################################"
  cd "$LAB"
  env LAB=$LAB ALGORITHM=$ALGO SEED=20260910 \
    HF_HOME=$HF_HOME TMPDIR=$TMPDIR VLLM_CACHE_ROOT=$LAB/cache/vllm \
    TRITON_CACHE_DIR=$TRITON_CACHE_DIR HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    RAY_TMPDIR=$RAY_TMPDIR VLLM_LOGGING_LEVEL=WARNING \
    TOTAL_TRAINING_STEPS=$STEPS TEST_FREQ=10 SAVE_FREQ=-1 \
    http_proxy=${http_proxy:-} https_proxy=${https_proxy:-} no_proxy=${no_proxy:-} \
    bash "$LAB/scripts/training/run_with_observer.sh" "$RID" \
         bash "$LAB/scripts/training/$SCRIPT" \
    > "$LAB/logs/${RID}_launcher.log" 2>&1
  echo "  exit=$? $(date +%H:%M:%S)"
  echo "  updates=$(wc -l < $LAB/experiments/$RID/metrics/verl_file_logger.jsonl 2>/dev/null || echo 0)"
}

run_arm M20_grpo run_r1_grpo_baseline.sh GRPO
run_arm M20_gspo run_gspo.sh             GSPO
run_arm M20_dapo run_r2_dapo.sh          DAPO

echo
echo "################ MATCHED-20 全部结束 $(date +%H:%M:%S) ################"
python3 - <<'PY'
import json, os
arms=["M20_grpo","M20_gspo","M20_dapo"]
L="/root/autodl-tmp/rl_lab/experiments/"
print(f"{'arm':10s}{'upd':>5s}{'reward_last':>12s}{'val@10':>9s}{'val@20':>9s}"
      f"{'ratio_p99':>11s}{'ratio_max':>11s}{'clipfrac':>11s}{'tokens':>12s}{'wall_s':>9s}")
print("-"*100)
for a in arms:
    p=L+a+"/metrics/verl_file_logger.jsonl"
    if not os.path.exists(p): print(f"{a:10s}  (无数据)"); continue
    rows=[json.loads(l) for l in open(p)]
    if not rows: print(f"{a:10s}  (空)"); continue
    d=rows[-1]["data"]
    pre = "actor/seqratio_" if any("seqratio" in k for k in d) else "actor/ratio_"
    def col(k):
        v=[r["data"].get(k) for r in rows if isinstance(r["data"].get(k),(int,float))]
        return v
    tok=sum(col("perf/total_num_tokens"))
    wall=sum(col("timing_s/step"))
    val=[(r["step"], r["data"].get("val-core/math_dapo/acc/mean@1")) for r in rows
         if isinstance(r["data"].get("val-core/math_dapo/acc/mean@1"),(int,float))]
    v10 = next((v for s,v in val if s<=10), None)
    v20 = next((v for s,v in reversed(val)), None)
    f=lambda v,w,p=4: f"{v:{w}.{p}g}" if isinstance(v,(int,float)) else f"{'--':>{w}}"
    print(f"{a:10s}{len(rows):>5d}{f(d.get('critic/rewards/mean'),12)}{f(v10,9)}{f(v20,9)}"
          f"{f(d.get(pre+'absdev_p99'),11)}{f(d.get(pre+'absdev_max'),11)}"
          f"{f(d.get('actor/pg_clipfrac'),11)}{f(tok,12,6)}{f(wall,9,5)}")
print()
print("注意：DAPO 每个 update 的生成 token 约为其他两臂的 2 倍（dynamic sampling 重采样），")
print("      所以 '相同 update 数' 不等于 '相同算力预算'。两个视角都要看。")
print("MATCHED20_DONE")
PY

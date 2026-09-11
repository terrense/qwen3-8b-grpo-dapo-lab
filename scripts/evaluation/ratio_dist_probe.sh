#!/usr/bin/env bash
# 带 ratio 分布观测的探测。回答上一轮留下的问题：
#   ppo_kl 的均值接近 0 推不出 rho ~= 1（正负会抵消），clipfrac 只数越过 ±0.2 的尾巴。
#   所以「GSPO 会测不出来」这个结论当时是没有证据的。现在直接量分布本身。
#
# 判据（看 |rho-1| 的分位数，这是最直接的宽窄度量）：
#   absdev_p99 < 1e-3   -> 分布确实极窄，GSPO/clip-higher 在这个配置下确实没东西可测
#   absdev_p99 ~ 1e-2   -> 有可观的离散，但离 clip 边界(0.2)还远
#   absdev_p99 > 0.1    -> 尾巴够肥，GSPO 的论点有发挥空间
set -uo pipefail
LAB=${LAB:-/root/autodl-tmp/rl_lab}
source "$LAB/env.sh"; export PATH=/root/miniconda3/bin:$PATH; lab_proxy_on >/dev/null
STEPS=${STEPS:-3}
#      名字         mini  说明
ARMS=("mini8  8  2个梯度步（R1/R2 用的就是这个）"
      "mini2  2  8个梯度步")
for arm in "${ARMS[@]}"; do
  set -- $arm; NAME=$1; MINI=$2
  echo "=========== ARM $NAME  (mini=$MINI, $((16/MINI)) 个梯度步) ==========="
  RID="rdist_$NAME"; DUMP=$LAB/logs/${RID}_dump; mkdir -p "$DUMP"
  TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=$MINI ACTOR_LR=1e-6 \
  TOTAL_TRAINING_STEPS=$STEPS TEST_FREQ=-1 SAVE_FREQ=-1 ROLLOUT_DUMP_DIR="$DUMP" \
  VERL_FILE_LOGGER_PATH=$LAB/logs/${RID}_metrics.jsonl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING \
    bash "$LAB/scripts/training/run_r1_grpo_baseline.sh" trainer.experiment_name="$RID" \
    > $LAB/logs/${RID}.log 2>&1
  echo "  exit=$?"
done
echo
echo "==================== RATIO 分布汇总 ===================="
python3 - <<'PY'
import json, glob
rows=[]
for f in sorted(glob.glob("/root/autodl-tmp/rl_lab/logs/rdist_*_metrics.jsonl")):
    arm=f.split("rdist_")[1].replace("_metrics.jsonl","")
    for r in [json.loads(l) for l in open(f)]:
        d=r["data"]; d["_arm"]=arm; d["_step"]=r["step"]; rows.append(d)
if not rows:
    print("没有数据"); raise SystemExit(1)
keys=["actor/ratio_absdev_p50","actor/ratio_absdev_p95","actor/ratio_absdev_p99","actor/ratio_absdev_max",
      "actor/ratio_frac_gt_1pct","actor/ratio_frac_gt_5pct","actor/ratio_frac_gt_20pct",
      "actor/ratio_p01","actor/ratio_p99","actor/ratio_min","actor/ratio_max",
      "actor/pg_clipfrac","actor/ppo_kl","actor/ratio_n_tokens"]
print(f"{'arm/step':12s}" + "".join(f"{k.split('/')[-1][:11]:>13s}" for k in keys[:7]))
for d in rows:
    print(f"{d['_arm']+'/'+str(d['_step']):12s}" + "".join(
        f"{d.get(k):13.4g}" if isinstance(d.get(k),(int,float)) else f"{'--':>13s}" for k in keys[:7]))
print()
print(f"{'arm/step':12s}" + "".join(f"{k.split('/')[-1][:11]:>13s}" for k in keys[7:]))
for d in rows:
    print(f"{d['_arm']+'/'+str(d['_step']):12s}" + "".join(
        f"{d.get(k):13.4g}" if isinstance(d.get(k),(int,float)) else f"{'--':>13s}" for k in keys[7:]))
print()
for arm in sorted({d['_arm'] for d in rows}):
    sub=[d for d in rows if d['_arm']==arm]
    p99=[d.get("actor/ratio_absdev_p99") for d in sub if isinstance(d.get("actor/ratio_absdev_p99"),(int,float))]
    mx=[d.get("actor/ratio_absdev_max") for d in sub if isinstance(d.get("actor/ratio_absdev_max"),(int,float))]
    if p99:
        v=max(p99)
        verdict=("极窄，GSPO/clip-higher 确实无从测起" if v<1e-3 else
                 "有离散但远未及 clip 边界 0.2" if v<0.05 else "尾巴够肥，GSPO 有发挥空间")
        print(f"{arm}: |rho-1| p99 最大 = {v:.3e}, absdev_max = {max(mx):.3e}  -> {verdict}")
print("RATIO_DIST_DONE")
PY

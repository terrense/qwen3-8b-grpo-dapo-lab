#!/usr/bin/env bash
# AUDIT 2026-09-10: historical probe; see analysis/ratio_probe_report.md.
# Signed ppo_kl does not determine the ratio distribution. The statements below
# and the hard-coded R1 summary are historical, superseded by the audited report.
# This script bypasses the flight recorder and truncates its summary on rerun.
# Do not rerun before adapting it to the authorized recorder workflow.
# Ratio sensitivity probe -- 找一个能让 rho 真正离开 1 的配置。
#
# 为什么需要这个：R1 和 R2 的 pg_clipfrac 都只有 ~1e-4，ppo_kl ~1e-5。
# rho 离 1 的距离比 clip 边界(0.2)小四个数量级，所以：
#   - clip-higher 测不出来（R2 已证实：上界 0.2->0.28，clipfrac 基本没动）
#   - GSPO 也会测不出来（它比的就是 token ratio vs sequence ratio 的分布）
#   - clipfrac / ppo_kl 这两个观测面等于是死的
#
# 设计：train_batch_size 固定 16 不动，只改 ppo_mini_batch_size。
# rollout 占 R1 wall time 的 53%，是贵的那部分；多几个梯度步只增加 actor update 时间。
# 这样能干净隔离出唯一关心的变量 —— old_logprob 和当前策略之间隔了几个梯度步。
#
# 参照臂 mini=8 / lr=1e-6（2 个梯度步）已经有 R1 的 20 个 update 数据，不重跑。

set -uo pipefail
LAB=${LAB:-/root/autodl-tmp/rl_lab}
source "$LAB/env.sh"
export PATH=/root/miniconda3/bin:$PATH
lab_proxy_on >/dev/null

STEPS=${STEPS:-3}
OUT=$LAB/analysis/ratio_probe.jsonl
: > "$OUT"

#      名字            mini  lr      说明
ARMS=(
  "mini4_lr1e6         4     1e-6    4个梯度步，LR不变"
  "mini2_lr1e6         2     1e-6    8个梯度步，LR不变"
  "mini8_lr5e6         8     5e-6    2个梯度步，LR x5 —— 隔离LR的影响"
)

for arm in "${ARMS[@]}"; do
  set -- $arm
  NAME=$1; MINI=$2; LR=$3
  echo "=================================================================="
  echo " ARM: $NAME   ppo_mini_batch_size=$MINI  lr=$LR  steps=$STEPS"
  echo "=================================================================="
  RID="probe_$NAME"
  DUMP=$LAB/logs/${RID}_dump; mkdir -p "$DUMP"
  LOG=$LAB/logs/${RID}.log

  TRAIN_BATCH_SIZE=16 PPO_MINI_BATCH_SIZE=$MINI ACTOR_LR=$LR \
  TOTAL_TRAINING_STEPS=$STEPS TEST_FREQ=-1 SAVE_FREQ=-1 \
  ROLLOUT_DUMP_DIR="$DUMP" \
  VERL_FILE_LOGGER_PATH=$LAB/logs/${RID}_metrics.jsonl \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING \
    bash "$LAB/scripts/training/run_r1_grpo_baseline.sh" \
      trainer.experiment_name="$RID" \
    > "$LOG" 2>&1
  CODE=$?
  echo "  exit=$CODE"

  python3 - "$NAME" "$MINI" "$LR" "$CODE" "$LAB/logs/${RID}_metrics.jsonl" "$OUT" <<'PY'
import json, sys
name, mini, lr, code, mpath, out = sys.argv[1:7]
rows = []
try:
    rows = [json.loads(l) for l in open(mpath)]
except Exception as e:
    print(f"  metrics 读不到: {e}")
rec = dict(arm=name, ppo_mini_batch_size=int(mini), lr=lr, exit_code=int(code),
           n_updates=len(rows))
if rows:
    def col(k):
        return [r["data"].get(k) for r in rows if isinstance(r["data"].get(k), (int, float))]
    for k, short in [("actor/pg_clipfrac","clipfrac"), ("actor/ppo_kl","ppo_kl"),
                     ("actor/pg_clipfrac_lower","clipfrac_lower"),
                     ("actor/grad_norm","grad_norm"), ("actor/entropy","entropy"),
                     ("critic/rewards/mean","reward"), ("timing_s/step","t_step"),
                     ("timing_s/update_actor","t_actor")]:
        v = col(k)
        if v:
            rec[short] = dict(min=min(v), max=max(v), mean=sum(v)/len(v), last=v[-1])
    c = rec.get("clipfrac", {})
    k = rec.get("ppo_kl", {})
    print(f"  clipfrac  max={c.get('max',float('nan')):.3e}  mean={c.get('mean',float('nan')):.3e}")
    print(f"  |ppo_kl|  max={max(abs(k.get('min',0)),abs(k.get('max',0))):.3e}")
    print(f"  t_step    mean={rec.get('t_step',{}).get('mean',float('nan')):.1f}s")
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
PY
done

echo
echo "=================== 汇总 ==================="
python3 - "$OUT" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
print(f"{'arm':18s} {'mini':>5s} {'lr':>6s} {'steps':>6s} {'clipfrac max':>14s} {'|ppo_kl| max':>14s} {'t_step':>8s}")
print("-"*80)
# R1 参照（已知）
print(f"{'R1 (参照)':18s} {8:>5d} {'1e-6':>6s} {'2':>6s} {1.53e-4:>14.3e} {4.40e-5:>14.3e} {'~90s':>8s}")
for r in rows:
    c=r.get("clipfrac",{}); k=r.get("ppo_kl",{})
    kk=max(abs(k.get('min',0)),abs(k.get('max',0))) if k else float('nan')
    gs=16//r["ppo_mini_batch_size"]
    print(f"{r['arm']:18s} {r['ppo_mini_batch_size']:>5d} {r['lr']:>6s} {gs:>6d} "
          f"{c.get('max',float('nan')):>14.3e} {kk:>14.3e} {r.get('t_step',{}).get('mean',float('nan')):>7.1f}s")
print()
print("判据：clipfrac 要到 1e-2 量级、|ppo_kl| 到 1e-3 量级，clipping 才算真正活跃。")
print("RATIO_PROBE_DONE")
PY

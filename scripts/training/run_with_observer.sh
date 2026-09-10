#!/usr/bin/env bash
# Unified launcher: every real training run goes through here, never directly.
#
#   run_with_observer.sh <RUN_ID> <training command...>
#
# Responsibilities:
#   A. create the run directory
#   B. record start time, command, resolved config, git SHAs, model/dataset ids, env
#   C. start the 10s system sampler
#   D. start the training process
#   E. tee stdout/stderr to a LOCAL log (never committed)
#   F. trap SIGINT/SIGTERM/EXIT and shut the observer down cleanly
#   G. on exit: stop observer, finalize metrics, plots, RUN_REPORT, git sync
#
# The trainer's exit code is preserved and re-raised. It is never swallowed.

set -uo pipefail

LAB=${LAB:-/root/autodl-tmp/rl_lab}
source "$LAB/env.sh"

if [ $# -lt 2 ]; then
    echo "usage: run_with_observer.sh <RUN_ID> <training command...>" >&2
    exit 2
fi

RUN_ID="$1"; shift
TRAIN_CMD=("$@")

STAMP=$(date +%Y%m%d-%H%M%S)
RUN_UID="${RUN_ID}-${STAMP}"
RUN_DIR="$LAB/experiments/$RUN_ID"
MON="$LAB/scripts/monitoring"
VENV_PY="$LAB/repos/verl/.venv/bin/python"      # has psutil + pynvml
TOOL_PY="/root/miniconda3/bin/python"           # has matplotlib

mkdir -p "$RUN_DIR"/{telemetry,metrics,trajectories,incidents,figures}
mkdir -p "$LAB/logs/$RUN_UID" "$LAB/status"

TRAIN_LOG="$LAB/logs/$RUN_UID/trainer.log"
OBS_LOG="$LAB/logs/$RUN_UID/observer.log"
SAMPLER_LOG="$LAB/logs/$RUN_UID/sampler.log"
VERL_JSONL="$RUN_DIR/metrics/verl_file_logger.jsonl"
DUMP_DIR="$LAB/logs/$RUN_UID/rollout_dump"      # LOCAL ONLY, never committed
mkdir -p "$DUMP_DIR"
# The training script must write its rollout dump where the observer reads it,
# otherwise group-signal stats and trajectory audit silently come back empty.
export ROLLOUT_DUMP_DIR="$DUMP_DIR"

# ---- B. provenance --------------------------------------------------------
VERL_SHA=$(git -C "$LAB/repos/verl" rev-parse HEAD 2>/dev/null || echo unknown)
LAB_SHA=$(git -C "$LAB" rev-parse HEAD 2>/dev/null || echo unknown)
MODEL_PATH=${MODEL_PATH:-$LAB/models/Qwen3-8B}
# INC-005: this used to default to data/raw/, which is NOT what any run trains on --
# every launcher uses the DEDUPLICATED splits (the raw file repeats each prompt 100x).
# Recording the raw hash made the manifest describe a file the run never touched.
DATA_PARQUET=${DATA_PARQUET:-$LAB/data/dapo_math_17k/train.parquet}
DATA_SHA=$( [ -f "$DATA_PARQUET" ] && sha256sum "$DATA_PARQUET" | cut -c1-16 || echo unknown )
DATA_VAL=${DATA_VAL:-$LAB/data/dapo_math_17k/val.parquet}
DATA_VAL_SHA=$( [ -f "$DATA_VAL" ] && sha256sum "$DATA_VAL" | cut -c1-16 || echo unknown )
MODEL_CFG_SHA=$( [ -f "$MODEL_PATH/config.json" ] && sha256sum "$MODEL_PATH/config.json" | cut -c1-16 || echo unknown )

printf '%q ' "${TRAIN_CMD[@]}" > "$RUN_DIR/command.txt"; echo >> "$RUN_DIR/command.txt"

RESTART_EPOCH=0
if [ -f "$RUN_DIR/run_manifest.json" ]; then
    RESTART_EPOCH=$("$TOOL_PY" -c "
import json,sys
try: print(json.load(open('$RUN_DIR/run_manifest.json')).get('restart_epoch',0)+1)
except Exception: print(0)" 2>/dev/null || echo 0)
    cp "$RUN_DIR/run_manifest.json" "$RUN_DIR/run_manifest.prev.json" 2>/dev/null || true
fi

cat > "$RUN_DIR/run_manifest.json" <<EOF
{
  "run_id": "$RUN_ID",
  "run_uid": "$RUN_UID",
  "start_ts": $(date +%s),
  "start_iso": "$(date -Is)",
  "restart_epoch": $RESTART_EPOCH,
  "algorithm": "${ALGORITHM:-GRPO}",
  "model": "Qwen/Qwen3-8B",
  "model_path": "$MODEL_PATH",
  "model_config_sha256_16": "$MODEL_CFG_SHA",
  "dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
  "dataset_train_parquet": "$DATA_PARQUET",
  "dataset_train_sha256_16": "$DATA_SHA",
  "dataset_val_parquet": "$DATA_VAL",
  "dataset_val_sha256_16": "$DATA_VAL_SHA",
  "verl_commit": "$VERL_SHA",
  "lab_commit": "$LAB_SHA",
  "seed": ${SEED:-20260910},
  "hardware": "4x NVIDIA H20 96GB, NV18 full mesh",
  "torch": "2.11.0+cu130",
  "vllm": "0.24.0",
  "trainer_log": "$TRAIN_LOG",
  "rollout_dump_dir": "$DUMP_DIR",
  "note": "GRPO systems pilot / rehearsal. NOT the medical paper's M3 checkpoint."
}
EOF
cp "$LAB/manifests/versions.txt" "$RUN_DIR/environment_manifest.txt" 2>/dev/null || true

echo "=================================================================="
echo " RUN_ID       : $RUN_ID"
echo " RUN_UID      : $RUN_UID"
echo " run dir      : $RUN_DIR"
echo " trainer log  : $TRAIN_LOG"
echo " restart epoch: $RESTART_EPOCH"
echo "=================================================================="

# ---- C. system sampler ----------------------------------------------------
setsid nohup "$VENV_PY" "$MON/system_sampler.py" \
    --run-dir "$RUN_DIR" --interval 10 > "$SAMPLER_LOG" 2>&1 < /dev/null &
SAMPLER_PID=$!
echo "[launcher] system sampler pid=$SAMPLER_PID (10s cadence)"

# ---- D/E. training --------------------------------------------------------
export VERL_FILE_LOGGER_PATH="$VERL_JSONL"
export VERL_FILE_LOGGER_ROOT="$RUN_DIR/metrics"

OBSERVER_PID=""
# INC-005: resolved_config.yaml used to stay a NOT RUN placeholder forever -- it was
# referenced by the report and every incident bundle but never written. Hydra does emit
# the composed config plus the override list; copy them out of the run it just created.
capture_resolved_config() {
    local hd
    hd=$(find "$LAB/repos/verl/outputs" -type d -name ".hydra" -newer "$RUN_DIR/command.txt"             2>/dev/null | sort | tail -1)
    if [ -n "$hd" ] && [ -f "$hd/config.yaml" ]; then
        cp "$hd/config.yaml"    "$RUN_DIR/resolved_config.yaml"
        [ -f "$hd/overrides.yaml" ] && cp "$hd/overrides.yaml" "$RUN_DIR/hydra_overrides.yaml"
        echo "[launcher] resolved config captured from $hd"
    else
        echo "[launcher] WARNING: no hydra config found; resolved_config.yaml left as-is" >&2
    fi
}

cleanup() {
    local code=$1
    echo "[launcher] cleanup (exit=$code)"
    capture_resolved_config
    [ -n "$OBSERVER_PID" ] && kill "$OBSERVER_PID" 2>/dev/null || true
    kill "$SAMPLER_PID" 2>/dev/null || true
    sleep 2
    "$TOOL_PY" "$MON/finalize_run.py" --run-dir "$RUN_DIR" --lab "$LAB" \
        --verl-jsonl "$VERL_JSONL" --dump-dir "$DUMP_DIR" --exit-code "$code" \
        >> "$OBS_LOG" 2>&1 || echo "[launcher] finalize failed (see $OBS_LOG)"
    "$TOOL_PY" "$MON/github_sync.py" --lab "$LAB" --run-dir "$RUN_DIR" \
        --reason "run-end-exit-$code" >> "$OBS_LOG" 2>&1 || echo "[launcher] git sync failed"
    echo "[launcher] done. RUN_REPORT: $RUN_DIR/RUN_REPORT.md"
}
trap 'cleanup 130; exit 130' INT
trap 'cleanup 143; exit 143' TERM

# sync at run start
"$TOOL_PY" "$MON/github_sync.py" --lab "$LAB" --run-dir "$RUN_DIR" \
    --reason "run-start-$RUN_UID" >> "$OBS_LOG" 2>&1 || true

set -o pipefail
"${TRAIN_CMD[@]}" > >(tee -a "$TRAIN_LOG") 2>&1 &
TRAIN_PID=$!
echo "[launcher] trainer pid=$TRAIN_PID"

# ---- observer (needs the trainer pid) -------------------------------------
setsid nohup "$TOOL_PY" "$MON/rl_observer.py" \
    --run-dir "$RUN_DIR" --lab "$LAB" --verl-jsonl "$VERL_JSONL" \
    --dump-dir "$DUMP_DIR" --trainer-log "$TRAIN_LOG" \
    --resolved-config "$RUN_DIR/resolved_config.yaml" \
    --interval 20 > "$OBS_LOG" 2>&1 < /dev/null &
OBSERVER_PID=$!
echo "[launcher] observer pid=$OBSERVER_PID"

wait "$TRAIN_PID"
EXIT_CODE=$?
echo "[launcher] trainer exited with code $EXIT_CODE"

cleanup "$EXIT_CODE"
exit "$EXIT_CODE"     # G. the trainer's exit code is never swallowed

#!/usr/bin/env bash
# R0 -- GRPO smoke test. Exactly 2 optimizer updates.
#
# Purpose is the PIPELINE, not the science: prove that one optimizer update's
# data flow is fully accounted for end to end before any long run inherits it.
#
# Length budget chosen from measurement, not intuition (length_budget_probe.json):
#   think@4096   acc 12.70%  trunc 87.11%  mixed 20.31%   <- original, unusable
#   think@16384  acc 56.25%  trunc 18.75%  mixed 31.25%   8820 tok/sample
#   nothink@4096 acc 31.25%  trunc  1.56%  mixed 43.75%   1477 tok/sample  <- CHOSEN
# Arm B gives the most GRPO gradient signal per generated token (~8x arm A) and
# 11x faster wall clock, which is what a cost-bounded pilot should optimise.
#
# Derived from the official example
#   repos/verl/examples/grpo_trainer/run_qwen3_8b_fsdp.sh
# scaled down. Every key below was verified against the real config at
# verl @ 1252cc71 (see analysis/verl_config_explained.md); none is invented.
#
# Launch via the flight recorder, never directly:
#   scripts/training/run_with_observer.sh R0_grpo_smoke \
#       bash scripts/training/run_r0_grpo_smoke.sh

set -xeuo pipefail

LAB=${LAB:-/root/autodl-tmp/rl_lab}
MODEL_PATH=${MODEL_PATH:-$LAB/models/Qwen3-8B}
DATA=$LAB/data/dapo_math_17k

# ---- R0 scale (deliberately tiny) ----------------------------------------
train_batch_size=${TRAIN_BATCH_SIZE:-8}        # prompts per optimizer update
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}  # == train_batch -> 1 grad step/update
rollout_n=${ROLLOUT_N:-4}                      # G, samples per prompt
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
actor_lr=${ACTOR_LR:-1e-6}                     # official example's value
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}
rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
total_training_steps=${TOTAL_TRAINING_STEPS:-2}
NGPUS=${NGPUS:-4}

RUN_DIR=$LAB/experiments/R0_grpo_smoke
DUMP_DIR=${ROLLOUT_DUMP_DIR:-$LAB/logs/R0_rollout_dump}
mkdir -p "$DUMP_DIR" "$RUN_DIR"

cd "$LAB/repos/verl"

# uv launcher exactly as the official example does it, so the Ray workers use
# the same locked interpreter as the driver.
LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")

"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="['$DATA/smoke_train.parquet']" \
    data.val_files="['$DATA/val.parquet']" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger='["console","file"]' \
    trainer.project_name=qwen3_8b_grpo_dapo_lab \
    trainer.experiment_name=R0_grpo_smoke \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.rollout_data_dir="$DUMP_DIR" \
    "${RAY[@]}" \
    "$@"

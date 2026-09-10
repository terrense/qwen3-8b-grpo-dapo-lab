#!/usr/bin/env bash
# R1 -- Vanilla GRPO baseline. 20 optimizer updates (phase-2 short run).
#
# This is the CONTROL arm for the whole algorithm suite. Every later comparison
# (Dr.GRPO / DAPO / GSPO) is matched against it, so its config is the one that
# must be defensible.
#
# Two config decisions carry evidence from R0:
#
#  1. ppo_mini_batch_size (8) < train_batch_size (16)  -> 2 gradient steps per
#     update. R0 ran 8 == 8 with ppo_epochs=1, which makes rho identically 1 and
#     forces ppo_kl / clipfrac to be structurally 0. That is fine for a pipeline
#     smoke test but useless as a baseline: the whole point of R1 is to observe
#     policy shift, and a GSPO-vs-GRPO ratio comparison against rho == 1 would be
#     vacuous. mini < train is what makes clipping and KL non-zero.
#
#  2. max_response_length = 8192, thinking disabled.
#     Measured (analysis/pre_rl_reward_distribution.md, length_budget_probe.json):
#       think@4096    87.11% truncation, 79.69% zero-std, acc 12.70%  <- unusable
#       think@16384   18.75% truncation, acc 56.25%, 8820 tok/sample  <- 11x cost
#       nothink@4096   1.56% truncation, acc 31.25%, 1477 tok/sample  <- best signal/token
#     R0 used nothink@4096 and response length rose 1532 -> 2041 tokens in two
#     updates, so the cap is doubled to 8192 to buy headroom while keeping the
#     cheap non-thinking regime.
#     NOTE (corrected after R1): the original justification here also cited
#     "truncation 3.13% -> 9.38%", which was a MISREAD of
#     response_length/clip_ratio -- that metric compares against the padded
#     tensor width, not the configured cap (INC-005). The length-growth argument
#     stands on response_length/mean alone; the truncation figure did not.
#     This value is FIXED across all algorithm arms for matched comparison.
#
# Launch through the flight recorder, never directly:
#   scripts/training/run_with_observer.sh R1_grpo_baseline \
#       bash scripts/training/run_r1_grpo_baseline.sh

set -xeuo pipefail

LAB=${LAB:-/root/autodl-tmp/rl_lab}
MODEL_PATH=${MODEL_PATH:-$LAB/models/Qwen3-8B}
DATA=$LAB/data/dapo_math_17k

train_batch_size=${TRAIN_BATCH_SIZE:-16}          # prompts (groups) per update
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}     # < train_batch -> 2 grad steps, rho != 1
rollout_n=${ROLLOUT_N:-8}                         # G
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

actor_lr=${ACTOR_LR:-1e-6}                        # official example's value
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.2}           # symmetric: this is the CONTROL

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
total_training_steps=${TOTAL_TRAINING_STEPS:-20}
test_freq=${TEST_FREQ:-10}                        # validation at update 10 and 20
save_freq=${SAVE_FREQ:--1}                        # no checkpoints; resume tested separately
NGPUS=${NGPUS:-4}

DUMP_DIR=${ROLLOUT_DUMP_DIR:-$LAB/logs/R1_rollout_dump}
mkdir -p "$DUMP_DIR"

cd "$LAB/repos/verl"

LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")

"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.filter_groups.enable=False \
    data.train_files="['$DATA/train.parquet']" \
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
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
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
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger='["console","file"]' \
    trainer.project_name=qwen3_8b_grpo_dapo_lab \
    trainer.experiment_name=R1_grpo_baseline \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.rollout_data_dir="$DUMP_DIR" \
    "${RAY[@]}" \
    "$@"

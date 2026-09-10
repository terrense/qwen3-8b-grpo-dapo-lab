#!/usr/bin/env bash
# R2 -- DAPO. Phase 1 is a 2-update smoke test; phase 2 is 20 updates matched to R1.
#
# WHY THE RECIPE ENTRY POINT: core verl's ray_trainer.py does NOT implement
# filter_groups resampling or overlong reward shaping -- grep finds the config
# schema only. The resampling loop lives in verl-recipe/dapo/dapo_ray_trainer.py,
# which overrides fit(). So DAPO must run through main_dapo.py, not main_ppo.
#
# COMPATIBILITY, checked before spending GPU time:
#   our verl 1252cc71 is 615 commits AHEAD of the recipe's rolling pin bcb63864.
#   Verified anyway: dapo_ray_trainer + main_dapo import cleanly, RayDAPOTrainer's
#   MRO is intact, and the recipe's own dapo_trainer.yaml already uses our current
#   config paths (reward.reward_manager.name / reward.reward_kwargs.overlong_buffer_cfg
#   / algorithm.filter_groups). Only the recipe's OLD shell scripts still use the
#   pre-rename reward_model.* paths -- those are not used here.
#
# MATCHED TO R1 (analysis/R1_baseline_report.md): same model, dedup dataset, seed,
# train_batch_size, rollout.n, mini-batch, response budget, thinking disabled, LR,
# temperature/top-p, TP. Differences are DAPO's four mechanisms plus the KL removal
# the published recipe specifies.
#
# KNOWN LIMITATION, stated up front: R1 measured pg_clipfrac at only ~1.5e-4, i.e.
# rho ~= 1 at lr 1e-6 with mini=8/train=16. Clip-Higher therefore has very little
# room to act in this configuration, and any clip-asymmetry effect will be muted.
# Dynamic sampling, token-level aggregation and overlong shaping are unaffected and
# remain fully measurable -- dynamic sampling is the dominant lever here given R1's
# 36.25% zero-variance groups.

set -xeuo pipefail

LAB=${LAB:-/root/autodl-tmp/rl_lab}
MODEL_PATH=${MODEL_PATH:-$LAB/models/Qwen3-8B}
DATA=$LAB/data/dapo_math_17k
RECIPE=$LAB/repos/verl-recipe/dapo

train_batch_size=${TRAIN_BATCH_SIZE:-16}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}
rollout_n=${ROLLOUT_N:-8}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
actor_lr=${ACTOR_LR:-1e-6}

# ---- DAPO mechanism 1: Clip-Higher ----
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_ratio_c=${CLIP_RATIO_C:-10.0}
# ---- DAPO mechanism 2: Dynamic Sampling ----
filter_groups_enable=${FILTER_GROUPS:-True}
filter_groups_metric=${FILTER_GROUPS_METRIC:-acc}
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}
# ---- DAPO mechanism 4: Overlong Reward Shaping ----
# reference recipe uses len = 20% of a 20480 cap; 2048 is 25% of our 8192 cap
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-2048}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

total_training_steps=${TOTAL_TRAINING_STEPS:-2}
test_freq=${TEST_FREQ:--1}
save_freq=${SAVE_FREQ:--1}
NGPUS=${NGPUS:-4}

DUMP_DIR=${ROLLOUT_DUMP_DIR:-$LAB/logs/R2_rollout_dump}
mkdir -p "$DUMP_DIR"

# INC-007: the recipe dapo_trainer.yaml declares
#   hydra.searchpath: [file://verl/trainer/config]
# -- a RELATIVE path, resolved against the CWD. Upstream runs
# `python3 -m recipe.dapo.main_dapo` from the verl repo root, where that path
# exists. Running main_dapo.py from the recipe directory makes Hydra fail with
# "Could not load ppo_trainer". So: cd to the verl root and invoke the recipe
# entry point by absolute path (hydra resolves config_path="config" relative to
# main_dapo.py itself, so the recipe own config is still found).
#
# ALSO: invoking the recipe by ABSOLUTE PATH from inside the uv project made
# Ray's uv runtime-env hook die with "path_or_uri must be a string, got NoneType"
# -- it could not locate the script within the project. Upstream runs
# `python3 -m recipe.dapo.main_dapo`, so repos/verl/recipe/dapo is symlinked to
# the recipe checkout and we use the module form, exactly as upstream does.
cd "$LAB/repos/verl"

LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")

"${LAUNCH[@]}" -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.filter_groups.enable=${filter_groups_enable} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="['$DATA/train.parquet']" \
    data.val_files="['$DATA/val.parquet']" \
    data.train_batch_size=${train_batch_size} \
    data.gen_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=True \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.reward_kwargs.overlong_buffer_cfg.log=True \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
    trainer.experiment_name=R2_dapo \
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

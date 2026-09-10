# Metric Dictionary

Canonical metric names used in `experiments/<RUN_ID>/metrics/training_metrics.csv`,
and the **real VeRL key** each one comes from at commit `1252cc71`.

> **Availability rule.** A metric the current VeRL build does not emit is written as an
> **empty cell** and rendered `unavailable`. It is **never** substituted with `0`.
> A zero and a missing measurement mean completely different things, and conflating
> them is how observability lies to you.

## Optimization

| canonical | VeRL key | notes |
|---|---|---|
| `policy_loss` | `actor/pg_loss` | policy-gradient loss |
| `learning_rate` | *probed* | VeRL publishes no canonical actor-LR key; the collector probes `actor/lr`, `actor/learning_rate`, `actor/optim/lr`, `lr`, `learning_rate` and reports `unavailable` if none exist |
| `grad_norm` | `actor/grad_norm` | pre-clip gradient norm |
| `entropy` | `actor/entropy` | policy entropy |
| `entropy_loss` | `actor/entropy_loss` | entropy term as a loss contribution |
| `mfu` | `actor/mfu` | model FLOPs utilization |

## Policy shift

| canonical | VeRL key | notes |
|---|---|---|
| `kl` | `actor/ppo_kl` | in-batch approximate KL between the sampling policy and the current policy |
| `clip_fraction_high` | `actor/pg_clipfrac` | **upper** clip fraction |
| `clip_fraction_low` | `actor/pg_clipfrac_lower` | **lower** clip fraction |
| `clip_fraction` | derived | `low + high` |
| `kl_penalty` / `kl_penalty_coeff` | `actor/reward_kl_penalty` / `..._coeff` | only when KL is applied in the reward |
| `importance_ratio_p01 … max` | **NOT EMITTED** | requires in-trainer instrumentation; deferred, not faked |

The official Qwen3-8B GRPO example sets `use_kl_loss=True`, `kl_loss_coef=0.001`,
`kl_loss_type=low_var_kl`, `use_kl_in_reward=False` — KL shapes the **gradient**, not
the reward, so the reward stays a clean correctness signal.

## Reward and score

| canonical | VeRL key |
|---|---|
| `reward_mean` / `reward_max` / `reward_min` | `critic/rewards/{mean,max,min}` |
| `score_mean` / `score_max` / `score_min` | `critic/score/{mean,max,min}` |
| `reward_std` | derived from the rollout dump |

With the `math_dapo` verifier the reward is **`+1.0` / `-1.0`** (not `+1/0`), so a
reward mean of `-0.75` means roughly 12% correct, not "75% of maximum".

## Advantage

| canonical | VeRL key |
|---|---|
| `advantage_mean` / `advantage_max` / `advantage_min` | `critic/advantages/{mean,max,min}` |
| `advantage_std` | derived |

## GRPO group signal (derived from `trainer.rollout_data_dir`)

| canonical | meaning |
|---|---|
| `n_groups` | prompts in the batch (one group per prompt) |
| `all_correct_groups` | every sample in the group correct |
| `all_wrong_groups` | every sample wrong |
| `mixed_groups` | group contains both |
| `zero_std_groups` | reward variance exactly 0 (= all-correct + all-wrong) |
| `zero_std_group_ratio` | `zero_std_groups / n_groups` |
| `effective_signal_groups` | `n_groups - zero_std_groups` |
| `effective_signal_fraction` | fraction of the nominal batch that actually produced gradient |

**Why `effective_signal_fraction` is the headline GRPO metric.** GRPO normalises
advantage within a group. If every sample in a group gets the same reward, the
normalised advantage is identically zero and that prompt contributes *nothing* —
not a weak signal, none. So the nominal prompt batch and the batch that actually
trains the policy can differ by a large factor, and only this metric shows the gap.

## Generation

| canonical | VeRL key | notes |
|---|---|---|
| `response_length_mean/max/min` | `response_length/{mean,max,min}` | |
| `truncation_rate` | `response_length/clip_ratio` | fraction hitting `max_response_length` |
| `prompt_length_clip_ratio` | `prompt_length/clip_ratio` | over-long prompts |
| `aborted_ratio` | `response/aborted_ratio` | |
| `response_length_median/p95` | derived from the rollout dump | |
| `eos_rate` | derived | `finish_reason == "stop"` |

## Verifier states (instrumentation, not a reward change)

`verifier_correct`, `verifier_incorrect`, `verifier_parse_failure`,
`verifier_exception`, `verifier_truncated`.

The official verifier maps *wrong answer* and *unparseable answer* to the same `-1.0`.
Splitting them is the only way to tell a policy regression from a formatting or
truncation regression.

## Timing (`timing_s/*`, from `_timer(...)` in `verl/trainer/ppo/ray_trainer.py`)

| canonical | VeRL key |
|---|---|
| `t_rollout` | `timing_s/gen` |
| `t_old_logprob` | `timing_s/old_log_prob` |
| `t_ref_logprob` | `timing_s/ref` |
| `t_adv` | `timing_s/adv` |
| `t_reward` | `timing_s/reward` |
| `t_actor_update` | `timing_s/update_actor` |
| `t_weight_sync` | `timing_s/update_weights` |
| `t_checkpoint` | `timing_s/save_checkpoint` |
| `t_step` | `timing_s/step` |

Derived shares `pct_rollout`, `pct_reward`, `pct_logprob`, `pct_actor_update`,
`pct_weight_sync`, `pct_other` answer *why* GPU utilization is what it is, instead of
merely reporting that it is low.

## Policy provenance

| canonical | meaning |
|---|---|
| `rollout_policy_step` | policy version that generated the batch |
| `consumer_update_step` | update consuming it |
| `policy_staleness_steps` | difference |

For synchronous GRPO this should be **constant**. A drifting value means rollouts are
being consumed by a different policy version than the one that produced them, which
invalidates the importance-ratio assumption. Recorded to be *verified*, not assumed.

---

## Thresholds are heuristics, not constants

Every threshold in `incident_detector.py` is a **diagnostic heuristic**, not a
theoretical constant. They exist to draw a human's attention, never to justify a
parameter change on their own.

| heuristic | value | rationale |
|---|---|---|
| `truncation_rate` warning | `> 0.30` | above this, reward increasingly measures length budget rather than correctness |
| `zero_std_group_ratio` warning | `> 0.70` | most of the batch is teaching nothing |
| `effective_signal_fraction` warning | `< 0.30` | same fact, stated as the useful fraction |
| `clip_fraction` warning | `> 0.40` | most tokens at the trust-region boundary |
| system disk | `> 90%` | RED |
| data disk | `> 95%` | RED |
| robust-z | `abs(z) > 4.0` vs rolling median (MAD-scaled), window 20, min history 6 | adaptive |

**Why adaptive rules dominate.** Early in a run the natural scale of reward, KL,
entropy and grad norm is unknown, so fixed thresholds either fire constantly or never.
Continuous metrics are judged against a rolling median with a MAD-based robust z-score,
which tolerates the heavy tails RL produces. Only quantities with genuine absolute
meaning — NaN/Inf, disk full, process death — get hard rules.

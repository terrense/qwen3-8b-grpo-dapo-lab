# R0 — GRPO Smoke Test Report

**Run UID** `R0_grpo_smoke-20260910-132022` · **Outcome: COMPLETED, exit code 0**
2/2 optimizer updates in **1 min 46 s** (10.0 min wall incl. model load and teardown)
VeRL `1252cc71` · lab commit `78b21d8` · seed `20260910`
Launched through `scripts/training/run_with_observer.sh` — the first real GRPO update in
this project was already under the flight recorder.

## Configuration

`train_batch_size=8` · `rollout.n=4` (G) · `ppo_mini_batch_size=8` · `ppo_epochs=1` ·
`max_prompt_length=2048` · `max_response_length=4096` · `enable_thinking=False` ·
`actor.optim.lr=1e-6` · `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type=low_var_kl` ·
`use_kl_in_reward=False` · `entropy_coeff=0` · `rollout.tensor_model_parallel_size=2` ·
`gpu_memory_utilization=0.6` · FSDP, actor `param_offload=False`, ref `param_offload=True` ·
4×H20. Resolved config: `experiments/R0_grpo_smoke/resolved_config.yaml`.

## The two updates

| metric | update 1 | update 2 |
|---|---|---|
| `reward_mean` (`critic/rewards/mean`) | **−0.1875** | **−0.3750** |
| `policy_loss` (`actor/pg_loss`) | 0.026233 | 0.028386 |
| `actor/lr` | 1e-06 | 1e-06 |
| `actor/grad_norm` | 0.254685 | 0.212263 |
| `actor/entropy` | 0.343033 | 0.341709 |
| **`actor/ppo_kl`** | **0.0** | **0.0** |
| **`actor/pg_clipfrac`** | **0.0** | **0.0** |
| **`actor/pg_clipfrac_lower`** | **0.0** | **0.0** |
| `actor/kl_loss` | 0.0 | 0.000311 |
| `critic/advantages/mean` | −0.026233 | −0.028386 |
| `critic/advantages/min` / `max` | **−1.5 / +1.5** | −1.5 / +1.5 |
| `response_length/mean` | 1532.2 | 2040.7 |
| `response_length/max` | 3509 | 4096 |
| `truncation_rate` (`response_length/clip_ratio`) | 3.13% | 9.38% |
| `perf/total_num_tokens` | 53,599 | 69,975 |
| `perf/throughput` | 206.8 tok/s | 418.1 tok/s |
| `perf/mfu/actor` | 0.567 | 0.648 |
| `actor/perf/max_memory_allocated_gb` | **40.52** | 40.52 |
| `timing_s/step` | 64.80 s | 41.84 s |

### GRPO group signal (derived from `trainer.rollout_data_dir`)

| | update 1 | update 2 |
|---|---|---|
| groups (prompts) | 8 | 8 |
| all-correct | 1 | 0 |
| all-wrong | 3 | 3 |
| **mixed** | **4** | **5** |
| zero-std | 4 | 3 |
| **zero_std_group_ratio** | **50.0%** | **37.5%** |
| **effective_signal_fraction** | **50.0%** | **62.5%** |
| mean group reward std | 0.450 | 0.592 |

Half to nearly two-thirds of the batch carried gradient. The 4096-token *thinking*
preflight produced 79.7% zero-variance groups and only 20.3% effective signal, so the
length-budget change did what the probe predicted.

## What the numbers verify about the pipeline

### 1. The importance ratio is exactly 1 — and that is correct, not a bug

`ppo_kl = 0` and both clip fractions `= 0` on both updates. With
`ppo_mini_batch_size == train_batch_size` (8 == 8) and `ppo_epochs = 1`, there is exactly
**one** gradient step per rollout batch, taken with the *same* policy that generated it.
So `old_logprob ≡ current_logprob`, ρ = exp(0) = 1, nothing can leave the clip range, and
the approximate KL between sampling and training policy is identically zero.

This is the degenerate on-policy case of the GRPO objective. Exact zeros here are the
strongest available evidence that old-logprob bookkeeping is correct — a *non-zero* value
in this configuration would have indicated a bug.

**Consequence for R1:** clipping and KL only become informative once
`ppo_mini_batch_size < train_batch_size` or `ppo_epochs > 1`. R1 must set that
deliberately, or `clipfrac` stays pinned at 0 and the policy-shift instrumentation
measures nothing. This also matters for the algorithm suite: **GSPO's whole claim is about
ratio behaviour, so any GSPO-vs-GRPO comparison run at ratio ≡ 1 would be vacuous.**

### 2. Rollout policy and training policy are the same weights

VeRL measures this natively:
`training/rollout_actor_probs_pearson_corr` = **0.99958 / 0.99957**,
`rollout_probs_diff_mean` = 0.0030 / 0.0029, `rollout_corr/kl` = 4.89e-4 / 3.73e-4.
The residual (max per-token diff 0.1436) is expected bf16 plus differing kernel paths
between the vLLM rollout engine and the FSDP training graph — not a sync fault.

### 3. Policy staleness is exactly 0

`training/off_policy/trajectory_staleness` mean/max/min = **0** on both updates, and
`trajectory_spans` = 1. Synchronous GRPO confirmed **by measurement**, not by
architectural assumption. Every trajectory is consumed by the policy version that produced it.

### 4. Advantage scale ±1.5 matches the source arithmetic exactly

In `compute_grpo_outcome_advantage` (`core_algos.py:304-322`), `norm_adv_by_std_in_grpo`
defaults to `true` and the code uses `torch.std`, which is the **unbiased (n−1)** sample
std. For G=4 with rewards in {+1,−1} and a 1-correct / 3-wrong group: mean = −0.5,
n−1 std = 1.0, advantage = 1.5 / 1.0 = **±1.5**. Observed extrema match.

Zero-variance groups take the `(score − mean) / (0 + 1e-6)` path with a numerator of
exactly 0 → advantage 0. No division blow-up, but also no signal — which is precisely the
quantity `effective_signal_fraction` tracks.

> This std division is exactly the term **Dr.GRPO** argues is a bias. R0 has now measured
> the baseline behaviour it would remove, which makes the Dr.GRPO comparison concrete
> rather than theoretical.

### 5. Prompt tokens do not enter the policy loss

`NaiveRewardManager` decodes only `response_ids[:valid_response_length]`
(`naive.py:39-56`), and the advantage is applied as `scores.unsqueeze(-1) * response_mask`
(`core_algos.py:322`) — the mask is what confines the advantage to response tokens.
`prompt_length/mean` (142.8) is tracked separately from `response_length/mean` (1532.2).
There are no tool/environment tokens in this math task.

## Where the wall time went

| stage | update 1 | update 2 |
|---|---|---|
| rollout (`timing_s/gen`) | 28.3% | **53.4%** |
| logprob (old + ref) | 39.1% | 14.2% |
| actor update | 12.8% | 22.8% |
| weight sync | 6.2% | 9.5% |
| other | 13.6% | 0.2% |

Update 1's logprob share is a one-off: `timing_s/ref` was 21.11 s on update 1 and 2.70 s
on update 2 — the reference model is CPU-offloaded (`ref.fsdp_config.param_offload=True`),
so the first call pays the host→device transfer. **Update 2 is the representative steady
state: rollout dominates at 53.4%.**

Peak allocated actor memory was **40.52 GiB of 95.07**, leaving substantial headroom for
larger batches, longer responses, or a critic (relevant to VAPO).

`timing_s/reward` is **not emitted** by this VeRL build — reward is computed inside the
agent/reward loop rather than as a separately timed trainer stage — so the verifier share
is recorded as `unavailable`, never as 0. Independently measured verifier cost is 0.02 ms
per sample, five orders of magnitude below rollout, so it is not a throughput factor.

## Incidents

**None raised by the detector.** It ran throughout; 43 system telemetry samples were
collected at the 10 s cadence.

Two defects were found in the **observability layer itself** (not in training):

**(a) Silent dump-path mismatch.** `run_with_observer.sh` set
`DUMP_DIR="$LAB/logs/$RUN_UID/rollout_dump"` but never exported it, while
`run_r0_grpo_smoke.sh` defaulted `ROLLOUT_DUMP_DIR` to `$LAB/logs/R0_rollout_dump`. The
trainer wrote to the second path, the observer read the first, and group statistics plus
trajectory sampling came back **silently empty rather than erroring**. Fixed by exporting
`ROLLOUT_DUMP_DIR` from the launcher; R0's group metrics above were then re-derived from
the dump that was actually written. Worth recording precisely because it failed silently —
the exact class of bug that would have made R1's group-signal analysis quietly meaningless.

**(b) A derived metric contradicting a measured one.** The collector originally *derived*
`policy_staleness_steps` from consecutive step numbers and produced `0, 1`, contradicting
VeRL's own `training/off_policy/trajectory_staleness` of `0, 0`. The derivation was removed
in favour of the native metric. Deriving a quantity the framework already measures is how
an observability layer invents facts.

## Verdict

**R0 PASSES.** The pipeline executes end to end, the update data flow is fully accounted
for, the importance-ratio and staleness invariants hold exactly, advantage arithmetic
matches the source, memory has headroom, and the flight recorder produced metrics,
figures, telemetry and a run report without touching the algorithm.

**R1 is not blocked by R0**, but two configuration decisions must be made deliberately
before it starts:

1. **`ppo_mini_batch_size < train_batch_size`** (or `ppo_epochs > 1`). Otherwise ρ ≡ 1 and
   clipping/KL are structurally zero — which would also invalidate the GSPO and DAPO
   clip-behaviour comparisons.
2. **A final length budget.** R0 used non-thinking @ 4096. Truncation rose 3.13% → 9.38%
   and `response_length/mean` went 1532 → 2041 across just two updates. Response-length
   growth under GRPO is itself a first-class research question here — it is the phenomenon
   Dr.GRPO attributes partly to optimization bias — so it must be measured, not merely
   capped away.

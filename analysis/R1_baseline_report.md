# R1 vanilla GRPO baseline — evidence and limitations

**Completed: 20/20 rollout-batch updates, launcher exit 0; no restart.**
Run UID: `R1_grpo_baseline-20260910-134534`. Recorded interval: 2026-09-10
13:45:35–14:21:57 Asia/Shanghai, **36.37 min** including finalization.
The run completed before this post-run audit. No R2 or further GPU experiment was started.

R1 establishes a working, nondegenerate GRPO control: clipping and signed approximate
KL are nonzero, useful within-group reward variation persists, and no sustained length
or entropy collapse is visible. **It does not establish a generalization improvement.**
Validation moves from 99/200 at update 10 to 97/200 at update 20; a pre-training
validation under this exact configuration was **NOT RUN**.

## Evidence and configuration

Native metrics: [verl_file_logger.jsonl](../experiments/R1_grpo_baseline/metrics/verl_file_logger.jsonl).
Audited measurements: [R1_audit.json](R1_audit.json).
Regenerated [RUN_REPORT.md](../experiments/R1_grpo_baseline/RUN_REPORT.md) and
[training_metrics.csv](../experiments/R1_grpo_baseline/metrics/training_metrics.csv).
Exact command-line overrides recovered from the original shell xtrace:
[launch_overrides.json](../experiments/R1_grpo_baseline/launch_overrides.json).

Verified overrides: Qwen3-8B; `train_batch_size=16`, `rollout.n=8`, mini-batch 8,
one PPO epoch; 8192 response-token budget; `enable_thinking=False`; LR 1e-6;
symmetric clips 0.2/0.2; token-mean loss; KL loss coefficient 0.001; no KL in reward;
group filtering disabled; temperature 1 / top-p 1; TP=2 on four H20s;
validation every 10 steps, no pre-training validation, `save_freq=-1`, resume disabled.
There are 20 trainer updates and two optimizer mini-batch steps per update by this configuration.
No checkpoint was requested or produced; checkpoint/resume testing: **NOT RUN**.

VeRL commit `1252cc71aa5bd82e5604322064d69bfe6454c660`; lab launch commit
`4382441ca545481b6a141449eab73b2093c2a686`; seed 20260910.
The training log selects the deduplicated `data/dapo_math_17k/train.parquet`,
not the raw 100-fold repeated dataset. All **320 sampled prompts are distinct**,
with exactly 8 responses each: **2560 rollouts**.

Provenance limitations discovered during audit:

- `resolved_config.yaml` originally contained only a NOT RUN placeholder. Full resolved
  configuration is **unavailable** there; recovered overrides are explicitly not a substitute
  for all resolved defaults.
- The launch manifest's dataset hash `534375d6bb8630d2` does not match the actual
  selected training file. The launcher's default hash path points to raw data. Actual current
  deduplicated train SHA-256 is `1b86ea9f4e194c75f4604565b23d98bc386e11c81775a75109068a49a0c065d9`;
  validation SHA-256 is `0af4d4bf0ba3eb1fefd882826fd69393d0f5236805751686c7958094c79890c0`.
  These are post-run hash checks, not retroactive launch-time measurements. Original manifest
  provenance was retained.
- Actual rollout directory comes from the manifest:
  `logs/R1_grpo_baseline-20260910-134534/rollout_dump`, not the handoff's fallback path.

## 1. Clipping and policy shift

Both `actor/pg_clipfrac` and `actor/ppo_kl` are nonzero on **all 20 updates**.
Clip fraction ranges **5.0718e-5–2.6074e-4** (0.0051%–0.0261%; mean 0.0154%).
Signed approximate KL ranges **−5.4436e-5 to +4.0170e-5**.
This is consistent with the intended mini-batch configuration making subsequent updates
compare different weights; the R0 all-zero invariant no longer applies.

Source inspection (`core_algos.py:1336–1360`) matters: `ppo_kl` is the sampled,
masked mean of old minus current log-probability. Small negative values are possible;
this is not a guaranteed nonnegative exact distribution KL. `actor/kl_loss` against
the fixed reference is a different quantity: **0.000143 → 0.002413**, maximum 0.003082.

`pg_clipfrac` counts where clipping changes the PPO objective, for either advantage
sign. `pg_clipfrac_lower` measures the vanilla dual-clip branch, and is zero throughout;
it is **not** the directional lower-ratio clip fraction. Directional upper/lower rates
and importance-ratio quantiles are **unavailable**. Reporting labels were corrected;
native records were not altered. No evidence of clipping saturation is present.

## 2. Reward and effective group signal

Training reward mean is **−0.06406** overall. First/last values are −0.5000 / +0.2656;
first-five/last-five means are −0.3156 / +0.0719. This is an upward descriptive training
trend with large batch fluctuations, not a matched-prompt estimate of learning.

All 2560 saved responses were re-scored on CPU using the **unchanged official verifier**
(SHA-256 `a1943c1379313cbde933687a496bd3071f9f830cbbb70158e615d21cf31cf713`): **zero score mismatches**.
States are **1198 verifier-positive (46.7969%)**, **1356 parsed but not matched**,
and **6 parse failures (0.2344%)**. Re-scoring raised no exception. Training-time
verifier exception and latency metrics remain **unavailable**; this audit measures
re-scoring, not historical runtime telemetry. Parsed-not-matched is not proof of a
mathematical error: formatting/normalization can still matter. No relaxed verifier was used.

Of 320 groups, **204 mixed (63.75%)**, 47 all-positive (14.6875%), and 69 all-negative
(21.5625%); zero-variance share is **36.25%**. Effective-signal fraction ranges
50%–81.25%, first 62.5%, last 56.25%. There is no sustained signal collapse.
Mixed groups provide nonzero GRPO reward advantages; this is not a measurement of
the final gradient norm of each group, and zero-advantage groups can still enter KL regularization.

## 3. Entropy and validation

Entropy is **0.35845 → 0.33841**, range 0.22545–0.44323. First-half/second-half means
are **0.32743 / 0.33402**: the whole trajectory does not support sustained entropy collapse.
Validation uses the configured dataset of 200 prompts, one greedy response per prompt
(rollout configuration defaults: temperature 0, `do_sample=False`, n=1).

| update | verifier accuracy | verifier reward |
|---|---:|---:|
| pre-training | NOT RUN | NOT RUN |
| 10 | 49.5% (99/200) | −0.01 |
| 20 | 48.5% (97/200) | −0.03 |

Entropy decline accompanied by validation improvement is **not observed**.
The two-answer validation difference alone does not establish degradation either.
Per-example validation outputs are unavailable for a paired change audit. Longer runs,
repeated seeds, and matched fixed-policy baselines are **NOT RUN**.

## 4. Response length and truncation

Mean response length is **1683.84 → 1508.27 tokens**; first-half/second-half means
1692.44 / 1656.43. Peaks occur at update 10 (2188.11) and update 19 (2147.10),
with no sustained growth over this short run. Exact token median/p95 and per-response
token counts are **unavailable** in the dump; character counts were not substituted.

The old `truncation_rate` mapping was misleading. In `metric_utils.py:459–460,576`,
the comparison target is `batch['responses'].shape[-1]`, which can be a dynamically
sized tensor width, not the configured 8192 cap. For example, update 10 reports 1/128
width hits even though its longest response is only 6625 tokens.

Using the actual 8192 override and native maximum, **configured-cap hits are 5/2560
(0.1953%)**: 3 at update 5, 1 at update 14, and 1 at update 20. Other updates have
zero cap hits. The original width-hit series totals 22/2560 (0.8594%) and overstates
this proxy. Exact truncation/finish-reason rate remains **unavailable**: hitting the
budget and finishing by EOS on its boundary cannot be separated. The corrected figure
therefore says **cap-hit proxy**, not exact truncation. This run does not show cap
contamination rising monotonically or dominating reward.

## 5. Wall-time attribution and resource cost

Shares below use sums of measured stage times, not the unweighted mean of per-step
percentages. Training-step time totals **1823.28 s (30.39 min)**. Validation is separately
recorded as **167.61 s**; adding it once gives 1990.89 s, consistent with the approximately
33:14 progress duration. Full manifest wall time is 2182.01 s.

| stage | seconds | share of recorded training-step time | share of full run wall |
|---|---:|---:|---:|
| Rollout | 751.24 | 41.20% | 34.43% |
| Old-policy logprob | 171.34 | 9.40% | 7.85% |
| Reference logprob | 188.03 | 10.31% | 8.62% |
| Actor update | 618.46 | 33.92% | 28.34% |
| Weight synchronization | 82.20 | 4.51% | 3.77% |
| Other within recorded training steps | 12.01 | 0.66% | 0.55% |
| Validation (separately timed) | 167.61 | excluded | 7.68% |
| Residual wall time: initialization / orchestration / teardown / finalization | 191.11 | excluded | 8.76% |

Verifier stage time is **unavailable**, not zero; it may be nested within other work.
The residual is arithmetic accounting, not independently measured substage attribution.
Rollout is the largest stage, followed by actor update; there is no basis for attributing
this run's bottleneck to the verifier. Reference logprob takes 29.82 s on the first update,
versus roughly 6–11 s later, consistent with initialization/offload overhead.

Native masked response-token total is **4,286,556** (may include EOS); total tokens
including repeated prompt tokens are **4,686,892**. Validation generated-token count is
**unavailable**, so 4.287M is the training output budget, not total run generation.
Allocated cost is **2.424 GPU-hours** (4 GPUs × 0.6061 h), about **¥24.24** at the
user-provided ¥40/node-hour rate for the recorded run interval; post-run idle time is excluded.
Peak actor allocated/reserved memory is 53.36 / 66.64 GiB. Telemetry GPU peaks must be
read as run-level samples, not per-update allocations: the existing collector's peak window
is a rolling wall-time query. GPU memory was released and all four GPUs were idle at audit.

## 6. Anomalies, checked in playbook order

All three detector bundles retain **Root cause: PENDING**. No training repair, learning-rate
change, verifier change, or restart was performed.

| detector event | evidence checked | warranted conclusion |
|---|---|---|
| update 10, length robust-z +6.16 | 2188.11 mean / 6625 max tokens; 0 cap hits, 0 parse failures, exact reward replay; 12/16 mixed groups; small KL/clip and grad 0.2354 | transient long batch; cap contamination is not supported here; causal attribution remains PENDING |
| update 14, length robust-z +5.30 | 2115.34 mean, 1 cap hit, 1 parse failure; 3 all-positive, 5 all-negative, 8 mixed groups; grad 0.2011 | limited length/format failures coexist with reward variation; not enough evidence to blame policy instability |
| update 18, grad robust-z +5.51 | grad 0.51845 versus 0.25767 at 17, then 0.25153 at 19; advantages still within ±2.47487; clip 0.000249, KL −2.159e-5; 0 cap hits and 0 parse failures | isolated gradient excursion; no simultaneous sustained KL/clip escalation or nonfinite signal; mechanism PENDING |

Across the full run: no nonfinite native scalar metrics, maximum measured policy staleness
0, fixed LR, 320 distinct prompts. Batch composition changes on every step, so prompt
difficulty and learning are confounded. Re-scoring rules out score drift on these saved
outputs; it does not prove the verifier is semantically complete. Sampled long outputs
include a correct completed geometry solution at update 10 and an incorrect long numerical
search at update 14. Those examples do not establish dataset-wide repetition or reward hacking.
Fixed-policy regeneration, altered length budgets, gradient replay and changed LR trials:
**NOT RUN**.

A further **exit-phase traceback** occurs after the 100% progress lines: `weakref._exitfunc`
→ `torch.library._del_library` → DataLoader signal handler reports a worker killed by signal.
The final validation and all 20 metric/dump records exist, and the launcher records exit 0.
This is documented as a cleanup anomaly; it is not silently called a wholly clean exit,
and the signal's cause (including possible OOM) is **PENDING** without OS-level evidence.
No restart was attempted. See [incident_log.md](incident_log.md), INC-005/INC-006.

## Reporting corrections and validation

Post-run changes affect only analysis/monitoring: preserve missing verifier timing,
include the two measured validation points, distinguish PPO clipping from dual clipping,
separate tensor-width hits from configured-cap hits, and preserve the original end timestamp
when regenerating reports. The original native metrics, training code, weights, and incident
root-cause fields were not changed. Original reporting artifacts were backed up locally on
the server under `tmp/codex_r1_original`.

CPU synthetic tests passed for missingness, measured zeros, clipping semantics, cap-hit
guarding, validation sparsity and collection; the real audit passed 20-step continuity,
128 responses / 16 groups / 8 samples per group, reward-mean agreement and exact verifier replay.
The public status page was stale at step 19 and has been marked completed at step 20.

## Per-update evidence

| update | reward mean | mixed groups | mean response tokens | cap hits | parse failures |
|---|---:|---:|---:|---:|---:|
| 1 | -0.5000 | 10/16 | 1683.8 | 0/128 | 0 |
| 2 | -0.1562 | 13/16 | 1643.8 | 0/128 | 1 |
| 3 | -0.4219 | 10/16 | 1695.0 | 0/128 | 0 |
| 4 | -0.2188 | 12/16 | 1447.7 | 0/128 | 0 |
| 5 | -0.2812 | 9/16 | 1804.3 | 3/128 | 3 |
| 6 | 0.3438 | 11/16 | 1384.5 | 0/128 | 0 |
| 7 | 0.0781 | 12/16 | 1803.2 | 0/128 | 0 |
| 8 | -0.1250 | 9/16 | 1608.4 | 0/128 | 0 |
| 9 | -0.1875 | 12/16 | 1665.6 | 0/128 | 0 |
| 10 | -0.2656 | 12/16 | 2188.1 | 0/128 | 0 |
| 11 | 0.1250 | 9/16 | 1615.0 | 0/128 | 0 |
| 12 | 0.0000 | 10/16 | 1697.5 | 0/128 | 0 |
| 13 | 0.1719 | 9/16 | 1591.3 | 0/128 | 0 |
| 14 | -0.2344 | 8/16 | 2115.3 | 1/128 | 1 |
| 15 | 0.0312 | 11/16 | 1788.3 | 0/128 | 0 |
| 16 | 0.1094 | 10/16 | 1566.3 | 0/128 | 1 |
| 17 | 0.3281 | 8/16 | 1343.0 | 0/128 | 0 |
| 18 | 0.0625 | 11/16 | 1192.2 | 0/128 | 0 |
| 19 | -0.4062 | 9/16 | 2147.1 | 0/128 | 0 |
| 20 | 0.2656 | 9/16 | 1508.3 | 1/128 | 0 |

## Next action

**STOP and await user direction.** R2 DAPO, Dr.GRPO, GSPO, VAPO, longer R1 and failure
injection are **NOT RUN**. Any next algorithm must pass its 2-update smoke gate before
20 updates; no automatic extension is authorized. This baseline is useful for observed
pipeline dynamics, with the provenance and instrumentation limitations above carried forward.

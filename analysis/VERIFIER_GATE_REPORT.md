# Verifier Gate Report

VeRL commit `1252cc71aa5bd82e5604322064d69bfe6454c660` · 2026-09-10
Evidence: [`verifier_unit_test.md`](verifier_unit_test.md) ·
[`verifier_semantics.md`](verifier_semantics.md) ·
[`reward_pipeline_trace.md`](reward_pipeline_trace.md) ·
[`verifier_cases.jsonl`](verifier_cases.jsonl)

```
PRIMARY_REWARD:                 RLVR / deterministic, rule-based
REWARD_FUNCTION:                verl.utils.reward_score.math_dapo.compute_score
                                (reached via verl.utils.reward_score.default_compute_score)
DATA_SOURCE_ROUTE:              "math_dapo"  -> reward_score/__init__.py:59
                                (AIME 2024 routes to the SAME verifier via
                                 data_source.startswith("aime"))
REWARD_MANAGER:                 NaiveRewardManager  (reward_manager.name=naive,
                                 reward_model.enable=false)

CORRECT_REWARD:                 +1.0
INCORRECT_REWARD:               -1.0
PARSE_FAILURE_REWARD:           -1.0   (indistinguishable from "incorrect" in the
                                        official return; separated only by our
                                        instrumentation via pred == "[INVALID]")

CORRECT_ACCEPT_RATE:            88.89%   (160/180)
WRONG_REJECT_RATE:             100.00%   (60/60)
PARSE_FAILURE_RATE:              0.00%   exceptions; 100% of malformed cases correctly
                                        FLAGGED as PARSE_FAILURE (100/100)
VERIFIER_EXCEPTION_RATE:         0.00%   (0/440 constructed, 0/512 real rollouts)
FORMAT_FALSE_NEGATIVE_RATE:      6.05%   (31/512 real rollouts: correct answer pushed
                                        outside the 300-char extraction window)
                                 0.00%   (trailing-period mode: real but dormant for
                                        this policy)

MEAN_LATENCY:                   0.017 ms (unit test) / 0.020 ms (real rollouts)
P95_LATENCY:                    0.014 ms (unit test) / 0.022 ms (real rollouts)
MAX_LATENCY:                    2.23 ms

LLM_JUDGE_USED_FOR_TRAINING:    NO
MATH_VERIFY_USED_FOR_BASELINE:  NO
```

## FINAL_DECISION: `VERIFIER_READY_FOR_R0`

The verifier itself passes. It is deterministic, exception-free, negligibly cheap, and
never accepts a wrong answer. Its two false-negative modes are characterised with
measured rates, and neither is a reason to swap in Math-Verify: the trailing-period mode
is dormant for this policy, and the 300-character extraction window is a truncation
issue that symbolic equivalence would not address.

**However — and this is separate from the verifier gate:**

## `R1 IS BLOCKED` — response-length budget, not the reward function

The fixed-policy rollout (64 prompts x G=8 = 512 samples, `max_response_length=4096`)
returned:

| metric | value |
|---|---|
| truncation_rate | **87.11%** |
| zero_std_group_ratio | **79.69%** |
| effective_signal_fraction | **20.31%** |
| accuracy | 12.70% |
| reward mean | −0.7461 |
| **genuinely wrong answers** | **2 / 512 (0.39%)** |

All 512 responses open a `<think>` block (Qwen3-8B's chat template default). The
response-length median sits *exactly* on the 4096 cap, and truncated samples are cut off
mid-reasoning with no `</think>`, no `Answer:` line and no `\boxed{}`. Essentially the
entire negative reward mass is "the model never finished speaking", not "the model was
wrong".

Per the stage-gate rule, R1 does not start on this configuration, and the fix is **not**
to relax the verifier — that would hide a prompt/output-format fault rather than repair
it. The length budget is being resolved by direct measurement in
[`../scripts/evaluation/length_budget_probe.py`](../scripts/evaluation/length_budget_probe.py)
(thinking @16384 vs non-thinking @4096); see
[`pre_rl_reward_distribution.md`](pre_rl_reward_distribution.md).

**R0 is not blocked by this.** R0 is a pipeline smoke test whose exit criterion is that
one optimizer update's data flow is fully accounted for — it does not require a healthy
reward distribution to be valid, and running it now surfaces any masking / logprob /
advantage bug before those would corrupt R1.

## Secondary audit — deliberately not yet invoked

MiniMax-M3 and DeepSeek-V4-Pro were **not called**. They remain available for a
read-only audit of at most 50-100 hand-picked cases (rule-verifier-says-incorrect but
plausible, parse failures, symbolic edge cases, verifier disagreements, suspicious
high-reward trajectories), at temperature 0, emitting only
`EQUIVALENT` / `NOT_EQUIVALENT` / `UNCERTAIN`, with endpoint, version, prompt, response
and latency recorded.

They would produce **no training reward** under any circumstances. On present evidence
there is little for them to adjudicate: 0% exceptions, 100% wrong-rejection, and an
all-integer answer set where symbolic equivalence is nearly vacuous.

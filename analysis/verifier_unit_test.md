# Verifier Unit Test — official `math_dapo`

Script: [`../scripts/evaluation/verifier_unit_test.py`](../scripts/evaluation/verifier_unit_test.py)
Raw records: [`verifier_cases.jsonl`](verifier_cases.jsonl)
VeRL commit `1252cc71` · seed `20260910` · **440 cases = 20 real DAPO-Math-17k problems x 22 variants**

The function under test is the one the trainer will actually call —
`default_compute_score(data_source="math_dapo", ...)` — not a reimplementation.

## Headline results

| statistic | value |
|---|---|
| **CORRECT_ACCEPT_RATE** | **88.89%** (160/180) |
| **WRONG_REJECT_RATE** | **100.00%** (60/60) |
| **PARSE_FAILURE_DETECTED** | **100.00%** (100/100 malformed cases correctly flagged) |
| **VERIFIER_EXCEPTION_RATE** | **0.00%** |
| truncation-window survival | **0.00%** (0/40) |
| latency mean / median / p95 / max | 0.017 / 0.012 / 0.014 / 2.23 ms |

Correct answers are never mistaken for wrong ones in the *reward direction* — every
one of the 60 clearly-wrong cases was rejected, and no case produced an exception.
The 11.11% shortfall in accept rate is a single, identifiable formatting mode.

## Why equivalence testing focuses on integer formatting

**All 17,917 unique ground truths in DAPO-Math-17k are plain integers** (verified by
classifying every deduplicated `reward_model.ground_truth`; `integer` = 100.0%).

So `1/2` vs `0.5`, `2/4` vs `1/2`, `x^2` vs `x*x` are **not** meaningful tests for this
dataset — constructing them would manufacture a failure mode that cannot occur here.
The equivalence classes that actually matter are integer *formatting* variants, which
is what was tested.

This also settles the Math-Verify question empirically: symbolic-equivalence power buys
almost nothing on an all-integer answer set.

## Per-variant outcome

| case | form | score | state |
|---|---|---|---|
| `A_canonical_correct` | `Answer: 34` | `+1.0` | OK_CORRECT |
| `B1_boxed_correct` | `$\boxed{34}$`, no `Answer:` line | `+1.0` | OK_CORRECT |
| `B2_dollar_wrapped` | `Answer: $34$` | `+1.0` | OK_CORRECT |
| **`B3_trailing_period`** | **`Answer: 34.`** | **`-1.0`** | **OK_INCORRECT** |
| `B4_bold_latex` | `Answer: \textbf{34}` | `+1.0` | OK_CORRECT |
| `B5_spaces` | `Answer:   34   ` | `+1.0` | OK_CORRECT |
| `B6_decimal_form` | `Answer: 34.0` | `-1.0` | OK_INCORRECT |
| `B7_plus_sign` | `Answer: +34` | `-1.0` | OK_INCORRECT |
| `B8_comma_grouped` | `Answer: 1,234` | `+1.0` | OK_CORRECT |
| `B9_text_wrapped` | `Answer: \text{34}` | `+1.0` | OK_CORRECT |
| `C1/C2_wrong` | wrong integer, plain and boxed | `-1.0` | OK_INCORRECT |
| `D1-D5_malformed` | no answer / empty / whitespace / unbalanced `\boxed{` / bare `Answer:` | `-1.0` | **PARSE_FAILURE** |
| `E_long_cot_correct` | long reasoning then `Answer: 34` | `+1.0` | OK_CORRECT |
| `F_long_cot_wrong` | long reasoning then wrong | `-1.0` | OK_INCORRECT |
| **`G_correct_but_outside_300char_window`** | correct answer + ~600 chars of trailing text | **`-1.0`** | **PARSE_FAILURE** |
| **`H_boxed_outside_window`** | boxed answer + trailing text | **`-1.0`** | **PARSE_FAILURE** |
| `I_think_block_then_answer` | `<think>...</think>` then `Answer: 34` | `+1.0` | OK_CORRECT |

## The two real false-negative modes

### 1. Trailing period — `Answer: 34.` scores `-1.0`

`SUBSTITUTIONS` contains `(".$", "$")`, which removes a period only when it precedes a
`$`. A bare trailing period survives normalisation and breaks the exact string equality
at `math_dapo.py:190`.

**Measured incidence in 512 real Qwen3-8B rollouts: 0.** The flaw is real but currently
dormant for this policy. It is recorded because RL shifts the output distribution — if
the policy drifts toward ending its answer line with a period, reward would fall for a
purely cosmetic reason and would look exactly like a policy regression.

### 2. The 300-character extraction window — 0/40 survived

`compute_score` truncates to `solution_str[-300:]` (`math_dapo.py:267`) before any
extraction, and the boxed fallback narrows further to `pred[-100:]` (`L211`). A correct
answer followed by more than ~300 characters of closing commentary is invisible.

The source comment justifies 300 chars because "the longest answer in MATH-500 has 159
characters" — but the cap is applied to the whole response *tail*, not to the answer.

**Measured incidence in real rollouts: 31/512 = 6.05%.** This one is active.

## State instrumentation

The official return exposes only `score` / `acc` / `pred`, and maps **both** "wrong
answer" and "unparseable answer" to `-1.0`. The unit test layers a state label derived
from that return without altering it:

`OK_CORRECT` · `OK_INCORRECT` · `PARSE_FAILURE` · `VERIFIER_EXCEPTION` · `TRUNCATED`

All 100 malformed cases were correctly separated as `PARSE_FAILURE` rather than being
silently absorbed into "the model was wrong". This split is what makes it possible to
tell a formatting regression from a policy regression during training.

## Exceptions

Zero exceptions across 440 constructed cases (including unbalanced `\boxed{`) and 512
real rollouts. `remove_boxed()`'s bare `assert`s are guarded in practice because
`last_boxed_only_string()` returns `None` unless it found a balanced expression.

This matters because **`NaiveRewardManager` wraps `compute_score` in no `try`/`except`**
(`naive.py:66-84`) — an exception would propagate rather than being converted into a
reward. Visible-crash-over-silent-contamination is the safer default, but it does mean
a single malformed sample could abort a step.

## Latency

Deterministic, pure-Python, no network: **0.017 ms mean**, 2.23 ms worst case. That is
roughly five orders of magnitude below rollout cost, so the verifier can be excluded as
a throughput bottleneck by inspection, and there is no timeout path to contaminate the
reward. R3-C must therefore *inject* verifier failures artificially to study them.

## Verdict

The verifier is **reliable in the reward direction** (100% wrong-rejection, 0%
exceptions, negligible latency) and its two false-negative modes are characterised with
measured rates. Neither is grounds for swapping in Math-Verify: mode 1 is dormant, and
mode 2 is an extraction-window issue that symbolic equivalence would not fix.

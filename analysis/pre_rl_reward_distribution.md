# Pre-RL Reward Distribution — fixed-policy rollout

Script: [`../scripts/evaluation/fixed_policy_rollout.py`](../scripts/evaluation/fixed_policy_rollout.py)
Raw dump: `pre_rl_rollouts.jsonl` (512 records) · stats: `pre_rl_rollout_stats.json`

**Config:** `Qwen/Qwen3-8B` (untrained, released post-trained weights) · 64 prompts x
**G=8** = 512 samples · temperature 1.0, top_p 1.0 · `max_prompt_length=2048` ·
**`max_response_length=4096`** · seed `20260910` · vLLM 0.24.0, bf16, TP=1 ·
scored with the **same** official `math_dapo` verifier R0 will use.

Wall time 747.4 s, 2,024,795 output tokens, 2709 tok/s.

---

## Verdict: **STOP — do not start R1 on this configuration**

This is not a marginal call. The batch is dominated by a systems fault, not by policy quality.

| metric | measured | gate |
|---|---|---|
| **truncation_rate** | **87.11%** (446/512) | heuristic warning at 30% |
| **zero_std_group_ratio** | **79.69%** (51/64) | heuristic warning at 70% |
| all-wrong groups | **75.00%** (48/64) | |
| all-correct groups | 4.69% (3/64) | |
| mixed groups | 20.31% (13/64) | |
| **effective_signal_fraction** | **20.31%** | heuristic warning at 30% |
| accuracy | 12.70% | |
| reward mean | **-0.7461** (scale is +1/-1) | |
| mean group std | 0.1567 | |

**Only 13 of 64 prompts would have contributed any gradient at all.** GRPO normalises
advantage within a group; a group whose samples all receive the same reward yields an
identically zero advantage. So the nominal prompt batch (64) and the batch that actually
trains the policy (13) differ by ~5x.

---

## Root cause: the response-length budget, not the policy or the verifier

Evidence, in order of decisiveness:

1. **All 512/512 responses begin with `<think>`.** Qwen3-8B's chat template enables
   thinking mode by default, and VeRL builds prompts with that same template.

2. **Response-length histogram is a wall at the cap:**

   | token bucket | count |
   |---|---|
   | 2048-2559 | 21 |
   | 2560-3071 | 15 |
   | 3072-3583 | 18 |
   | 3584-4095 | 12 |
   | **4096+** | **446** |

   Mean 3955, **median 4096**, p95 4096, max 4096. The median sits exactly *on* the cap —
   the distribution is censored, not centred.

3. **Truncated samples are cut off mid-reasoning, still inside the `<think>` block.**
   A representative tail ends `"...Therefore, for each choice of Paper"` — no `</think>`,
   no `Answer:` line, no `\boxed{}`. The verifier is handed a fragment that contains no
   answer by construction.

4. **Verifier state distribution makes the mechanism explicit:**

   | state | count | share |
   |---|---|---|
   | `TRUNCATED` | 442 | 86.33% |
   | `OK_CORRECT` | 65 | 12.70% |
   | `PARSE_FAILURE` | 3 | 0.59% |
   | `OK_INCORRECT` | **2** | **0.39%** |

   **Only 2 of 512 samples were genuinely wrong answers.** Essentially the entire
   negative reward mass is "the model never finished speaking", not "the model was wrong".

5. **Even successful responses barely fit:** correct answers averaged 3070 tokens
   (median 3101, max 4096) against a 4096 cap.

### Format false negatives (upper bound)

256/512 (50.00%) of `-1.0` samples contained the ground-truth integer *somewhere* in
their text. Decomposed by cause:

| cause | count | share |
|---|---|---|
| no answer marker emitted (still reasoning when cut) | 217 | 42.38% |
| correct answer pushed outside the 300-char extraction window | 31 | 6.05% |
| formatting mismatch | 0 | 0.00% |

This 50% is an **upper bound, not a false-negative rate** — a long chain of thought
routinely mentions the correct number mid-derivation without concluding it, so
"contains the integer" does not mean "answered correctly". Reported as an upper bound
deliberately; treating it as the true rate would overstate the verifier's fault.

The decomposition is the useful part: the dominant term (42.38%) is *truncation*, and
the verifier's own extraction window accounts for a much smaller 6.05%.

---

## What this rules out

- **Not a verifier reliability problem.** 0% exception rate, 100% wrong-rejection,
  0.020 ms mean latency (see [`verifier_unit_test.md`](verifier_unit_test.md)).
  Loosening the verifier would *hide* this fault, not fix it — and is explicitly
  forbidden as a response to a prompt/output-format mismatch.
- **Not a policy quality problem.** Only 2/512 samples were actually wrong answers.
- **Not sampling temperature.** Temperature 1.0 is the intended exploration setting;
  the failure is that generation is cut off, not that it is too random.
- **Not dataset difficulty** — at least, difficulty is not yet *measurable*, because
  the policy is not being allowed to finish. Difficulty can only be assessed after the
  length budget is fixed.

## What must change before R1

The fix is to the **length budget / thinking-mode contract**, not to the reward function.
Two candidates, measured head-to-head in
[`../scripts/evaluation/length_budget_probe.py`](../scripts/evaluation/length_budget_probe.py):

- **A** — thinking mode on (template default), `max_response_length = 16384`
- **B** — thinking mode off (`enable_thinking=False`), `max_response_length = 4096`

Results are recorded in `length_budget_probe.json`. The R1 configuration is chosen from
that measurement rather than from intuition, because the two options have very different
throughput and KV-cache costs and the choice changes what the experiment is even studying.

## Note for R3-A

R3-A was designed to *inject* truncation contamination deliberately (4096 -> 1024) to
show that a reward drop can be a length fault rather than policy degradation. This
preflight produced the same phenomenon **accidentally, in the intended baseline config**,
which is a stronger demonstration than the planned injection: it is exactly the class of
failure catalogued as "apparent training failure that is really a systems fault". The
measurements here become the natural control for R3-A.

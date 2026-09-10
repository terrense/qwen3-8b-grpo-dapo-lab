# Verifier Semantics — official `math_dapo` at VeRL `1252cc71`

All values below were read from the current source **and** confirmed empirically by
440 constructed cases (`verifier_unit_test.md`, `verifier_cases.jsonl`). Nothing is
assumed from the DAPO paper or from older VeRL releases.

## Return contract

`verl/utils/reward_score/math_dapo.py:249-279` returns a **dict**:

```python
{"score": reward,   # float
 "acc":   correct,  # bool
 "pred":  pred}     # str | None -- the extracted answer, or "[INVALID]" / None
```

## Reward values — measured, not assumed

| outcome | `score` | `acc` | `pred` |
|---|---|---|---|
| correct | **`+1.0`** | `True` | extracted answer string |
| incorrect | **`-1.0`** | `False` | extracted answer string |
| **no answer extractable** | **`-1.0`** | `False` | `"[INVALID]"` or `None` |

`math_dapo.py:272` — `reward = 1.0 if correct else -1.0`.

So the scale is **+1 / −1, not +1 / 0**. Two consequences that matter for GRPO:

1. A group where every sample is wrong has mean `−1.0` and **std 0** → the
   group-normalised advantage is identically zero → that prompt contributes **no
   gradient at all**. It is not a "small negative push"; it is silence.
2. **Parse failure is scored identically to a wrong answer.** The official return
   gives no way to separate them from `score` alone — only `pred == "[INVALID]"`
   reveals it. This is why our instrumentation records a separate state.

## Extraction: two mechanisms, in order

**1. Minerva-style (primary).** `is_correct_minerva`, `math_dapo.py:165-190`
Pattern: `r"(?i)Answer\s*:\s*([^\n]+)"`, taking the **last** match. This matches the
DAPO prompt's own instruction ("The last line of your response should be of the form
`Answer: $Answer`"), so prompt and verifier are contractually aligned.

**2. `\boxed{...}` (fallback).** `is_correct_strict_box`, `math_dapo.py:193-217`
Used only when Minerva extraction yields `"[INVALID]"` (`L240-245`). Confirmed useful
in practice: real rollouts that ended in `$\boxed{5}$` with no `Answer:` line were
scored correctly.

Comparison in both paths is **exact string equality after normalisation**
(`pred == gt`, `L190` / `L217`) — there is **no symbolic algebra**.

## Normalisation

`normalize_final_answer`, `math_dapo.py:124-162`: strips everything before `=`,
applies a `SUBSTITUTIONS` table (removes spaces, `\$`, `an `, `a `, ...) and a
`REMOVED_EXPRESSIONS` list (units like `cm`, `degrees`, `dollars`, `\text{}` wrappers,
`^\circ`, ...), unwraps `$...$`, `\text{}`, `\textbf{}`, `\overline{}`, `\boxed{}`,
normalises shorthand `\fracab -> \frac{a}{b}`, and strips thousands separators from
pure digit strings (`L159-160`).

### Measured equivalence behaviour

Every ground truth in DAPO-Math-17k is a **plain integer** (17,917 / 17,917 verified),
so integer *formatting* is the equivalence class that actually matters — symbolic
equivalence like `1/2` vs `0.5` essentially never arises in this dataset.

| response form | accepted? |
|---|---|
| `Answer: 34` | **yes** |
| `$\boxed{34}$` (no `Answer:` line) | **yes** (boxed fallback) |
| `Answer: $34$` | **yes** |
| `Answer: \textbf{34}` | **yes** |
| `Answer: \text{34}` | **yes** |
| `Answer:   34   ` (padding) | **yes** |
| `Answer: 1,234` (comma-grouped) | **yes** |
| **`Answer: 34.`** (trailing period) | **NO — scored `-1.0`** |
| `Answer: 34.0` | no |
| `Answer: +34` | no |

**The trailing-period case is a genuine false negative.** `SUBSTITUTIONS` contains
`(".$", "$")`, which only removes a period immediately before a `$`; a bare trailing
period survives normalisation and breaks exact equality. `Answer: 34.` is a
completely natural way for a model to end a sentence.

Measured incidence in 512 real Qwen3-8B rollouts: **0 cases** — this model does not
end its answer line with a period. So the flaw is real but currently **not active**
for this policy. It is recorded because it could activate at any point during RL as
the output distribution shifts, and it would then look exactly like a reward drop.

## Two truncation windows — the dominant practical risk

Before any extraction happens:

- `compute_score` keeps only **`solution_str[-300:]`** (`math_dapo.py:267`)
- the boxed fallback further narrows to **`pred[-100:]`** (`math_dapo.py:211`)

The 300-character cap is justified in a source comment as "the longest answer in
MATH-500 has 159 characters" — that reasoning is about the *answer*, but the cap is
applied to the *whole response tail*. Any correct answer followed by more than ~300
characters of closing commentary is invisible to the verifier.

Measured: **0 / 40** constructed cases with a correct answer pushed just outside the
window were credited. All 40 became `PARSE_FAILURE` → `-1.0`.

In real rollouts this fired **31 / 512 (6.05%)** of the time.

## Latency

Deterministic, pure-Python, no network:

| | unit test (440 cases) | real rollouts (512) |
|---|---|---|
| mean | 0.017 ms | 0.020 ms |
| median | 0.012 ms | 0.016 ms |
| p95 | 0.014 ms | 0.022 ms |
| max | 2.23 ms | 1.62 ms |

Verifier cost is ~5 orders of magnitude below rollout cost. **Reward computation will
never be the bottleneck**, and there is no timeout path to contaminate — which is
exactly why R3-C has to *inject* verifier failures artificially to study them.

## Math-Verify is deliberately NOT used for the baseline

`verl/utils/reward_score/math_verify.py` exists in this checkout, but the
`math_dapo` branch does not reference it — Math-Verify appears only as a
**commented-out optional override** inside the `lighteval/MATH` branch
(`__init__.py:50-57`). Selecting it would be a deviation from the official DAPO
reward path.

The baseline reproduces the official path. A verifier swap would only be justified by
evidence of symbolic-equivalence false negatives, and the dataset's all-integer ground
truth makes that class of error close to irrelevant here. If it is ever tested, it goes
in a dedicated `R_verifier_ablation` — never mixed into a GRPO/DAPO algorithm comparison,
because then two variables move at once.

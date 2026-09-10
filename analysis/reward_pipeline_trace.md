# RLVR Reward Pipeline Trace

Traced against the checked-out source at VeRL commit
**`1252cc71aa5bd82e5604322064d69bfe6454c660`**. Every claim below cites a real file
and line range in that checkout. Nothing here is from memory of the DAPO paper or of
older VeRL releases.

## Decision recorded

This project's **primary training reward is deterministic, rule-based RLVR**
(Reinforcement Learning with Verifiable Rewards). It is **not** an LLM judge and
**not** a reward model, because DAPO-Math-17k and AIME both ship ground truth, and a
deterministic verifier avoids API latency, timeout contamination, judge stochasticity
and endpoint/version drift — all of which would blur the line between *verifier error*
and *policy error*, which is precisely the distinction this lab exists to study.

MiniMax-M3 / DeepSeek-V4-Pro are **not called** anywhere in the training loop.

## The call chain

```
dataset row  (data_source = "math_dapo")
  |
  v
NaiveRewardManager.run_single()
  verl/experimental/reward_loop/reward_manager/naive.py:34-99
  |   decodes ONLY the response tokens:
  |     valid_response_length = attention_mask[-response_length:].sum()      (L39)
  |     valid_response_ids    = response_ids[:valid_response_length]         (L40-41)
  |     response_str = tokenizer.decode(valid_response_ids,
  |                                     skip_special_tokens=True)            (L54-56)
  |   -> the prompt is NEVER passed to the verifier
  v
default_compute_score(data_source, solution_str, ground_truth, extra_info)
  verl/utils/reward_score/__init__.py:14-107
  |   dispatch (L59):
  |     elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] \
  |          or data_source.startswith("aime"):
  |         from . import math_dapo
  |         res = math_dapo.compute_score(solution_str, ground_truth)
  v
math_dapo.compute_score(solution_str, ground_truth,
                        strict_box_verify=False, pause_tokens_index=None)
  verl/utils/reward_score/math_dapo.py:249-279
  |
  |-- L267:  solution_str = solution_str[-300:]        <-- ONLY the last 300 CHARACTERS
  |
  |-- verify()                                          math_dapo.py:220-246
  |     |-- is_correct_minerva()                        math_dapo.py:165-190
  |     |     answer_pattern = r"(?i)Answer\s*:\s*([^\n]+)"          (L166)
  |     |     match = re.findall(pattern, solution_str); takes the LAST match  (L180-181)
  |     |     extracted = match[-1] if match else "[INVALID]"
  |     |     pred = normalize_final_answer(extracted)               (L182)
  |     |     gt   = normalize_final_answer(gt)                      (L188)
  |     |     return (pred == gt), pred          <-- EXACT STRING EQUALITY  (L190)
  |     |
  |     |-- if pred != "[INVALID]": return that result               (L240-241)
  |     |
  |     `-- fallback is_correct_strict_box()             math_dapo.py:193-217
  |           pred = pred[-100:]                         <-- last 100 chars  (L211)
  |           boxed = last_boxed_only_string(pred)                    (L214)
  |           extracted = remove_boxed(boxed) if boxed is not None    (L215)
  |           return (1 if extracted == gt else -1), extracted        (L217)
  |
  `-- L272:  reward = 1.0 if correct else -1.0
      L275:  return {"score": reward, "acc": correct, "pred": pred}
  v
NaiveRewardManager reads the dict                       naive.py:88-97
    score = result["score"]                              (L90)
    every key of the dict is copied into reward_extra_info  (L91-92)
    reward = score                                       (L97)
```

## Routing confirmation

- Our dataset's `data_source` is **`math_dapo`** for all 17,917 unique prompts
  (verified against the parquet; single distinct value). It therefore matches the
  `math_dapo` branch at `__init__.py:59` exactly.
- **AIME 2024 will route to the identical verifier**, because the same branch also
  accepts `data_source.startswith("aime")`. Train and validation share one reward
  implementation — no train/val verifier mismatch is possible here.

## Reward manager selection

`verl/trainer/config/_generated_ppo_trainer.yaml`:

```yaml
reward:
  reward_manager:
    source: register
    name: naive          # -> NaiveRewardManager
  reward_model:
    enable: false        # no reward model anywhere in the loop
```

`resolve_reward_manager_cls()` (`verl/trainer/ppo/reward.py:89-108`) resolves
`source: register` through `verl.experimental.reward_loop.reward_manager.get_reward_manager_cls`,
and `NaiveRewardManager` is registered with `@register("naive")`
(`.../reward_manager/naive.py:23-24`). `load_reward_manager()`
(`verl/trainer/ppo/reward.py:111-151`) falls back to `default_compute_score` when no
custom `compute_score` is configured (L134-149), which is our case.

## Exception handling: there is none on this path

`NaiveRewardManager.run_single` calls `self.compute_score(...)` with **no `try`/`except`**
(`naive.py:66-84`). The only `except Exception` in `verl/experimental/reward_loop/reward_loop.py:181`
guards HTTP calls to a *remote reward router*, which we do not use. `verl/workers/reward_manager/prime.py`
does wrap scoring, but that is a different manager.

**Consequence:** a verifier exception propagates rather than being silently converted
into a reward. That is arguably the safer default — a crash is visible, whereas
`exception -> 0.0` would be invisible contamination — but it also means a single
malformed sample can abort a step. Measured exception rate on 440 constructed cases
and 512 real rollouts: **0.00%** (see `verifier_unit_test.md`).

`remove_boxed()` (`math_dapo.py:50-62`) uses bare `assert`s, but it is only reachable
when `last_boxed_only_string()` returned non-`None`, and that function returns `None`
unless it found a balanced `\boxed{...}` (`L37-47`). The asserts are therefore guarded
in practice, which the empirical exception rate confirms.

## What the trainer sees

Because `compute_score` returns a dict, `NaiveRewardManager` copies **every** key into
`reward_extra_info` (`naive.py:91-92`). So `acc` and `pred` are carried alongside
`score` and are available for logging — this is what makes non-invasive verifier-state
instrumentation possible without touching the algorithm.

## Instrumentation layered on top (diagnosis only)

The official return exposes only correct/incorrect via `score` and `acc`. Our
instrumentation adds a state label **derived from the official return, never replacing it**:

| state | derivation |
|---|---|
| `OK_CORRECT` | `acc` truthy |
| `OK_INCORRECT` | `acc` falsy **and** `pred` is a real extracted answer |
| `PARSE_FAILURE` | `pred` is `None` / `"[INVALID]"` — nothing could be extracted |
| `VERIFIER_EXCEPTION` | `compute_score` raised |
| `TRUNCATED` | rollout-level: `finish_reason == "length"` (not inferable from the verifier) |

This matters because the official reward maps **both** "wrong answer" and "could not
parse an answer" to the same `-1.0`. Without this split, a formatting or truncation
regression is indistinguishable from the policy getting worse — the exact confusion
this lab is built to eliminate.

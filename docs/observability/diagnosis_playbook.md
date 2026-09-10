# Diagnosis Playbook

A decision tree for RL failures, not a list of parameters to jiggle.

> **`reward` is not ground truth.** It is the *output* of a long pipeline. When it
> moves the wrong way, the policy is the **last** thing to suspect, not the first.

"Reward dropped, so lower the LR" is not a diagnosis. Every branch below tells you
which metric to read first, what the competing hypotheses are, and **how to falsify
each one**.

---

## Master tree: reward went down

```
reward drops
|
+-- verifier_parse_failure or verifier_exception up?
|      -> REWARD PIPELINE FAULT, not a policy fault
|      falsify: hold the policy fixed, re-score the SAME rollouts with the previous
|               verifier build. If reward returns, the policy never changed.
|
+-- truncation_rate up / response_length_p95 pinned at max_response_length?
|      -> LENGTH CONTAMINATION
|      falsify: raise max_response_length and re-score the same prompts. If accuracy
|               recovers with no weight change, the policy was never worse.
|      (This exact failure was observed here in the pre-RL rollout -- 87.1%
|       truncation at 4096 -- see analysis/pre_rl_reward_distribution.md.)
|
+-- KL up AND clip_fraction up AND grad_norm up, together?
|      -> POLICY UPDATE INSTABILITY (LR hypothesis)
|      falsify: replay the same batch at a lower LR. If grad norm and KL fall back
|               into their prior band, it is the step size, not the data.
|
+-- KL stable AND validation stable?
|      -> BATCH COMPOSITION or REWARD NOISE
|      falsify: compare the prompt-id distribution of this batch against previous
|               ones; check whether a harder shard entered.
|
+-- zero_std_group_ratio up / effective_signal_fraction down?
|      -> EFFECTIVE SIGNAL COLLAPSE
|      falsify: the reward may be flat because the batch is uniformly too easy or
|               too hard. Split the batch by all-correct vs all-wrong -- the two
|               have opposite fixes.
|
`-- none of the above
       -> check policy provenance (policy_staleness_steps) and old-logprob
          correctness before touching any hyperparameter.
```

---

## Per-symptom reference

### reward up but validation down
**First metrics:** validation accuracy, `truncation_rate`, response length, format success.
**Hypotheses:** (a) reward hacking — the policy found a way to satisfy the verifier without solving the problem; (b) train/val distribution mismatch; (c) overfitting to the prompt subset.
**Falsify:** read the highest-reward trajectories by hand. Reward hacking is always visible in the text. Check whether reward correlates with length rather than correctness.
**Fix:** tighten the verifier contract, or add a format/length penalty — but only after confirming the mechanism.

### KL spike
**First metrics:** `kl`, `clip_fraction_high`, `grad_norm`, LR schedule position.
**Hypotheses:** (a) step size too large; (b) a batch of unusually high-advantage samples; (c) stale rollouts inflating the importance ratio.
**Falsify:** check `policy_staleness_steps` — if it drifted, (c). If clipfrac saturated at the same step, (a). If advantage max spiked alone, (b).

### entropy collapse
**First metrics:** `entropy`, reward, validation, response length.
**Hypotheses:** (a) healthy convergence to a confident policy; (b) premature determinism / mode collapse.
**Falsify:** entropy falling *with* validation rising is (a). Entropy falling with flat or falling validation is (b). Entropy alone is not a verdict.

### clip fraction saturation
**First metrics:** `clip_fraction_low` vs `clip_fraction_high` separately.
**Interpretation:** high upper-clip means many tokens want to increase probability more than the trust region allows — the update is being throttled. Persistent saturation means the effective step is smaller than the nominal LR suggests, so raising LR further does nothing except distort the objective.

### response length explosion
**First metrics:** length mean/p95, `truncation_rate`, reward-vs-length correlation.
**Hypotheses:** (a) length is being rewarded as a proxy for correctness; (b) the model is degenerating into repetition.
**Falsify:** compute Pearson/Spearman of reward vs length. Read the longest trajectories — repetition is obvious on sight.

### zero-variance groups dominate
**First metrics:** `all_correct_group_ratio` vs `all_wrong_group_ratio`.
**Interpretation:** these have **opposite** fixes. All-correct means the data is too easy for this policy — make it harder. All-wrong means it is too hard *or* the reward pipeline is broken — check verifier states before concluding difficulty.

### grad norm spike
**First metrics:** `grad_norm`, `advantage_max`, `policy_loss`, non-finite checks.
**Hypotheses:** (a) outlier advantage; (b) LR too high; (c) numerical instability.
**Falsify:** if `advantage_max` spiked in the same step, (a). If NaN/Inf appear anywhere, (c) and stop immediately.

### GPU utilization low
**First metrics:** `pct_rollout`, `pct_reward`, `pct_logprob`, `pct_actor_update`, `pct_weight_sync`.
**Interpretation:** never report "GPU util 40%" without attribution. The share table says whether you are generation-bound, verifier-bound, sync-bound or data-bound. For this stack the verifier costs ~0.02 ms/sample and can be excluded immediately.

### rollout suddenly slow
**First metrics:** `t_rollout`, response length, `truncation_rate`, KV-cache pressure, GPU memory.
**Hypotheses:** (a) responses got longer, so there is genuinely more to generate; (b) KV cache thrashing; (c) an engine-level stall.
**Falsify:** normalise by generated tokens — `t_rollout / generated_tokens`. If that is flat, it is (a) and nothing is wrong.

### training stall
**Detection:** adaptive — `max(3 x median update time, 10 min)`. A fixed 5-minute alarm is wrong for RL, where one legitimate update can exceed it.
**Captured automatically:** process snapshot, `nvidia-smi`, `ray status`, last 200 log lines, telemetry window.
**Never** auto-restart. A restart destroys the evidence.

### checkpoint / resume mismatch
**First metrics:** restored `global_step`, LR, optimizer state, policy version, first post-resume metrics.
**Trap:** a resumed run whose LR schedule silently restarted looks like a mysterious mid-run behaviour change. Always compare the first post-resume update against the last pre-resume one, and keep the `RESTART` boundary visible in every figure.

---

## Environment traps specific to this node

- **Split-routed network** — `github.com` / `huggingface.co` need the academic proxy;
  `download.pytorch.org` is blocked *by* it (INC-001).
- **`git` subprocesses do not inherit proxy env vars** — configure per-host in git's own
  config (INC-002).
- **PyPI is ~40 KB/s here and `uv sync --frozen` ignores `UV_DEFAULT_INDEX`** (INC-003).
- **System disk is 30 GB** — every cache is redirected to the data disk; `>80%` is a stop.
- **Data disk writes at ~226 MB/s** — full checkpoints cost minutes, not seconds.

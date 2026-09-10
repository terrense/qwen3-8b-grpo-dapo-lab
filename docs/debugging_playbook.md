# RL Debugging Playbook

The rule this lab runs on:

> **`reward` is not ground truth.** A reward curve is the *output* of a long pipeline. When it
> moves the wrong way, the policy is the **last** thing to suspect, not the first.

## When any of these fire, do not touch a hyperparameter yet

- reward suddenly drops
- reward plateaus
- KL explodes
- entropy collapses
- clip fraction spikes
- gradient norm spikes
- response length explodes
- truncation rate suddenly rises

## Check in this order

| # | Layer | What to actually look at |
|---|---|---|
| 1 | **environment / verifier** | verifier success rate, timeout rate, exception rate, latency p95. A verifier that silently returns 0 on timeout looks exactly like a policy collapse. |
| 2 | **data batch composition** | which prompts entered this batch; did the sampler drift to a harder/easier shard; duplicate prompts |
| 3 | **truncation / length cap** | truncation rate, response length p95 vs `max_response_length`. A length cap cutting off answers destroys reward without any policy change. |
| 4 | **policy-version / rollout freshness** | which policy version generated these rollouts vs which one is being updated |
| 5 | **old logprob correctness** | is `old_logprob` the generating policy's, recomputed or cached; importance ratio should sit near 1.0 at the first inner epoch |
| 6 | **token masking** | prompt tokens excluded, response tokens included, padding masked, truncated responses flagged |
| 7 | **reward normalisation / group variance** | zero-std group ratio, all-correct vs all-wrong split |
| 8 | **optimizer** | grad norm, clipping, Adam state after a resume |
| 9 | **LR** | schedule position, warmup, whether a restart reset it |
| 10 | **KL / clipping** | coefficient, clip ratio, low/high clip fractions |

Only after 1-9 are cleared is it legitimate to conclude "this is an optimization problem".

## Signatures worth memorising

| Symptom | Usually means |
|---|---|
| reward drops, truncation rate rises at the same step | length cap, not policy degradation (see R3-truncation) |
| reward drops, verifier timeout rate rises | verifier contamination (see R3-verifier-timeout) |
| zero-std group ratio near 1.0 | GRPO has no gradient signal — dataset too easy or too hard for this policy |
| importance ratio far from 1.0 on the first inner epoch | stale rollouts or an old-logprob bug |
| grad norm spike + KL spike + clipfrac spike together | LR too high (see R3-high-lr) |
| entropy collapse with flat reward | premature determinism, not learning |
| GPU util low, rollout wall time dominant | generation-bound, not training-bound — tune rollout, not the optimizer |

## Environment-level traps specific to this node

- **Split-routed network.** `github.com`/`huggingface.co` need the academic proxy;
  `download.pytorch.org` is blocked *by* it. See INC-001.
- **`git` subprocesses do not reliably inherit proxy env vars.** Configure the proxy per-host in
  git's own config. See INC-002.
- **System disk is 30 GB.** Every cache (HF, pip, uv, triton, torch extensions, Ray temp, wandb)
  is redirected to the 1.3 TB data disk. Check `df -h /` before any large operation; over 80 % is
  a stop condition.
- **Data disk writes at ~226 MB/s.** Full checkpoints are minutes, not seconds. Keep at most two
  resumable checkpoints.

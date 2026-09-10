# Experiment Plan

## What this is, and what it is not

This is a **GRPO systems pilot / rehearsal** on `Qwen/Qwen3-8B`, run directly on the released
post-trained checkpoint. The author's medical paper design is
`Qwen3-8B -> SFT (M1) -> DPO (M2) -> GRPO (M3)`; **nothing produced here is that M3** and nothing
here may be reported as it. The deliverable of this project is *engineering understanding and a
reproducible record*, not a checkpoint and not a SOTA number.

Out of scope for this project: SFT, DPO, VL models, medical multi-turn environments.

## Stage gates

Each stage must pass before the next starts. Any failure is diagnosed before anything is re-run —
"just try again" is not an allowed response to a failure.

| Gate | Requirement |
|---|---|
| G0 hardware | 4 GPUs visible, BF16 OK, NCCL correctness OK, NCCL bandwidth sane, disks sane |
| G1 environment | VeRL env built from the official lock; import + CUDA + vLLM smoke test passes |
| G2 data | DAPO-Math-17k + AIME 2024 prepared, schema confirmed against the real files |
| G3 verifier | verifier unit test passes with distinct states; latency recorded |
| G4 reward distribution | fixed-policy rollout shows a workable zero-std group ratio |
| G5 R0 | 2-5 optimizer updates end-to-end, full data flow verified |
| G6 R1 | vanilla GRPO baseline stable and instrumented |
| G7 R2 | DAPO matched-budget vs R1 |
| G8 R3 | three failure injections, each on a *copy*, never on the baseline |

## Runs

### R0 — GRPO smoke test
Prove the pipeline, not the science. Small prompt batch (8-16), `rollout.n=4`, short responses,
2-5 optimizer updates. The exit criterion is that one full update's data flow has been verified
end to end:

`prompt ids -> rollout policy version -> generated response -> reward -> group mean/std ->
normalised advantage -> old logprob -> recomputed current logprob -> importance ratio -> clipping
-> KL -> policy loss -> grad norm -> optimizer step`

plus explicit confirmation that prompt tokens are excluded from the policy loss, response tokens
included, padding masked, truncated responses flagged. On OOM: record peak VRAM **and the stage it
occurred in** (rollout / logprob / reference / backward / optimizer / weight sync) before changing
anything. Enabling CPU offload is not the first response to OOM.

### R1 — vanilla GRPO baseline
Long enough to read dynamics, not to converge. Target: prompt batch ~32, `rollout.n=8`,
`max_response_length` ~4096, temperature 1.0, LR at the official example's order of magnitude
(`1e-6`). Checkpoint at 20 updates for a staged review, then continue toward 80-150 updates if the
GPU-hour estimate justifies it.

Metrics recorded every step, grouped as: optimization (loss, LR, grad norm) / GRPO (reward mean &
std, group std, zero-std ratio, advantage distribution) / policy shift (KL, clipfrac low & high,
importance-ratio quantiles) / exploration (entropy) / generation (length mean, median, p95, max,
truncation rate, EOS rate, format success) / correctness (train reward accuracy, validation
accuracy, all-correct, all-wrong, mixed group ratios) / systems (rollout, verifier, ref-logprob,
update and checkpoint seconds, tokens/s, GPU util, VRAM, RAM, disk).

Trajectories are saved periodically — high reward, low reward, sudden-change, longest, and
truncated. Averages alone are not enough to diagnose anything.

### R2 — DAPO matched budget
Only after R1 is healthy. Four mechanisms, each read from the *current* source rather than from
memory of the paper: Clip-Higher (asymmetric clipping, ~0.2 low / ~0.28 high), Dynamic Sampling,
token-level policy-gradient loss aggregation, and overlong reward shaping.

Matched against R1 on: model, dataset, seed policy, approximate optimizer updates, prompt
distribution, validation protocol. Because dynamic sampling changes real rollout cost, generated
tokens, wall-clock **and** GPU-hours are all reported alongside the reward. The deliverable is
`analysis/R1_vs_R2.md`, which must explain *why* the dynamics differ — "DAPO reward is higher" is
not an acceptable conclusion.

### R3 — failure injection
Run on copies of the stable config. **Never on the baseline**, and never sharing code or output
directories with it.

- **R3-A truncation contamination** — `max_response_length` 4096 -> 1024, short matched run.
  Question: is the reward drop actually policy degradation?
- **R3-B excessive LR** — 3-5x the stable LR, few updates, stop the moment divergence is clear.
  Watch grad norm, KL, clipfrac, importance ratio, entropy.
- **R3-C verifier timeout contamination** — deterministic hash-seeded failure of ~5 % of verifier
  requests. Phase 1 deliberately mishandles it as `timeout -> reward 0` and measures how that
  poisons group std, zero-std ratio and advantages; phase 2 fixes it to mask/invalidate the sample
  with bounded retries, and compares.

## Logging design for the medical paper

Even though this pilot is math, the log schema is chosen so it transfers to medical multi-turn
GRPO unchanged: total reward, reward components, group variance, KL, clipping fraction, entropy,
response length, termination reason, and reward-hacking evidence.

Every run persists: run id, VeRL commit, model revision, dataset revision, seed, fully resolved
config, hardware snapshot, environment lock, checkpoint ids, and restart boundaries. **Two
restarted runs are never stitched into one curve** — a restart must leave a visible boundary.

## Checkpoint policy

At most 2 full resumable checkpoints (latest + previous), plus a small number of model-only ones.
Disk is checked before every save. Checkpoint/resume is itself an experiment: save, terminate the
trainer cleanly, resume, and verify that global step, optimizer state, LR schedule, policy
weights, RNG state and data position all came back correctly.

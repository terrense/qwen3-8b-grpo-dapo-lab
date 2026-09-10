# R3_truncation — Failure injection: truncation contamination

> **Status: NOT RUN.** This file is the pre-registered plan. Every section from
> "Observed behavior" onward is filled in only from real run output. Nothing here may be
> reported as a result until it has actually executed.

## Objective

Determine whether a reward drop caused by a length cap is distinguishable from genuine policy degradation.

## Hypothesis

Cutting max_response_length from the stable value to 1024 drops reward sharply while entropy and KL stay comparatively calm; truncation rate and response-length p95 identify the true cause.

## Why this experiment exists

This is the most common misdiagnosis in LLM RL: a systems or config fault read as the model getting worse.

## Independent variables

max_response_length: stable value -> 1024. Everything else matched.

## Controlled variables

Model Qwen/Qwen3-8B, dataset BytedTsinghua-SIA/DAPO-Math-17k, validation AIME 2024,
seed policy, seed, prompt distribution, validation protocol and hardware are held fixed unless
listed as independent above.

## Hardware

4 x NVIDIA H20 96GB, full NV18 NVLink mesh. See [../../docs/hardware_environment.md](../../docs/hardware_environment.md).

## Model

Qwen/Qwen3-8B post-trained release. Revision recorded in [../../manifests/model.md](../../manifests/model.md).

## Dataset

See [../../manifests/dataset.md](../../manifests/dataset.md).

## Full training command

```bash
# TO BE FILLED from the actual launch
```

## Resolved configuration

resolved_config.yaml in this directory, written by the trainer rather than by hand.

## Expected behavior

Truncation rate jumps immediately at step 1; answer correctness falls while format correctness holds up; KL stays modest.

## Metrics

results_summary.md plus this run CSV/JSONL under analysis/. Figures are regenerated
from those files by [../../scripts/plotting/](../../scripts/plotting/) into figures/generated/.

## Observed behavior

NOT RUN.

## Failure / incident

NOT RUN.

## Root-cause hypotheses

NOT RUN.

## Evidence

NOT RUN.

## Confirmed root cause

NOT RUN.

## Fix

NOT RUN.

## Post-fix evidence

NOT RUN.

## Conclusion

NOT RUN.

## What I learned

NOT RUN.

## Next experiment

R3_high_lr.

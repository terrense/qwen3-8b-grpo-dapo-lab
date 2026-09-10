# R1_grpo_baseline — Vanilla GRPO baseline

> **Status: NOT RUN.** This file is the pre-registered plan. Every section from
> "Observed behavior" onward is filled in only from real run output. Nothing here may be
> reported as a result until it has actually executed.

## Objective

Produce a vanilla GRPO run long enough to read real training dynamics, and to act as the matched-budget control for R2.

## Hypothesis

Vanilla GRPO on DAPO-Math-17k shows a non-trivial zero-variance group ratio that caps the usable gradient signal, and rollout dominates wall time over the actor update.

## Why this experiment exists

It is the control. Without a healthy, fully instrumented vanilla baseline, no DAPO claim in R2 is interpretable.

## Independent variables

None. This is the reference configuration.

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

Reward rises slowly; entropy declines; a measurable fraction of groups are all-correct or all-wrong and contribute no advantage.

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

R2_dapo, matched on budget.
